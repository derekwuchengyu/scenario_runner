#!/usr/bin/env python

# Copyright (c) 2020-2021 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example control for vehicles which
does not use CARLA's vehicle engine.

Limitations:
- Does not respect any traffic regulation: speed limit, priorities, etc.
- Can only consider obstacles in forward facing reaching (i.e. in tight corners obstacles may be ignored).
"""

import math
import sys
import numpy as np
import traceback

import carla

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.actorcontrols.visualizer import Visualizer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.tools.util import strtobool
from agents.navigation.basic_agent import LocalPlanner
from agents.navigation.local_planner import RoadOption


# ---------------------------------------------------------------------------
# Throttle CarlaDataProvider's "Actor(id=X) not found!" stdout spam.
# ---------------------------------------------------------------------------
# Diagnosis showed the spam source is atomic_behaviors.py:566
# (ChangeActorTargetSpeed.update) — when an actor is destroyed (collision /
# off-road / etc.) that atomic keeps polling CarlaDataProvider.get_location
# every tick before py_trees stops it, and the unconditional `print('...not
# found!')` inside CarlaDataProvider floods stdout with thousands of
# identical lines.
#
# We patch the three accessor staticmethods at module-import time so a
# missing actor is logged at most once per id per process, then silenced.
# Behavior is otherwise preserved: cached path delegates to the original
# (unchanged); missing-actor path returns the same fallback (None / 0.0)
# the original would have returned.
#
# The patch lives here (not in carla_data_provider.py) to keep the change
# scoped to this controller — installation is a side-effect of importing
# this module, which always happens during scenario build before any
# atomic ticks.
# ---------------------------------------------------------------------------
def _install_missing_actor_warning_throttle():
    if getattr(CarlaDataProvider, '_missing_actor_throttle_installed', False):
        return
    CarlaDataProvider._warned_missing_actor_ids = set()

    def _wrap(method_name, map_attr_name, fallback):
        original = getattr(CarlaDataProvider, method_name)

        def wrapped(actor):
            actor_map = getattr(CarlaDataProvider, map_attr_name)
            # Cached path: delegate to original — no print, returns cache.
            if actor in actor_map:
                return original(actor)
            # Same-id-different-handle path: also delegate (original loops
            # the map by id and returns correctly without printing).
            actor_id = getattr(actor, 'id', None)
            if actor_id is not None:
                for key in actor_map:
                    if getattr(key, 'id', None) == actor_id:
                        return original(actor)
            # Genuinely missing: warn at most once per id, then silent.
            if actor_id is not None and actor_id not in CarlaDataProvider._warned_missing_actor_ids:
                CarlaDataProvider._warned_missing_actor_ids.add(actor_id)
                try:
                    actor_repr = str(actor)
                except Exception:
                    actor_repr = 'Actor(id={})'.format(actor_id)
                print('srunner.scenariomanager.carla_data_provider.{}: {} '
                      'not found! (further warnings for this actor suppressed)'.format(
                          method_name, actor_repr))
            return fallback

        wrapped.__name__ = method_name
        wrapped.__wrapped__ = original
        setattr(CarlaDataProvider, method_name, staticmethod(wrapped))

    _wrap('get_location', '_actor_location_map', None)
    _wrap('get_transform', '_actor_transform_map', None)
    _wrap('get_velocity', '_actor_velocity_map', 0.0)

    # Reset the warned-set on cleanup so a subsequent scenario in the same
    # process can re-warn once per missing actor (CARLA actor ids reset
    # between scenarios, so without this the second run would silently
    # collide on previously-warned ids).
    if not getattr(CarlaDataProvider, '_cleanup_resets_warned_set', False):
        original_cleanup = CarlaDataProvider.cleanup

        def cleanup_with_warn_reset():
            CarlaDataProvider._warned_missing_actor_ids.clear()
            return original_cleanup()

        cleanup_with_warn_reset.__wrapped__ = original_cleanup
        CarlaDataProvider.cleanup = staticmethod(cleanup_with_warn_reset)
        CarlaDataProvider._cleanup_resets_warned_set = True

    CarlaDataProvider._missing_actor_throttle_installed = True


_install_missing_actor_warning_throttle()


class SimpleVehicleControl(BasicControl):

    """
    Controller class for vehicles derived from BasicControl.

    The controller directly sets velocities in CARLA, therefore bypassing
    CARLA's vehicle engine. This allows a very precise speed control, but comes
    with limitations during cornering.

    In addition, the controller can consider blocking obstacles, which are
    classified as dynamic (i.e. vehicles, bikes, pedestrians). Activation of this
    features is controlled by passing proper arguments to the class constructor.
    The collision detection uses CARLA's obstacle sensor (sensor.other.obstacle),
    which checks for obstacles in the direct forward channel of the vehicle, i.e.
    there are limitation with sideways obstacles and while cornering.

    The controller can also respect red traffic lights, and use a given deceleration
    value for more realistic behavior. Both behaviors are activated via the corresponding
    controller arguments.

    Args:
        actor (carla.Actor): Vehicle actor that should be controlled.
        args (dictionary): Dictonary of (key, value) arguments to be used by the controller.
                           May include: (consider_obstacles, true/false)     - Enable consideration of obstacles
                                        (proximity_threshold, distance)      - Distance in front of actor in which
                                                                               obstacles are considered
                                        (waypoint_reached_threshold, distance) - Distance between actor and waypoint
                                                                               determining reached or not
                                        (consider_trafficlights, true/false) - Enable consideration of traffic lights
                                        (max_deceleration, float)            - Use a reasonable deceleration value for
                                                                               this vehicle
                                        (max_acceleration, float)            - Use a reasonable acceleration value for
                                                                               this vehicle
                                        (attach_camera, true/false)          - Attach OpenCV display to actor
                                                                               (useful for debugging)

    Attributes:

        _generated_waypoint_list (list of carla.Transform): List of target waypoints the actor
            should travel along. A waypoint here is of type carla.Transform!
            Defaults to [].
        _last_update (float): Last time step the update function (tick()) was called.
            Defaults to None.
        _consider_obstacles (boolean): Enable/Disable consideration of obstacles
            Defaults to False.
        _proximity_threshold (float): Distance in front of actor in which obstacles are considered
            Defaults to infinity.
        _waypoint_reached_threshold(float): Distance between actor and waypoint determining reached or not
            Defaults to 4.0 meters.
        _cv_image (CV Image): Contains the OpenCV image, in case a debug camera is attached to the actor
            Defaults to None.
        _camera (sensor.camera.rgb): Debug camera attached to actor
            Defaults to None.
        _obstacle_sensor (sensor.other.obstacle): Obstacle sensor attached to actor
            Defaults to None.
        _obstacle_distance (float): Distance of the closest obstacle returned by the obstacle sensor
            Defaults to infinity.
        _obstacle_actor (carla.Actor): Closest obstacle returned by the obstacle sensor
            Defaults to None.
    """

    def __init__(self, actor, args=None):
        super(SimpleVehicleControl, self).__init__(actor)
        self._generated_waypoint_list = []
        self._last_update = None
        self._consider_traffic_lights = False
        self._consider_obstacles = False
        self._proximity_threshold = 15
        self._waypoint_reached_threshold = 1
        self._max_acceleration = 5
        self._max_deceleration = 3
        self._speed_limit = 40 / 3.6

        self._obstacle_sensor = None
        self._obstacle_distance = float('inf')
        self._obstacle_actor = None

        self._visualizer = None

        self._brake_lights_active = False

        self._setinitsp = False
        self._prev_target_speed = 0.0
        self._start_time = None
        self._has_explicit_waypoint_plan = bool(self._waypoints)

        self._role_name = self._actor.attributes.get('role_name', 'Unknow')
        # An actor is treated as a timed-replay agent when its motion is fully
        # described by a (time, waypoint) Polyline trajectory and we just play
        # it back. Two classes of actor are explicitly NOT timed-replay:
        #   - Ego: driven by an external agent / planner.
        #   - Any role with a NURBS spec registered by the parser (e.g. Agent1
        #     in this scenario): it self-drives along the baked dense polyline
        #     through LocalPlanner + obstacle sensing, just like Ego.
        # This gating is critical for obstacle braking, which is disabled in
        # timed-replay mode (line ~158 + the `not use_timed_trajectory` guard
        # in _set_new_velocity).
        nurbs_roles = set()
        try:
            from srunner.tools.openscenario_parser import OpenScenarioParser
            nurbs_roles = set(OpenScenarioParser.nurbs_spec_by_role.keys())
        except Exception:
            pass
        self._use_timed_trajectory = (
            self._role_name != 'Ego' and self._role_name not in nurbs_roles
        )

        # Logging mode: 'info' (default) shows runtime status,
        # 'debug' additionally prints verbose per-tick traces.
        self._log_mode = 'info'
        if args and 'log_mode' in args:
            mode = str(args['log_mode']).strip().lower()
            if mode in ('info', 'debug'):
                self._log_mode = mode

        # Per-actor log filter. `log_actors` may be a comma-separated string
        # (e.g. "Ego,Agent1") or an iterable of role names. Special values
        # 'all' / '' / None mean "log every actor". Case-insensitive match.
        self._log_actors = None  # None == log all
        if args and 'log_actors' in args:
            raw = args['log_actors']
            if isinstance(raw, str):
                names = [n.strip() for n in raw.split(',') if n.strip()]
            else:
                names = [str(n).strip() for n in raw if str(n).strip()]
            if names and not (len(names) == 1 and names[0].lower() == 'all'):
                self._log_actors = {n.lower() for n in names}
        self._log_enabled_for_actor = (
            self._log_actors is None
            or self._role_name.lower() in self._log_actors
        )

        if self._actor.attributes['role_name'] != 'Ego':
            self._actor.set_simulate_physics(False)
            self._actor.set_location(self._actor.get_location() + carla.Location(z=-100))

        if not self._use_timed_trajectory:
            self._log_info("Actor {} with role name {} is using SimpleVehicleControl, activating obstacle consideration.".format(self._actor.id, self._role_name))
            args['consider_obstacles'] = 'True'

        if args and 'consider_obstacles' in args and strtobool(args['consider_obstacles']):
            self._consider_obstacles = True
            bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.other.obstacle')
            if 'proximity_threshold' in args:
                self._proximity_threshold = float(args['proximity_threshold'])
            bp.set_attribute('distance', '100')
            bp.set_attribute('hit_radius', '1.5')
            bp.set_attribute('only_dynamics', 'True')
            bp.set_attribute('sensor_tick', '0.05')
            self._obstacle_sensor = CarlaDataProvider.get_world().spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=self._actor.bounding_box.extent.x, z=1.0)
                ),
                attach_to=self._actor,
            )
            self._obstacle_sensor.listen(lambda event: self._on_obstacle(event))  # pylint: disable=unnecessary-lambda

        if args and 'consider_trafficlights' in args and strtobool(args['consider_trafficlights']):
            self._consider_traffic_lights = True

        if args and 'waypoint_reached_threshold' in args:
            self._waypoint_reached_threshold = float(args['waypoint_reached_threshold'])

        if args and 'max_deceleration' in args:
            self._max_deceleration = float(args['max_deceleration'])

        if args and 'max_acceleration' in args:
            self._max_acceleration = float(args['max_acceleration'])

        if args and 'attach_camera' in args and strtobool(args['attach_camera']):
            self._visualizer = Visualizer(self._actor)

        self._log_info("Initialized SimpleVehicleControl for actor {} with role name {}".format(self._actor.id, self._role_name))


        self._local_planner = LocalPlanner(  # pylint: disable=undefined-variable
            self._actor, opt_dict={
                'target_speed': self._target_speed * 3.6,
                })

        if self._waypoints:
            self._update_plan()

    def _update_plan(self):
        """
        Update the plan (waypoint list) of the LocalPlanner
        """
        plan = []
        for transform in self._waypoints:
            waypoint = CarlaDataProvider.get_map().get_waypoint(
                transform.location, project_to_road=True, lane_type=carla.LaneType.Any)
            plan.append((waypoint, RoadOption.LANEFOLLOW))
        self._local_planner.set_global_plan(plan)

    def update_waypoints(self, waypoints, times=None, start_time=None):
        """Override BasicControl.update_waypoints to bake NURBS control points into a
        dense polyline before storing them.

        The OSC parser registers a NURBS spec (order/weights/knots) per role_name in
        OpenScenarioParser.nurbs_spec_by_role at scenario-build time. When the
        FollowTrajectoryAction's atomic eventually fires, ChangeActorWaypoints calls
        this method with the control point Positions already converted to
        carla.Transform. We treat them as NURBS control points, sample the curve
        roughly every metre and replace _waypoints with the dense polyline so the
        downstream waypoint-following logic in run_step can drive the actor along
        the curve.
        """
        spec = None
        try:
            from srunner.tools.openscenario_parser import OpenScenarioParser
            spec = OpenScenarioParser.nurbs_spec_by_role.get(self._role_name)
        except Exception:
            spec = None

        if spec is not None and len(waypoints) >= spec.get('order', 0):
            dense = self._bake_nurbs_polyline(waypoints, spec)
            super().update_waypoints(dense, times=None, start_time=start_time)
            self._times = []
            try:
                # Keep the LocalPlanner in sync so its done() flag won't
                # prematurely terminate the actor in run_step.
                self._update_plan()
            except Exception:
                pass
            self._log_info(
                "[NURBS] {} baked {} control points -> {} dense waypoints".format(
                    self._role_name, len(waypoints), len(dense)))
        else:
            super().update_waypoints(waypoints, times=times, start_time=start_time)

        self._has_explicit_waypoint_plan = bool(self._waypoints)

        # OSC Polyline trajectories occasionally encode an unwrapped initial
        # heading that contradicts the trajectory's actual direction of travel
        # (e.g. spawn faces NW while the polyline marches east). Without a
        # correction, signed_distance to the next waypoint stays negative,
        # the ReverseGuard in _set_new_velocity zeros target_speed, and
        # _setinitsp never flips True — so set_simulate_physics is never
        # called and the actor sits frozen at the spawn pose until teardown.
        # Realign the body to the first non-trivial segment direction once,
        # only when no motion has started yet.
        self._maybe_snap_yaw_to_first_segment()

    def _maybe_snap_yaw_to_first_segment(self):
        """Snap actor yaw to face the first non-trivial trajectory segment.

        Conservative: only fires for timed-replay agents that haven't started
        moving, never for Ego, and only when the misalignment exceeds 90° so
        agents with sane spawn headings are left untouched.
        """
        if self._role_name == 'Ego' or self._setinitsp:
            return
        if not self._use_timed_trajectory or not self._times:
            return
        if not self._waypoints or len(self._waypoints) < 2 or self._actor is None:
            return

        p0 = self._waypoints[0].location
        seg_dx = seg_dy = 0.0
        for i in range(1, len(self._waypoints)):
            dx = self._waypoints[i].location.x - p0.x
            dy = self._waypoints[i].location.y - p0.y
            if math.hypot(dx, dy) >= 0.5:
                seg_dx, seg_dy = dx, dy
                break
        if seg_dx == 0.0 and seg_dy == 0.0:
            return

        target_yaw = math.degrees(math.atan2(seg_dy, seg_dx))
        try:
            current_transform = self._actor.get_transform()
        except RuntimeError:
            return
        current_yaw = current_transform.rotation.yaw
        delta = target_yaw - current_yaw
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
        if abs(delta) <= 90.0:
            return

        new_rotation = carla.Rotation(
            pitch=current_transform.rotation.pitch,
            yaw=target_yaw,
            roll=current_transform.rotation.roll,
        )
        try:
            self._actor.set_transform(
                carla.Transform(current_transform.location, new_rotation))
        except RuntimeError:
            return
        self._log_info(
            "[YawSnap] {} spawn yaw {:.1f} misaligned with first segment {:.1f} "
            "(delta={:.1f}) - snapped to {:.1f}".format(
                self._role_name, current_yaw, target_yaw, delta, target_yaw))

    @staticmethod
    def _nurbs_basis(t, i, k, knots):
        """Cox-de Boor recursion for B-spline basis function N_{i,k}(t).

        Mirrors esmini's RoadManager.cpp NurbsShape::CoxDeBoor — half-open intervals,
        zero-denominator branches dropped.
        """
        if k == 1:
            return 1.0 if (knots[i] <= t < knots[i + 1]) else 0.0
        den1 = knots[i + k - 1] - knots[i]
        den2 = knots[i + k] - knots[i + 1]
        val = 0.0
        if den1 > 0.0:
            val += ((t - knots[i]) / den1) * SimpleVehicleControl._nurbs_basis(t, i, k - 1, knots)
        if den2 > 0.0:
            val += ((knots[i + k] - t) / den2) * SimpleVehicleControl._nurbs_basis(t, i + 1, k - 1, knots)
        return val

    def _eval_nurbs_point(self, t, ctrl_xyz, weights, knots, order):
        """Evaluate the rational B-spline at parameter t. Returns (x, y, z)."""
        eps = 1e-9
        t = max(knots[0], min(t, knots[-1] - eps))
        bases = [SimpleVehicleControl._nurbs_basis(t, i, order, knots)
                 for i in range(len(ctrl_xyz))]
        rw = sum(b * w for b, w in zip(bases, weights))
        if rw < eps:
            return ctrl_xyz[0]
        x = y = z = 0.0
        for (px, py, pz), b, w in zip(ctrl_xyz, bases, weights):
            c = b * w / rw
            x += c * px
            y += c * py
            z += c * pz
        return (x, y, z)

    def _bake_nurbs_polyline(self, ctrl_transforms, spec, step_len=1.0):
        """Sample the NURBS curve at ~step_len metre spacing and return a list of
        carla.Transform whose yaw points to the next sample (last sample inherits
        from a forward-extrapolated tangent).
        """
        order = int(spec['order'])
        weights = list(spec['weights'])
        knots = list(spec['knots'])
        ctrl_xyz = [(t.location.x, t.location.y, t.location.z) for t in ctrl_transforms]
        if len(weights) != len(ctrl_xyz):
            weights = [1.0] * len(ctrl_xyz)

        # Knot vector validity: needs len(ctrl) + order entries for a clamped curve.
        if len(knots) != len(ctrl_xyz) + order or len(ctrl_xyz) < order:
            self._log_info(
                "[NURBS] {} invalid spec (cps={}, order={}, knots={}); using control "
                "points as polyline".format(
                    self._role_name, len(ctrl_xyz), order, len(knots)))
            return list(ctrl_transforms)

        # Rough chord length to choose sample count.
        rough = 0.0
        for i in range(1, len(ctrl_xyz)):
            rough += math.hypot(ctrl_xyz[i][0] - ctrl_xyz[i - 1][0],
                                ctrl_xyz[i][1] - ctrl_xyz[i - 1][1])
        n_steps = max(int(1 + rough / step_len), 4)
        t_max = knots[-1]
        p_step = t_max / n_steps

        samples = [self._eval_nurbs_point(i * p_step, ctrl_xyz, weights, knots, order)
                   for i in range(n_steps + 1)]

        transforms = []
        for i, (x, y, z) in enumerate(samples):
            if i < len(samples) - 1:
                nx, ny, _ = samples[i + 1]
            else:
                px, py, _ = samples[i - 1]
                nx, ny = 2 * x - px, 2 * y - py
            yaw = math.degrees(math.atan2(ny - y, nx - x))
            transforms.append(carla.Transform(carla.Location(x=x, y=y, z=z),
                                              carla.Rotation(yaw=yaw)))
        return transforms

    def _on_obstacle(self, event):
        """
        Callback for the obstacle sensor

        Sets _obstacle_distance and _obstacle_actor according to the closest obstacle
        found by the sensor.
        """
        if not event:
            return

        self._log_debug(f"DEBUG: {self._actor.attributes.get('role_name')} 偵測到: {event.other_actor.type_id}, 距離: {event.distance}")

        # 排除偵測到自己
        if event.other_actor.id == self._actor.id:
            return

        # 更新偵測到的最近距離與對象
        self._obstacle_distance = event.distance
        self._obstacle_actor = event.other_actor

        # 記錄最後一次偵測到障礙物的時間戳記，用於衰減邏輯
        self._last_obstacle_timestamp = GameTime.get_time()

    def _log_info(self, msg):
        """Print when log mode is 'info' or 'debug' and this actor is in the log filter."""
        if not self._log_enabled_for_actor:
            return
        if self._log_mode in ('info', 'debug'):
            print(msg)

    def _log_debug(self, msg):
        """Print only when log mode is 'debug', throttled to once per 1s per call site."""
        if self._log_mode != 'debug' or not self._log_enabled_for_actor:
            return
        frame = sys._getframe(1)
        key = (frame.f_code.co_filename, frame.f_lineno)
        now = GameTime.get_time()
        if not hasattr(self, '_debug_throttle'):
            self._debug_throttle = {}
        if now - self._debug_throttle.get(key, -1.0) >= 1:
            print(msg)
            self._debug_throttle[key] = now

    def reset(self):
        """
        Reset the controller
        """
        if self._actor is not None:
            self._log_info(f"--- Actor {self._actor.id} is being RESET/DESTROYED ---")
            if self._log_mode == 'debug':
                traceback.print_stack()  # 印出是誰呼叫了這個銷毀動作
        if self._visualizer:
            self._visualizer.reset()
        if self._obstacle_sensor:
            try:
                self._obstacle_sensor.destroy()
            except RuntimeError:
                pass
            self._obstacle_sensor = None
        if self._actor and self._actor.is_alive:
            self._actor = None

    def _cleanup_actor(self, destroy=True):
        """Tear down sensors and (optionally) destroy the underlying actor.

        Called when the controller decides the actor is finished — e.g. trajectory
        completed, off-road, or already missing in CARLA. Idempotent.
        """
        if self._obstacle_sensor is not None:
            try:
                self._obstacle_sensor.destroy()
            except RuntimeError:
                pass
            self._obstacle_sensor = None

        if destroy and self._actor is not None:
            try:
                if self._actor.is_alive:
                    self._actor.destroy()
            except RuntimeError:
                pass
        self._actor = None

    def _finish_explicit_waypoint_plan(self):
        """Finish a FollowTrajectoryAction whose explicit route has been consumed."""
        self._reached_goal = True
        if self._role_name != 'Ego' and self._setinitsp:
            self._cleanup_actor(destroy=True)

    def _should_finish_explicit_waypoint_plan(self):
        # Agent1 follows a NURBS spec and should keep coasting along the lane
        # past the end of its plan via the auto-generate-from-map branch, so
        # it is explicitly excluded from the "finish + destroy" path.
        return (self._has_explicit_waypoint_plan
                and self._role_name == 'Ego'
                and self._setinitsp)

    def run_step(self):
        """Tick wrapper that catches the "destroyed actor" race.

        CARLA can destroy an actor between two ticks (e.g. via remove_actor_by_id
        from another scenario hook) while ``actor.is_alive`` still reports True
        for one extra tick. Any subsequent ``actor.get_location()`` /
        ``get_transform()`` raises RuntimeError("trying to operate on a
        destroyed actor"), which would crash the entire scenario tree.

        We treat that exception exactly the same as the explicit "actor is
        gone" guard at the top of _run_step_inner: mark the goal reached, drop
        sensors, and let the tree move on.
        """
        try:
            self._run_step_inner()
        except RuntimeError as exc:
            if 'destroyed actor' in str(exc):
                self._log_info(
                    f"[Destroyed] {self._role_name} actor went away mid-tick: {exc}")
                self._reached_goal = True
                self._cleanup_actor(destroy=False)
                return
            raise

    def _run_step_inner(self):
        """
        Execute on tick of the controller's control loop

        If _waypoints are provided, the vehicle moves towards the next waypoint
        with the given _target_speed, until reaching the final waypoint. Upon reaching
        the final waypoint, _reached_goal is set to True.

        If _waypoints is empty, the vehicle moves in its current direction with
        the given _target_speed.

        For further details see :func:`_set_new_velocity`
        """

        # If the actor is gone (destroyed externally, fell off the world, etc.)
        # there is nothing to control — mark the trajectory finished and bail
        # out cleanly so the parent tick doesn't crash on a None location.
        # `is_alive` itself can raise RuntimeError on a fully-destroyed handle,
        # so defensively wrap the check.
        try:
            actor_dead = self._actor is None or not self._actor.is_alive
        except RuntimeError:
            actor_dead = True
        if actor_dead:
            self._reached_goal = True
            self._cleanup_actor(destroy=False)
            return

        # Even when `is_alive` reports True, CarlaDataProvider may have already
        # dropped the actor from its location map (it removes anything whose
        # `is_alive` was False on the previous on_carla_tick — there is a
        # one-tick race). Calling get_location() in that window prints
        # "Actor(id=X) not found!" on stdout *every* tick until is_alive
        # finally flips. Pre-check the map and bail silently if missing.
        loc_map = CarlaDataProvider._actor_location_map
        in_map = self._actor in loc_map
        if not in_map:
            actor_id = self._actor.id
            for key in loc_map:
                if key.id == actor_id:
                    in_map = True
                    break
        if not in_map:
            # CarlaDataProvider has dropped the actor — treat as destroyed
            # and skip the noisy get_location call.
            self._reached_goal = True
            self._cleanup_actor(destroy=False)
            return

        actor_location = CarlaDataProvider.get_location(self._actor)
        if actor_location is None:
            # CARLA can't locate the actor — treat as off-road / destroyed.
            self._reached_goal = True
            self._cleanup_actor(destroy=True)
            return

        # Off-road / fell-through-the-map detection: only after the actor has
        # been activated (set_init_speed). Before that, non-Ego actors are
        # parked at z ≈ -100 by __init__ on purpose (waiting for their trigger),
        # so this guard must not fire during the parking period.
        if (self._setinitsp and actor_location.z < -10.0
                and self._role_name != 'Ego'):
            self._log_info(f"[OffRoad] {self._role_name} dropped to z={actor_location.z:.2f}, destroying.")
            self._reached_goal = True
            self._cleanup_actor(destroy=True)
            return

        # Don't bootstrap motion before the actor is ready. Two cases:
        #   1. Still parked at the __init__ z=-100 drop: enabling physics
        #      here lets the off-road guard destroy us next tick.
        #   2. Timed-replay agent whose FollowTrajectory atomic hasn't yet
        #      called update_waypoints: the OSC SpawnActor TeleportAction
        #      may have set a heading that doesn't match the trajectory
        #      direction (unwrapped heading data), so any motion now drifts
        #      us away from the trajectory.
        # In both cases the SpeedAction may have already set _target_speed>0,
        # and the "no waypoints" map-following branch below would otherwise
        # call set_simulate_physics(True) and start moving us. Just wait.
        # 解決_f16131 Agent3 出場即destroy問題
        if not self._setinitsp and self._role_name != 'Ego':
            if actor_location.z < -10.0:
                return
            if self._use_timed_trajectory and not self._times:
                return

        role_name = self._role_name
        use_timed_trajectory = self._use_timed_trajectory and bool(self._times)

        control = None
        if not use_timed_trajectory:
            # keep local planner in sync with xosc target speed (km/h)
            self._local_planner.set_speed(self._target_speed * 3.6)

            control = self._local_planner.run_step(debug=False)

            # Deliberately do NOT consult `_local_planner.done()` to mark the
            # goal reached — Agent1 (and Ego) should coast along the road past
            # the end of their plan. Once `self._waypoints` empties, the
            # `if not self._waypoints:` branch below auto-generates further
            # waypoints from the map and keeps the actor following the lane.

            # # Agent1 到目的地就消失
            # if (not use_timed_trajectory
            #         and self._setinitsp
            #         and self._local_planner.done()):
            #     self._reached_goal = True
            

        if self._reached_goal:
            # Reached the goal — release the actor so it stops being ticked and
            # downstream code doesn't see a half-alive entity. Keep Ego alive
            # because the StopTrigger's ReachPositionCondition still queries it.
            # Only destroy if the actor had actually started moving; otherwise
            # we'd kill actors whose trigger event hasn't fired yet.
            if role_name != 'Ego' and self._setinitsp:
                self._cleanup_actor(destroy=True)
            return

        if self._visualizer:
            self._visualizer.render()

        self._reached_goal = False

        if use_timed_trajectory and self._waypoints and self._times:
            if self._start_time is None:
                self._start_time = GameTime.get_time()

            sim_time = GameTime.get_time() - self._start_time

            time_index = None
            for i in range(len(self._times)):
                if self._times[i] >= sim_time:
                    time_index = i
                    break

            if time_index is None:
                time_index = len(self._times) - 1

            if time_index >= len(self._waypoints):
                time_index = len(self._waypoints) - 1

            # If trajectory time has elapsed and we've already passed (or reached) the final
            # waypoint, switch to lane-following coast mode — query the CARLA map for the
            # continuation of the current lane and drive towards those waypoints instead of
            # projecting straight forward (which made the actor shoot off-road on any curve).
            # Coast speed = last linear speed; if stopped, fall back to _target_speed.
            # Not passing time_index/sim_time to _set_new_velocity skips the trajectory-time
            # speed recalculation block and uses the coast_speed we wrote into _target_speed.
            # 避免 ego 開得比 real-world raw data慢, agent就消失，ego等 agent 消失再到終點便success
            if sim_time >= self._times[-1] and self._waypoints:
                last_wp_location = self._waypoints[-1].location
                last_wp_distance = last_wp_location.distance(self._actor.get_location())
                last_wp_signed = self._signed_distance_to_location(last_wp_location)
                if last_wp_signed < 0.0 or last_wp_distance < self._waypoint_reached_threshold:
                    velocity = self._actor.get_velocity()
                    coast_speed = math.hypot(velocity.x, velocity.y)
                    if coast_speed < 0.5:
                        coast_speed = max(self._target_speed, 0.0)
                    self._target_speed = coast_speed

                    # replayer 結束繼續自動沿車道走
                    carla_map = CarlaDataProvider.get_map()
                    if not self._generated_waypoint_list:
                        seed_wp = carla_map.get_waypoint(self._actor.get_location())
                    else:
                        seed_wp = carla_map.get_waypoint(self._generated_waypoint_list[-1].location)
                    while seed_wp is not None and len(self._generated_waypoint_list) < 50:
                        next_wps = seed_wp.next(2.0)
                        if not next_wps:
                            break
                        self._generated_waypoint_list.append(next_wps[0].transform)
                        seed_wp = next_wps[0]

                    while (self._generated_waypoint_list and
                           self._generated_waypoint_list[0].location.distance(self._actor.get_location()) < 0.5):
                        self._generated_waypoint_list = self._generated_waypoint_list[1:]

                    if self._generated_waypoint_list:
                        direction_norm = self._set_new_velocity(
                            self._offset_waypoint(self._generated_waypoint_list[0]))
                        if direction_norm < 2.0:
                            self._generated_waypoint_list = self._generated_waypoint_list[1:]
                    else:
                        # 找不到路的話就直走，避免停在原地不動
                        # End of road / no map data — fall back to straight-line coast so
                        # the actor keeps moving instead of stalling.
                        forward_vec = self._actor.get_transform().get_forward_vector()
                        next_location = self._actor.get_location() + carla.Location(
                            x=forward_vec.x * 5.0,
                            y=forward_vec.y * 5.0,
                            z=0.0,
                        )
                        self._set_new_velocity(next_location)
                    return

            original_index = time_index
            if not self._use_timed_trajectory:
                time_index = self._find_ahead_waypoint_index(time_index)
                if time_index != original_index:
                    self._log_debug(f"[ReverseGuard] {self._role_name} shift waypoint index {original_index}->{time_index} at t={sim_time:.2f}")

            direction_norm = self._set_new_velocity(
                self._offset_waypoint(self._waypoints[time_index]),
                time_index=time_index,
                sim_time=sim_time,
            )

            if sim_time >= self._times[-1] and direction_norm < self._waypoint_reached_threshold:
                self._reached_goal = True
            return

        if not self._waypoints:
            if self._should_finish_explicit_waypoint_plan():
                self._finish_explicit_waypoint_plan()
                return

            # No waypoints are provided, so we have to create a list of waypoints internally
            # get next waypoints from map, to avoid leaving the road
            self._reached_goal = False

            map_wp = None
            if not self._generated_waypoint_list:
                map_wp = CarlaDataProvider.get_map().get_waypoint(CarlaDataProvider.get_location(self._actor))
            else:
                map_wp = CarlaDataProvider.get_map().get_waypoint(self._generated_waypoint_list[-1].location)
            while len(self._generated_waypoint_list) < 50:
                map_wps = map_wp.next(2.0)
                if map_wps:
                    self._generated_waypoint_list.append(map_wps[0].transform)
                    map_wp = map_wps[0]
                else:
                    break

            # Remove all waypoints that are too close to the vehicle
            while (self._generated_waypoint_list and
                   self._generated_waypoint_list[0].location.distance(self._actor.get_location()) < 0.5):
                self._generated_waypoint_list = self._generated_waypoint_list[1:]

            if not self._generated_waypoint_list:
                if role_name == 'Ego':
                    # Don't park the Ego — the StopTrigger's ReachPosition needs it
                    # to keep moving, and the TLE timer requires continuous speed>0.
                    # Mirror the else-branch fallback: project a target straight ahead.
                    forward_vec = self._actor.get_transform().get_forward_vector()
                    next_location = actor_location + carla.Location(
                        x=forward_vec.x * 5.0,
                        y=forward_vec.y * 5.0,
                        z=0.0,
                    )
                    self._set_new_velocity(next_location)
                    return
                # Non-Ego: end of road / no reachable waypoints — finish.
                self._reached_goal = True
                return

            direction_norm = self._set_new_velocity(self._offset_waypoint(self._generated_waypoint_list[0]))
            if direction_norm < 2.0:
                self._generated_waypoint_list = self._generated_waypoint_list[1:]
                
        else:
            # When changing from "free" driving without pre-defined waypoints to a defined route with waypoints
            # it may happen that the first few waypoints are too close to the ego vehicle for obtaining a
            # reasonable control command. Therefore, we drop these waypoints first.
            while self._waypoints and self._waypoints[0].location.distance(self._actor.get_location()) < 0.3:
                self._waypoints = self._waypoints[1:]

            if role_name in ['Ego', 'Agent1']:
                self._drop_waypoints_behind()

            self._reached_goal = False
            if not self._waypoints:
                if self._should_finish_explicit_waypoint_plan():
                    self._finish_explicit_waypoint_plan()
                    return
                self._log_debug(f"{self._role_name}: No more waypoints. Keep moving with current target speed.")
                current_location = CarlaDataProvider.get_location(self._actor)
                forward_vec = self._actor.get_transform().get_forward_vector()
                next_location = current_location + carla.Location(
                    x=forward_vec.x * 5.0,
                    y=forward_vec.y * 5.0,
                    z=0.0,
                )
                direction_norm = self._set_new_velocity(next_location)
            else:
                target_transform = self._waypoints[0]

                direction_norm = self._set_new_velocity(self._offset_waypoint(target_transform))
                if direction_norm < self._waypoint_reached_threshold:
                    self._log_debug(f"DEBUG: {self._role_name} reached waypoint. Remaining: {len(self._waypoints)}")
                    self._waypoints = self._waypoints[1:]
                    if not self._waypoints:
                        if self._should_finish_explicit_waypoint_plan():
                            self._finish_explicit_waypoint_plan()
                            return
                        self._log_debug(f"{self._role_name}: No more waypoints. Keep moving with current target speed.")
                        current_location = CarlaDataProvider.get_location(self._actor)
                        forward_vec = self._actor.get_transform().get_forward_vector()
                        next_location = current_location + carla.Location(
                            x=forward_vec.x * 5.0,
                            y=forward_vec.y * 5.0,
                            z=0.0,
                        )
                        direction_norm = self._set_new_velocity(next_location)

        

    def _offset_waypoint(self, transform):
        """
        Given a transform (which should be the position of a waypoint), displaces it to the side,
        according to a given offset

        Args:
            transform (carla.Transform): Transform to be moved

        returns:
            offset_location (carla.Transform): Moved transform
        """
        if self._offset == 0:
            offset_location = transform.location
        else:
            right_vector = transform.get_right_vector()
            offset_location = transform.location + carla.Location(x=self._offset*right_vector.x,
                                                                  y=self._offset*right_vector.y)

        return offset_location

    def _signed_distance_to_location(self, location):
        current_location = self._actor.get_location()
        forward_vec = self._actor.get_transform().get_forward_vector()
        direction = location - current_location
        return direction.x * forward_vec.x + direction.y * forward_vec.y

    def _find_ahead_waypoint_index(self, start_index, max_lookahead=10):
        if not self._waypoints:
            return start_index

        end_index = min(len(self._waypoints), start_index + max_lookahead + 1)
        for i in range(start_index, end_index):
            if self._signed_distance_to_location(self._waypoints[i].location) >= 0.0:
                return i
        return start_index


    def _drop_waypoints_behind(self):
        """Drop waypoints that are behind the vehicle to avoid reverse motion."""
        if not self._waypoints:
            return

        current_location = self._actor.get_location()
        forward_vec = self._actor.get_transform().get_forward_vector()

        while self._waypoints:
            direction = self._waypoints[0].location - current_location
            signed_distance = direction.x * forward_vec.x + direction.y * forward_vec.y
            if signed_distance >= 0.0:
                break
            self._waypoints = self._waypoints[1:]

    def _set_new_velocity(self, next_location, time_index=None, sim_time=None):
        """
        Calculate and set the new actor veloctiy given the current actor
        location and the _next_location_

        If _consider_obstacles is true, the speed is adapted according to the closest
        obstacle in front of the actor, if it is within the _proximity_threshold distance.
        If _consider_trafficlights is true, the vehicle will enforce a stop at a red
        traffic light.
        If _max_deceleration is set, the vehicle will reduce its speed according to the
        given deceleration value.
        If the vehicle reduces its speed, braking lights will be activated.

        Args:
            next_location (carla.Location): Next target location of the actor

        returns:
            direction (carla.Vector3D): Length of direction vector of the actor
        """
        # Defensive: actor may have been destroyed between run_step()'s entry guard
        # and reaching this method (e.g. via reset() called from another thread).
        if self._actor is None or not self._actor.is_alive:
            self._reached_goal = True
            return 0.0
        # Times have been specified, modify the speed accordingly


        # if self._times:
        #     # print(self._role_name, self._times)
        #     if self._start_time is not None:
        #         plan_len = len(self._local_planner.get_plan())
        #         current_index = len(self._waypoints) - plan_len
        #         print(f"  DEBUG: {self._role_name} [{GameTime.get_time():<5.2f}]: target_time: {self._times[current_index]}")
        #         if current_index < len(self._times):
        #             target_time = self._times[current_index]
        #             delta_time = target_time - (GameTime.get_time() - self._start_time)
        #             target_location = self._local_planner.get_plan()[0][0].transform.location
        #             target_distance = self._actor.get_location().distance(target_location)
        #             self._target_speed = target_distance / max(delta_time, 0.001)

        use_timed_trajectory = self._use_timed_trajectory and bool(self._times)

        if use_timed_trajectory and self._start_time is not None and time_index is not None and sim_time is not None:
            target_time = self._times[time_index]

            # === 2. 時間差 ===
            delta_time = max(target_time - sim_time, 0.01)

            # === 3. 位置 ===
            current_location = self._actor.get_location()
            target_location = next_location

            signed_distance = self._signed_distance_to_location(target_location)

            target_speed = signed_distance / delta_time  # m/s
            self._target_speed = target_speed

            if signed_distance < 0.0 and self._role_name not in ['Ego', 'Agent1']:
                if not hasattr(self, '_last_reverse_debug_time'):
                    self._last_reverse_debug_time = -1.0
                if sim_time - self._last_reverse_debug_time > 0.5:
                    self._log_info(f"[ReverseGuard] {self._role_name} behind target idx={time_index} sd={signed_distance:.2f} t={sim_time:.2f}")
                    self._last_reverse_debug_time = sim_time
                self._target_speed = 0.0

            self._target_speed = max(self._target_speed, 0.0)

            # === DEBUG ===
            if self._log_mode == 'debug':
                if self._role_name in ['Agent1', 'Ego']:
                    self._log_debug(
                        f"[{GameTime.get_time():.2f}] "
                        f"idx={time_index} "
                        f"t={target_time:.2f} "
                        f"v={self._target_speed:.2f} m/s "
                        f"sd={signed_distance:.2f}m "
                        f"dt={delta_time:.3f}s"
                        f" loc=({current_location.x:.2f}, {current_location.y:.2f}) -> "
                        f"({target_location.x:.2f}, {target_location.y:.2f})"
                    )
                else:
                    pass
                    # self._log_debug(f"  DEBUG: {self._role_name} Starting timer at {GameTime.get_time():.2f}")

        if self._role_name in ['Ego'] and self._log_mode == 'debug':
            signed_distance = self._signed_distance_to_location(next_location)
            self._log_debug(
                f"[{GameTime.get_time():.2f}] "
                f"v={self._target_speed:.2f} m/s "
                f"sd={signed_distance:.2f}m "
                f" loc=({self._actor.get_location().x:.2f}, {self._actor.get_location().y:.2f}) -> "
                f"({next_location.x:.2f}, {next_location.y:.2f})"
            )

        current_time = GameTime.get_time()
        if not self._use_timed_trajectory:
            target_speed = self._target_speed
        else:
            target_speed = min(self._target_speed, self._speed_limit)
        # # 限制 target_speed 的變化量：根據上一次的 target speed 與最大加/減速度
        # dt = current_time - self._last_update if self._last_update else 0.05
        # lower_bound = self._prev_target_speed - self._max_deceleration * dt
        # upper_bound = self._prev_target_speed + self._max_acceleration * dt
        # # clamp target_speed 到允許範圍
        # target_speed = max(lower_bound, min(target_speed, upper_bound))
        #

        



        if not self._last_update:
            self._last_update = current_time

        current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

        if self._consider_obstacles and not use_timed_trajectory:
            # --- 障礙物衰減與煞車邏輯 ---
            # 1. 衰減邏輯：超過 0.2 秒沒有新的偵測事件即視為障礙物已消失。
            # Sensor tick = 0.05s → 4 ticks 沒有 event 才衰減，仍能避開單 tick race，
            # 但 resume 反應從原本 0.5s 縮到 0.2s。
            decayed_this_tick = False
            if hasattr(self, '_last_obstacle_timestamp'):
                if current_time - self._last_obstacle_timestamp > 0.2:
                    if self._obstacle_actor is not None or self._obstacle_distance < float('inf'):
                        decayed_this_tick = True
                    self._obstacle_distance = float('inf')
                    self._obstacle_actor = None

            # 2. 判斷是否需要煞車
            # 使用 self._proximity_threshold 作為觸發點（建議設為 10.0 ~ 15.0）
            brake_branch = None
            other_role = None
            other_type = None
            current_speed_other = None
            if self._obstacle_distance < self._proximity_threshold:
                distance = max(self._obstacle_distance, 0.01)  # 避免除以零
                if self._obstacle_actor is not None:
                    try:
                        other_role = self._obstacle_actor.attributes.get('role_name', '?')
                        other_type = self._obstacle_actor.type_id
                    except RuntimeError:
                        other_role = '<destroyed>'

                # A. 緊急煞車：距離太近 (例如小於 5 公尺
                if distance < 5.0:
                    target_speed = 0
                    brake_branch = 'EMERGENCY'

                # B. 線性減速：在感應範圍內根據距離調整速度
                else:
                    other_velocity = self._obstacle_actor.get_velocity()
                    current_speed_other = math.sqrt(other_velocity.x**2 + other_velocity.y**2)

                    # 如果我們比對方快，則需要減速
                    if current_speed > current_speed_other:
                        # 簡單的線性插值：距離越近，速度越接近對方的速度
                        gap_ratio = (distance - 5) / (self._proximity_threshold - 5)
                        gap_ratio = max(-1, min(1, gap_ratio))

                        # 目標速度 = 對方速度 + (原本目標速度與對方速度的差額 * 距離權重)
                        target_speed = current_speed_other + (target_speed - current_speed_other) * gap_ratio
                        brake_branch = 'LINEAR_DECEL'
                    else:
                        brake_branch = 'NO_ADJUST'

            # Throttled obstacle-state log (~1 Hz per actor) + always-on transition
            # events. Helps debug "Agent1 brakes and stays stopped" without flooding.
            if not hasattr(self, '_last_obstacle_log_time'):
                self._last_obstacle_log_time = -1.0
                self._prev_brake_branch = None
            transition = (brake_branch != self._prev_brake_branch) or decayed_this_tick
            if transition or (current_time - self._last_obstacle_log_time) > 1.0:
                if brake_branch is not None:
                    self._log_debug(
                        f"[Obstacle] {self._role_name} br={brake_branch} "
                        f"dist={self._obstacle_distance:.2f} thr={self._proximity_threshold:.1f} "
                        f"other=({other_role}/{other_type}) "
                        f"v_self={current_speed:.2f} v_other={current_speed_other if current_speed_other is not None else 'n/a'} "
                        f"tgt_speed={target_speed:.2f}"
                    )
                elif decayed_this_tick:
                    self._log_debug(
                        f"[Obstacle] {self._role_name} DECAYED (no sensor event >0.5s) — resuming"
                    )
                self._last_obstacle_log_time = current_time
            self._prev_brake_branch = brake_branch

            self._log_debug(f"  Distance: {self._obstacle_distance}, Threshold: {self._proximity_threshold}")

        if self._consider_traffic_lights and not use_timed_trajectory:
            if (self._actor.is_at_traffic_light() and
                    self._actor.get_traffic_light_state() == carla.TrafficLightState.Red):
                target_speed = 0

        speed_epsilon = 0.1

        if target_speed + speed_epsilon < current_speed:
            if not self._brake_lights_active:
                self._brake_lights_active = True
                light_state = self._actor.get_light_state()
                light_state |= carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if self._max_deceleration is not None:
                if not self._use_timed_trajectory:
                    self._log_debug(f"  Applying deceleration: {(current_time - self._last_update) * self._max_deceleration:.2f} m/s^2")
                target_speed = max(target_speed, current_speed - (current_time -
                                                                  self._last_update) * self._max_deceleration)
        else:
            if self._brake_lights_active:
                self._brake_lights_active = False
                light_state = self._actor.get_light_state()
                light_state &= ~carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if self._max_acceleration is not None:
                lower_bound = 1 if current_speed < 1 else 0.0
                delta_speed = max((current_time - self._last_update) * self._max_acceleration, lower_bound)
                tmp_speed = min(target_speed, current_speed + delta_speed)
                target_speed = tmp_speed

        self._log_debug(f"[{GameTime.get_time():<5.2f}]{self._role_name} Target speed: {target_speed} Current speed: {current_speed}")

        # set new linear velocity
        velocity = carla.Vector3D(0, 0, 0)
        actor_loc = CarlaDataProvider.get_location(self._actor)
        if actor_loc is None:
            # Actor disappeared mid-tick — abort gracefully.
            self._reached_goal = True
            return 0.0
        direction = next_location - actor_loc
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm > 1e-3:
            velocity.x = direction.x / direction_norm * target_speed
            velocity.y = direction.y / direction_norm * target_speed


        if not self._setinitsp and self._target_speed > 0:
            # 讓agent直接以指定起始速度開始移動，以符合scenario

            forward_vec = self._actor.get_transform().get_forward_vector()
            vx = forward_vec.x * self._target_speed
            vy = forward_vec.y * self._target_speed
            vz = self._actor.get_velocity().z  # 獲取當前垂直速度，保留重力影響
            self._actor.set_target_velocity(carla.Vector3D(vx, vy, vz))
            self._setinitsp = True
            if self._start_time is None:
                self._start_time = GameTime.get_time()
            self._log_info(f"================={self._role_name} set_target_velocity {self._target_speed} {self._actor.get_transform().rotation.yaw} - {vx}, {vy} {self._start_time}=================")
            self._actor.set_simulate_physics(True)
            if self._use_timed_trajectory:
                self._log_debug(f"Actor {self._actor.id} with role name {self._actor.attributes.get('role_name', 'default')} is set to non-collision mode for initial speed setting.")
        else:
            forward_vec = self._actor.get_transform().get_forward_vector()
            forward_xy = np.array([forward_vec.x, forward_vec.y])

            # Bicycle-model velocity: always project linear velocity onto the
            # body's forward axis (magnitude preserved). Otherwise — for timed
            # replay agents in particular — set_target_velocity points the
            # body straight at the next waypoint while the body itself is
            # mid-rotation, causing visible side-slip on corners and the
            # off-road guard then destroys motorcycle-grade tight turns. With
            # bicycle model, the body chases the trajectory tangent below via
            # angular velocity, and motion follows the body. No side-slip.
            # Magnitude must follow the dynamically-computed target_speed
            # (post deceleration / traffic-light / lead-vehicle), NOT the
            # original cruise speed self._target_speed — otherwise an agent
            # that is decelerating to a stop sees target_speed -> 0 collapse
            # velocity.{x,y} to ~0, then the fallback snaps it back up to
            # self._target_speed and the actor lurches forward right before
            # stopping. Using target_speed here keeps the stop smooth while
            # still preserving forward-axis projection (no side-slip on
            # corners, which is what this block was originally added for).
            # 速度向量投影到車輛的前進方向上，並且保持目標速度的大小。這樣可以確保車輛在轉彎時不會出現側滑現象。
            speed = max(target_speed, 0.0)
            forward_norm = forward_xy / (np.linalg.norm(forward_xy) + 1e-6)
            velocity.x = forward_norm[0] * speed
            velocity.y = forward_norm[1] * speed

            # Preserve vertical velocity so gravity isn't overwritten each tick
            # (otherwise the actor floats when it leaves the road).
            velocity.z = self._actor.get_velocity().z
            self._actor.set_target_velocity(velocity)

        # set new angular velocity
        current_yaw = CarlaDataProvider.get_transform(self._actor).rotation.yaw

        # Heading target.
        # For timed-replay agents we deliberately use the SEGMENT tangent
        # (waypoints[i] - waypoints[i-1]) instead of the actor->waypoint
        # direction. Segment tangent depends only on the trajectory geometry,
        # so on a straight section it is constant and the body cannot twitch;
        # near a tight waypoint it cannot flip (the cause of the previous
        # "spin in place" regression on motorcycles).
        target_heading = None
        if (self._use_timed_trajectory
                and time_index is not None
                and time_index >= 1
                and self._waypoints
                and time_index < len(self._waypoints)):
            prev_loc = self._waypoints[time_index - 1].location
            cur_loc = self._waypoints[time_index].location
            tdx = cur_loc.x - prev_loc.x
            tdy = cur_loc.y - prev_loc.y
            if math.hypot(tdx, tdy) > 1e-3:
                target_heading = math.degrees(math.atan2(tdy, tdx))

        if target_heading is not None:
            delta_yaw = target_heading - current_yaw
        elif self._waypoints:
            delta_yaw = math.degrees(math.atan2(direction.y, direction.x)) - current_yaw
        else:
            new_yaw = CarlaDataProvider.get_map().get_waypoint(next_location).transform.rotation.yaw
            delta_yaw = new_yaw - current_yaw

        if math.fabs(delta_yaw) > 360:
            delta_yaw = delta_yaw % 360
        if delta_yaw > 180:
            delta_yaw = delta_yaw - 360
        elif delta_yaw < -180:
            delta_yaw = delta_yaw + 360

        angular_velocity = carla.Vector3D(0, 0, 0)
        if target_speed < 0.5 or direction_norm < 0.5:
            angular_velocity.z = 0
        else:
            turn_time = max(direction_norm / max(target_speed, 0.1), 0.05)
            yaw_rate = delta_yaw / turn_time

            # Cap selection:
            #   Non-timed (Ego / Agent1-NURBS): fixed 90°/s.
            #   Timed-replay: base 25°/s for straights (eliminates twitch),
            #     dynamically lifted up to 180°/s when |delta_yaw| is large
            #     so motorcycles can actually rotate through tight corners
            #     without sliding sideways into the off-road guard.
            if self._use_timed_trajectory:
                max_yaw_rate = max(25.0, min(180.0, abs(delta_yaw) * 4.0))
            else:
                max_yaw_rate = 90.0
            yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, yaw_rate))

            # Universal low-pass filter to suppress per-tick noise on both
            # the segment-transition jumps (timed agents) and the LocalPlanner
            # waypoint hand-offs (Ego/Agent1).
            if not hasattr(self, '_prev_yaw_rate'):
                self._prev_yaw_rate = 0.0
            alpha = 0.3 if not self._use_timed_trajectory else 0.4
            yaw_rate = alpha * yaw_rate + (1 - alpha) * self._prev_yaw_rate
            self._prev_yaw_rate = yaw_rate

            angular_velocity.z = yaw_rate
        self._actor.set_target_angular_velocity(angular_velocity)

        self._last_update = current_time

        return direction_norm
