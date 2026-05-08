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

        self._role_name = self._actor.attributes.get('role_name', 'Unknow')
        # Agent1 may either be AI-driven (no trajectory) or follow a timed
        # trajectory like a replay actor. The actual mode is decided at run
        # time by whether `self._times` ends up populated; obstacle logic is
        # gated on `not use_timed_trajectory` so it auto-disables in
        # follow-trajectory mode.
        self._use_timed_trajectory = self._role_name != 'Ego' and bool(self._times)

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

    def run_step(self):
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
        if self._actor is None or not self._actor.is_alive:
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

        role_name = self._role_name
        use_timed_trajectory = self._use_timed_trajectory and bool(self._times)

        control = None
        if not use_timed_trajectory:
            # keep local planner in sync with xosc target speed (km/h)
            self._local_planner.set_speed(self._target_speed * 3.6)

            control = self._local_planner.run_step(debug=False)

            # Check if the actor reached the end of the plan. Only consider this
            # AFTER the actor has been activated — LocalPlanner.done() is True
            # for an empty plan, so before the trigger fires we'd otherwise
            # mark non-Ego actors as finished while they're still parked.
            if (not use_timed_trajectory
                    and self._setinitsp
                    and self._local_planner.done()):
                self._reached_goal = True

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
            # waypoint, finish — otherwise the controller keeps driving towards a point that's
            # behind the actor and the angular-velocity loop spins it 180° (= circling bug).
            if sim_time >= self._times[-1] and self._waypoints:
                last_wp_location = self._waypoints[-1].location
                last_wp_distance = last_wp_location.distance(self._actor.get_location())
                last_wp_signed = self._signed_distance_to_location(last_wp_location)
                if last_wp_signed < 0.0 or last_wp_distance < self._waypoint_reached_threshold:
                    self._reached_goal = True
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
            # 1. 衰減邏輯：如果超過 0.5 秒沒有新的偵測事件，視為障礙物已消失
            if hasattr(self, '_last_obstacle_timestamp'):
                if current_time - self._last_obstacle_timestamp > 0.5:
                    self._obstacle_distance = float('inf')
                    self._obstacle_actor = None

            # 2. 判斷是否需要煞車
            # 使用 self._proximity_threshold 作為觸發點（建議設為 10.0 ~ 15.0）
            if self._obstacle_distance < self._proximity_threshold:
                distance = max(self._obstacle_distance, 0.01)  # 避免除以零

                # A. 緊急煞車：距離太近 (例如小於 5 公尺
                if distance < 5.0:
                    target_speed = 0

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
            velocity_xy = np.array([velocity.x, velocity.y])

            if not self._use_timed_trajectory:
                # set_target_velocity bypasses tyre dynamics, so a velocity that
                # points at the next waypoint while the body is mid-rotation
                # produces visible side-slip / drift. Project the linear velocity
                # onto the body's forward axis (bicycle-model style) — the yaw
                # rate below does the steering.
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                forward_norm = forward_xy / (np.linalg.norm(forward_xy) + 1e-6)
                velocity.x = forward_norm[0] * speed
                velocity.y = forward_norm[1] * speed
            elif np.dot(velocity_xy, forward_xy) < 0.0:
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                if speed < 1e-3:
                    speed = max(self._target_speed, self._speed_limit, 0.0)
                forward_norm = forward_xy / (np.linalg.norm(forward_xy) + 1e-6)
                velocity.x = forward_norm[0] * speed
                velocity.y = forward_norm[1] * speed

            # Preserve vertical velocity so gravity isn't overwritten each tick
            # (otherwise the actor floats when it leaves the road).
            velocity.z = self._actor.get_velocity().z
            self._actor.set_target_velocity(velocity)

        # Replay agents (timed trajectory, no obstacle handling) cannot follow
        # sharp trajectory turns within the 50°/s yaw cap; they slide sideways,
        # leave the road and the off-road guard kills them. For these actors,
        # snap the yaw to the future trajectory tangent so the body forward
        # axis matches the velocity direction we just set.

        # set new angular velocity
        current_yaw = CarlaDataProvider.get_transform(self._actor).rotation.yaw
        # When we have a waypoint list, use the direction between the waypoints to calculate the heading (change)
        # otherwise use the waypoint heading directly
        if self._waypoints:
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
            # Ego/Agent1 use forward-projected velocity (bicycle model), so the
            # only way they can negotiate a corner is via yaw rate. Give them a
                            # higher cap; other actors keep the original 50°/s.
            max_yaw_rate = 90.0 if not self._use_timed_trajectory else 50.0
            yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, yaw_rate))

            # Low-pass filter yaw_rate for Ego/Agent1 to remove the head jitter
            # caused by step changes in delta_yaw on waypoint switching and at
            # the direction_norm < 0.5 cliff.
            if not self._use_timed_trajectory:
                if not hasattr(self, '_prev_yaw_rate'):
                    self._prev_yaw_rate = 0.0
                alpha = 0.3
                yaw_rate = alpha * yaw_rate + (1 - alpha) * self._prev_yaw_rate
                self._prev_yaw_rate = yaw_rate

            angular_velocity.z = yaw_rate
        self._actor.set_target_angular_velocity(angular_velocity)

        self._last_update = current_time

        return direction_norm