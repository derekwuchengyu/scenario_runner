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
        self._use_timed_trajectory = self._role_name not in ['Ego', 'Agent1']


        if self._actor.attributes['role_name'] != 'Ego':
            self._actor.set_simulate_physics(False)
            self._actor.set_location(self._actor.get_location() + carla.Location(z=-100))
        
        if self._role_name in ['Ego', 'Agent1']:
            print("Actor {} with role name {} is using SimpleVehicleControl, activating obstacle consideration.".format(self._actor.id, self._role_name))
            args['consider_obstacles'] = 'True'


        if args and 'consider_obstacles' in args and strtobool(args['consider_obstacles']):
            self._consider_obstacles = strtobool(args['consider_obstacles'])
            bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.other.obstacle')
            if args and 'proximity_threshold' in args:
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
            self._consider_traffic_lights = strtobool(args['consider_trafficlights'])

        if args and 'waypoint_reached_threshold' in args:
            self._waypoint_reached_threshold = float(args['waypoint_reached_threshold'])

        if args and 'max_deceleration' in args:
            self._max_deceleration = float(args['max_deceleration'])

        if args and 'max_acceleration' in args:
            self._max_acceleration = float(args['max_acceleration'])

        if args and 'attach_camera' in args and strtobool(args['attach_camera']):
            self._visualizer = Visualizer(self._actor)

        # print Actor.role_name using Simple Control
        print("Initialized SimpleVehicleControl for actor {} with role name {}".format(self._actor.id, self._role_name))


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
        
        print(f"DEBUG: {self._actor.attributes.get('role_name')} 偵測到: {event.other_actor.type_id}, 距離: {event.distance}")
        
        # 排除偵測到自己
        if event.other_actor.id == self._actor.id:
            return

        # 更新偵測到的最近距離與對象
        self._obstacle_distance = event.distance
        self._obstacle_actor = event.other_actor
        
        # 記錄最後一次偵測到障礙物的時間戳記，用於衰減邏輯
        self._last_obstacle_timestamp = GameTime.get_time()

    def reset(self):
        """
        Reset the controller
        """
        print(f"--- DEBUG: Actor {self._actor.id} is being RESET/DESTROYED ---")
        traceback.print_stack() # 這會印出是誰呼叫了這個銷毀動作
        if self._visualizer:
            self._visualizer.reset()
        if self._obstacle_sensor:
            self._obstacle_sensor.destroy()
            self._obstacle_sensor = None
        if self._actor and self._actor.is_alive:
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

        # if self._actor is None or not self._actor.is_alive:
        #     return
        role_name = self._role_name
        use_timed_trajectory = self._use_timed_trajectory and bool(self._times)

        control = None
        if not use_timed_trajectory:
            # keep local planner in sync with xosc target speed (km/h)
            if role_name in ['Ego', 'Agent1']:
                self._local_planner.set_speed(self._target_speed * 3.6)

            control = self._local_planner.run_step(debug=False)

            # Check if the actor reached the end of the plan
            if role_name not in ['Ego', 'Agent1'] and self._local_planner.done():
                self._reached_goal = True

        if self._reached_goal:
            # Reached the goal, so stop
            # velocity = carla.Vector3D(0, 0, 0)
            # self._actor.set_target_velocity(velocity)
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

            original_index = time_index
            if self._role_name not in ['Ego', 'Agent1']:
                time_index = self._find_ahead_waypoint_index(time_index)
                if time_index != original_index:
                    print(f"[ReverseGuard] {self._role_name} shift waypoint index {original_index}->{time_index} at t={sim_time:.2f}")

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
                print("Agent5: No more waypoints. Keep moving with current target speed.")
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
                    print(f"DEBUG: Agent5 reached waypoint. Remaining: {len(self._waypoints)}")
                    self._waypoints = self._waypoints[1:]
                    if not self._waypoints:
                        print("Agent5: No more waypoints. Keep moving with current target speed.")
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
        # Times have been specified, modify the speed accordingly


        # if self._times:
        #     # print(self._actor.attributes.get('role_name', 'Unknow'), self._times)
        #     if self._start_time is not None:
        #         plan_len = len(self._local_planner.get_plan())
        #         current_index = len(self._waypoints) - plan_len
        #         print(f"  DEBUG: {self._actor.attributes.get('role_name', 'Unknow')} [{GameTime.get_time():<5.2f}]: target_time: {self._times[current_index]}")
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
                    print(f"[ReverseGuard] {self._role_name} behind target idx={time_index} sd={signed_distance:.2f} t={sim_time:.2f}")
                    self._last_reverse_debug_time = sim_time
                self._target_speed = 0.0

            self._target_speed = max(self._target_speed, 0.0)

            # === DEBUG ===
            if self._actor.attributes.get('role_name', 'Unknow') in ['Agent1','Agent2']:
                print(
                    f"[{GameTime.get_time():.2f}] "
                    f"idx={time_index} "
                    f"t={target_time:.2f} "
                    f"v={self._target_speed:.2f} m/s "
                    f"sd={signed_distance:.2f}m "
                    f"dt={delta_time:.3f}s"
                    f" loc=({current_location.x:.2f}, {current_location.y:.2f}) -> "
                    f"({target_location.x:.2f}, {target_location.y:.2f})"
                )
            # else:
            #     print(f"  DEBUG: {self._actor.attributes.get('role_name', 'Unknow')} Starting timer at {GameTime.get_time():.2f}")
            #     self._start_time = GameTime.get_time()

        current_time = GameTime.get_time()
        if self._role_name in ['Ego', 'Agent1']:
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
            # If distance is less than the proximity threshold, adapt velocity
            current_time = GameTime.get_time()
            target_speed = self._target_speed if self._role_name in ['Ego', 'Agent1'] else min(self._target_speed, self._speed_limit)
            current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

            # --- 障礙物衰減與煞車邏輯 ---
            if self._consider_obstacles:
                # 1. 衰減邏輯：如果超過 0.5 秒沒有新的偵測事件，視為障礙物已消失
                if hasattr(self, '_last_obstacle_timestamp'):
                    if current_time - self._last_obstacle_timestamp > 0.5:
                        self._obstacle_distance = float('inf')
                        self._obstacle_actor = None

                # 2. 判斷是否需要煞車
                # 使用 self._proximity_threshold 作為觸發點（建議設為 10.0 ~ 15.0）
                if self._obstacle_distance < self._proximity_threshold:
                    distance = max(self._obstacle_distance, 0.01) # 避免除以零
                    
                    # A. 緊急煞車：距離太近 (例如小於 3 米)
                    if distance < 3.0:
                        target_speed = 0
                    
                    # B. 線性減速：在感應範圍內根據距離調整速度
                    else:
                        # 獲取對方的速度
                        other_velocity = self._obstacle_actor.get_velocity()
                        current_speed_other = math.sqrt(other_velocity.x**2 + other_velocity.y**2)
                        
                        # 如果我們比對方快，則需要減速
                        if current_speed > current_speed_other:
                            # 簡單的線性插值：距離越近，速度越接近對方的速度
                            gap_ratio = (distance - 3.0) / (self._proximity_threshold - 3.0)
                            gap_ratio = max(0, min(1, gap_ratio)) # 限制在 0~1
                            
                            # 目標速度 = 對方速度 + (原本目標速度與對方速度的差額 * 距離權重)
                            target_speed = current_speed_other + (target_speed - current_speed_other) * gap_ratio
                else:
                    # 距離足夠遠，維持原速
                    pass
                if self._actor.attributes.get('role_name', 'Unknow') in ['Agent5']:
                    print(f"  Distance: {self._obstacle_distance}, Threshold: {self._proximity_threshold}")

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
                if self._actor.attributes.get('role_name', 'Unknow') in ['Agent5']:
                    print(f"  Applying deceleration: {(current_time - self._last_update) * self._max_deceleration:.2f} m/s^2")
                target_speed = max(target_speed, current_speed - (current_time -
                                                                  self._last_update) * self._max_deceleration)
        else:
            if self._brake_lights_active:
                self._brake_lights_active = False
                light_state = self._actor.get_light_state()
                light_state &= ~carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if self._max_acceleration is not None:
                # print("delta speed ", (current_time -self._last_update) * self._max_acceleration)
                lower_bound = 1 if current_speed < 1 else 0.0
                delta_speed = max((current_time -self._last_update) * self._max_acceleration, lower_bound)
                tmp_speed = min(target_speed, current_speed + delta_speed)
                target_speed = tmp_speed

        if self._actor.attributes.get('role_name', 'Unknow') in ['Agent5'] or True:
            # Print debug info every 0.5 seconds
            if not hasattr(self, '_last_print_time'):
                self._last_print_time = current_time
            if current_time - self._last_print_time >= 0.5:
                print(f"[{GameTime.get_time():<5.2f}]{self._actor.attributes.get('role_name', 'Unknow')} Target speed: {target_speed} Current speed: {current_speed}")
                self._last_print_time = current_time

        # set new linear velocity
        velocity = carla.Vector3D(0, 0, 0)
        direction = next_location - CarlaDataProvider.get_location(self._actor)
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm > 1e-3:
            velocity.x = direction.x / direction_norm * target_speed
            velocity.y = direction.y / direction_norm * target_speed


        if not self._setinitsp and self._target_speed > 0:

            forward_vec = self._actor.get_transform().get_forward_vector()
            vx = forward_vec.x * self._target_speed
            vy = forward_vec.y * self._target_speed
            vz = self._actor.get_velocity().z # 獲取當前垂直速度，保留重力影響
            self._actor.set_target_velocity(carla.Vector3D(vx, vy, vz))
            self._setinitsp = True
            if self._start_time is None:
                self._start_time = GameTime.get_time()
            print(f"==============================={self._actor.attributes.get('role_name', 'Unknow')} set_target_velocity {self._target_speed} {self._actor.get_transform().rotation.yaw} - {vx}, {vy} {self._start_time}====================================")
            self._actor.set_simulate_physics(True)
            # if self._actor.attributes.get('role_name', 'Unknow') not in ['Ego', 'Agent1']:
            #     print(f"Actor {self._actor.id} with role name {self._actor.attributes.get('role_name', 'default')} is set to non-collision mode for initial speed setting.")
            #     # self._actor.set_collisions(False)
            #     # self._actor.set_simulate_physics(True)
        else:
            forward_vec = self._actor.get_transform().get_forward_vector()
            forward_xy = np.array([forward_vec.x, forward_vec.y])
            velocity_xy = np.array([velocity.x, velocity.y])

            if np.dot(velocity_xy, forward_xy) < 0.0:
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                if speed < 1e-3:
                    speed = max(self._target_speed, self._speed_limit, 0.0)
                forward_norm = forward_xy / (np.linalg.norm(forward_xy) + 1e-6)
                velocity.x = forward_norm[0] * speed
                velocity.y = forward_norm[1] * speed

            self._actor.set_target_velocity(velocity)

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
            max_yaw_rate = 50 #if role_name in ['Ego', 'Agent1'] else 35.0
            yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, yaw_rate))
            angular_velocity.z = yaw_rate
        self._actor.set_target_angular_velocity(angular_velocity)

        self._last_update = current_time

        return direction_norm