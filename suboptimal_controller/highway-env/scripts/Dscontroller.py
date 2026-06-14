import numpy as np
from typing import List, Tuple, Dict, Union
# the advantage of using this DS controller is that,
#1  this DS controller focus on finding the backward gap 
#2  
#3 
#4 
class DSLaneChangeController:
    def __init__(self,v0=15, T=4, a_max=1.5, b=2.0, delta=4.0, s0=2.0, 
                 t_safe=2.0, a_comfort=1.0, max_lane_index: int = 2, lane_width: float = 4.0, ds_threshold: float =50.0):
        # this ds_threshold means that if ds value is larger then certain value, might needs to consider not to make a lane change 
        self.v0 = v0
        self.T = T
        self.a_max = a_max
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.t_safe = t_safe
        self.a_comfort = a_comfort
        self.max_lane_index = max_lane_index
        self.lane_width = lane_width
        self.vehicle_length = 5
        self.safety_lane_change_TTC = 2.0  # Safety time threshold for lane change
        self.ds_threshold = ds_threshold  # Threshold for DS to justify a lane change,if the dsvalue is larger then this threshold, this lane change maneuver is too difficult
        self.lane_bound=[-2,2,6,10]
        self.safety_distance=15

    def parse_state(self, state: List[float]) -> Tuple[Tuple[float, float, float, int], List[Tuple[float, float, float, int]]]:
        ego_presence, ego_x, ego_y, ego_vx, ego_vy = state[:5]
        ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
        ego_lane_index = self.get_lane_index(ego_y)
        ego_state = (ego_x, ego_y, ego_speed, ego_lane_index)

        surrounding_vehicles = []
        for i in range(1, 5):
            start_idx = i * 5
            presence, x, y, vx, vy = state[start_idx:start_idx + 5]
            if presence:
                speed = np.sqrt(vx**2 + vy**2)
                lane_index = self.get_lane_index(y)
                surrounding_vehicles.append((x, y, speed, lane_index))
        
        return ego_state, surrounding_vehicles
    
    def get_lane_index(self,y_value):
        for i in range(len(self.lane_bound) - 1):
            if self.lane_bound[i] < y_value <= self.lane_bound[i + 1]:
                return i
        # If y is not within any defined range, return -1 (indicating outside known lanes).
        return -1       
    
    def compute_ds(self, ego_state, target_vehicle) -> float:
        #print(ego_state) 
        #print(target_vehicle)
        x_c, _, v_c, _= ego_state
        x_t, _, v_t, _= target_vehicle

        relative_speed = v_c - v_t
        safety_distance = self.safety_distance#(relative_speed * self.t_safe) / 2+ 10
        #print(safety_distance)
        distance_gap = abs(x_c - x_t)
        safety_distance += distance_gap
       # print(safety_distance)
        return safety_distance
    
    def classify_vehicles(self, 
                          ego_state: Tuple[float, float, float, int], 
                          surrounding_vehicles: List[Tuple[float, float, float, int]]) -> Dict[str, List[Tuple[float, float, float, int, float]]]:
        classified_vehicles = {
            "left_behind": [], "right_behind": [],
            "left_adjacent": [], "right_adjacent": [],
            "left_forward": [], "right_forward": [],
            "behind": [], "forward": []
        }
        #print(ego_state)
        ego_x, _, ego_speed, ego_lane_index = ego_state
        safety_distance = self.vehicle_length

        for vehicle in surrounding_vehicles:
            x, y, speed, lane_index = vehicle
            relative_speed = speed - ego_speed
            longitudinal_distance = x - ego_x

            if lane_index == ego_lane_index + 1:  # Right lane
                if longitudinal_distance > safety_distance:
                    classified_vehicles["right_forward"].append((x, y, speed, lane_index))
                elif longitudinal_distance < -safety_distance:
                    classified_vehicles["right_behind"].append((x, y, speed, lane_index))
                else:
                    classified_vehicles["right_adjacent"].append((x, y, speed, lane_index))
            
            elif lane_index == ego_lane_index - 1:  # Left lane
                if longitudinal_distance > safety_distance:
                    classified_vehicles["left_forward"].append((x, y, speed, lane_index))
                elif longitudinal_distance < -safety_distance:
                    classified_vehicles["left_behind"].append((x, y, speed, lane_index))
                else:
                    classified_vehicles["left_adjacent"].append((x, y, speed, lane_index))

            elif lane_index == ego_lane_index:  # Same lane
                if longitudinal_distance > 0:
                    classified_vehicles["forward"].append((x, y, speed, lane_index))
                elif longitudinal_distance < 0:
                    classified_vehicles["behind"].append((x, y, speed, lane_index))
        #print(classified_vehicles)
        # here if there are multiple vehicle on the lane, reorder the vehicle by longitudinal relative distance to the ego vehicle 
        return classified_vehicles
    def calculate_ds(self, front_vehicle, vehicles_in_target_lane: List[Tuple[float, float, float, int]]) -> float:
        """
        Calculate DS value to determine if the lane change is possible.
        Iteratively check gaps until an acceptable one is found.
        :param ego_state: Ego vehicle's state (x, y, speed, lane_index).
        :param front_vehicle: The vehicle directly in front of the ego in the target lane.
        :param vehicles_in_target_lane: List of vehicles in the target lane.
        :return: DS value. A value of 0 means a safe lane change is possible.
        """

        # If there are no other vehicles in the target lane, the lane change is automatically safe.
        if not vehicles_in_target_lane:
            return 0

        # Sort the vehicles in the target lane based on their longitudinal position (x value) from high to low.
        vehicles_in_target_lane.sort(key=lambda v: v[0], reverse=True)

        # Start with the front vehicle and look for gaps with successive vehicles.
        last_vehicle = front_vehicle

        for next_vehicle in vehicles_in_target_lane:
            # Calculate the gap between the last vehicle and the next vehicle.
            gap = next_vehicle[0] - last_vehicle[0]
            safety_gap = self.safety_distance # 
            if gap > 0: # 
                continue
            gap=abs(gap)
            # If the gap is large enough for the ego vehicle to fit, calculate DS.
            if gap >= safety_gap:
                # DS is the safety distance required for a safe lane change.
                #print(next_vehicle)
                ds_value = self.compute_ds(front_vehicle, next_vehicle)
                return ds_value

            # Update last_vehicle for the next iteration.
            last_vehicle = next_vehicle

        # If no suitable gap is found, calculate DS using the gap between the last vehicle and the next.
        #print(last_vehicle)
        ds_value = self.compute_ds(front_vehicle, last_vehicle)
        # If DS is within the safe limit, return 0 indicating a safe lane change.
        if ds_value <= self.vehicle_length:
            return 0

        return ds_value


    def evaluate_lane_change(self, ego_state, classified_vehicles) -> Union[str, None]:
        # ds value can not be larger then self.threshold, if larger then this threshold, then 
        best_decision = 0
        min_ds_value = 10000# stands for inf
        ego_ds_value=25
        
        front_key="forward"
        front_vehicles=classified_vehicles.get(front_key, []) 
        front_vehicles.sort(key=lambda v: v[0], reverse=False) 
        closest_front_vehicle = front_vehicles[0] if front_vehicles else None
        if closest_front_vehicle:
            ego_ds_value=closest_front_vehicle[0]- ego_state[0]
            if ego_ds_value>self.ds_threshold:   # free lane 
                best_decision=0

                #min_ds_value=self.ds_threshold
            else:
                # Check both left and right lanes for potential lane changes.
                for lane_offset, lane_label in [(1, "right")]:
                    
                    target_lane_index = ego_state[3] + lane_offset
                    if target_lane_index < 0 or target_lane_index > self.max_lane_index: #target lane not available
                        best_decision=0
                        #min_ds_value=self.ds_threshold
                    else:
                        forward_key = f"{lane_label}_forward"
                        behind_key = f"{lane_label}_behind"
                        adjacent_key = f"{lane_label}_adjacent"
                        vehicles_in_target_lane = classified_vehicles.get(forward_key, []) + classified_vehicles.get(adjacent_key, [])+ classified_vehicles.get(behind_key, [])
                        if vehicles_in_target_lane==None: # target lane free
                            min_ds_value=self.safety_distance
                            #print(f"lane change {lane_label}")
                            best_decision = lane_offset
                        else:
                            ds_value = self.calculate_ds(closest_front_vehicle, vehicles_in_target_lane)  # if free,re
                            if ds_value >self.ds_threshold: 
                                # motivation is not enough, 1 take too much effort to lane change, 2 not maintaining the safety distance 
                                best_decision=0
                                #min_ds_value=self.ds_threshold
                            else:
                            # now here, both ds_value and the ego ds value are smaller then threshold, we are determined to lane change 
                                if ds_value<min_ds_value:
                                    min_ds_value=ds_value
                                    #print(f"lane change {lane_label}")
                                    best_decision = lane_offset
        else:
            ego_ds_value=0
            best_decision=0
            #min_ds_value=0


        return best_decision,min_ds_value,ego_ds_value
    
    def control(self,allow_lane_change, env,state: List[float]) -> str:
        ego_state, surrounding_vehicles = self.parse_state(state)
        #print(ego_state)
        classified_vehicles = self.classify_vehicles(ego_state, surrounding_vehicles)

        # Evaluate potential lane change options.
        #print(ego_state)
        target_lane_offset,ds_value,ego_ds_value = self.evaluate_lane_change(ego_state, classified_vehicles)
        #print("change_lane")
        #print(change_lane)
        RL_ds_weight=(ds_value-ego_ds_value)/self.ds_threshold
        if target_lane_offset and allow_lane_change:
            #print("change_lane control")
           # print(ego_ds_value)
            #print(ds_value)
            target_speed, target_lane=self.change_lane_control(env,target_lane_offset,ds_value,ego_ds_value)
            if target_lane==1:
                print("here")
            return target_speed, target_lane,RL_ds_weight
        else:
        # If no lane change is appropriate, use lane keep control.
           #print("lane_keep control")
           #print(env)
           #print(state)
           target_speed, target_lane=self.lane_keep_control(env,state)
           if target_lane==1:
                print("here")
           return target_speed, target_lane,0

    def lane_keep_control(self, env,state: List[float]) -> str:
        ego_state, surrounding_vehicles = self.parse_state(state)
        classified_vehicles = self.classify_vehicles(ego_state, surrounding_vehicles)
        forward_vehicles = classified_vehicles.get("forward", [])
        ds_target_speed=0
        target_lane=env.vehicle.lane_index[2]
        if not forward_vehicles:
            target_speed=12.5
        else:
            closest_vehicle = min(forward_vehicles, key=lambda v: v[0])
            ego_x, _, ego_speed, _ = ego_state
            forward_x, _, forward_speed, _ = closest_vehicle
            gap = forward_x - ego_x
            safety_distance = self.safety_distance
            #if gap 
            if gap < safety_distance:
                target_speed=10
            else:
                target_speed=12.5
        return target_speed, target_lane

    def change_lane_control(self,env,target_lane_offset,ds_value,ego_ds_value):
        # if there is front vehicle 
        #if or self.safety_distance>ego_ds_value
        target_speed=12.5
        target_lane=env.vehicle.lane_index[2]
        if ego_ds_value<ds_value :
            target_speed=10
        else:
            target_lane=env.vehicle.lane_index[2]+target_lane_offset
        return target_speed, target_lane
    
def interperate_ds_command(env,target_speed, target_lane):
    print(env.vehicle.speed)

    # List of discretized acceleration values
    discretized_index_speed = [10, 12.5, 15]

    # Find the closest discretized value
    discretized_speed = min(discretized_index_speed, key=lambda x: abs(env.vehicle.speed - x))

    if np.abs(env.vehicle.target_lane_index[2]-target_lane)==2:
        print("stop")
    if discretized_speed ==target_speed and env.vehicle.lane_index[2]==target_lane:
        return "IDLE"
    elif discretized_speed ==target_speed and env.vehicle.lane_index[2]>target_lane:
        return "LANE_LEFT"
    elif discretized_speed ==target_speed and env.vehicle.lane_index[2]<target_lane:
        return "LANE_RIGHT"
    elif discretized_speed <target_speed and env.vehicle.lane_index[2]==target_lane:
        return "FASTER"
    elif discretized_speed <target_speed and env.vehicle.lane_index[2]>target_lane:
        return "FASTER and LANE_LEFT",
    elif discretized_speed <target_speed and env.vehicle.lane_index[2]<target_lane:
        return "FASTER and LANE_RIGHT"
    elif discretized_speed >target_speed and env.vehicle.lane_index[2]==target_lane:
        return "SLOWER"
    elif discretized_speed >target_speed and env.vehicle.lane_index[2]>target_lane:
        return "SLOWER and LANE_LEFT"
    elif discretized_speed >target_speed and env.vehicle.lane_index[2]<target_lane:
        return "SLOWER and LANE_RIGHT"
    '''
def interperate_ds_command(env,command):

    current_speed=env.vehicle.speed
    current_target_speed=env.vehicle.target_speed
    current_lane=env.vehicle.lane_index
    delta=1
    if command =="slower":
        if env.vehicle.target_speed>10 and current_speed<current_target_speed+delta:
            return "SLOWER"
        else:
            return "IDLE"
    elif command=="faster":
        if env.vehicle.target_speed<12.5 and current_speed>current_target_speed-delta:
            return "FASTER"
        else:
             return "IDLE"

    elif command=="lane change left":
        return "LANE_LEFT"
    elif command=="lane change right":
        return "LANE_RIGHT"
    elif command=="slower curse":
         if env.vehicle.target_speed>7.5 and current_speed<current_target_speed+delta:
            return "SLOWER"
         else:
            return "IDLE"         

'''


GOALS_ALL = {
0: 'LANE_LEFT',
1: 'IDLE',
2: 'LANE_RIGHT',
3: 'FASTER',
4: 'SLOWER',
5: "FASTER and LANE_LEFT",
6: "FASTER and LANE_RIGHT",
7: "SLOWER and LANE_LEFT",
8: "SLOWER and LANE_RIGHT",
}
def get_goal_index(goal_string):
    for key, value in GOALS_ALL.items():
        if value == goal_string:
            return key