from html.entities import name2codepoint
from itertools import filterfalse
from pickle import FALSE
from re import A
from this import s
import numpy as np
from gym.envs.registration import register
from gym import spaces
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import itertools


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                #"features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False
            },
            "action": {
                "type": "GOAL",
            },
            
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 1000,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 0.5,
            #"reward_distance_range":[0,8],# distance will affect the reward.
            #"reward_speed_range": [10,15],
            "offroad_terminal": False,
            #"off_lane_reward":-0.7,
            "lane_centering_cost": 4,
            "crash":-10,
            "lane_center_reward":0.05,
            "steering_reward":0.05,
            "high_speed_reward": 1.5, 


        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=15),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
    


    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        difficultylevel=3
        defaultspacing=0.95
        if difficultylevel>=3:
            trapnumbers=difficultylevel-2
        elif difficultylevel==2:
            trapnumbers=1
            defaultspacing=0.01
        else:
            trapnumbers=0

        self.controlled_vehicles = []
        vehicle = Vehicle.create_random( # 在这里create vehicle
            self.road,
            speed=10,# set the initial speed 
            lane_id=0,# set the initial lane 
            spacing=self.config["ego_spacing"] # nothing
        )
        vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)
        self.trapped_vehicles=[]
        #fromtrap
        if difficultylevel>0:
            vehicle = Vehicle.create_trapvehicle( # 在这里create vehicle
                    self.road,
                    speed=11,# set the initial speed 
                    lane_id=0,# set the initial lane 
                     # nothing
                     spacing=0.8
                )

        # 差不多是所在lane中的vehicle 
        #vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        self.road.vehicles.append(vehicle)
        self.trapped_vehicles.append(vehicle)
        # side trap
        for trapcar in range(trapnumbers):
            vehicle = Vehicle.create_trapvehicle( # 在这里create vehicle
                self.road,
                speed=11,# set the initial speed 
                lane_id=1,# set the initial lane 
                 # nothing
                 spacing=defaultspacing
            )
            #vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.road.vehicles.append(vehicle)
            self.trapped_vehicles.append(vehicle)

        for others in other_per_controlled:
            for _ in range(others):
                vehicle = other_vehicles_type.create_other_traffic(self.road,speed=11,spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def goal_diff(self):


        target_lane = self.road.network.get_lane(self.vehicle.target_lane_index)
        _,goal_lateral_distance = target_lane.local_coordinates(self.vehicle.position)
        #print(goal_lateral_distance)
        #print(self.vehicle.target_speed)
        #print(self.vehicle.speed)
        speediff=self.vehicle.target_speed-self.vehicle.speed* np.cos(self.vehicle.heading)
        #print([goal_lateral_distance, speediff])
        # normalize
        goal_lateral_distance=utils.lmap(goal_lateral_distance,[-12,12],[-1,1]) # threelane and width 4 
        speediff=utils.lmap(speediff,[-80,80],[-1,1])
        return [goal_lateral_distance, speediff]

    def lat_center(self):
        none,lat = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lat=utils.lmap(lat,[-12,12],[-1,1])
        return lat
    def out_of_trap(self):
        egovehiclex= np.max([v.position[0] for v in self.controlled_vehicles])
        #print(egovehiclex)
        trapped_vehiclesx = np.max([v.position[0] for v in self.trapped_vehicles])
        #print(trapped_vehiclesx)
        #print(1)        
        return self._is_terminal() and egovehiclex>trapped_vehiclesx

    def egovehiclespeed(self):
        return  self.vehicle.speed * np.cos(self.vehicle.heading)

    def addicitional_obs(self):
        add_obs=np.concatenate([[self.vehicle.on_road],self.goal_diff(),[self.lat_center()]])
        return add_obs

    def _reward(self, action: Action) -> float:
        forward_speed = self.vehicle.speed #* np.cos(self.vehicle.heading)
        low_speed =self.vehicle.speed * np.cos(self.vehicle.heading)<=5
        steeringreward=-abs(np.sin(self.vehicle.heading))*self.config["steering_reward"]
        #print(forward_speed)
        if 12.5<=forward_speed<=15:
            speed_reward=self.config["high_speed_reward"]*(0.32*forward_speed-3.8)
        elif forward_speed>15:
            speed_reward=self.config["high_speed_reward"]*np.exp(-(forward_speed-15)**2) 
        elif forward_speed<12.5:
            speed_reward=self.config["high_speed_reward"]*(2/75*forward_speed-2/15)
        elif forward_speed<=5:
            speed_reward=0
        #the close vehicle reward added
        front_vehicle,rear_vehicle=self.road.neighbour_vehicles(self.vehicle,self.vehicle.lane_index)
        dangerdistance= self.vehicle.speed*2
        alertdistance=25
        distance1= abs(self.vehicle.lane_distance_to(front_vehicle))
        alert= False
        danger=False
        reward_distance=0
        '''
        if distance1<alertdistance and distance1>dangerdistance and self.vehicle.crashed ==False:
            alert=True
            reward_distance=self.config["reward_distance"]*(distance1/10-2.5)
   '''
        if distance1<=dangerdistance and self.vehicle.crashed ==False:
            danger=True         
        B,lateralAA = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering_reward =self.config["lane_center_reward"]*np.exp(-1.5*lateralAA**2)    
        stopped= self.vehicle.speed * np.cos(self.vehicle.heading)<=0

        # this is the environment reward 
        env_reward = \
            + speed_reward\
            + lane_centering_reward\
            + steeringreward      
        env_reward = utils.lmap(env_reward, 
                          [
                           -self.config["steering_reward"]*np.sin(np.pi / 50),
                           +self.config["lane_center_reward"]
                           +self.config["high_speed_reward"]],
                          [0, 1])
        env_reward = 0 if  low_speed or danger else env_reward
        reward = self.config["crash"] if (not self.vehicle.on_road) or self.vehicle.crashed or stopped else env_reward
        return  reward
    

    def reach_end_crash(self) -> bool:
        stopped= self.vehicle.speed * np.cos(self.vehicle.heading)<=0
        inverse=np.cos(self.vehicle.heading)<=0
        #print(inverse)
        return self.vehicle.crashed or \
               stopped or inverse or (not self.vehicle.on_road)




    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        stopped= self.vehicle.speed * np.cos(self.vehicle.heading)<=0
        inverse=np.cos(self.vehicle.heading)<=0

        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or stopped or (not self.vehicle.on_road)

    def set_goal(self,goal_1=None):

        return self.vehicle.set_goal(goal_1)
  

    def goal_reached(self,goal:int)-> bool:
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        B,lateralAA = self.vehicle.lane.local_coordinates(self.vehicle.position)
        target_lane = self.road.network.get_lane(self.vehicle.target_lane_index)
        _,goal_lateral_distance = target_lane.local_coordinates(self.vehicle.position)
        lane_diff=abs(goal_lateral_distance)
        #if vehicle.target_speedr


        if goal==5 or goal==6 or goal==7 or goal==8 :
            return abs(forward_speed-self.vehicle.target_speed)<=0.3\
                and lane_diff<=0.3 
        elif goal== 0 or goal==2:
            return lane_diff<=0.3 
        else:
            return  abs(forward_speed-self.vehicle.target_speed)<=0.3



    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 15,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 150,  # [s]
            "ego_spacing": 100,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)
