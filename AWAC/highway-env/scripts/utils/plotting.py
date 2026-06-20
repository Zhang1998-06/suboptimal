import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards","crash_times","trap","average_speed","distance","episode_low_level_reward","goal_reached_times"])

def plot_training_episode_stats(stats, smoothing_window=10, mean_number=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    x=np.array([i for i in range(len(stats.episode_lengths))])
    plt.scatter(x,stats.episode_lengths)
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Length")
    plt.title("Episode Length ")
    #crash 
    fig2 = plt.figure(figsize=(10,5))
    x=np.array([i for i in range(len(stats.crash_times))])
    plt.scatter(x,stats.crash_times)
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode crash")
    plt.title("Epsiode crash")
    #trap
    fig3 = plt.figure(figsize=(10,5))
    x=np.array([i for i in range(len(stats.trap))])
    plt.scatter(x,stats.trap)
    plt.xlabel("Epsiode")
    plt.ylabel("get out of trap")
    plt.title("successfully get out of trap")


    # Plot the episode reward over time
    fig4 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_reward=np.average(stats.episode_rewards.reshape(-1,mean_number),axis=1)
    mean_axis=np.array([i*mean_number for i in range(mean_reward.shape[0])])
    plt.plot(rewards_smoothed,label="smoothed_episode_reward")
    plt.plot(mean_axis,mean_reward,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))


    fig9 = plt.figure(figsize=(10,5))
    rewards_smoothed2 = pd.Series(stats.episode_low_level_reward).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_reward2=np.average(stats.episode_low_level_reward.reshape(-1,mean_number),axis=1)
    mean_axis2=np.array([i*mean_number for i in range(mean_reward2.shape[0])])
    plt.plot(rewards_smoothed2,label="smoothed_episode_reward")
    plt.plot(mean_axis2,mean_reward2,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Episode lowlevel Reward over Time (Smoothed over window size {})".format(smoothing_window))
    # plot the crash_times 


    fig5 = plt.figure(figsize=(10,5))
    crash_times_smoothed = pd.Series(stats.crash_times).rolling(smoothing_window, min_periods=smoothing_window).mean()
    meancrash_times_reward=np.average(stats.crash_times.reshape(-1,mean_number),axis=1)
    meancrash_times_axis=np.array([i*mean_number for i in range(meancrash_times_reward.shape[0])])
    plt.plot(crash_times_smoothed,label="smoothed_episode_reward")
    plt.plot(meancrash_times_axis,meancrash_times_reward,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode crash_times (Smoothed)")
    plt.title("Episode crash_times (Smoothed over window size {})".format(smoothing_window))
    #plot the trap


    fig6 = plt.figure(figsize=(10,5))
    rewardstrap_smoothed = pd.Series(stats.trap).rolling(smoothing_window, min_periods=smoothing_window).mean()
    meantrap_reward=np.average(stats.trap.reshape(-1,mean_number),axis=1)
    meantrap_axis=np.array([i*mean_number for i in range(meantrap_reward.shape[0])])
    plt.plot(rewardstrap_smoothed,label="smoothed_episode_reward")
    plt.plot(meantrap_axis,meantrap_reward,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode trap(Smoothed)")
    plt.title("Episode trap over Time (Smoothed over window size {})".format(smoothing_window))
    # plot the average speed 


    fig7 = plt.figure(figsize=(10,5))
    average_speed_smoothed = pd.Series(stats.average_speed).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_average_speed=np.average(stats.average_speed.reshape(-1,mean_number),axis=1)
    meanaverage_speed_axis=np.array([i*mean_number for i in range(mean_average_speed.shape[0])])
    plt.plot(average_speed_smoothed,label="smoothed_episode_reward")
    plt.plot(meanaverage_speed_axis,mean_average_speed,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode average_speed (Smoothed)")
    plt.title("Episode average_speed over Time (Smoothed over window size {})".format(smoothing_window))
    # plot the distance

    fig8 = plt.figure(figsize=(10,5))
    distance_smoothed = pd.Series(stats.distance).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_distance=np.average(stats.distance.reshape(-1,mean_number),axis=1)
    meandistance_axis=np.array([i*mean_number for i in range(mean_distance.shape[0])])
    plt.plot(distance_smoothed,label="smoothed_episode_reward")
    plt.plot(meandistance_axis,mean_distance,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode distance (Smoothed)")
    plt.title("Episode distance over Time (Smoothed over window size {})".format(smoothing_window))

    fig10 = plt.figure(figsize=(10,5))
    goal_reached_times_smoothed = pd.Series(stats.goal_reached_times).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_goal_reached_times=np.average(stats.goal_reached_times.reshape(-1,mean_number),axis=1)
    mean_goal_reached_axis=np.array([i*mean_number for i in range(mean_goal_reached_times.shape[0])])
    plt.plot(goal_reached_times_smoothed,label="smoothed_episode_reward")
    plt.plot(mean_goal_reached_axis,mean_goal_reached_times,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode goal reached times (Smoothed)")
    plt.title("Episode goal reached times over Time (Smoothed over window size {})".format(smoothing_window))

    return fig1, fig2, fig3, fig4, fig5, fig6 ,fig7, fig8,fig9, fig10 #goal_reached_times

EpisodestatsStats = namedtuple("Stats",[
    "episode_lengths",
    "episode_rewards",
    "episode_average_speed",
    "episode_distance",
    "episode_trap",
    "episode_crash",
    "episode_min_headway",
    "episode_min_ttc",
    "episode_max_lateral_jerk",
    "episode_lane_invasion",
])

def plot_episode_stats(stats, smoothing_window=5, mean_number=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    x=np.array([i for i in range(len(stats.episode_lengths))])
    plt.scatter(x,stats.episode_lengths)
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Length")
    plt.title("Episode Length ")


    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    mean_reward=np.average(stats.episode_rewards.reshape(-1,mean_number),axis=1)
    mean_axis=np.array([i*mean_number for i in range(mean_reward.shape[0])])
    plt.plot(rewards_smoothed,label="smoothed_episode_reward")

    plt.plot(mean_axis,mean_reward,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))




    fig3 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_average_speed)
    plt.xlabel("Epsiode")
    plt.ylabel("Episode_average_speed")
    plt.title("Episode_average_speed")
   

    fig4 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_distance)
    plt.xlabel("Epsiode")
    plt.ylabel("Episode_distance")
    plt.title("Episode_distance")
 

    fig5 = plt.figure(figsize=(10,5))
    plt.scatter(x,stats.episode_crash)
    plt.xlabel("Epsiode")
    plt.ylabel("episode_crash")
    plt.title("episode_crash")

    fig6 = plt.figure(figsize=(10,5))
    plt.scatter(x,stats.episode_trap)
    plt.xlabel("Epsiode")
    plt.ylabel("episode_trap")
    plt.title("episode_trap") 

    fig7 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_min_headway)
    plt.xlabel("Episode")
    plt.ylabel("Min headway (m)")
    plt.title("Episode min headway")

    fig8 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_min_ttc)
    plt.xlabel("Episode")
    plt.ylabel("Min TTC (s)")
    plt.title("Episode min TTC")

    fig9 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_max_lateral_jerk)
    plt.xlabel("Episode")
    plt.ylabel("Max lateral jerk")
    plt.title("Episode max lateral jerk")

    fig10 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lane_invasion)
    plt.xlabel("Episode")
    plt.ylabel("Lane invasion time (s)")
    plt.title("Episode lane invasion time")

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10

def plot_visited_states(visits, num_episodes):
    n_thousand_episode = int(np.floor(num_episodes / 1000))
    eps = list(range(1, n_thousand_episode + 1))

    plt.figure(figsize=(10,5))
    for i_state in range(2, 6):
        state_label = "State %d" % (i_state + 1)
        plt.plot(eps, visits[:, i_state]/1000, label=state_label)

    plt.legend()
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.1, 1.2)
    plt.xlim(1, 12)
    plt.title("Number of visits (for States 3 to 6) averaged over 1000 episodes")
    plt.grid(True)
