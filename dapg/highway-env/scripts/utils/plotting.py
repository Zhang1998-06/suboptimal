import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards","crash_times","trap",
                                   "average_speed","distance","episode_low_level_reward","goal_reached_times",
                                   "entropy","exp_entropy"])

# def plot_exploration_stats(stats, smoothing_window=10, mean_number=10, noshow=False):
#     """
#     Figure 11 : 每集平均策略熵 H(π)
#     Figure 12 : 每集 exp(H)（有效动作数）
#     """
#     # ---------- 计算 H_avg 与 exp(H) ----------
#     episode_lengths = np.array(stats.episode_lengths, dtype=np.float32)
#     accumulated_H   = np.array(stats.goal_reached_times, dtype=np.float32)
#     # 防止除零
#     episode_lengths[episode_lengths == 0] = 1.0

#     H_avg  = accumulated_H / episode_lengths          # 平均熵
#     exp_H  = np.exp(H_avg)                            # 有效动作数

#     # ---------- Figure 11 : H(π) ----------------
#     fig11 = plt.figure(figsize=(10, 5))
#     x = np.arange(len(H_avg))

#     plt.scatter(x, H_avg, s=8, alpha=0.6, label="H(π) per episode")

#     H_smoothed = pd.Series(H_avg).rolling(
#         smoothing_window, min_periods=smoothing_window).mean()
#     plt.plot(H_smoothed, label=f"smoothed (window={smoothing_window})")

#     if len(H_avg) >= mean_number:
#         H_block = H_avg.reshape(-1, mean_number).mean(axis=1)
#         x_block = np.arange(len(H_block)) * mean_number
#         plt.plot(x_block, H_block, label=f"mean per {mean_number} eps")

#     plt.xlabel("Episode")
#     plt.ylabel("Average Entropy  $H(\\pi)$")
#     plt.title("Figure 11  —  Policy Entropy per Episode")
#     plt.legend()

#     # ---------- Figure 12 : exp(H) --------------
#     fig12 = plt.figure(figsize=(10, 5))
#     plt.scatter(x, exp_H, s=8, alpha=0.6, label="exp(H) per episode")

#     exp_smoothed = pd.Series(exp_H).rolling(
#         smoothing_window, min_periods=smoothing_window).mean()
#     plt.plot(exp_smoothed, label=f"smoothed (window={smoothing_window})")

#     if len(exp_H) >= mean_number:
#         exp_block = exp_H.reshape(-1, mean_number).mean(axis=1)
#         x_block2  = np.arange(len(exp_block)) * mean_number
#         plt.plot(x_block2, exp_block, label=f"mean per {mean_number} eps")

#     plt.xlabel("Episode")
#     plt.ylabel(r"Effective Action Count  $e^{H}$")
#     plt.title("Figure 12  —  exp(H) per Episode")
#     plt.legend()

#     if not noshow:
#     return fig11, fig12


# import numpy as np
# from collections import namedtuple


#     # 假设你之前保存的文件路径就是 stats_savepath
#     stats_savepath =  '0hdqn.csv'

#     # 1) 读入并转置
#     data = np.loadtxt(stats_savepath, delimiter=",")   # 原 shape=(8,3000)
#     data = data.T                                     # 现在 shape=(3000,8)

#     # 2) 构造 namedtuple
#     stats = EpisodeStats(
#         episode_lengths          = data[:, 0],
#         episode_rewards          = data[:, 1],
#         crash_times              = data[:, 2],
#         trap                     = data[:, 3],
#         average_speed            = data[:, 4],
#         distance                 = data[:, 5],
#         episode_low_level_reward = data[:, 6],
#         goal_reached_times       = data[:, 7]
#     )

#     # 3) 再调用你的绘图函数就能看到 3000 条 episode 了
#     fig11, fig12 = plot_exploration_stats(stats)
def plot_training_episode_stats(stats, smoothing_window=10, mean_number=10, noshow=False):
    def block_mean(values, block_size):
        values = np.asarray(values)
        if block_size <= 0 or len(values) == 0:
            return np.array([]), np.array([])
        block_size = min(block_size, len(values))
        usable = (len(values) // block_size) * block_size
        if usable == 0:
            return np.array([]), np.array([])
        means = values[:usable].reshape(-1, block_size).mean(axis=1)
        axis = np.arange(len(means)) * block_size
        return means, axis

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
    mean_reward, mean_axis = block_mean(stats.episode_rewards, mean_number)
    plt.plot(rewards_smoothed,label="smoothed_episode_reward")
    if len(mean_reward) > 0:
        plt.plot(mean_axis,mean_reward,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))


    fig9 = plt.figure(figsize=(10,5))
    rewards_smoothed2 = pd.Series(stats.episode_low_level_reward).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_reward2, mean_axis2 = block_mean(stats.episode_low_level_reward, mean_number)
    plt.plot(rewards_smoothed2,label="smoothed_episode_reward")
    if len(mean_reward2) > 0:
        plt.plot(mean_axis2,mean_reward2,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Episode lowlevel Reward over Time (Smoothed over window size {})".format(smoothing_window))
    # plot the crash_times 


    fig5 = plt.figure(figsize=(10,5))
    crash_times_smoothed = pd.Series(stats.crash_times).rolling(smoothing_window, min_periods=smoothing_window).mean()
    meancrash_times_reward, meancrash_times_axis = block_mean(stats.crash_times, mean_number)
    plt.plot(crash_times_smoothed,label="smoothed_episode_reward")
    if len(meancrash_times_reward) > 0:
        plt.plot(meancrash_times_axis,meancrash_times_reward,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode crash_times (Smoothed)")
    plt.title("Episode crash_times (Smoothed over window size {})".format(smoothing_window))
    #plot the trap


    fig6 = plt.figure(figsize=(10,5))
    rewardstrap_smoothed = pd.Series(stats.trap).rolling(smoothing_window, min_periods=smoothing_window).mean()
    meantrap_reward, meantrap_axis = block_mean(stats.trap, mean_number)
    plt.plot(rewardstrap_smoothed,label="smoothed_episode_reward")
    if len(meantrap_reward) > 0:
        plt.plot(meantrap_axis,meantrap_reward,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode trap(Smoothed)")
    plt.title("Episode trap over Time (Smoothed over window size {})".format(smoothing_window))
    # plot the average speed 


    fig7 = plt.figure(figsize=(10,5))
    average_speed_smoothed = pd.Series(stats.average_speed).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_average_speed, meanaverage_speed_axis = block_mean(stats.average_speed, mean_number)
    plt.plot(average_speed_smoothed,label="smoothed_episode_reward")
    if len(mean_average_speed) > 0:
        plt.plot(meanaverage_speed_axis,mean_average_speed,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode average_speed (Smoothed)")
    plt.title("Episode average_speed over Time (Smoothed over window size {})".format(smoothing_window))
    # plot the distance

    fig8 = plt.figure(figsize=(10,5))
    distance_smoothed = pd.Series(stats.distance).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_distance, meandistance_axis = block_mean(stats.distance, mean_number)
    plt.plot(distance_smoothed,label="smoothed_episode_reward")
    if len(mean_distance) > 0:
        plt.plot(meandistance_axis,mean_distance,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode distance (Smoothed)")
    plt.title("Episode distance over Time (Smoothed over window size {})".format(smoothing_window))

    fig10 = plt.figure(figsize=(10,5))
    goal_reached_times_smoothed = pd.Series(stats.goal_reached_times).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mean_goal_reached_times, mean_goal_reached_axis = block_mean(stats.goal_reached_times, mean_number)
    plt.plot(goal_reached_times_smoothed,label="smoothed_episode_reward")
    if len(mean_goal_reached_times) > 0:
        plt.plot(mean_goal_reached_axis,mean_goal_reached_times,label="mean_reward for 100 episodes")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode goal reached times (Smoothed)")
    plt.title("Episode goal reached times over Time (Smoothed over window size {})".format(smoothing_window))

    fig11 = plt.figure(figsize=(10,5))
    distance_smoothed = pd.Series(stats.entropy).rolling(smoothing_window, min_periods=smoothing_window).mean()

    mean_distance, meandistance_axis = block_mean(stats.entropy, mean_number)
    plt.plot(distance_smoothed,label="entropy")

    if len(mean_distance) > 0:
        plt.plot(meandistance_axis,mean_distance,label="entropy")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode entropy(Smoothed)")
    plt.title("Episode entropy over Time (Smoothed over window size {})".format(smoothing_window))


    fig12 = plt.figure(figsize=(10,5))
    distance_smoothed = pd.Series(stats.exp_entropy).rolling(smoothing_window, min_periods=smoothing_window).mean()

    mean_distance, meandistance_axis = block_mean(stats.exp_entropy, mean_number)
    plt.plot(distance_smoothed,label="exp_entropy")

    if len(mean_distance) > 0:
        plt.plot(meandistance_axis,mean_distance,label="exp_entropy")
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode exp_entropy(Smoothed)")
    plt.title("Episode exp_entropy over Time (Smoothed over window size {})".format(smoothing_window))
    return fig1, fig2, fig3, fig4, fig5, fig6 ,fig7, fig8,fig9, fig10, fig11, fig12  #goal_reached_times

EpisodestatsStats = namedtuple("Stats",["episode_lengths", "episode_rewards",
                                   "episode_average_speed","episode_distance","episode_trap",
                                   "episode_crash"])

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

    return fig1, fig2, fig3, fig4, fig5, fig6

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
