import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.ndimage import gaussian_filter1d  # updated import

# File paths for each dataset
endpath = 'set1/'
hierarchcial = 'set2/'
tejas = 'set3/'
sac = 'set4/'

def read_data(runs, episode):
    # -------- 1) 预分配数组（原逻辑 + 新增两项） --------
    # Set1: Ours
    endtoendreward        = np.zeros((runs, episode))
    endtoendlengths       = np.zeros((runs, episode))
    endtoenddistance      = np.zeros((runs, episode))
    endtoendcrash_times   = np.zeros((runs, episode))
    endtoendtrap          = np.zeros((runs, episode))
    endtoendaverage_speed = np.zeros((runs, episode))
    endtoendentropy       = np.zeros((runs, episode))          # 新
    endtoendexp_entropy   = np.zeros((runs, episode))          # 新
    # Set2: GAIL
    savelowreward        = np.zeros((runs, episode))
    savelowlengths       = np.zeros((runs, episode))
    savelowdistance      = np.zeros((runs, episode))
    savelowcrash_times   = np.zeros((runs, episode))
    savelowtrap          = np.zeros((runs, episode))
    savelowaverage_speed = np.zeros((runs, episode))
    savelowentropy       = np.zeros((runs, episode))           # 新
    savelowexp_entropy   = np.zeros((runs, episode))           # 新
    # Set3: CQL
    tejasreward        = np.zeros((runs, episode))
    tejaslengths       = np.zeros((runs, episode))
    tejasdistance      = np.zeros((runs, episode))
    tejascrash_times   = np.zeros((runs, episode))
    tejastrap          = np.zeros((runs, episode))
    tejasaverage_speed = np.zeros((runs, episode))
    tejasentropy       = np.zeros((runs, episode))             # 新
    tejasexp_entropy   = np.zeros((runs, episode))             # 新
    # Set4: SAC
    sacreward        = np.zeros((runs, episode))
    saclengths       = np.zeros((runs, episode))
    sacdistance      = np.zeros((runs, episode))
    saccrash_times   = np.zeros((runs, episode))
    sactrap          = np.zeros((runs, episode))
    sacaverage_speed = np.zeros((runs, episode))
    sacentropy       = np.zeros((runs, episode))               # 新
    sacexp_entropy   = np.zeros((runs, episode))               # 新

    # -------- 2) 每个 run 读入 csv --------
    for i in range(runs):
        # --- Set1: Ours ---
        end_data = np.loadtxt(endpath + str(i) + 'hdqn.csv', delimiter=",")

        EndStats = namedtuple("EndStats",["episode_lengths", "episode_rewards","crash_times","trap",
                                   "average_speed","distance","episode_low_level_reward","goal_reached_times",
                                   "entropy","exp_entropy"])
        end_stats = EndStats(*end_data)
        endtoendreward[i] = end_stats.episode_rewards
        endtoendlengths[i] = end_stats.episode_lengths
        endtoenddistance[i] = end_stats.distance
        endtoendcrash_times[i] = end_stats.crash_times
        endtoendtrap[i] = end_stats.trap
        endtoendaverage_speed[i] = end_stats.average_speed
        endtoendentropy[i]       = end_stats.entropy              # 新
        endtoendexp_entropy[i]   = end_stats.exp_entropy          # 新

        # --- Set2: GAIL ---
        savelow_data = np.loadtxt(hierarchcial + str(i) + 'hdqn.csv', delimiter=",")

        SaveStats = namedtuple("SaveStats",["episode_lengths", "episode_rewards","crash_times","trap",
                                   "average_speed","distance","episode_low_level_reward","goal_reached_times",
                                   "entropy","exp_entropy"])              
        save_stats = SaveStats(*savelow_data)
        savelowreward[i] = save_stats.episode_rewards
        savelowlengths[i] = save_stats.episode_lengths
        savelowdistance[i] = save_stats.distance
        savelowcrash_times[i] = save_stats.crash_times
        savelowtrap[i] = save_stats.trap
        savelowaverage_speed[i] = save_stats.average_speed
        savelowentropy[i]       = save_stats.entropy               # 新
        savelowexp_entropy[i]   = save_stats.exp_entropy           # 新

        # --- Set3: CQL ---
        tejas_data = np.loadtxt(tejas + str(i) + 'hdqn.csv', delimiter=",")
        TejasStats= namedtuple("TejasStats",["episode_lengths", "episode_rewards","crash_times","trap",
                                   "average_speed","distance","episode_low_level_reward","goal_reached_times",
                                   "entropy","exp_entropy"])
        tejas_stats = TejasStats(*tejas_data)
        tejasreward[i] = tejas_stats.episode_rewards
        tejaslengths[i] = tejas_stats.episode_lengths
        tejasdistance[i] = tejas_stats.distance
        tejascrash_times[i] = tejas_stats.crash_times
        tejastrap[i] = tejas_stats.trap
        tejasaverage_speed[i] = tejas_stats.average_speed
        tejasentropy[i]       = tejas_stats.entropy                # 新
        tejasexp_entropy[i]   = tejas_stats.exp_entropy            # 新

        # --- Set4: SAC ---
        sac_data = np.loadtxt(sac + str(i) + 'end.csv', delimiter=",")
        SacStats = namedtuple("SacStats",["episode_lengths", "episode_rewards","crash_times","trap","average_speed","distance","entropy","exp_entropy"])
        sac_stats = SacStats(*sac_data)
        sacreward[i] = sac_stats.episode_rewards
        saclengths[i] = sac_stats.episode_lengths
        sacdistance[i] = sac_stats.distance
        saccrash_times[i] = sac_stats.crash_times
        sactrap[i] = sac_stats.trap
        sacaverage_speed[i] = sac_stats.average_speed
        sacentropy[i]       = sac_stats.entropy                    # 新
        sacexp_entropy[i]   = sac_stats.exp_entropy                # 新

    # -------- 3) 打包返回（原列表后追加两项） --------
    endtoendtrap  = np.zeros((runs, episode))
    endtoendlist = [endtoendreward, endtoendlengths, endtoenddistance,
                    endtoendcrash_times, endtoendtrap, endtoendaverage_speed,
                    endtoendentropy, endtoendexp_entropy]          # 新
    savelowlist  = [savelowreward, savelowlengths, savelowdistance,
                    savelowcrash_times, savelowtrap, savelowaverage_speed,
                    savelowentropy, savelowexp_entropy]            # 新
    tejaslist    = [tejasreward, tejaslengths, tejasdistance,
                    tejascrash_times, tejastrap, tejasaverage_speed,
                    tejasentropy, tejasexp_entropy]                # 新
    saclist      = [sacreward, saclengths, sacdistance,
                    saccrash_times, sactrap, sacaverage_speed,
                    sacentropy, sacexp_entropy]                    # 新
    return endtoendlist, savelowlist, tejaslist, saclist


def smooth_series(data, window, sigma):
    """Apply a centered rolling mean followed by a Gaussian filter for extra smoothing."""
    s = pd.Series(data).rolling(window=window, min_periods=window, center=True).mean()
    s = s.fillna(method='bfill').fillna(method='ffill')
    s = gaussian_filter1d(s, sigma=sigma)
    return s

def plot_method(xaxis, data, label, window, sigma, alpha=0.3):
    """Compute mean and std across runs, smooth them, and plot the mean with ± std fill."""
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    mean_smoothed = smooth_series(mean_data, window, sigma)
    lower_smoothed = smooth_series(mean_data + std_data, window, sigma)
    upper_smoothed = smooth_series(mean_data - std_data, window, sigma)
    plt.fill_between(xaxis, lower_smoothed, upper_smoothed, alpha=alpha, linewidth=0)
    plt.plot(xaxis, mean_smoothed, label=label)

def variance_rewardresult(endtoendreward, savelowreward, tejasreward, sacreward, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendreward, "2-0-1000", window, sigma)
    plot_method(xaxis, savelowreward, "adaptive_in_reward", window, sigma)
    plot_method(xaxis, tejasreward, "adaptive_in_sac_alpha", window, sigma)
    plot_method(xaxis, sacreward, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Reward (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_Reward.png')
    return fig

def variance_crashresult(endtoendcrash_times, savelowcrash_times, tejascrash_times, saccrash_times, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendcrash_times, "2-0-1000", window, sigma)
    plot_method(xaxis, savelowcrash_times, "adaptive_in_reward", window, sigma)
    plot_method(xaxis, tejascrash_times, "adaptive_in_sac_alpha", window, sigma)
    plot_method(xaxis, saccrash_times, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Crash Times (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_crash.png')
    return fig

def variance_lengthresult(endtoendlengths, savelowlengths, tejaslengths, saclengths, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendlengths, "2-0-1000", window, sigma)
    plot_method(xaxis, savelowlengths, "adaptive_in_reward", window, sigma)
    plot_method(xaxis, tejaslengths, "adaptive_in_sac_alpha", window, sigma)
    plot_method(xaxis, saclengths, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Length (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_length.png')
    return fig

def variance_trapresult(endtoendtrap, savelowtrap, tejastrap, sactrap, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendtrap, "2-0-1000", window, sigma)
    plot_method(xaxis, savelowtrap, "adaptive_in_reward", window, sigma)
    plot_method(xaxis, tejastrap, "adaptive_in_sac_alpha", window, sigma)
    plot_method(xaxis, sactrap, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Trap (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_trap.png')
    return fig

def variance_speedresult(endtoendaverage_speed, savelowaverage_speed, tejasaverage_speed, sacaverage_speed, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendaverage_speed, "2-0-1000", window, sigma)
    plot_method(xaxis, savelowaverage_speed, "adaptive_in_reward", window, sigma)
    plot_method(xaxis, tejasaverage_speed, "adaptive_in_sac_alpha", window, sigma)
    plot_method(xaxis, sacaverage_speed, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Average Speed (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('average_speed.png')
    return fig

def variance_distanceresult(endtoenddistance, savelowdistance, tejasdistance, sacdistance, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoenddistance, "2-0-1000", window, sigma)
    plot_method(xaxis, savelowdistance, "adaptive_in_reward", window, sigma)
    plot_method(xaxis, tejasdistance, "adaptive_in_sac_alpha", window, sigma)
    plot_method(xaxis, sacdistance, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Distance (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_distance.png')
    return fig
def variance_entropyresult(endtoendentropy, savelowentropy, tejasentropy,
                            sacentropy, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendentropy, "2-0-1000", window, sigma)
    plot_method(xaxis, savelowentropy, "adaptive_in_reward", window, sigma)
    plot_method(xaxis, tejasentropy, "adaptive_in_sac_alpha", window, sigma)
    plot_method(xaxis, sacentropy, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Entropy $H_{avg}$ (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_entropy.png')
    return fig


def variance_exp_entropyresult(endtoendexp_entropy, savelowexp_entropy,
                                tejasexp_entropy, sacexp_entropy,
                                runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendexp_entropy, "2-0-1000", window, sigma)
    plot_method(xaxis, savelowexp_entropy, "adaptive_in_reward", window, sigma)
    plot_method(xaxis, tejasexp_entropy, "adaptive_in_sac_alpha", window, sigma)
    plot_method(xaxis, sacexp_entropy, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel(r"Effective Actions $e^{H_{avg}}$ (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_exp_entropy.png')
    return fig

if __name__ == '__main__':
    runs = 5
    episode = 3000
    smoothing_window = 10
    gaussian_sigma = 6

    endtoendlist, savelowlist, tejaslist, saclist = read_data(runs, episode)
    (endtoendreward, endtoendlengths, endtoenddistance,
     endtoendcrash_times, endtoendtrap, endtoendaverage_speed,
     endtoendentropy, endtoendexp_entropy) = endtoendlist

    (savelowreward, savelowlengths, savelowdistance,
     savelowcrash_times, savelowtrap, savelowaverage_speed,
     savelowentropy, savelowexp_entropy) = savelowlist

    (tejasreward, tejaslengths, tejasdistance,
     tejascrash_times, tejastrap, tejasaverage_speed,
     tejasentropy, tejasexp_entropy) = tejaslist

    (sacreward, saclengths, sacdistance,
     saccrash_times, sactrap, sacaverage_speed,
     sacentropy, sacexp_entropy) = saclist

    # -------- 原 6 幅图调用保持不变 --------
    fig2 = variance_rewardresult(endtoendreward, savelowreward, tejasreward, sacreward,
                                 runs, episode, smoothing_window, gaussian_sigma); fig2.show()
    fig3 = variance_crashresult(endtoendcrash_times, savelowcrash_times, tejascrash_times, saccrash_times,
                                 runs, episode, smoothing_window, gaussian_sigma); fig3.show()
    fig4 = variance_lengthresult(endtoendlengths, savelowlengths, tejaslengths, saclengths,
                                 runs, episode, smoothing_window, gaussian_sigma); fig4.show()
    fig5 = variance_trapresult(endtoendtrap, savelowtrap, tejastrap, sactrap,
                                 runs, episode, smoothing_window, gaussian_sigma); fig5.show()
    fig6 = variance_speedresult(endtoendaverage_speed, savelowaverage_speed,
                                 tejasaverage_speed, sacaverage_speed,
                                 runs, episode, smoothing_window, gaussian_sigma); fig6.show()
    fig7 = variance_distanceresult(endtoenddistance, savelowdistance, tejasdistance, sacdistance,
                                 runs, episode, smoothing_window, gaussian_sigma); fig7.show()

    # -------- 新增 2 幅图 --------
    fig8 = variance_entropyresult(endtoendentropy, savelowentropy,
                                  tejasentropy, sacentropy,
                                  runs, episode, smoothing_window, gaussian_sigma); fig8.show()
    fig9 = variance_exp_entropyresult(endtoendexp_entropy, savelowexp_entropy,
                                      tejasexp_entropy, sacexp_entropy,
                                      runs, episode, smoothing_window, gaussian_sigma); fig9.show()