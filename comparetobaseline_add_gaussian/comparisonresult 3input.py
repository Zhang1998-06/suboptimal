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
    # Initialize arrays for each metric for each method
    # Set1: Ours
    endtoendreward = np.zeros((runs, episode))
    endtoendlengths = np.zeros((runs, episode))
    endtoenddistance = np.zeros((runs, episode))
    endtoendcrash_times = np.zeros((runs, episode))
    endtoendtrap = np.zeros((runs, episode))
    endtoendaverage_speed = np.zeros((runs, episode))
    # Set2: GAIL
    savelowreward = np.zeros((runs, episode))
    savelowlengths = np.zeros((runs, episode))
    savelowdistance = np.zeros((runs, episode))
    savelowcrash_times = np.zeros((runs, episode))
    savelowtrap = np.zeros((runs, episode))
    savelowaverage_speed = np.zeros((runs, episode))
    # Set3: CQL
    tejasreward = np.zeros((runs, episode))
    tejaslengths = np.zeros((runs, episode))
    tejasdistance = np.zeros((runs, episode))
    tejascrash_times = np.zeros((runs, episode))
    tejastrap = np.zeros((runs, episode))
    tejasaverage_speed = np.zeros((runs, episode))
    # Set4: SAC
    sacreward = np.zeros((runs, episode))
    saclengths = np.zeros((runs, episode))
    sacdistance = np.zeros((runs, episode))
    saccrash_times = np.zeros((runs, episode))
    sactrap = np.zeros((runs, episode))
    sacaverage_speed = np.zeros((runs, episode))

    for i in range(runs):
        # --- Set1: Ours ---
        end_data = np.loadtxt(endpath + str(i) + 'hdqn.csv', delimiter=",")
        if end_data.shape[0] > 6:
            print(f"Warning: {endpath + str(i) + 'hdqn.csv'} has extra rows. Using only the first 6 rows.")
            end_data = end_data[:6]
        EndStats = namedtuple("EndStats", ["episode_lengths", "episode_env_rewards", "crash_times", "trap", "average_speed", "distance"])
        end_stats = EndStats(*end_data)
        endtoendreward[i] = end_stats.episode_env_rewards
        endtoendlengths[i] = end_stats.episode_lengths
        endtoenddistance[i] = end_stats.distance
        endtoendcrash_times[i] = end_stats.crash_times
        endtoendtrap[i] = end_stats.trap
        endtoendaverage_speed[i] = end_stats.average_speed

        # --- Set2: GAIL ---
        savelow_data = np.loadtxt(hierarchcial + str(i) + 'hdqn.csv', delimiter=",")
        if savelow_data.shape[0] > 6:
            print(f"Warning: {hierarchcial + str(i) + 'hdqn.csv'} has extra rows. Using only the first 6 rows.")
            savelow_data = savelow_data[:6]
        SaveStats = namedtuple("SaveStats", ["episode_lengths", "episode_env_rewards", "crash_times", "trap", "average_speed", "distance"])
        save_stats = SaveStats(*savelow_data)
        savelowreward[i] = save_stats.episode_env_rewards
        savelowlengths[i] = save_stats.episode_lengths
        savelowdistance[i] = save_stats.distance
        savelowcrash_times[i] = save_stats.crash_times
        savelowtrap[i] = save_stats.trap
        savelowaverage_speed[i] = save_stats.average_speed

        # --- Set3: CQL ---
        tejas_data = np.loadtxt(tejas + str(i) + 'hdqn.csv', delimiter=",")
        if tejas_data.shape[0] > 6:
            print(f"Warning: {tejas + str(i) + 'hdqn.csv'} has extra rows. Using only the first 6 rows.")
            tejas_data = tejas_data[:6]
        TejasStats = namedtuple("TejasStats", ["episode_lengths", "episode_env_rewards", "crash_times", "trap", "average_speed", "distance"])
        tejas_stats = TejasStats(*tejas_data)
        tejasreward[i] = tejas_stats.episode_env_rewards
        tejaslengths[i] = tejas_stats.episode_lengths
        tejasdistance[i] = tejas_stats.distance
        tejascrash_times[i] = tejas_stats.crash_times
        tejastrap[i] = tejas_stats.trap
        tejasaverage_speed[i] = tejas_stats.average_speed

        # --- Set4: SAC ---
        sac_data = np.loadtxt(sac + str(i) + 'end.csv', delimiter=",")
        if sac_data.shape[0] > 6:
            print(f"Warning: {sac + str(i) + 'end.csv'} has extra rows. Using only the first 6 rows.")
            sac_data = sac_data[:6]
        SacStats = namedtuple("SacStats", ["episode_lengths", "episode_env_rewards", "crash_times", "trap", "average_speed", "distance"])
        sac_stats = SacStats(*sac_data)
        sacreward[i] = sac_stats.episode_env_rewards
        saclengths[i] = sac_stats.episode_lengths
        sacdistance[i] = sac_stats.distance
        saccrash_times[i] = sac_stats.crash_times
        sactrap[i] = sac_stats.trap
        sacaverage_speed[i] = sac_stats.average_speed

    endtoendlist = [endtoendreward, endtoendlengths, endtoenddistance, endtoendcrash_times, endtoendtrap, endtoendaverage_speed]
    savelowlist = [savelowreward, savelowlengths, savelowdistance, savelowcrash_times, savelowtrap, savelowaverage_speed]
    tejaslist = [tejasreward, tejaslengths, tejasdistance, tejascrash_times, tejastrap, tejasaverage_speed]
    saclist = [sacreward, saclengths, sacdistance, saccrash_times, sactrap, sacaverage_speed]
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
    plot_method(xaxis, endtoendreward, "Ours", window, sigma)
    plot_method(xaxis, savelowreward, "GAIL", window, sigma)
    plot_method(xaxis, tejasreward, "CQL", window, sigma)
    plot_method(xaxis, sacreward, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Reward (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_Reward.png')
    return fig

def variance_crashresult(endtoendcrash_times, savelowcrash_times, tejascrash_times, saccrash_times, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendcrash_times, "Ours", window, sigma)
    plot_method(xaxis, savelowcrash_times, "GAIL", window, sigma)
    plot_method(xaxis, tejascrash_times, "CQL", window, sigma)
    plot_method(xaxis, saccrash_times, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Crash Times (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_crash.png')
    return fig

def variance_lengthresult(endtoendlengths, savelowlengths, tejaslengths, saclengths, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendlengths, "Ours", window, sigma)
    plot_method(xaxis, savelowlengths, "GAIL", window, sigma)
    plot_method(xaxis, tejaslengths, "CQL", window, sigma)
    plot_method(xaxis, saclengths, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Length (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_length.png')
    return fig

def variance_trapresult(endtoendtrap, savelowtrap, tejastrap, sactrap, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendtrap, "Ours", window, sigma)
    plot_method(xaxis, savelowtrap, "GAIL", window, sigma)
    plot_method(xaxis, tejastrap, "CQL", window, sigma)
    plot_method(xaxis, sactrap, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Trap (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_trap.png')
    return fig

def variance_speedresult(endtoendaverage_speed, savelowaverage_speed, tejasaverage_speed, sacaverage_speed, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoendaverage_speed, "Ours", window, sigma)
    plot_method(xaxis, savelowaverage_speed, "GAIL", window, sigma)
    plot_method(xaxis, tejasaverage_speed, "CQL", window, sigma)
    plot_method(xaxis, sacaverage_speed, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Average Speed (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('average_speed.png')
    return fig

def variance_distanceresult(endtoenddistance, savelowdistance, tejasdistance, sacdistance, runs, episode, window, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_method(xaxis, endtoenddistance, "Ours", window, sigma)
    plot_method(xaxis, savelowdistance, "GAIL", window, sigma)
    plot_method(xaxis, tejasdistance, "CQL", window, sigma)
    plot_method(xaxis, sacdistance, "SAC", window, sigma)
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Distance (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_distance.png')
    return fig

if __name__ == '__main__':
    runs = 5
    episode = 3000
    smoothing_window = 10
    gaussian_sigma = 6

    endtoendlist, savelowlist, tejaslist, saclist = read_data(runs, episode)
    (endtoendreward, endtoendlengths, endtoenddistance, 
     endtoendcrash_times, endtoendtrap, endtoendaverage_speed) = endtoendlist
    (savelowreward, savelowlengths, savelowdistance, 
     savelowcrash_times, savelowtrap, savelowaverage_speed) = savelowlist
    (tejasreward, tejaslengths, tejasdistance, 
     tejascrash_times, tejastrap, tejasaverage_speed) = tejaslist
    (sacreward, saclengths, sacdistance, 
     saccrash_times, sactrap, sacaverage_speed) = saclist

    # (Optional) Compute average values over the last 50 episodes for summary
    rewardend = np.average(endtoendreward[:, -50], axis=0)
    rewardgail = np.average(savelowreward[:, -50], axis=0)
    rewardcql = np.average(tejasreward[:, -50], axis=0)
    rewardSAC = np.average(sacreward[:, -50], axis=0)

    trapend = np.average(endtoendtrap[:, -50], axis=0)
    trapgail = np.average(savelowtrap[:, -50], axis=0)
    trapcql = np.average(tejastrap[:, -50], axis=0)
    trapSAC = np.average(sactrap[:, -50], axis=0)

    speedend = np.average(endtoendaverage_speed[:, -50], axis=0)
    speedgail = np.average(savelowaverage_speed[:, -50], axis=0)
    speedcql = np.average(tejasaverage_speed[:, -50], axis=0)
    speedSAC = np.average(sacaverage_speed[:, -50], axis=0)

    fig2 = variance_rewardresult(endtoendreward, savelowreward, tejasreward, sacreward, runs, episode, smoothing_window, gaussian_sigma)
    fig2.show()
    fig3 = variance_crashresult(endtoendcrash_times, savelowcrash_times, tejascrash_times, saccrash_times, runs, episode, smoothing_window, gaussian_sigma)
    fig3.show()
    fig4 = variance_lengthresult(endtoendlengths, savelowlengths, tejaslengths, saclengths, runs, episode, smoothing_window, gaussian_sigma)
    fig4.show()
    fig5 = variance_trapresult(endtoendtrap, savelowtrap, tejastrap, sactrap, runs, episode, smoothing_window, gaussian_sigma)
    fig5.show()
    fig6 = variance_speedresult(endtoendaverage_speed, savelowaverage_speed, tejasaverage_speed, sacaverage_speed, runs, episode, smoothing_window, gaussian_sigma)
    fig6.show()
    fig7 = variance_distanceresult(endtoenddistance, savelowdistance, tejasdistance, sacdistance, runs, episode, smoothing_window, gaussian_sigma)
    fig7.show()
