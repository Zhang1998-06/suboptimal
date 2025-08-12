import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # updated import

# File paths for each method
endpath = 'nooffline/'
hierarchcial = 'nopanalty/'
tejas = 'ours/'

def read_data(runs, episode):
    # Initialize arrays for each metric and method
    endtoendreward = np.zeros((runs, episode))
    endtoendlengths = np.zeros((runs, episode))
    endtoenddistance = np.zeros((runs, episode))
    endtoendcrash_times = np.zeros((runs, episode))
    endtoendtrap = np.zeros((runs, episode))
    endtoendaverage_speed = np.zeros((runs, episode))

    savelowreward = np.zeros((runs, episode))
    savelowlengths = np.zeros((runs, episode))
    savelowdistance = np.zeros((runs, episode))
    savelowcrash_times = np.zeros((runs, episode))
    savelowtrap = np.zeros((runs, episode))
    savelowaverage_speed = np.zeros((runs, episode))

    tejasreward = np.zeros((runs, episode))
    tejaslengths = np.zeros((runs, episode))
    tejasdistance = np.zeros((runs, episode))
    tejascrash_times = np.zeros((runs, episode))
    tejastrap = np.zeros((runs, episode))
    tejasaverage_speed = np.zeros((runs, episode))

    for i in range(runs):
        # --- For end-to-end method ---
        end_data = np.loadtxt(endpath + str(i) + 'hdqn.csv', delimiter=",")
        # If the CSV file has more rows than expected, slice to the first 6 rows.
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

        # --- For offline reply buffer method ---
        save_data = np.loadtxt(hierarchcial + str(i) + 'hdqn.csv', delimiter=",")
        if save_data.shape[0] > 6:
            print(f"Warning: {hierarchcial + str(i) + 'hdqn.csv'} has extra rows. Using only the first 6 rows.")
            save_data = save_data[:6]
        SaveStats = namedtuple("SaveStats", ["episode_lengths", "episode_env_rewards", "crash_times", "trap", "average_speed", "distance"])
        save_stats = SaveStats(*save_data)
        savelowreward[i] = save_stats.episode_env_rewards
        savelowlengths[i] = save_stats.episode_lengths
        savelowdistance[i] = save_stats.distance
        savelowcrash_times[i] = save_stats.crash_times
        savelowtrap[i] = save_stats.trap
        savelowaverage_speed[i] = save_stats.average_speed

        # --- For "ours" method ---
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

    endtoendlist = [endtoendreward, endtoendlengths, endtoenddistance, endtoendcrash_times, endtoendtrap, endtoendaverage_speed]
    savelowlist = [savelowreward, savelowlengths, savelowdistance, savelowcrash_times, savelowtrap, savelowaverage_speed]
    tejaslist = [tejasreward, tejaslengths, tejasdistance, tejascrash_times, tejastrap, tejasaverage_speed]

    return endtoendlist, savelowlist, tejaslist

def smooth_series(data, window, sigma):
    """
    Apply a centered rolling average followed by a Gaussian filter for extra smoothing.
    """
    series = pd.Series(data)
    # Apply rolling average with a centered window
    smoothed = series.rolling(window=window, min_periods=window, center=True).mean()
    # Fill in any boundary NaNs
    smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
    # Further smooth with a Gaussian filter
    smoothed = gaussian_filter1d(smoothed, sigma=sigma)
    return smoothed

def plot_metric(xaxis, data, label, window, sigma, alpha):
    """
    Compute mean and standard deviation of the given data, smooth both, 
    then plot the smoothed mean along with a filled area for ± the smoothed standard deviation.
    """
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    
    mean_smoothed = smooth_series(mean_data, window, sigma)
    lower_bound_smoothed = smooth_series(mean_data + std_data, window, sigma)
    upper_bound_smoothed = smooth_series(mean_data - std_data, window, sigma)
    
    plt.fill_between(xaxis, lower_bound_smoothed, upper_bound_smoothed, alpha=alpha, linewidth=0)
    plt.plot(xaxis, mean_smoothed, label=label)

def variance_rewardresult(endtoendreward, savelowreward, tejasreward, runs, episode, window, alpha, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_metric(xaxis, tejasreward, "Ours", window, sigma, alpha)
    plot_metric(xaxis, endtoendreward, "With soft constraint", window, sigma, alpha)
    plot_metric(xaxis, savelowreward, "With offline reply buffer", window, sigma, alpha)
    
    
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Reward (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_Reward.png')
    return fig

def variance_crashresult(endtoendcrash_times, savelowcrash_times, tejascrash_times, runs, episode, window, alpha, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_metric(xaxis, tejascrash_times, "Ours", window, sigma, alpha)
    plot_metric(xaxis, endtoendcrash_times, "With soft constraint", window, sigma, alpha)
    plot_metric(xaxis, savelowcrash_times, "With offline reply buffer", window, sigma, alpha)

    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Crash Times (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_crash.png')
    return fig

def variance_lengthresult(endtoendlengths, savelowlengths, tejaslengths, runs, episode, window, alpha, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_metric(xaxis, tejaslengths, "Ours", window, sigma, alpha)
    plot_metric(xaxis, endtoendlengths, "With soft constraint", window, sigma, alpha)
    plot_metric(xaxis, savelowlengths, "With offline reply buffer", window, sigma, alpha)

    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Length (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_length.png')
    return fig

def variance_trapresult(endtoendtrap, savelowtrap, tejastrap, runs, episode, window, alpha, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_metric(xaxis, tejastrap, "Ours", window, sigma, alpha)
    plot_metric(xaxis, endtoendtrap, "With soft constraint", window, sigma, alpha)
    plot_metric(xaxis, savelowtrap, "With offline reply buffer", window, sigma, alpha)

    
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Trap (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_trap.png')
    return fig

def variance_speedresult(endtoendaverage_speed, savelowaverage_speed, tejasaverage_speed, runs, episode, window, alpha, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_metric(xaxis, tejasaverage_speed, "Ours", window, sigma, alpha)
    plot_metric(xaxis, endtoendaverage_speed, "With soft constraint", window, sigma, alpha)
    plot_metric(xaxis, savelowaverage_speed, "With offline reply buffer", window, sigma, alpha)

    
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Average Speed (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('average_speed.png')
    return fig

def variance_distanceresult(endtoenddistance, savelowdistance, tejasdistance, runs, episode, window, alpha, sigma=2):
    xaxis = np.linspace(1, episode, episode)
    fig = plt.figure(figsize=(10, 5))
    plot_metric(xaxis, tejasdistance, "Ours", window, sigma, alpha)
    plot_metric(xaxis, endtoenddistance, "With soft constraint", window, sigma, alpha)
    plot_metric(xaxis, savelowdistance, "With offline reply buffer", window, sigma, alpha)

    
    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Episode Distance (Smoothed)", fontsize=17)
    plt.legend(fontsize=12)
    fig.savefig('env_distance.png')
    return fig

if __name__ == '__main__':
    runs = 5
    episode = 3000
    # Adjust the smoothing parameters as needed:
    smoothing_window = 15  # larger window for more smoothing
    alpha = 0.3            # transparency for fill_between
    gaussian_sigma = 6     # sigma for additional Gaussian smoothing
    
    # Read data from files for all methods
    endtoendlist, savelowlist, tejaslist = read_data(runs, episode)
    endtoendreward, endtoendlengths, endtoenddistance, endtoendcrash_times, endtoendtrap, endtoendaverage_speed = endtoendlist
    savelowreward, savelowlengths, savelowdistance, savelowcrash_times, savelowtrap, savelowaverage_speed = savelowlist
    tejasreward, tejaslengths, tejasdistance, tejascrash_times, tejastrap, tejasaverage_speed = tejaslist

    # (Optional) Compute averages over the last 50 episodes if needed
    rewardend = np.average(endtoendreward[:, -50], axis=0)
    rewardhdqn = np.average(savelowreward[:, -50], axis=0)
    rewardtajas = np.average(tejasreward[:, -50], axis=0)
    trapend = np.average(endtoendtrap[:, -50], axis=0)
    traphdqn = np.average(savelowtrap[:, -50], axis=0)
    traptajas = np.average(tejastrap[:, -50], axis=0)
    speedend = np.average(endtoendaverage_speed[:, -50], axis=0)
    speedhdqn = np.average(savelowaverage_speed[:, -50], axis=0)
    speedtajas = np.average(tejasaverage_speed[:, -50], axis=0)

    # Generate and show the smoothed plots for each metric
    fig2 = variance_rewardresult(endtoendreward, savelowreward, tejasreward, runs, episode, smoothing_window, alpha, gaussian_sigma)
    fig2.show()
    fig3 = variance_crashresult(endtoendcrash_times, savelowcrash_times, tejascrash_times, runs, episode, smoothing_window, alpha, gaussian_sigma)
    fig3.show()
    fig4 = variance_lengthresult(endtoendlengths, savelowlengths, tejaslengths, runs, episode, smoothing_window, alpha, gaussian_sigma)
    fig4.show()
    fig5 = variance_trapresult(endtoendtrap, savelowtrap, tejastrap, runs, episode, smoothing_window, alpha, gaussian_sigma)
    fig5.show()
    fig6 = variance_speedresult(endtoendaverage_speed, savelowaverage_speed, tejasaverage_speed, runs, episode, smoothing_window, alpha, gaussian_sigma)
    fig6.show()
    fig7 = variance_distanceresult(endtoenddistance, savelowdistance, tejasdistance, runs, episode, smoothing_window, alpha, gaussian_sigma)
    fig7.show()
