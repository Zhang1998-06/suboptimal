import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

#there might be two different way of plotting this 
# mean+min/max
# mean + variance

endpath='set1/'
hierarchcial='set2/'
tejas='set3/'
sac='set4/'
def read_data(runs=None,episode=None):


    endtoendreward=np.zeros((runs,episode))
    endtoendlengths=np.zeros((runs,episode))
    endtoenddistance=np.zeros((runs,episode))
    endtoendcrash_times=np.zeros((runs,episode))
    endtoendtrap=np.zeros((runs,episode))
    endtoendaverage_speed=np.zeros((runs,episode))


    savelowreward=np.zeros((runs,episode))
    savelowlengths=np.zeros((runs,episode))
    savelowdistance=np.zeros((runs,episode))
    savelowcrash_times=np.zeros((runs,episode))
    savelowtrap=np.zeros((runs,episode))
    savelowaverage_speed=np.zeros((runs,episode))

    tejasreward=np.zeros((runs,episode))
    tejaslengths=np.zeros((runs,episode))
    tejasdistance=np.zeros((runs,episode))
    tejascrash_times=np.zeros((runs,episode))
    tejastrap=np.zeros((runs,episode))
    tejasaverage_speed=np.zeros((runs,episode))

    sacreward=np.zeros((runs,episode))
    saclengths=np.zeros((runs,episode))
    sacdistance=np.zeros((runs,episode))
    saccrash_times=np.zeros((runs,episode))
    sacstrap=np.zeros((runs,episode))
    sacaverage_speed=np.zeros((runs,episode))

    for i in range(runs):
        endstoendtats=np.loadtxt(endpath+str(i)+'hdqn.csv',delimiter=",")
        endstats = namedtuple("endstoendtats",["episode_lengths",  "episode_env_rewards","crash_times","trap","average_speed","distance"])
        endstats=endstats(endstoendtats[0],endstoendtats[1],endstoendtats[2],endstoendtats[3],endstoendtats[4],endstoendtats[5])
        endtoendreward[i]=endstats.episode_env_rewards
        endtoendlengths[i]=endstats.episode_lengths
        endtoenddistance[i]=endstats.distance
        endtoendcrash_times[i]=endstats.crash_times
        endtoendtrap[i]=endstats.trap
        endtoendaverage_speed[i]=endstats.average_speed
        endtoendlist=[endtoendreward,endtoendlengths,endtoenddistance,endtoendcrash_times,endtoendtrap,endtoendaverage_speed]


        savelowtats=np.loadtxt(hierarchcial+str(i)+'hdqn.csv',delimiter=",")
        htats = namedtuple("savelowtats",["episode_lengths",  "episode_env_rewards","crash_times","trap","average_speed","distance"])
        htats=htats(savelowtats[0],savelowtats[1],savelowtats[2],savelowtats[3],savelowtats[4],savelowtats[5])
        savelowreward[i]=htats.episode_env_rewards
        savelowlengths[i]=htats.episode_lengths
        savelowdistance[i]=htats.distance
        savelowcrash_times[i]=htats.crash_times
        savelowtrap[i]=htats.trap
        savelowaverage_speed[i]=htats.average_speed
        savelowlist=[savelowreward,savelowlengths, savelowdistance,savelowcrash_times,savelowtrap,savelowaverage_speed]

        tejasstats=np.loadtxt(tejas+str(i)+'hdqn.csv',delimiter=",")
        Tstats = namedtuple("tejastats",["episode_lengths",  "episode_env_rewards","crash_times","trap","average_speed","distance"])
        Ttats=Tstats(tejasstats[0],tejasstats[1],tejasstats[2],tejasstats[3],tejasstats[4],tejasstats[5])
        tejasreward[i]=Ttats.episode_env_rewards
        tejaslengths[i]=Ttats.episode_lengths
        tejasdistance[i]=Ttats.distance
        tejascrash_times[i]=Ttats.crash_times
        tejastrap[i]=Ttats.trap
        tejasaverage_speed[i]=Ttats.average_speed
        tejaslist=[tejasreward,tejaslengths, tejasdistance,tejascrash_times,tejastrap,tejasaverage_speed]

        sacstats=np.loadtxt(sac+str(i)+'end.csv',delimiter=",")
        sastats = namedtuple("sac",["episode_lengths",  "episode_env_rewards","crash_times","trap","average_speed","distance"])
        satats=sastats(sacstats[0],sacstats[1],sacstats[2],sacstats[3],sacstats[4],sacstats[5])
        sacreward[i]=satats.episode_env_rewards
        saclengths[i]=satats.episode_lengths
        sacdistance[i]=satats.distance
        saccrash_times[i]=satats.crash_times
        sacstrap[i]=satats.trap
        sacaverage_speed[i]=satats.average_speed
        saclist=[sacreward,saclengths, sacdistance,saccrash_times,sacstrap,sacaverage_speed]
    return endtoendlist,savelowlist,tejaslist,saclist
    
def variance_rewardresult(endtoendreward,savelowreward,tajasreward,SACreward,runs,episode,smoothing_window):
#endtoendreward,hdqn_reward,highlevel_reward,runs,episode,smoothing_window
    xaxis=np.linspace(1,episode,episode)
    fig2 = plt.figure(figsize=(10,5))

    endmean_run=np.average(endtoendreward,axis=0)
    endmeanrun_smoothed = pd.Series(endmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endvariance=np.std(endtoendreward,axis=0)
    endlower=endmean_run+endvariance
    endlower_smoothed= pd.Series(endlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endupper=endmean_run-endvariance
    endupper_smoothed= pd.Series(endupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,endlower_smoothed,endupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,endmeanrun_smoothed,label="Ours")

    hdqnmean_run=np.average(savelowreward,axis=0)
    hdqnmeanrun_smoothed = pd.Series(hdqnmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnvariance=np.std(savelowreward,axis=0)
    hdqnlower=hdqnmean_run+hdqnvariance
    hdqnlower_smoothed= pd.Series(hdqnlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnupper=hdqnmean_run-hdqnvariance
    hdqnupper_smoothed= pd.Series(hdqnupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,hdqnlower_smoothed,hdqnupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,hdqnmeanrun_smoothed,label="GAIL")

    tajasmean_run=np.average(tajasreward,axis=0)
    tajasmeanrun_smoothed = pd.Series(tajasmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasvariance=np.std(tajasreward,axis=0)
    tajaslower=tajasmean_run+tajasvariance
    tajaslower_smoothed= pd.Series(tajaslower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasupper=tajasmean_run-tajasvariance
    tajasupper_smoothed= pd.Series(tajasupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,tajaslower_smoothed,tajasupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,tajasmeanrun_smoothed,label="CQL")

    SACmean_run=np.average(SACreward,axis=0)
    SACmeanrun_smoothed = pd.Series(SACmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACvariance=np.std(SACreward,axis=0)
    SAClower=SACmean_run+SACvariance
    SAClower_smoothed= pd.Series(SAClower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACupper=SACmean_run-SACvariance
    SACupper_smoothed= pd.Series(SACupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,SAClower_smoothed,SACupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,SACmeanrun_smoothed,label="SAC")

    plt.xlabel("Epsiode",fontsize=17)
    plt.ylabel("Epsiode Reward (Smoothed)",fontsize=17)
    plt.legend(fontsize=12)
    fig2.savefig('env_Reward.png')
    return fig2
def variance_crashresult(endtoendreward,savelowreward,tajasreward,SACreward,runs,episode,smoothing_window):
#endtoendreward,hdqn_reward,highlevel_reward,runs,episode,smoothing_window
    xaxis=np.linspace(1,episode,episode)
    fig3 = plt.figure(figsize=(10,5))

    endmean_run=np.average(endtoendreward,axis=0)
    endmeanrun_smoothed = pd.Series(endmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endvariance=np.std(endtoendreward,axis=0)
    endlower=endmean_run+endvariance
    endlower_smoothed= pd.Series(endlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endupper=endmean_run-endvariance
    endupper_smoothed= pd.Series(endupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,endlower_smoothed,endupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,endmeanrun_smoothed,label="Ours")

    hdqnmean_run=np.average(savelowreward,axis=0)
    hdqnmeanrun_smoothed = pd.Series(hdqnmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnvariance=np.std(savelowreward,axis=0)
    hdqnlower=hdqnmean_run+hdqnvariance
    hdqnlower_smoothed= pd.Series(hdqnlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnupper=hdqnmean_run-hdqnvariance
    hdqnupper_smoothed= pd.Series(hdqnupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,hdqnlower_smoothed,hdqnupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,hdqnmeanrun_smoothed,label="GAIL")

    tajasmean_run=np.average(tajasreward,axis=0)
    tajasmeanrun_smoothed = pd.Series(tajasmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasvariance=np.std(tajasreward,axis=0)
    tajaslower=tajasmean_run+tajasvariance
    tajaslower_smoothed= pd.Series(tajaslower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasupper=tajasmean_run-tajasvariance
    tajasupper_smoothed= pd.Series(tajasupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,tajaslower_smoothed,tajasupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,tajasmeanrun_smoothed,label="CQL")

    SACmean_run=np.average(SACreward,axis=0)
    SACmeanrun_smoothed = pd.Series(SACmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACvariance=np.std(SACreward,axis=0)
    SAClower=SACmean_run+SACvariance
    SAClower_smoothed= pd.Series(SAClower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACupper=SACmean_run-SACvariance
    SACupper_smoothed= pd.Series(SACupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,SAClower_smoothed,SACupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,SACmeanrun_smoothed,label="SAC")

    plt.xlabel("Epsiode",fontsize=17)
    plt.ylabel("Epsiode crashtimes (Smoothed)",fontsize=17)
    plt.legend(fontsize=12)
    fig3.savefig('env_crash.png')
    return fig3

def variance_lengthresult(endtoendreward,savelowreward,tajasreward,SACreward,runs,episode,smoothing_window):
#endtoendreward,hdqn_reward,highlevel_reward,runs,episode,smoothing_window
    xaxis=np.linspace(1,episode,episode)
    fig4 = plt.figure(figsize=(10,5))

    endmean_run=np.average(endtoendreward,axis=0)
    endmeanrun_smoothed = pd.Series(endmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endvariance=np.std(endtoendreward,axis=0)
    endlower=endmean_run+endvariance
    endlower_smoothed= pd.Series(endlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endupper=endmean_run-endvariance
    endupper_smoothed= pd.Series(endupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,endlower_smoothed,endupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,endmeanrun_smoothed,label="Ours")

    hdqnmean_run=np.average(savelowreward,axis=0)
    hdqnmeanrun_smoothed = pd.Series(hdqnmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnvariance=np.std(savelowreward,axis=0)
    hdqnlower=hdqnmean_run+hdqnvariance
    hdqnlower_smoothed= pd.Series(hdqnlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnupper=hdqnmean_run-hdqnvariance
    hdqnupper_smoothed= pd.Series(hdqnupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,hdqnlower_smoothed,hdqnupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,hdqnmeanrun_smoothed,label="GAIL")

    tajasmean_run=np.average(tajasreward,axis=0)
    tajasmeanrun_smoothed = pd.Series(tajasmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasvariance=np.std(tajasreward,axis=0)
    tajaslower=tajasmean_run+tajasvariance
    tajaslower_smoothed= pd.Series(tajaslower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasupper=tajasmean_run-tajasvariance
    tajasupper_smoothed= pd.Series(tajasupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,tajaslower_smoothed,tajasupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,tajasmeanrun_smoothed,label="CQL")

    SACmean_run=np.average(SACreward,axis=0)
    SACmeanrun_smoothed = pd.Series(SACmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACvariance=np.std(SACreward,axis=0)
    SAClower=SACmean_run+SACvariance
    SAClower_smoothed= pd.Series(SAClower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACupper=SACmean_run-SACvariance
    SACupper_smoothed= pd.Series(SACupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,SAClower_smoothed,SACupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,SACmeanrun_smoothed,label="SAC")
    plt.xlabel("Epsiode",fontsize=17)
    plt.ylabel("Epsiode length (Smoothed)",fontsize=17)
    plt.legend(fontsize=12)
    fig4.savefig('env_length.png')
    return fig4

def variance_trapresult(endtoendreward,savelowreward,tajasreward,SACreward,runs,episode,smoothing_window):
#endtoendreward,hdqn_reward,highlevel_reward,runs,episode,smoothing_window
    xaxis=np.linspace(1,episode,episode)
    fig5 = plt.figure(figsize=(10,5))

    endmean_run=np.average(endtoendreward,axis=0)
    endmeanrun_smoothed = pd.Series(endmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endvariance=np.std(endtoendreward,axis=0)
    endlower=endmean_run+endvariance
    endlower_smoothed= pd.Series(endlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endupper=endmean_run-endvariance
    endupper_smoothed= pd.Series(endupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,endlower_smoothed,endupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,endmeanrun_smoothed,label="Ours")

    hdqnmean_run=np.average(savelowreward,axis=0)
    hdqnmeanrun_smoothed = pd.Series(hdqnmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnvariance=np.std(savelowreward,axis=0)
    hdqnlower=hdqnmean_run+hdqnvariance
    hdqnlower_smoothed= pd.Series(hdqnlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnupper=hdqnmean_run-hdqnvariance
    hdqnupper_smoothed= pd.Series(hdqnupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,hdqnlower_smoothed,hdqnupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,hdqnmeanrun_smoothed,label="GAIL")

    tajasmean_run=np.average(tajasreward,axis=0)
    tajasmeanrun_smoothed = pd.Series(tajasmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasvariance=np.std(tajasreward,axis=0)
    tajaslower=tajasmean_run+tajasvariance
    tajaslower_smoothed= pd.Series(tajaslower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasupper=tajasmean_run-tajasvariance
    tajasupper_smoothed= pd.Series(tajasupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,tajaslower_smoothed,tajasupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,tajasmeanrun_smoothed,label="CQL")

    SACmean_run=np.average(SACreward,axis=0)
    SACmeanrun_smoothed = pd.Series(SACmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACvariance=np.std(SACreward,axis=0)
    SAClower=SACmean_run+SACvariance
    SAClower_smoothed= pd.Series(SAClower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACupper=SACmean_run-SACvariance
    SACupper_smoothed= pd.Series(SACupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,SAClower_smoothed,SACupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,SACmeanrun_smoothed,label="SAC")
    plt.xlabel("Epsiode",fontsize=17)
    plt.ylabel("Epsiode trap (Smoothed)",fontsize=17)
    plt.legend(fontsize=12)
    fig5.savefig('env trap.png')
    return fig5

def variance_speedresult(endtoendreward,savelowreward,tajasreward,SACreward,runs,episode,smoothing_window):
#endtoendreward,hdqn_reward,highlevel_reward,runs,episode,smoothing_window
    xaxis=np.linspace(1,episode,episode)
    fig6 = plt.figure(figsize=(10,5))

    endmean_run=np.average(endtoendreward,axis=0)
    endmeanrun_smoothed = pd.Series(endmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endvariance=np.std(endtoendreward,axis=0)
    endlower=endmean_run+endvariance
    endlower_smoothed= pd.Series(endlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endupper=endmean_run-endvariance
    endupper_smoothed= pd.Series(endupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,endlower_smoothed,endupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,endmeanrun_smoothed,label="Ours")

    hdqnmean_run=np.average(savelowreward,axis=0)
    hdqnmeanrun_smoothed = pd.Series(hdqnmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnvariance=np.std(savelowreward,axis=0)
    hdqnlower=hdqnmean_run+hdqnvariance
    hdqnlower_smoothed= pd.Series(hdqnlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnupper=hdqnmean_run-hdqnvariance
    hdqnupper_smoothed= pd.Series(hdqnupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,hdqnlower_smoothed,hdqnupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,hdqnmeanrun_smoothed,label="GAIL")

    tajasmean_run=np.average(tajasreward,axis=0)
    tajasmeanrun_smoothed = pd.Series(tajasmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasvariance=np.std(tajasreward,axis=0)
    tajaslower=tajasmean_run+tajasvariance
    tajaslower_smoothed= pd.Series(tajaslower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasupper=tajasmean_run-tajasvariance
    tajasupper_smoothed= pd.Series(tajasupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,tajaslower_smoothed,tajasupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,tajasmeanrun_smoothed,label="CQL")

    SACmean_run=np.average(SACreward,axis=0)
    SACmeanrun_smoothed = pd.Series(SACmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACvariance=np.std(SACreward,axis=0)
    SAClower=SACmean_run+SACvariance
    SAClower_smoothed= pd.Series(SAClower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACupper=SACmean_run-SACvariance
    SACupper_smoothed= pd.Series(SACupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,SAClower_smoothed,SACupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,SACmeanrun_smoothed,label="SAC")
    plt.xlabel("Epsiode",fontsize=17)
    plt.ylabel("Epsiode average_speed (Smoothed)",fontsize=17)
    plt.legend(fontsize=12)
    fig6.savefig('average_speed.png')
    return fig6

def variance_distanceresult(endtoendreward,savelowreward,tajasreward,SACreward,runs,episode,smoothing_window):
#endtoendreward,hdqn_reward,highlevel_reward,runs,episode,smoothing_window
    xaxis=np.linspace(1,episode,episode)
    fig7 = plt.figure(figsize=(10,5))

    endmean_run=np.average(endtoendreward,axis=0)
    endmeanrun_smoothed = pd.Series(endmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endvariance=np.std(endtoendreward,axis=0)
    endlower=endmean_run+endvariance
    endlower_smoothed= pd.Series(endlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    endupper=endmean_run-endvariance
    endupper_smoothed= pd.Series(endupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,endlower_smoothed,endupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,endmeanrun_smoothed,label="Ours")

    hdqnmean_run=np.average(savelowreward,axis=0)
    hdqnmeanrun_smoothed = pd.Series(hdqnmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnvariance=np.std(savelowreward,axis=0)
    hdqnlower=hdqnmean_run+hdqnvariance
    hdqnlower_smoothed= pd.Series(hdqnlower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    hdqnupper=hdqnmean_run-hdqnvariance
    hdqnupper_smoothed= pd.Series(hdqnupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,hdqnlower_smoothed,hdqnupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,hdqnmeanrun_smoothed,label="GAIL")

    tajasmean_run=np.average(tajasreward,axis=0)
    tajasmeanrun_smoothed = pd.Series(tajasmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasvariance=np.std(tajasreward,axis=0)
    tajaslower=tajasmean_run+tajasvariance
    tajaslower_smoothed= pd.Series(tajaslower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    tajasupper=tajasmean_run-tajasvariance
    tajasupper_smoothed= pd.Series(tajasupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,tajaslower_smoothed,tajasupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,tajasmeanrun_smoothed,label="CQL")

    SACmean_run=np.average(SACreward,axis=0)
    SACmeanrun_smoothed = pd.Series(SACmean_run).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACvariance=np.std(SACreward,axis=0)
    SAClower=SACmean_run+SACvariance
    SAClower_smoothed= pd.Series(SAClower).rolling(smoothing_window, min_periods=smoothing_window).mean()
    SACupper=SACmean_run-SACvariance
    SACupper_smoothed= pd.Series(SACupper).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #fig,ax= plt.subplots()
    plt.fill_between(xaxis,SAClower_smoothed,SACupper_smoothed,alpha=0.5,linewidth=0)
    plt.plot(xaxis,SACmeanrun_smoothed,label="SAC")
    plt.xlabel("Epsiode",fontsize=17)
    plt.ylabel("Epsiode distance (Smoothed)",fontsize=17)
    plt.legend(fontsize=12)
    fig7.savefig('env distance.png')
    return fig7



runs=5
episode=2000
smoothing_window=10

endtoendlist,savelowlist,tajaslist,SAClist=read_data(runs,episode)
endtoendreward,endtoendlengths,endtoenddistance,endtoendcrash_times,endtoendtrap,endtoendaverage_speed=endtoendlist
savelowreward,savelowlengths, savelowdistance,savelowcrash_times,savelowtrap,savelowaverage_speed=savelowlist
tajasreward,tajaslengths, tajasdistance,tajascrash_times,tajastrap,tajasaverage_speed=tajaslist
SACreward,SAClengths, SACdistance,SACcrash_times,SACtrap,SACaverage_speed=SAClist
rewardend=np.average(endtoendreward[:,-50],axis=0)
rewardhdqn=np.average(savelowreward[:,-50],axis=0)
rewardtajas=np.average(tajasreward[:,-50],axis=0)

SACtajas=np.average(SACreward[:,-50],axis=0)

trapend=np.average(endtoendtrap[:,-50],axis=0)
traphdqn=np.average(savelowtrap[:,-50],axis=0)
traptajas=np.average(tajastrap[:,-50],axis=0)
trapSAC=np.average(SACtrap[:,-50],axis=0)

speedend=np.average(endtoendaverage_speed[:,-50],axis=0)
speedhdqn=np.average(savelowaverage_speed[:,-50],axis=0)
speedtajas=np.average(tajasaverage_speed[:,-50],axis=0)
speedSAC=np.average(SACaverage_speed[:,-50],axis=0)

fig2=variance_rewardresult(endtoendreward,savelowreward,tajasreward,SACreward,runs,episode,smoothing_window=10)
fig2.show()
fig3=variance_crashresult(endtoendcrash_times,savelowcrash_times,tajascrash_times,SACcrash_times,runs,episode,smoothing_window=10)
fig3.show()
fig4=variance_lengthresult(endtoendlengths,savelowlengths,tajaslengths,SAClengths,runs,episode,smoothing_window=10)
fig4.show()
fig5=variance_trapresult(endtoendtrap,savelowtrap,tajastrap,SACtrap,runs,episode,smoothing_window=10)
fig5.show()
fig6=variance_speedresult(endtoendaverage_speed,savelowaverage_speed,tajasaverage_speed,SACaverage_speed,runs,episode,smoothing_window=10)
fig6.show()
fig7=variance_distanceresult(endtoenddistance,savelowdistance,tajasdistance,SACdistance,runs,episode,smoothing_window=10)
fig7.show()
