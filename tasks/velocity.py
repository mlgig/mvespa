import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
import statsmodels.api as sm
import pingouin as pg
from scipy.signal import savgol_filter
from scipy.spatial import distance
from scipy.signal import resample
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import utils, PTM
from utils import plot_ba, get_ratings
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_performance(P, task, ptm='barbell'):
    ptm_opts = {
        'barbell': PTM.barbell(),
         # 'gravity': PTM.gravity(visualize=False),
        'height': PTM.height(task=task) 
    }
    ptm = ptm_opts[ptm]
    mean_vel_dict = {}
    peak_vel_dict = {}
    for a in range(len(P)):
        p = P[a]
        omc = savgol_filter(p['left']['coda'][:,2], 21, 2)
        mmc = savgol_filter(p['left']['op'][:,1], 21, 2)
        coda_seg, coda_reps = utils.get_reps(omc, fps=100, plot=False, d=1,
                                            forward=1.5, rewind=1.5)
        op_seg, op_reps = utils.get_reps(mmc, fps=30, plot=False, d=1,
                                        forward=1.5, rewind=1.5)
        # select last three reps
        coda_seg, coda_reps = coda_seg[-3:], coda_reps[-3:]
        op_seg, op_reps = op_seg[-3:], op_reps[-3:]
        for i in range(3):
            coda_ohp = deepcopy(coda_reps[i])[:-20]
            op_ohp = deepcopy(op_reps[i])
            coda_ohp -= coda_ohp.min()
            op_ohp -= op_ohp.min()
            op_conv = deepcopy(op_ohp) * ptm[a]
            op_conv = resample(op_conv, len(coda_ohp))[:-20]
            op_conv = utils.smooth_by_resampling(op_conv)
            # convert both from mm to m
            op_conv *= 0.001
            op_vel = savgol_filter(np.gradient(op_conv)*100, 21, 2)
            op_vel_pos = op_vel[4:np.argmax(op_conv)]
            op_zero_vel = np.argmin(abs(op_vel_pos[:np.argmax(op_vel_pos)]))
            op_peak = np.round(max(op_vel_pos), 2)
            op_mean = np.round(op_vel_pos[op_zero_vel:].mean(), 2)
            coda_ohp *= 0.001 # convert from mm to m
            coda_vel = savgol_filter(np.gradient(coda_ohp)*100, 21, 2)
            coda_vel_pos = coda_vel[:np.argmax(coda_ohp)]
            coda_zero_vel = np.argmin(abs(coda_vel_pos[:np.argmax(coda_vel_pos)]))
            coda_peak = np.round(max(coda_vel_pos), 2)
            coda_mean = np.round(coda_vel_pos[coda_zero_vel:].mean(), 2)
            coda_key = f'coda_ohp_{i+1}'
            op_key = f'op_ohp_{i+1}'
            if f'{coda_key}_mv' in mean_vel_dict.keys():
                mean_vel_dict[f'{coda_key}_mv'].append(coda_mean)
                peak_vel_dict[f'{coda_key}_pv'].append(coda_peak)
            else:
                mean_vel_dict[f'{coda_key}_mv'] = [coda_mean]
                peak_vel_dict[f'{coda_key}_pv'] = [coda_peak]
            if f'{op_key}_mv' in mean_vel_dict.keys():
                mean_vel_dict[f'{op_key}_mv'].append(op_mean)
                peak_vel_dict[f'{op_key}_pv'].append(op_peak)
            else:
                mean_vel_dict[f'{op_key}_mv'] = [op_mean]
                peak_vel_dict[f'{op_key}_pv'] = [op_peak]           
    peak_vel_df = pd.DataFrame(data=peak_vel_dict)
    mean_vel_df = pd.DataFrame(data=mean_vel_dict)
    return peak_vel_df, mean_vel_df

def flatten_reps(df, device, variant):
    return np.array([df[f'{device}_ohp_{i}_{variant}'] for i in [1,2,3]]).flatten()

def get_velocity(subjects, task, ptm='barbell'):
    peak_vel_df, mean_vel_df = get_performance(subjects, task=task, ptm=ptm)
    peak = {
        'omc': flatten_reps(peak_vel_df, 'coda', variant='pv'),
        'mmc': flatten_reps(peak_vel_df, 'op', variant='pv') 
    }
    mean = {
        'omc': flatten_reps(mean_vel_df, 'coda', variant='mv'),
        'mmc': flatten_reps(mean_vel_df, 'op', variant='mv') 
    }
    return peak, mean, peak_vel_df, mean_vel_df

def ba_plots(subjects, task, task_label, ptm):
    peak, mean, *dfs = get_velocity(subjects, task, ptm=ptm)
    fig, axs = plt.subplots(1,2, figsize=(10,3), dpi=100)
    plot_ba(mean['omc'], mean['mmc'], title='mean velocities (m/s)', ax=axs[0])
    plot_ba(peak['omc'], peak['mmc'], title='peak velocities (m/s)', ax=axs[1], label_y=False)
    fig.suptitle(f'{task_label} - {ptm} PTM')
    fig.supxlabel('Means')
    plt.show()
    return peak, mean, dfs

def get_retest(df, device, metric_variant):
    L = df.shape[0]
    reps = []
    for i in range(1,4):
        rep = pd.DataFrame({
            'sn': np.arange(1, L+1),
            'score': df[f'{device}_ohp_{i}_{metric_variant}'],
            'rater': [f'rep{i}' for j in range(1,L+1)]
        })
        reps.append(rep)
    jumps_icc = pd.concat(reps)
    icc = pg.intraclass_corr(data=jumps_icc, targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_icc(x, y, devices=['OMC', 'MMC']):
    df = pd.concat([get_ratings(x, devices[0]),
                           get_ratings(y, devices[1])])
    icc = pg.intraclass_corr(data=df, targets='rep',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_metrics(subjects, task, task_label, ptm='barbell'):
    peak, mean, dfs = ba_plots(subjects, task, task_label, ptm=ptm)
    peak_df, mean_df = dfs
    peak_vel = {
        'Metric': 'Peak velocity',
        'Task': task_label,
        'PTM Ref': ptm,
        'Ground Truth': 'Optical motion capure',
        'MAE': np.round(MAE(peak['omc'], peak['mmc']) ,2),
        'Reliability': get_retest(peak_df, device='op', metric_variant='pv'),
        'ICC': get_icc(peak['omc'], peak['mmc'])
    }
    mean_vel = {
        'Metric': 'Mean velocity',
        'Task': task_label,
        'PTM Ref': ptm,
        'Ground Truth': 'Optical motion capure',
        'MAE': np.round(MAE(mean['omc'], mean['mmc']) ,2),
        'Reliability': get_retest(mean_df, device='op', metric_variant='mv'),
        'ICC': get_icc(mean['omc'], mean['mmc'])
    }
    metrics_df = pd.DataFrame(peak_vel, index=[0])
    metrics_df.loc[1] = mean_vel
    return metrics_df
