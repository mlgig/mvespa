import numpy as np
import pandas as pd
import statsmodels.api as sm
import pingouin as pg
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, resample
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import utils, PTM
from utils import plot_ba, get_ratings, fill_nan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def retrieve_angles(p, idx):
    start = 2500 if idx == 5 else 1500
    coda_hip = fill_nan(p['nordic']['hip']['coda'])[start:]
    coda_knee = fill_nan(p['nordic']['knee']['coda'])[start:]
    coda_ankle = fill_nan(p['nordic']['ankle']['coda'])[start:]
    coda_angles = utils.get_angle(coda_hip, coda_knee, coda_ankle,
                                 ndims=3)
    op_hip = utils.smooth_array(p['nordic']['hip']['op'], (0,1))[600:]
    op_knee = utils.smooth_array(p['nordic']['knee']['op'], (0,1))[600:]
    op_ankle = utils.smooth_array(p['nordic']['ankle']['op'], (0,1))[600:]
    op_angles = utils.get_angle(op_hip, op_knee, op_ankle)
    
    return coda_angles, op_angles

def get_participant_reps(p, a):
    omc, mmc = retrieve_angles(p, idx=a+3)
    omc = savgol_filter(omc, 5, 2)
    mmc = savgol_filter(mmc, 5, 2)
    coda_seg, coda_reps = utils.get_reps(omc, fps=100, plot=False, d=5,
                                        forward=0.1, rewind=2.5)
    op_seg, op_reps = utils.get_reps(mmc, fps=30, plot=False, d=5,
                                    forward=0.1, rewind=2.5)
    # select first three reps
    coda_seg, coda_reps = coda_seg[:3], coda_reps[:3]
    op_seg, op_reps = op_seg[:3], op_reps[:3]
    return coda_reps, op_reps

def get_performance(P):
    mean_vel_dict = {}
    rom_dict = {}
    for a in range(len(P)):
        coda_reps, op_reps = get_participant_reps(P[a], a)
        for i in range(3):
            coda_rep = coda_reps[i] - min(coda_reps[i])
            op_rep = op_reps[i] - min(op_reps[i])
            coda_rom = np.round(max(coda_rep), 2)
            op_rep = resample(op_rep, len(coda_rep))[:-5]
            op_rep = savgol_filter(op_rep, 21, 2)
            op_rom = np.round(max(op_rep), 2)
            coda_vel = savgol_filter(np.gradient(coda_rep)*100, 5, 2)[10:]
            coda_mean = np.round(coda_vel.mean(), 2)
            op_vel = savgol_filter(np.gradient(op_rep)*100, 21, 2)[10:]
            op_mean = np.round(op_vel.mean(), 2)
            coda_key = f'coda_rep_{i+1}'
            op_key = f'op_rep_{i+1}'
            if coda_key in rom_dict.keys():
                mean_vel_dict[f'{coda_key}'].append(coda_mean)
                rom_dict[f'{coda_key}'].append(coda_rom)
            else:
                mean_vel_dict[f'{coda_key}'] = [coda_mean]
                rom_dict[f'{coda_key}'] = [coda_rom]
            if op_key in rom_dict.keys():
                mean_vel_dict[f'{op_key}'].append(op_mean)
                rom_dict[f'{op_key}'].append(op_rom)
            else:
                mean_vel_dict[f'{op_key}'] = [op_mean]
                rom_dict[f'{op_key}'] = [op_rom]           
    rom_df = pd.DataFrame(data=rom_dict)
    vel_df = pd.DataFrame(data=mean_vel_dict)
    return rom_df, vel_df     

def flatten_reps(df, device):
    return np.array([df[f'{device}_rep_{i}'] for i in [1,2,3]]).flatten()

def ba_plots(subjects):
    rom_df, vel_df = get_performance(subjects)
    roms = {
        'omc': flatten_reps(rom_df, 'coda'),
        'mmc': flatten_reps(rom_df, 'op')
    }
    vel = {
        'omc': flatten_reps(vel_df, 'coda'),
        'mmc': flatten_reps(vel_df, 'op')
    }
    fig, axs = plt.subplots(1,2, figsize=(10,3), dpi=100)
    plot_ba(roms['omc'], roms['mmc'],
        title='range of motion (degs)', ax=axs[0])
    plot_ba(vel['omc'], vel['mmc'],
        title='angular velocity (deg/s)', ax=axs[1], label_y=False)
    fig.suptitle('Nordic curl')
    fig.supxlabel('Means')
    plt.show()
    return roms, vel, rom_df, vel_df

def get_icc(x, y, devices=['OMC', 'MMC']):
    df = pd.concat([get_ratings(x, devices[0]),
                           get_ratings(y, devices[1])])
    icc = pg.intraclass_corr(data=df, targets='rep',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_retest(df, device):
    L = df.shape[0]
    reps = []
    for i in range(1,4):
        rep = pd.DataFrame({
            'sn': np.arange(1, L+1),
            'score': df[f'{device}_rep_{i}'],
            'rater': [f'rep{i}' for j in range(1,L+1)]
        })
        reps.append(rep)
    jumps_icc = pd.concat(reps)
    icc = pg.intraclass_corr(data=jumps_icc, targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_metrics(subjects):
    roms, vels, rom_df, vel_df = ba_plots(subjects)
    rom = {
        'Metric': 'Range of motion',
        'Task': 'Nordic curl',
        'PTM Ref': 'N/A',
        'Ground Truth': 'Optical motion capture',
        'MAE': np.round(MAE(roms['omc'], roms['mmc']),2),
        'Reliability': get_retest(rom_df, device='op'),
        'ICC': get_icc(roms['omc'], roms['mmc'])
    }
    vel = {
        'Metric': 'Angular velocity',
        'Task': 'Nordic curl',
        'PTM Ref': 'N/A',
        'Ground Truth': 'Optical motion capture',
        'MAE': np.round(MAE(vels['omc'], vels['mmc']),2),
        'Reliability': get_retest(vel_df, device='op'),
        'ICC': get_icc(vels['omc'], vels['mmc'])
    }
    metrics_df = pd.DataFrame(rom, index=[0])
    metrics_df.loc[1] = vel
    return metrics_df
