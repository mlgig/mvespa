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

segs = {
    'coda': [
        [400, 650, 860, 1100], #3
        [700, 1050, 1350, 1650], #4
        [650, 980, 1250, 1500], #5
        [450, 650, 810, 1150], #6
        [650, 1050, 1400, 1800], #7
        [500, 800, 1020, 1250], #8
        [700, 1050, 1400, 1750], #9
        [600, 800, 950, 1150], #10
        [650, 1100, 1550, 2100], #11
        [650, 800, 1000, 1200], #12
        [620, 920, 1200, 1500], #13
        [620, 950, 1300, 1600], #14
        [880, 1250, 1600, 1950], #15
        [500, 880, 1220, 1570], #16,
        [700, 1010, 1290, 1580], #17
        [600, 1000, 1400, 1750] #18
    ],
    'op': [
        [300, 390, 450, 530], #3
        [250, 380, 480, 580], #4
        [280, 400, 470, 570], #5
        [200, 260, 310, 380], #6
        [200, 340, 440, 550], #7
        [200, 290, 355, 430], #8
        [160, 275, 375, 500], #9
        [175, 238, 288, 348], #10
        [280, 420, 555, 750], #11
        [225, 300, 350, 425], #12
        [225, 325, 405, 500], #13
        [200, 280, 390, 490], #14
        [270, 410, 510, 620], #15
        [170, 290, 400, 500], #16
        [260, 360, 445, 540], #17
        [180, 325, 445, 560] #18
    ]
}

def interpolate(arr):
    if arr.ndim == 1:
        new_arr = np.where(
            arr > arr.mean(),
            arr.min(), arr)
    else:
        new_arr = np.zeros_like(arr)
        for i in range(arr.ndim):
            x = arr[:, i]
            new_arr[:, i] = np.where(
                x > x.mean(), x.min(), x)
    return new_arr

def sync(arr, fps, rw=1, fw=1):
    t = int(fps*0.5)
    trim = arr[t:-t]
    minima = trim.argmin() + t
    left_bound = minima - (fps * rw)
    right_bound = minima + (fps * fw)
    seg = arr[left_bound:right_bound+1]
    return seg

def get_roms(p, s):
    parts = ['hip', 'knee', 'ankle']
    coda_segs, op_segs = segs['coda'], segs['op']
    op_parts_reps = [
        utils.segment_squat(p['slsquat'][part]['op'], op_segs[s])
        for part in parts]
    coda_parts_reps = [
        utils.segment_squat(fill_nan(p['slsquat'][part]['coda']),
                            coda_segs[s]) for part in parts]
    op_roms = [utils.get_angle(
            op_parts_reps[0][i], op_parts_reps[1][i],
            interpolate(op_parts_reps[2][i]), ndims=2)
            for i in range(3)]
    coda_roms = [utils.get_angle(
            coda_parts_reps[0][i], coda_parts_reps[1][i],
            coda_parts_reps[2][i], ndims=3)
            for i in range(3)]
    return coda_roms, op_roms

def get_performance(P):
    rom_dict = {}
    mean_vel_dict = {}
    for s, p in enumerate(P):
        coda_roms, op_roms = get_roms(p, s)
        for k in range(3):
            op_rom = utils.flip_axis(savgol_filter(op_roms[k], 21, 2))
            coda_rom = utils.flip_axis(savgol_filter(coda_roms[k], 5, 2))
            op_peaks, _ = find_peaks(op_rom, prominence=0.9,
                                distance=len(op_rom)/1.5)
            coda_peaks, _ = find_peaks(coda_rom, prominence=0.9,
                                distance=len(coda_rom)/1.5)
            if s+3 in [4,5,14,17]:
                if s+3 == 4 and k==2:
                    coda_peaks = np.array([156])
                if s+3 == 5:
                    if k==0:
                        coda_peaks = np.array([177])
                    if k==1:
                        coda_peaks = np.array([140])
                if s+3 == 14:
                    if k==1:
                        coda_peaks = np.array([152])
                    if k==2:
                        coda_peaks = np.array([165])
                if s+3 == 17:
                    if k==0:
                        coda_peaks = np.array([198])
            # start from the peak and look backward to 0.8 secs
            coda_start = max(0, coda_peaks[0]-80)
            coda_rom = coda_rom[coda_start:coda_peaks[0]]
            op_start = max(0, op_peaks[0]-24)
            op_rom = op_rom[op_start:op_peaks[0]]
            coda_rom = resample(coda_rom, len(op_rom))
            op_rom -= min(op_rom)
            coda_rom -= min(coda_rom)
            op_rom = savgol_filter(op_rom, 5, 2)
            op_vel = np.round(np.gradient(op_rom)*30, 2)
            op_vel = np.where(op_vel < 0, 0, op_vel)
            coda_vel = np.round(np.gradient(coda_rom)*30, 2)
            coda_vel = np.where(coda_vel < 0, 0, coda_vel)
            coda_mean_vel = np.round(coda_vel[2:-2].mean(), 2)
            op_mean_vel = np.round(op_vel[2:-2].mean(), 2)
            coda_max_rom = np.round(max(coda_rom[2:-2]), 2)
            op_max_rom = np.round(max(op_rom[2:-2]), 2)
            coda_key = f'coda_rep_{k+1}'
            op_key = f'op_rep_{k+1}'
            if coda_key in rom_dict.keys():
                mean_vel_dict[f'{coda_key}'].append(coda_mean_vel)
                rom_dict[f'{coda_key}'].append(coda_max_rom)
            else:
                mean_vel_dict[f'{coda_key}'] = [coda_mean_vel]
                rom_dict[f'{coda_key}'] = [coda_max_rom]
            if op_key in rom_dict.keys():
                mean_vel_dict[f'{op_key}'].append(op_mean_vel)
                rom_dict[f'{op_key}'].append(op_max_rom)
            else:
                mean_vel_dict[f'{op_key}'] = [op_mean_vel]
                rom_dict[f'{op_key}'] = [op_max_rom]
    rom_df = pd.DataFrame(data=rom_dict)
    mean_vel_df = pd.DataFrame(data=mean_vel_dict)
    return rom_df, mean_vel_df           

def flatten_reps(df, device):
    return np.array([df[f'{device}_rep_{i}'] for i in [1,2,3]]).flatten()

def ba_plots(subjects):
    rom_df, mean_vel_df = get_performance(subjects)
    roms = {
        'omc': flatten_reps(rom_df, 'coda'),
        'mmc': flatten_reps(rom_df, 'op')
    }
    vels = {
        'omc': flatten_reps(mean_vel_df, 'coda'),
        'mmc': flatten_reps(mean_vel_df, 'op')
    }
    fig, axs = plt.subplots(1,2, figsize=(10,3), dpi=100)
    plot_ba(roms['omc'], roms['mmc'], title='range of motion (degs)', ax=axs[0])
    plot_ba(vels['omc'], vels['mmc'], title='mean angular velocity (degs/s)',
                ax=axs[1], label_y=False)
    fig.suptitle('Single leg squat')
    fig.supxlabel('Means')
    plt.show()
    return roms, vels, rom_df, mean_vel_df

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
        'Task': 'Single leg squat',
        'PTM Ref': 'N/A',
        'Ground Truth': 'Optical motion capture',
        'MAE': np.round(MAE(roms['omc'], roms['mmc']),2),
        'Reliability': get_retest(rom_df, device='op'),
        'ICC': get_icc(roms['omc'], roms['mmc'])
    }
    vel = {
        'Metric': 'Angular velocity',
        'Task': 'Single leg squat',
        'PTM Ref': 'N/A',
        'Ground Truth': 'Optical motion capture',
        'MAE': np.round(MAE(vels['omc'], vels['mmc']),2),
        'Reliability': get_retest(vel_df, device='op'),
        'ICC': get_icc(vels['omc'], vels['mmc'])
    }
    metrics_df = pd.DataFrame(rom, index=[0])
    metrics_df.loc[1] = vel
    return metrics_df
