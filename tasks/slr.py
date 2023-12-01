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

def get_participant_reps(p, a):
    coda_hip = utils.smooth_array(p['hip']['coda'], [0,1,2])
    coda_ankle = utils.smooth_array(p['ankle']['coda'], [0,1,2])
    coda_ankle_ref = np.ones_like(coda_ankle)*coda_ankle[0]
    op_hip = utils.smooth_array(p['hip']['op'], [0,1]) #* height_ptm[a] 
    op_ankle = utils.smooth_array(p['ankle']['op'], [0,1]) #* height_ptm[a]
    op_ankle_ref = np.ones_like(op_ankle)*op_ankle[0]
    coda_angle = utils.get_angle(coda_ankle_ref,coda_hip,coda_ankle,ndims=3)
    omc = savgol_filter(coda_angle, 21, 2)
    op_angle = utils.get_angle(op_ankle_ref, op_hip, op_ankle)
    mmc = savgol_filter(op_angle, 21, 2)
    if a+3 == 11:
        omc = omc[1000:]
        mmc = mmc[400:]
    if a+3 == 15:
        omc = omc[2500:]
        mmc = mmc[750:]
    coda_seg, coda_reps = utils.get_reps(omc, fps=100, plot=False, d=6,
                                        forward=3.7, rewind=3.7)
    op_seg, op_reps = utils.get_reps(mmc, fps=30, plot=False, d=6,
                                    forward=3.7, rewind=3.7)
    # select last three reps
    coda_seg, coda_reps = coda_seg[-3:], coda_reps[-3:]
    op_seg, op_reps = op_seg[-3:], op_reps[-3:]
    return coda_reps, op_reps

def get_ranges_of_motion(P):
    rom_dict = {}
    exclude = [5,7,9,10,12,13,16,17,18]
    for a in range(len(P)):
        if a+3 in exclude:
            continue
        coda_reps, op_reps = get_participant_reps(P[a], a)
        for i in range(3):
            coda_rep = coda_reps[i] - min(coda_reps[i])
            op_rep = op_reps[i] - min(op_reps[i])
            op_rep = resample(op_rep, len(coda_rep))
            coda_key = f'coda_{i+1}'
            op_key = f'op_{i+1}'
            coda_peak = np.round(max(coda_rep), 2)
            op_peak = np.round(max(op_rep), 2)
            if coda_key in rom_dict.keys():
                rom_dict[coda_key].append(coda_peak)
            else:
                rom_dict[coda_key] = [coda_peak]
            if op_key in rom_dict.keys():
                rom_dict[op_key].append(op_peak)
            else:
                rom_dict[op_key] = [op_peak]          
    return pd.DataFrame(data=rom_dict)

def flatten_reps(df, device):
    return np.array([df[f'{device}_{i}'] for i in [1,2,3]]).flatten()

def ba_plots(subjects):
    rom_df = get_ranges_of_motion(subjects)
    roms = {
        'omc': flatten_reps(rom_df, 'coda'),
        'mmc': flatten_reps(rom_df, 'op')
    }
    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi=100)
    plot_ba(roms['omc'], roms['mmc'],
        title='Straight leg raise range of motion (degs)', ax=ax)
    fig.suptitle('')
    fig.supxlabel('Means')
    plt.show()
    return roms, rom_df

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
            'score': df[f'{device}_{i}'],
            'rater': [f'rep{i}' for j in range(1,L+1)]
        })
        reps.append(rep)
    jumps_icc = pd.concat(reps)
    icc = pg.intraclass_corr(data=jumps_icc, targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_metrics(subjects):
    roms, rom_df = ba_plots(subjects)
    rom = {
        'Metric': 'Range of motion',
        'Task': 'Straight leg raise',
        'PTM Ref': 'N/A',
        'Ground Truth': 'Optical motion capture',
        'MAE': np.round(MAE(roms['omc'], roms['mmc']),2),
        'Reliability': get_retest(rom_df, device='op'),
        'ICC': get_icc(roms['omc'], roms['mmc'])
    }
    metrics_df = pd.DataFrame(rom, index=[0])
    return metrics_df
