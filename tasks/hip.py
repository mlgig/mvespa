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
from utils import plot_ba, get_ratings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_participant_reps(p, a, task):
    coda_knee = p[task]['knee']['coda'][:,[1,2]]
    coda_ankle = p[task]['ankle']['coda'][:,[1,2]]
    coda_ankle_ref = np.ones_like(coda_ankle)*coda_ankle[0]
    coda_knee_ref = np.ones_like(coda_knee)*coda_knee[0]
    op_knee = p[task]['knee']['op']
    op_ankle = p[task]['ankle']['op']
    op_ankle_ref = np.ones_like(op_ankle)*op_ankle[0]
    op_knee_ref = np.ones_like(op_knee)*op_knee[0]
    omc = utils.intersection_angle(line1=[coda_ankle_ref, coda_knee_ref],
                                line2=[coda_ankle, coda_knee])
    mmc = utils.intersection_angle(line1=[op_ankle_ref, op_knee_ref],
                                line2=[op_ankle, op_knee])
    coda_seg, coda_reps = utils.get_reps(omc, fps=100, plot=False,
                                            d=6 if a in [4,5] else 8,
                                            forward=3.7, rewind=3.7)
    op_seg, op_reps = utils.get_reps(mmc, fps=30, plot=False, d=7,
                                    forward=3.7, rewind=3.7)
    # select last three reps
    coda_seg, coda_reps = coda_seg[-3:], coda_reps[-3:]
    op_seg, op_reps = op_seg[-3:], op_reps[-3:]
    return coda_reps, op_reps

def get_range_of_motion(P):
    rom_dict = {'hir': {}, 'her': {}}
    for a in range(len(P)):
        p = P[a]
        for t, task in enumerate(['hir', 'her']):
            if (a+3 == 9 and task=='her') or a+3==12:
                continue
            coda_reps, op_reps = get_participant_reps(p, a, task)
            for i in range(3):
                coda_rep = coda_reps[i] - min(coda_reps[i])
                op_rep = op_reps[i] - min(op_reps[i])
                op_rep = resample(op_rep, len(coda_rep))
                coda_key = f'coda_{task}_{i+1}'
                op_key = f'op_{task}_{i+1}'
                coda_peak = np.round(max(coda_rep), 2)
                op_peak = np.round(max(op_rep), 2)
                if coda_key in rom_dict[task].keys():
                    rom_dict[task][coda_key].append(coda_peak)
                else:
                    rom_dict[task][coda_key] = [coda_peak]
                if op_key in rom_dict[task].keys():
                    rom_dict[task][op_key].append(op_peak)
                else:
                    rom_dict[task][op_key] = [op_peak]
    her_df = pd.DataFrame(data=rom_dict['her'])
    hir_df = pd.DataFrame(data=rom_dict['hir'])
    return her_df, hir_df

def flatten_reps(df, device, task):
    return np.array([df[f'{device}_{task}_{i}'] for i in [1,2,3]]).flatten()

def ba_plots(subjects):
    her_df, hir_df = get_range_of_motion(subjects)
    hers = {
        'omc': flatten_reps(her_df, 'coda', 'her'),
        'mmc': flatten_reps(her_df, 'op', 'her')
    }
    hirs = {
        'omc': flatten_reps(hir_df, 'coda', 'hir'),
        'mmc': flatten_reps(hir_df, 'op', 'hir')
    }
    # the her<->hir flip here is intentional
    # to correct a flip during data collection
    fig, axs = plt.subplots(1,2, figsize=(10,3), dpi=100)
    plot_ba(hers['omc'], hers['mmc'],
        title='internal rotation (degs)', ax=axs[0])
    plot_ba(hirs['omc'], hirs['mmc'],
        title='external rotation', ax=axs[1], label_y=False)
    fig.suptitle('Hip rotation')
    fig.supxlabel('Means')
    plt.show()
    return hers, hirs, her_df, hir_df

def get_icc(x, y, devices=['OMC', 'MMC']):
    df = pd.concat([get_ratings(x, devices[0]),
                           get_ratings(y, devices[1])])
    icc = pg.intraclass_corr(data=df, targets='rep',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_retest(df, device, task):
    L = df.shape[0]
    reps = []
    for i in range(1,4):
        rep = pd.DataFrame({
            'sn': np.arange(1, L+1),
            'score': df[f'{device}_{task}_{i}'],
            'rater': [f'rep{i}' for j in range(1,L+1)]
        })
        reps.append(rep)
    jumps_icc = pd.concat(reps)
    icc = pg.intraclass_corr(data=jumps_icc, targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_metrics(subjects):
    hers, hirs, her_df, hir_df = ba_plots(subjects)
    # the her<->hir flip here is intentional
    # to correct a flip during data collection
    her = {
        'Metric': 'Range of motion',
        'Task': 'Hip internal rotation',
        'PTM Ref': 'N/A',
        'Ground Truth': 'Optical motion capture',
        'MAE': np.round(MAE(hers['omc'], hers['mmc']),2),
        'Reliability': get_retest(her_df, device='op', task='her'),
        'ICC': get_icc(hers['omc'], hers['mmc'])
    }
    hir = {
        'Metric': 'Range of motion',
        'Task': 'Hip external rotation',
        'PTM Ref': 'N/A',
        'Ground Truth': 'Optical motion capture',
        'MAE': np.round(MAE(hirs['omc'], hirs['mmc']),2),
        'Reliability': get_retest(hir_df, device='op', task='hir'),
        'ICC': get_icc(hirs['omc'], hirs['mmc'])
    }
    metrics_df = pd.DataFrame(her, index=[0])
    metrics_df.loc[1] = hir
    return metrics_df
