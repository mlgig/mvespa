import numpy as np
import pandas as pd
import statsmodels.api as sm
import pingouin as pg
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
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

def get_mmc_values(p, mm_px):
    task = 'rjt'
    part = 'hip'
    op = utils.smoothen(p[task][part]['op'][:,1]) * mm_px
    op -= min(op)
    # find peaks
    op_rest = op[:50].mean()
    op_peaks, _ = find_peaks(op, height=op[0]+60, distance=10)
    op_jh = (op[op_peaks][:10] - op_rest)/10
    # get flight times
    op_ft = utils.rjt_flight_times(op, op_rest+15, fps=30)[:10]
    # get contact times
    op_ct = utils.rjt_contact_times(op, op_peaks, op_rest, fps=30)[:10]
    return op_ft, op_ct, op_jh

def get_force_plate_values(p):
    task = 'rjt'
    F = p[task]['force']
    f1 = F[1] + F[3]
    f2 = F[2] + F[4]
    f = f1 if f1.max() > f2.max() else f2
    force = f - f.min()
    thresh = min(force)
    ct = np.diff(np.where(force<=thresh+10)[0])
    ft = np.diff(np.where(force>thresh+30)[0])
    fp_ct = np.round(ct[ct>100]/1000, 2)[:10]
    fp_ft = np.round(ft[ft>100]/1000, 2)[:10]
    fp_jh = np.round(utils.jump_heights(fp_ft)*100, 2)
    return fp_ft, fp_ct, fp_jh

def get_performance(P, ptm):
    FP_JH = []
    FP_CT = []
    FP_FT = []
    OP_JH = []
    OP_CT = []
    OP_FT = []
    ptm_opts = {
        'gravity': PTM.gravity(visualize=False),
        'height': PTM.height(task='rjt') 
    }
    ptm = ptm_opts[ptm]
    for a, p in enumerate(P):
        op_ft, op_ct, op_jh = get_mmc_values(p, ptm[a])
        OP_FT.append(op_ft)
        OP_CT.append(op_ct)
        OP_JH.append(op_jh)
        fp_ft, fp_ct, fp_jh = get_force_plate_values(p)
        FP_FT.append(fp_ft)
        FP_CT.append(fp_ct)
        FP_JH.append(fp_jh)
    OP_FT = np.array(OP_FT)
    OP_CT = np.array(OP_CT)
    OP_JH = np.round(np.array(OP_JH), 2)
    FP_FT = np.array(FP_FT)
    FP_CT = np.array(FP_CT)
    FP_JH = np.array(FP_JH)
    op_cols = [f'o{i+1}' for i in range(10)]
    fp_cols = [f'f{i+1}' for i in range(10)]
    flight_df = pd.DataFrame.from_records(np.hstack([FP_FT,OP_FT]),
                                            columns=fp_cols+op_cols)
    contact_df = pd.DataFrame.from_records(np.hstack([FP_CT,OP_CT]),
                                            columns=fp_cols+op_cols)
    jump_df = pd.DataFrame.from_records(np.hstack([FP_JH,OP_JH]),
                                            columns=fp_cols+op_cols)
    return flight_df, contact_df, jump_df

def flatten_reps(df, col):
    return np.array(
        [df[f'{col}{i+1}'] for i in range(10)]
    ).flatten()

def ba_plots(subjects, ptm):
    flight_df, contact_df, _ = get_performance(subjects, ptm)
    flight_times = {
        'fp': flatten_reps(flight_df, 'f'),
        'mmc': flatten_reps(flight_df, 'o')
    }
    contact_times = {
        'fp': flatten_reps(contact_df, 'f'),
        'mmc': flatten_reps(contact_df, 'o')
    }
    fig, axs = plt.subplots(1,2, figsize=(10,3), dpi=100)
    plot_ba(flight_times['fp'], flight_times['mmc'],
        title='flight times (secs)', ax=axs[0])
    plot_ba(contact_times['fp'], contact_times['mmc'],
        title='contact times (secs)', ax=axs[1], label_y=False)
    fig.suptitle('Repeated jump test')
    fig.supxlabel('Means')
    plt.show()
    return flight_times, contact_times, flight_df, contact_df

def get_icc(x, y, devices=['FP', 'MMC']):
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
            'score': df[f'{device}{i}'],
            'rater': [f'rep{i}' for j in range(1,L+1)]
        })
        reps.append(rep)
    jumps_icc = pd.concat(reps)
    icc = pg.intraclass_corr(data=jumps_icc, targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_metrics(subjects, ptm='height'):
    flight_times, contact_times, *dfs = ba_plots(subjects, ptm)
    flight_df, contact_df, = dfs
    ft = {
        'Metric': 'Flight time',
        'Task': 'Repeated jump test',
        'PTM Ref': ptm,
        'Ground Truth': 'Force plates',
        'MAE': np.round(MAE(flight_times['fp'], flight_times['mmc']),2),
        'Reliability': get_retest(flight_df, device='o'),
        'ICC': get_icc(flight_times['fp'], flight_times['mmc'])
    }
    ct = {
        'Metric': 'Contact time',
        'Task': 'Repeated jump test',
        'PTM Ref': ptm,
        'Ground Truth': 'Force plates',
        'MAE': np.round(MAE(contact_times['fp'], contact_times['mmc']),2),
        'Reliability': get_retest(contact_df, device='o'),
        'ICC': get_icc(contact_times['fp'], contact_times['mmc'])
    }
    metrics_df = pd.DataFrame(ft, index=[0])
    metrics_df.loc[1] = ct
    return metrics_df
