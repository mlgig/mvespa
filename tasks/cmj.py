import numpy as np
import pandas as pd
import statsmodels.api as sm
import pingouin as pg
from scipy.signal import savgol_filter
from scipy.signal import resample
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import utils, PTM
from utils import plot_ba
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def visualize_seg(P):
    p = P[2]
    coda_seg, coda_reps = utils.get_reps(p['bl']['hip']['coda'][:,2][250:],
                            fps=100, d=5, plot_peaks=True)
    op_seg, op_reps = utils.get_reps(p['bl']['hip']['op'][:,1], fps=30, d=5)
    fig, axs = plt.subplots(1,3, figsize=(7, 3), dpi=400, sharey=True)
    for i in range(3):
        coda_jump = savgol_filter(coda_reps[i], 9, 2)
        coda_jump -= min(coda_jump)
        op_jump = savgol_filter(op_reps[i], 9, 2)
        _, coda_factor = utils.scale_filter(coda_jump)
        op_jump = resample(op_jump, len(coda_jump))
        op_jump, _ = utils.scale_filter(op_jump)
        op_jump /= coda_factor
        axs[i].plot(coda_jump, label='OMC')
        axs[i].plot(op_jump, ':', label='MMC')
        axs[i].set_title(f'Jump {i+1}')
    fig.supylabel('hip y-position')
    fig.supxlabel('Frame number')
    fig.tight_layout()
    sns.despine()
    plt.legend()
    plt.show()
    return op_reps

def visualize_cmj(P, ptm):
    height_ptm = PTM.height(task='cmj', start=0, end=32, skip=2)
    for a in [0]:
        p = deepcopy(P[a])
        _, op_reps = utils.get_reps(p['bl']['hip']['op'][:,1],
                            fps=30, plot=False, d=5)
        fig, axs = plt.subplots(2, len(op_reps), figsize=(7,4),
                                dpi=300, sharey='row', sharex='col')
        for i, task in enumerate(['bl', 'ul']):
            omc = p[task]['hip']['coda'][:,2]
            mmc = p[task]['hip']['op'][:,1]
            coda_seg, coda_reps = utils.get_reps(omc, fps=100, plot=False,
                                rewind=1, forward=1, d=5)
            op_seg, op_reps = utils.get_reps(mmc, fps=30, plot=False,
                                rewind=1, forward=1, d=5)
            # temporary: remove first rep for P08
            if a in [8,14] and task=='ul':
                coda_seg, coda_reps = coda_seg[1:], coda_reps[1:]
                op_seg, op_reps = op_seg[1:], op_reps[1:]
            
            omc_part = p[task]['toe']['coda'][:,2]
            omc_interp = resample(omc_part[::4], len(omc_part))
            mmc_part = p[task]['toe']['op'][:,1]
            mmc_interp = resample(mmc_part[::4], len(mmc_part))
            num_reps = min(len(coda_reps), len(op_reps))
            for j in range(num_reps):
                c1, c2 = coda_seg[j]
                o1, o2 = op_seg[j]
                coda_jump = omc_part[c1:c2]
                op_jump = mmc_interp[o1:o2]
                coda_jump -= min(coda_jump)
                coda_jump = savgol_filter(coda_jump, 5, 2)
                op_jump -= min(op_jump)
                op_jump = savgol_filter(op_jump, 5, 2)
                # mmpx = utils.px_to_metric(op_jump, 30, verbose=False)
                mmpx_g = max(ptm[a], ptm.mean())
                mmpx_h = height_ptm[a]
                op_jump_g = op_jump * mmpx_g
                op_jump_h = op_jump * mmpx_h
                op_jump_g = resample(op_jump_g, len(coda_jump))
                op_jump_h = resample(op_jump_h, len(coda_jump))
                coda_h = utils.ts_jump_height(coda_jump, fps=100)
                op_h_gravity = utils.ts_jump_height(op_jump_g, fps=100)
                op_h_body_height = utils.ts_jump_height(op_jump_h, fps=100)
                utils.ts_subplots(axs[i,j], [coda_jump, op_jump_g, op_jump_h], xlabel='',
                            labels=['OMC', f'PTM$_g$',f'PTM$_h$'], mode='compare',
                        ylabel=f'{"bilateral" if task=="bl" else "unilateral"}' if j==0 else '',
                        legend='upper right' if j==2 and i==0 else False, bbox_to_anchor=(1.15, 1.15),
                            title=f'Jump {j+1}' if i==0 else '', titlepad=10)
        sns.despine()
        ptmg, ptmh = np.round(ptm[a],2), np.round(height_ptm[a],2)
        fig.suptitle(f'Countermovement jump (BL and UL)\nPTM$_g$ = {ptmg} mm/px\
        PTM$_h$ = {ptmh} mm/px')
        fig.supxlabel('Frame')
        fig.supylabel('Toe vertical displacement (mm)')
        plt.tight_layout()
        plt.show()

def get_performance(subjects, task, part, reps=3, plot=False, ptm='height'):
    ptm_opts = {
        'gravity': PTM.gravity(visualize=False),
        'height': PTM.height(task='cmj', start=0, end=32, skip=2) 
    }
    ptm = ptm_opts[ptm]
    mae_dict = {}
    jumps_dict = {}
    for s, subject in enumerate(subjects):
        # Sync and calibrate based on hip
        _, op_reps = utils.get_reps(
            subject['bl']['hip']['op'][:,1], fps=30, plot=False, d=5)
        omc = subject[task]['hip']['coda'][:,2]
        mmc = subject[task]['hip']['op'][:,1]
        coda_seg, _ = utils.get_reps(omc, fps=100, plot=False,
                               rewind=1, forward=1, d=5)
        op_seg, _ = utils.get_reps(mmc, fps=30, plot=False,
                             rewind=1, forward=1, d=5)
        # temporary: remove first rep for P08 and P14
        if s in [8,14] and task=='ul':
            coda_seg = coda_seg[1:]
            op_seg = op_seg[1:]
        if plot:
            fig, axs = plt.subplots(1, reps, figsize=(12,3), dpi=100);
        #for p, part in enumerate(parts):
        if 'rom' in part:
            omc_part = subject[task][part]['coda']
            mmc_part = subject[task][part]['op']
        else:
            omc_part = subject[task][part]['coda'][:,2]
            omc_interp = resample(omc_part[::4], len(omc_part))
            mmc_part = subject[task][part]['op'][:,1]
            mmc_interp = resample(mmc_part[::4], len(mmc_part))
        op_mae = 0
        coda_reps_list = []
        op_reps_list = []
        for j in range(reps):
            c1, c2 = coda_seg[j]
            o1, o2 = op_seg[j]
            coda_jump = omc_part[c1:c2]
            # coda_jump = savgol_filter(coda_jump, 3, 2)
            op_jump = mmc_interp[o1:o2]
            # op_jump = savgol_filter(op_jump, 3, 2)
            op_jump -= min(op_jump)
            coda_jump -= min(coda_jump)
            px_mm = ptm[s]
            op_jump *= px_mm
            coda_h = utils.ts_jump_height(coda_jump, fps=100)
            coda_reps_list.append(coda_h)
            op_h = utils.ts_jump_height(op_jump, fps=100)
            op_reps_list.append(op_h)
            op_jump = resample(op_jump, len(coda_jump))
            op_mae += MAE(coda_jump/10, op_jump/10)
            part_mae = np.round(op_mae/len(coda_seg),2)
            if plot:
                utils.ts_subplots(axs[j], [coda_jump, op_jump], xlabel='Frame',
                        labels=['OMC', f'MMC'],
                        ylabel=part if j==0 else '', mode='compare',
                        legend='lower left' if j==2 else False, bbox_to_anchor=(0.8, 0.8),
                        title=f'Jump {j+1}' if i==0 else '', titlepad=10)
                tex = f'OMC: {coda_h}cm\nMMC: {op_h}cm'
                axs[j].text(8, max(coda_jump)-2, tex, ha='left')    
            op_key = f'op_{task}_{part}'
            if op_key in mae_dict.keys():
                mae_dict[op_key].append(part_mae)
            else:
                mae_dict[op_key] = [part_mae]
        if plot:
            sns.despine()
            fig.suptitle(f'Participant {s+3} with {scaling} scaling ({task})')
            plt.tight_layout()
            plt.show()
        coda_key = f'coda_{task}_{part}_{j+1}'
        op_key = f'op_{task}_{part}_{j+1}'
        if op_key in jumps_dict.keys():
            jumps_dict[op_key].append(op_reps_list)
        else:
            jumps_dict[op_key] = [op_reps_list]
        if coda_key in jumps_dict.keys():
            jumps_dict[coda_key].append(coda_reps_list)
        else:
            jumps_dict[coda_key] = [coda_reps_list]
    return mae_dict, jumps_dict

def get_jump_heights(subjects, ptm):
    bl_maes, bl_jumps = get_performance(subjects, 'bl', 'toe', ptm=ptm)
    ul_maes, ul_jumps = get_performance(subjects, 'ul', 'toe', ptm=ptm)
    errors_dict = {**bl_maes, **ul_maes}
    jumps_dict = {**bl_jumps, **ul_jumps}
    jumps_df = pd.DataFrame(data=jumps_dict)
    mae_df = pd.DataFrame(data=errors_dict)

    # Force plate jumps
    force_ft = utils.force_flight_time(subjects, thresh='auto')
    ftbl_list = [f'FT_BL_{i}' for i in range(1,4)]
    ftul_list = [f'FT_UL_{i}' for i in range(1,4)]
    jhbl_list = [f'JH_BL_{i}' for i in range(1,4)]
    jhul_list = [f'JH_UL_{i}' for i in range(1,4)]
    cols = ftbl_list + ftul_list + jhbl_list + jhul_list
    force_ft_bl = np.array(force_ft['bl'])/1000
    force_ft_ul = np.array(force_ft['ul'])/1000
    force_jh_bl = utils.jump_heights(force_ft_bl) * 100
    force_jh_ul = utils.jump_heights(force_ft_ul) * 100
    force_jh_bl = np.round(force_jh_bl, 2)
    force_jh_ul = np.round(force_jh_ul, 2)
    force_df = pd.DataFrame({
        'fp_bl': force_jh_bl.tolist(),
        'fp_ul': force_jh_ul.tolist()
    })
    fp_mean_jumps_df = force_df.applymap(
        lambda x: np.round(np.array(x).mean(),2))
    mean_jumps_df = jumps_df.applymap(
        lambda x: np.round(np.array(x).mean(),2))
    all_mean_jumps_df = pd.concat([fp_mean_jumps_df, mean_jumps_df], axis=1)
    all_jumps_df = pd.concat([force_df['fp_bl'], jumps_df.iloc[:, 0],
                             force_df['fp_ul'], jumps_df.iloc[:, 2]], axis=1)
    return all_jumps_df

def get_all_jumps(df, cols):
    # temp_df = pd.read_csv('cleaned_jumps_1Feb2023.csv', usecols=cols)
    jumps_lst = []
    df[cols].applymap(lambda x: jumps_lst.extend(x))
    return np.array(jumps_lst).flatten()

def flatten_jump_reps(df):
    # convert all the jumps to a 1d array
    fp_jumps_bl = get_all_jumps(df, cols=['fp_bl'])
    fp_jumps_ul = get_all_jumps(df, cols=['fp_ul'])
    fp_jumps_arr = np.concatenate([fp_jumps_bl,fp_jumps_ul])
    ptm_jumps_arr = get_all_jumps(df, cols=['op_bl_toe_3', 'op_ul_toe_3'])
    bls = ['bilateral' for i in fp_jumps_bl]
    uls = ['unilateral' for i in fp_jumps_ul]
    ba_df = pd.DataFrame({
        'task': bls+uls,
        'FP': fp_jumps_arr,
        'PTM': ptm_jumps_arr 
    })
    return ba_df

def ba_plots(df, title):
    ba_df = flatten_jump_reps(df)
    sns.set_style('white')
    fig, axs = plt.subplots(1,2, figsize=(10,3), dpi=100)
    for t, task in enumerate(['bilateral', 'unilateral']):
        task_df = ba_df[ba_df['task']==task]
        plot_ba(task_df['FP'], task_df['PTM'], title=task, ax=axs[t], label_y=t==0)
    fig.supxlabel('Means')
    fig.suptitle(f'Countermovement jump - {title}')
    fig.tight_layout()
    sns.despine()
    plt.show()

def get_mae(ba_df, task):
    task_df = ba_df[ba_df['task']==task]
    mae = MAE(task_df['FP'], task_df['PTM'])
    return np.round(mae, 2)

def get_retest(df, task):
    df = df[[f'op_{task}_toe_3']]
    lst = []
    for item in df.itertuples():
        item = item[1]
        if len(item) > 2:
            lst.append(item)
    three_reps = np.array(lst)
    L, _ = three_reps.shape
    rep1 = pd.DataFrame({
        'jump': np.arange(1, L+1),
        'score': three_reps[:,0],
        'rater': ['rep1' for i in range(1,L+1)]
    })
    rep2 = pd.DataFrame({
        'jump': np.arange(1, L+1),
        'score': three_reps[:,1],
        'rater': ['rep2' for i in range(1,L+1)]
    })
    rep3 = pd.DataFrame({
        'jump': np.arange(1, L+1),
        'score': three_reps[:,2],
        'rater': ['rep3' for i in range(1,L+1)]
    })
    jumps_icc = pd.concat([rep1, rep2, rep3])
    icc = pg.intraclass_corr(data=jumps_icc, targets='jump',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_ratings(ba_df, task):
    task_df = ba_df[ba_df['task']==task]
    L = len(task_df)+1
    fp_ratings = pd.DataFrame({
        'jump': np.arange(1, L),
        'score': task_df['FP'].to_numpy(),
        'rater': ['FP' for i in range(1,L)]
    })
    ptm_ratings = pd.DataFrame({
        'jump': np.arange(1, L),
        'score': task_df['PTM'].to_numpy(),
        'rater': ['PTM' for i in range(1,L)]
    })
    return fp_ratings, ptm_ratings

def get_icc(ba_df, task):
    ratings1, ratings2 = get_ratings(ba_df, task)
    jumps_icc = pd.concat([ratings1, ratings2])
    icc = pg.intraclass_corr(data=jumps_icc, targets='jump',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'][1], 2)

def get_metrics(df, ptm):
    ba_df = flatten_jump_reps(df)
    bl_metrics = {
        'Metric': 'Jump height',
        'Task': 'CMJ bilateral',
        'PTM Ref': ptm,
        'Ground Truth': 'Force plates',
        'MAE': get_mae(ba_df, task='bilateral'),
        'Reliability': get_retest(df, task='bl'),
        'ICC': get_icc(ba_df, task='bilateral')
    }
    ul_metrics = {
        'Metric': 'Jump height',
        'Task': 'CMJ unilateral',
        'PTM Ref': ptm,
        'Ground Truth': 'Force plates',
        'MAE': get_mae(ba_df, task='unilateral'),
        'Reliability': get_retest(df, task='ul'),
        'ICC': get_icc(ba_df, task='unilateral')
    }
    metrics_df = pd.DataFrame(bl_metrics, index=[0])
    metrics_df.loc[1] = ul_metrics
    return metrics_df