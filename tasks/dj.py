import numpy as np
import pandas as pd
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

# Manual segmentation using matplotlib widget interactive plot
segs = {
    'bl': {
        'coda': np.array([
                [405,1455,2633],
                [651,2201,3338],
                [575,1669,2663],
                [605,1480,2301],
                [588,1970,3166],
                [2042,3953,5088],
                [1146,2602,3608],
                [4997,5960,6814],
                [1050,2547,3817],
                [914,1958,3314],
                [1242,2259,3235],
                [973,1732,2397],
                [1220,3329,4576],
                [1131,3691,4781],
                [1266,2636,3570],
                [833,1750,2515]
            ]),
        'op': np.array([
                [356,668,1025],
                [408,871,1214],
                [339,669,966],
                [298,553,802],
                [290,702,1058],
                [724,1296,1637],
                [416,855,1155],
                [1699,1987,2246],
                [384,837,1213],
                [393,707,1118],
                [473,788,1081],
                [391,622,819],
                [504,1139,1511],
                [459,1227,1554],
                [484,895,1176],
                [349,623,853]
            ])
    },
    'ul': {
        'coda': np.array([
                [379,1386,2328],
                [501,1682,2757],
                [2408,4011,5234],
                [646,1947,2858],
                [3306,4705,6410],
                [876,1825,2731],
                [1039,2082,3243],
                [1030,3005,4031],
                [1222,2471,3669],
                [1033,2139,3182],
                [813,2807,4629],
                [901,1730,2586],
                [1218,2695,3723],
                [1367,3806,4826],
                [1254,2392,3419],
                [916,1741,2592]
            ]),
        'op': np.array([
                [502,803,1083],
                [364,713,1035],
                [870,1351,1718],
                [413,803,1076],
                [1095,1515,2029],
                [343,629,902],
                [394,713,1056],
                [399,991,1300],
                [443,816,1182],
                [440,774,1086],
                [367,965,1514],
                [417,667,922],
                [459,902,1211],
                [540,1273,1574],
                [474,813,1124],
                [162,409,665]
            ])
    }
}

def get_performance(subjects, task, segs, reps=3, ptm='height'):
    ptm_opts = {
        'gravity': PTM.gravity(visualize=False),
        'height': PTM.height(task='dj', start=0, end=32, skip=2) 
    }
    ptm = ptm_opts[ptm]
    jump_dict = {}
    ft_dict = {}
    ct_dict = {}
    for s, subject in enumerate(subjects):
        if s+3 == 8:
            continue
        omc = savgol_filter(subject[task]['toe']['coda'][:,2], 5, 2)
        mmc = savgol_filter(subject[task]['toe']['op'][:,1], 5, 2)
        for i in range(reps):
            coda0 = segs[task]['coda'][s][i]
            coda1 = coda0 + int((1.5 * 100)) # 1.5 secs * 100 fps
            op0 = segs[task]['op'][s][i]
            op1 = op0 + int((1.5 * 30)) # 1.5 secs * 30 fps
            coda_rep = omc[coda0:coda1]
            op_rep = mmc[op0:op1]
            coda_rep -= min(coda_rep)
            op_rep -= min(op_rep)
            op_rep *= ptm[s]
            op_rep = resample(op_rep, len(coda_rep))
            op_rep = np.where(op_rep>0, op_rep, 0)
            # fixes
            if task == 'ul':
                if s+3 in [7,9] and i==0: #participant 7,9 rep 1
                    op_rep[100:] = 0
                if s+3 == 10 and i==0: #participant 10 rep 1
                    op_rep[100:] = 0
                    coda_rep[100:] = 0
                if s+3 == 10 and i==1: #participant 10 rep 2
                    op_rep[100:] = 0
                if s+3 == 12 and i==1: #participant 12 rep 2
                    op_rep[105:] = 0
            op_vel = abs(savgol_filter(np.gradient(op_rep), 21, 2))[:-10]
            op_vel, _ = utils.scale_filter(op_vel)
            h, d = (0.5, 15) if task == 'bl' else (0.2, 10)
            op_times, op_vel_peaks = utils.time_from_velocity(op_vel, h, d)
            
            # Manually fix UL rep 2 for participant 10
            if task=='ul' and s+3 == 10 and i==1:
                op_vel_peaks = np.array([16,59,70])
                op_times = np.round(np.diff(op_vel_peaks)/100, 2)
                
            # select last 2 for op_times
            op_times = op_times[-2:]
            o1, o2 = op_vel_peaks[-2:]
            op_h = np.round((max(op_rep[o1:o2]) - op_rep[o2])/10, 2)
            op_key = f'op_{task}_{i+1}'
            if op_key in jump_dict.keys():
                jump_dict[op_key].append(op_h)
                ct_dict[op_key].append(op_times[0])
                ft_dict[op_key].append(op_times[1])
            else:
                jump_dict[op_key] = [op_h]
                ct_dict[op_key] = [op_times[0]]
                ft_dict[op_key] = [op_times[1]]
    return jump_dict, ct_dict, ft_dict

def flatten_reps(df, col):
    return np.array([df[f'{col}_{i}'] for i in [1,2,3]]).flatten()

def get_mmc_jumps_df(subjects, ptm):
    bl_jumps, bl_ct, bl_ft = get_performance(subjects, 'bl', segs=segs, ptm=ptm)
    ul_jumps, ul_ct, ul_ft = get_performance(subjects, 'ul', segs=segs, ptm=ptm)
    jumps_dict = {**bl_jumps, **ul_jumps}
    contact_dict = {**bl_ct, **ul_ct}
    flight_dict = {**bl_ft, **ul_ft}
    jumps_df = pd.DataFrame(data=jumps_dict)
    contact_df = pd.DataFrame(data=contact_dict)
    flight_df = pd.DataFrame(data=flight_dict)
    return jumps_df, contact_df, flight_df

def get_mmc_jumps(subjects, ptm):
    jumps_df, contact_df, flight_df = get_mmc_jumps_df(subjects, ptm)
    op_ft_bl = flatten_reps(flight_df, 'op_bl')
    op_ft_ul = flatten_reps(flight_df, 'op_ul')
    op_ct_bl = flatten_reps(contact_df, 'op_bl')
    op_ct_ul = flatten_reps(contact_df, 'op_ul')
    op_jh_bl = flatten_reps(jumps_df, 'op_bl')
    op_jh_ul = flatten_reps(jumps_df, 'op_ul')
    return op_ft_bl, op_ft_ul, op_ct_bl, op_ct_ul, op_jh_bl, op_jh_ul

def get_fp_jumps_df(P):
    ft_dict = {'fp_bl_1':[], 'fp_bl_2':[], 'fp_bl_3':[],
           'fp_ul_1':[], 'fp_ul_2':[], 'fp_ul_3':[]}
    ct_dict = {'fp_bl_1':[], 'fp_bl_2':[], 'fp_bl_3':[],
            'fp_ul_1':[], 'fp_ul_2':[], 'fp_ul_3':[]}
    jh_dict = {'fp_bl_1':[], 'fp_bl_2':[], 'fp_bl_3':[],
            'fp_ul_1':[], 'fp_ul_2':[], 'fp_ul_3':[]}
    for a in range(len(P)):
        if a+3 == 8:
            continue
        p = P[a]
        for i, task in enumerate(['bl', 'ul']):
            F = p[task]['force']
            f1 = F[1] + F[3]
            f2 = F[2] + F[4]
            f = f1 if f1.max() > f2.max() else f2
            force = f - f.min()
            # fp is at 1000fps vs 100fps coda
            fp_segs = segs[task]['coda'] * 10
            for j in range(3):
                fp0 = fp_segs[a][j]
                fp1 = fp0 + int((1.5 * 1000)) # 1.5 secs * 1000 fps
                grf = force[fp0:fp1]
                grf -= min(grf)
                thresh = grf[:20].mean()
                ct = np.diff(np.where(grf<=thresh+15)[0])
                ft = np.diff(np.where(grf>thresh+15)[0])
                contact_time = np.round(ct[ct>20][0]/1000, 2)
                flight_time = np.round(ft[ft>20][0]/1000, 2)
                jump_height = np.round(
                    utils.jump_heights(flight_time)*100, 2)
                ft_dict[f'fp_{task}_{j+1}'].append(flight_time)
                ct_dict[f'fp_{task}_{j+1}'].append(contact_time)
                jh_dict[f'fp_{task}_{j+1}'].append(jump_height)
    fp_contact_df = pd.DataFrame(data=ct_dict)
    fp_flight_df = pd.DataFrame(data=ft_dict)
    fp_jump_df = pd.DataFrame(data=jh_dict)
    return fp_contact_df, fp_flight_df, fp_jump_df

def get_fp_jumps(subjects):
    fp_contact_df, fp_flight_df, fp_jump_df = get_fp_jumps_df(subjects)
    fp_ft_bl = flatten_reps(fp_flight_df, 'fp_bl')
    fp_ft_ul = flatten_reps(fp_flight_df, 'fp_ul')
    fp_ct_bl = flatten_reps(fp_contact_df, 'fp_bl')
    fp_ct_ul = flatten_reps(fp_contact_df, 'fp_ul')
    fp_jh_bl = flatten_reps(fp_jump_df, 'fp_bl')
    fp_jh_ul = flatten_reps(fp_jump_df, 'fp_ul')
    return fp_ft_bl, fp_ft_ul, fp_ct_bl, fp_ct_ul, fp_jh_bl, fp_jh_ul

def plot_task(bl1,bl2,ul1,ul2, task, unit, ptm='height'):
    fig, axs = plt.subplots(1,2, figsize=(10,3), dpi=100)
    plot_ba(bl1, bl2, title='bilateral', ax=axs[0])
    plot_ba(ul1, ul2, title='unilateral', ax=axs[1], label_y=False)
    fig.suptitle(f'Drop jump {task} ({unit}) - {ptm} PTM')
    fig.supxlabel('Means')
    plt.show()

def ba_plots(subjects, ptm='height', plot='jump height'):
    op_ft_bl, op_ft_ul, op_ct_bl, op_ct_ul, op_jh_bl, op_jh_ul = get_mmc_jumps(subjects, ptm)
    fp_ft_bl, fp_ft_ul, fp_ct_bl, fp_ct_ul, fp_jh_bl, fp_jh_ul = get_fp_jumps(subjects)
    if 'time' in plot:
        plot_task(fp_ft_bl,op_ft_bl,fp_ft_ul,op_ft_ul, 'flight time', 'secs')
        plot_task(fp_ct_bl,op_ct_bl,fp_ct_ul,op_ct_ul, 'contact time', 'secs')
    else: # plot=='jump height'
        plot_task(fp_jh_bl,op_jh_bl,fp_jh_ul,op_jh_ul, 'jump height', 'cm', ptm)

def get_icc(x, y, devices=['FP', 'MMC']):
    jumps_icc = pd.concat([get_ratings(x, devices[0]),
                           get_ratings(y, devices[1])])
    icc = pg.intraclass_corr(data=jumps_icc, targets='rep',
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

def get_metrics(subjects, ptm):
    jumps_df, contact_df, flight_df = get_mmc_jumps_df(subjects, ptm)
    op_ft_bl, op_ft_ul, op_ct_bl, op_ct_ul, op_jh_bl, op_jh_ul = get_mmc_jumps(subjects, ptm)
    fp_ft_bl, fp_ft_ul, fp_ct_bl, fp_ct_ul, fp_jh_bl, fp_jh_ul = get_fp_jumps(subjects)
    jump_bl_metrics = {
        'Metric': 'Jump height',
        'Task': 'Drop jump bilateral',
        'PTM Ref': ptm,
        'Ground Truth': 'Force plates',
        'MAE':  np.round(MAE(fp_jh_bl, op_jh_bl), 2),
        'Reliability': get_retest(jumps_df, 'op', 'bl'),
        'ICC': get_icc(fp_jh_bl, op_jh_bl)
    }
    jump_ul_metrics = {
        'Metric': 'Jump height',
        'Task': 'Drop jump unilateral',
        'PTM Ref': ptm,
        'Ground Truth': 'Force plates',
        'MAE': np.round(MAE(fp_jh_ul, op_jh_ul), 2),
        'Reliability': get_retest(jumps_df, 'op', 'ul'),
        'ICC': get_icc(fp_jh_ul, op_jh_ul)
    }
    jump_metrics_df = pd.DataFrame(jump_bl_metrics, index=[0])
    jump_metrics_df.loc[1] = jump_ul_metrics
    
    if ptm=='height':
        ct_bl_metrics = {
            'Metric': 'Contact time',
            'Task': 'Drop jump bilateral',
            'PTM Ref': ptm,
            'Ground Truth': 'Force plates',
            'MAE':  np.round(MAE(fp_ct_bl, op_ct_bl),2),
            'Reliability': get_retest(contact_df, 'op', 'bl'),
            'ICC': get_icc(fp_ct_bl, op_ct_bl)
        }
        ct_ul_metrics = {
            'Metric': 'Contact time',
            'Task': 'Drop jump unilateral',
            'PTM Ref': ptm,
            'Ground Truth': 'Force plates',
            'MAE': np.round(MAE(fp_ct_ul, op_ct_ul), 2),
            'Reliability': get_retest(contact_df, 'op', 'ul'),
            'ICC': get_icc(fp_ct_ul, op_ct_ul)
        }
        ct_metrics_df = pd.DataFrame(ct_bl_metrics, index=[0])
        ct_metrics_df.loc[1] = ct_ul_metrics

        ft_bl_metrics = {
            'Metric': 'Flight time',
            'Task': 'Drop jump bilateral',
            'PTM Ref': ptm,
            'Ground Truth': 'Force plates',
            'MAE':  np.round(MAE(fp_ft_bl, op_ft_bl), 2),
            'Reliability': get_retest(flight_df, 'op', 'bl'),
            'ICC': get_icc(fp_ft_bl, op_ft_bl)
        }
        ft_ul_metrics = {
            'Metric': 'Flight time',
            'Task': 'Drop jump unilateral',
            'PTM Ref': ptm,
            'Ground Truth': 'Force plates',
            'MAE': np.round(MAE(fp_ft_ul, op_ft_ul), 2),
            'Reliability': get_retest(flight_df, 'op', 'ul'),
            'ICC': get_icc(fp_ft_ul, op_ft_ul)
        }
        ft_metrics_df = pd.DataFrame(ft_bl_metrics, index=[0])
        ft_metrics_df.loc[1] = ft_ul_metrics
    if ptm=='height':
        return jump_metrics_df, ft_metrics_df, ct_metrics_df
    else:
        return jump_metrics_df