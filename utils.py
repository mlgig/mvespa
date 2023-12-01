import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.spatial import distance
from scipy import stats
import statsmodels.api as sm
from scipy.constants import g
from scipy.signal import find_peaks, peak_widths
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelextrema
from scipy.signal import resample
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def euclidean(x, y, axis=None):
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x - y, axis=axis)

def flip_axis(d, ax=1):
    if d.ndim == 1:
        d = d.max() - d
    else:
        max_val = d[:,ax].max()
        d[:,ax] = max_val - d[:,ax]
    return d
    
def scale_filter(ts, f=None):
    if f is not None:
        ts = savgol_filter(ts, f, 2)
    ts = ts.reshape(-1, 1)
    scaler = MinMaxScaler().fit(ts)
    scaled = scaler.transform(ts)
    scales = scaler.scale_ 
    return scaled.flatten(), scales

def jump_heights(t):
    # (Jump height = (gravity*(flight time)^2)/8)
    return (g/8) * (t**2)

def height_with_minima(ts, part, device, fps, n,
                    plot=True, figsize=(10,5), dpi=(100)):
    """
        Inputs:
            ts: input time series
            part: body part
            device: video or coda
            fps: freq of the time series
            n: order (number of points to consider on each side)
            g: accel. due to gravity. Default is 9.80665
            plot: (boolean). Whether to show plot. Default: True
            figsize: size of the plot.
            dpi: dpi of the plot
        Returns:
            Numpy Array of jump times (secs)
            Numpy Array of jump heights (meters)
    """
    minimas = argrelextrema(ts, np.less, order=n)[0]
    widths = []
    for i in range(1, len(minimas)):
        widths.append(minimas[i]-minimas[i-1])
    widths = np.array(widths)
    # Get jump time in seconds based on frame rate
    jump_times = widths/fps
    # (Jump height = (gravity*(flight time)^2)/8)
    jumps = jump_heights(jump_times)
    print(f'minima: {minimas}\nwidths: {widths}\nj_times: {jump_times}')
    
    if plot:
        plt.figure(figsize=(figsize), dpi=dpi)
        plt.plot(ts)
        plt.plot(minimas, ts[minimas], 'x', label=f'{part} motion')
        plt.xlabel('Frame number')
        plt.ylabel(f'{part} vertical displacement')
        plt.title(f'Vertical displacements from {device}: {100*(jump_heights.round(2))} cm')
        plt.show()

    return jump_times, jumps

def get_flight_time(ts, fps, d, rel_height):
    peaks, _ = find_peaks(ts, height=max(ts)/1.5,
                         distance=d)
    widths = peak_widths(ts, peaks, rel_height=rel_height)
    # Get jump time in seconds based on frame rate
    jump_times = (np.array(widths)/fps)[0]

    return jump_times, widths

def height_with_peak(ts, fps, d, rel_height, ax=None, part='', device=''):
    """
        Inputs:
            ts: input time series
            part: body part
            device: video or coda
            fps: freq of the time series
            plot: (boolean). Whether to show plot. Default: True
            figsize: size of the plot.
            dpi: dpi of the plot
        Returns:
            Numpy Array of jump times (secs)
            Numpy Array of jump heights (meters)
    """
    jump_times, widths = get_flight_time(ts, fps, d, rel_height)
    # (Jump height = (gravity*(flight time)^2)/8)
    jumps = jump_heights(jump_times)
    # print(f'widths: {widths[0]}\nj_times: {jump_times}')
    line_y = widths[1:][0]
    xmins = widths[1:][1]
    xmaxs = widths[1:][2]
    
    if ax is not None:
        peaks, _ = find_peaks(ts, height=max(ts)/1.5,
                         distance=d)
        ax.plot(ts)
        ax.plot(peaks, ts[peaks], 'x', label=f'{part} motion')
        for i in range(len(jump_times)):
            ax.hlines(y=line_y[i], xmin=xmins[i], xmax=xmaxs[i], color='red')
            ax.text(xmins[i], line_y[i], f' {np.round(jump_times[i],2)} s',
                   fontsize='small', c='red', va='bottom', ha='left')

    return jump_times, jumps, ax

def smoothen(data, chunk_size=2, n=2, inset=2):
    spikes = 0
    # Perform z-score smoothening n times
    while n > 0:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        z = abs((data - mean)/std)
        thresh = z.mean(axis=0)
        # replace spikes with surrounding data
        unshifted = np.roll(data, -1, axis=0)
        shifted = np.roll(data, 1, axis=0)
        rep = (unshifted + shifted) / 2
        # Allow below thresh to pass through
        # Otherwise, replace with rep
        spikes += np.where(z <= thresh)[0].shape[0]
        smooth = np.where(z <= thresh, data, rep)
        smooth[:inset] = data[:inset]
        smooth[-inset] = data[-inset]
        data = smooth
        n -= 1

    return data

def get_subject_data(frames, subject_id, task='',
                     joints=None, key="keypoints"):
    """
    BODY KEYPOINTS
    The first 26 coords are the body keypoints
    [x, y, confidence] ==> 26 x 3 = 78
    body_kp_slice = keypoints[:78]
       
    Returns body keypoints in batches of 3 [x, y, confidence]
    
    """
    body_df = pd.DataFrame(columns=['id', 'task']+joints)
    subject = {**{'id': subject_id, 'task': task},
               **{joint: [] for joint in joints}}
    
    for frame in frames:
        keypoints = frame[key]
        # get body keypoints
        for j in range(0, len(joints)):
            idx = j * 3
            if (keypoints != []):
                kp = [keypoints[idx:idx+3]]
                subject[joints[j]] += kp
            else:
                subject[joints[j]] += [[]]
                print("Returning empty coordinates")
        new_subject = pd.DataFrame(subject)
    body_df = pd.concat([body_df, new_subject], ignore_index=True)
                
    return body_df

def preview_ts(fig, ts, labels, title, unit='',
               mode='multi', subplot=(111), s=12,
              sharex=None, sharey=None):
    ax = fig.add_subplot(subplot, sharex=sharex,
                        sharey=sharey)
    lines = ['-', ':', '--', '+']
    if mode == 'single':
        ax.plot(ts, label=labels[0])
    elif mode == 'multi':
        for i in range(len(labels)):
            ax.plot(ts[:,i], lines[i], label=labels[i])
    elif mode == 'compare':
        for i in range(len(labels)):
            ax.plot(ts[i], lines[i], label=labels[i])
    else:
        print('Unrecognized mode.')
    ax.legend()
    ax.set_xlabel('Frame', fontsize=s)
    position = 'Position' if unit == '' else f'Position {unit}'
    ax.set_ylabel(position, fontsize=s)
    ax.set_title(title, fontsize=s)
    
    return fig, ax

## Some additional functions referenced in the functions above
def ts_subplots(ax, ts, **kwargs):
    """
        kwargs:
            labels: for each plotted line
            title: for each axis
            mode: 'single', 'multi', or 'compare'
            fontsize, xlabel, ylabel
            legend: display legend (True or False)
    """
    defaultKwargs = {'labels': ['' for s in ts],
                     'single_label': '',
                     'title': '',
                     'mode': 'multi', 'fontsize': 14,
                     'xlabel': '', 'ylabel': '', 'legend': 'lower right',
                     'bbox_to_anchor': (1., 1.), 'titlepad': 30.,
                     'linestyle': '-', 'c': 'C0'}
    kwargs = {**defaultKwargs, **kwargs}
    lines = ['-', ':', '--', '-.']
    s = kwargs['fontsize']
    labels = kwargs['labels']
    if kwargs['mode'] == 'single':
        ax.plot(ts, label=kwargs['single_label'],
            linestyle=kwargs['linestyle'],
               c=kwargs['c'])
    elif kwargs['mode'] == 'multi':
        for i in range(len(labels)):
            ax.plot(ts[:,i], lines[i],
                    label=labels[i])
    elif kwargs['mode'] == 'compare':
        for i in range(len(labels)):
            ax.plot(ts[i], lines[i], label=labels[i])
    else:
        print('Unrecognized mode.')
    
    if kwargs['legend']:
        ax.legend(loc=kwargs['legend'],
                 bbox_to_anchor=kwargs['bbox_to_anchor'])
    ax.set_xlabel(kwargs['xlabel'], fontsize=s)
    ax.set_ylabel(kwargs['ylabel'], fontsize=s)
    ax.set_title(kwargs['title'], fontsize=s, pad=kwargs['titlepad'])
    
    return ax

def get_angle(a, b, c, ndims=2):
    """
        Inputs: a, b, c - the three coord points
        b is the point in the middle of a and c
        We want to find the angle where b is the vertex
        
        Returns
        - angle in degrees formed by a and c at b
        - total length of vectors ba and bc
    """
    # a, b, c are in the form (x, y, confidence)
    # so we need only x and y [:2]
    a = a[:, :ndims] 
    b = b[:, :ndims] 
    c = c[:, :ndims] 

    # get the vectors
    ba = a - b
    bc = c - b
    
    # get the norms
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    lengths = norm_ba + norm_bc

    # get angle from cosine rule
    dot_products = (ba * bc).sum(axis=1)
    cosine_angles = dot_products / (norm_ba * norm_bc)
    cosine_angles = np.clip(cosine_angles, -1, 1)
    radian_angles = np.arccos(cosine_angles)
    degree_angles = np.degrees(radian_angles)

    degree_angles = np.nan_to_num(degree_angles)
    degree_angles = smooth_by_resampling(degree_angles)

    return degree_angles

def get_reps(ts, fps, plot=False, smooth=5, **kwargs):
    """
        Inputs:
            ts: input time series
            fps: frequency of the time series
        kwargs:
            rewind=1.: how many seconds to capture backward
            forward=1.: how many seconds to capture forward.
            thresh=2: min peak height=max(ts)/thresh.
            d=2.5: number of secs separating each rep
        Returns (segs, reps)
            seg: list of Array [start, end] indices for segmenting time series.
            reps: list of segmented time series
    """
    # ts -= min(ts) # set coordinate origin to zero
    if smooth:
        ts = smoothen(ts, n=smooth)
    rest = ts[:15].mean()
    auto_thresh = rest + (ts.max() - rest)/2
    defaultKwargs = {'rewind': 1.5, 'forward': 1,
                     'thresh': auto_thresh, 'd': 2.5,
                     'plot_peaks': False,
                     'expected_reps': 3}
    kwargs = {**defaultKwargs, **kwargs}
    # ts = np.nan_to_num(ts)
    peaks, _ = find_peaks(ts, distance=fps*kwargs['d'], height=kwargs['thresh'])
    # peaks = find_peaks_cwt(ts, kwargs['width'])


    bounds = [np.array([max(i-(kwargs['rewind']*fps),0),
                        i+(kwargs['forward']*fps)],
                      dtype='i') for i in peaks]
    if kwargs['plot_peaks']:
        fig = plt.figure(figsize=(7, 3), dpi=100)
        plt.plot(ts)
        plt.plot(peaks, ts[peaks], 'x')
        for b in bounds:
            plt.vlines(b, min(ts), max(ts), color='green', linestyles='--')
        fig.supylabel('displacement')
        #plt.xlabel(' Frame number')
        sns.despine()
        plt.savefig('rep_peaks.svg', pad_inches=0)
        fig.tight_layout()
        plt.show()
    
    if len(peaks) <= kwargs['expected_reps']:
        reps = [ts[int(b[0]):int(b[1])] for b in bounds]
        segs = bounds
    else:
        reps = []
        segs = []
        for b in bounds:
            rep = ts[int(b[0]):int(b[1])]
            if len(rep) == 0:
                continue
            # is_dominant = np.diff(rep).std() > 1
            # not_flat = np.ptp(rep)/rep.mean() >= 0.6
            above_threshold = max(rep) >= kwargs['thresh']
            if above_threshold:
                reps.append(rep)
                segs.append(b)

    while len(reps) > kwargs['expected_reps']+1:
        kwargs['thresh'] += 1
        reps = []
        segs = []
        for b in bounds:
            rep = ts[int(b[0]):int(b[1])]
            if len(rep) == 0:
                continue
            above_threshold = max(rep) >= kwargs['thresh']
            if above_threshold:
                reps.append(rep)
                segs.append(b)
        
    if plot:
        fig, axs = plt.subplots(1, len(reps),
                                figsize=(12, 4) or kwargs['figsize'],
                                dpi=100, sharey='row')
        for i in range(len(reps)):
            ts_subplots(axs[i], reps[i], title=f'Repetition {i+1}',
                   mode='single', legend=False, titlepad=10.)
        axs[0].set_ylabel('Position')
        fig.supxlabel('Frame')
        sns.despine()
        plt.show()
        
    return segs, reps

def get_reps_with_minima(ts, fps, plot='True', smooth=5, **kwargs):
    """
        Inputs:
            ts: input time series
            fps: frequency of the time series
        kwargs:
            rw=1.: how many seconds to capture backward
            fw=1.: how many seconds to capture forward.
            thresh=2: min peak height=max(ts)/thresh.
            d=2.5: number of secs separating each rep
        Returns (segs, reps)
            seg: list of Array [start, end] indices for segmenting time series.
            reps: list of segmented time series
    """
    ts -= min(ts) # set coordinate origin to zero
    if smooth:
        ts = smoothen(ts, n=smooth)
    defaultKwargs = {'rw': 2, 'fw': 2,
                     'thresh': 20, 'd': 1.5,
                     'plot_peaks': False,
                     'expected_reps': 3}
    kwargs = {**defaultKwargs, **kwargs}
    # peaks = argrelextrema(ts, np.less, order=fps*kwargs['d'])[0]
    ts_copy = flip_axis(deepcopy(ts))
    peaks, _ = find_peaks(ts_copy, distance=fps*kwargs['d'])
    forward = fps * kwargs['fw']
    rewind = fps * kwargs['rw']
    bounds = []
    for peak in peaks:
        left_window = ts[max(peak-rewind, 0):peak]
        rw = np.argmax(left_window)
        left_bound = peak - (len(left_window)-rw)
        while left_bound > 0 and ts[left_bound-1] > ts[left_bound]:
            left_bound -= 1
        right_window = ts[peak:peak+forward]
        fw = np.argmax(right_window)
        right_bound = peak + fw
        while (right_bound+1) < len(ts) and (ts[right_bound+1] > ts[right_bound]):
            right_bound += 1
        bounds.append([left_bound, right_bound])
    bounds = np.array(bounds, dtype='i')
    if kwargs['plot_peaks']:
        plt.figure(figsize=(10, 3), dpi=100)
        plt.plot(ts)
        plt.plot(peaks, ts[peaks], 'x')
        for b in bounds:
            plt.vlines(b, 1, max(ts), color='green', linestyles='--')
        sns.despine()
        plt.show()
    
    thresh = max(ts)/2
    reps = []
    segs = []
    for b in bounds:
        rep = ts[int(b[0]):int(b[1])]
        if len(rep) == 0:
            continue
        below_threshold = min(rep) < thresh
        if below_threshold:
            reps.append(rep)
            segs.append(b)
                
    while len(reps) > kwargs['expected_reps']:
        thresh -= 1
        reps = []
        segs = []
        for b in bounds:
            rep = ts[int(b[0]):int(b[1])]
            if len(rep) == 0:
                continue
            below_threshold = min(rep) < thresh
            if below_threshold:
                reps.append(rep)
                segs.append(b)
        
    if plot:
        fig, axs = plt.subplots(1, len(reps),
                                figsize=(12, 4) or kwargs['figsize'],
                                dpi=100, sharey='row')
        for i in range(len(reps)):
            ts_subplots(axs[i], reps[i], title=f'Repetition {i+1}',
                   mode='single', legend=False, titlepad=10.)
        axs[0].set_ylabel('Position')
        fig.supxlabel('Frame')
        sns.despine()
        plt.show()
        
    return segs, reps

def sync_with_minima(ts1, fps1, ts2, fps2, **kwargs):
    """
        Inputs:
            ts: input time series
            fps: frequency of the time series
        kwargs:
            rw=1.: how many seconds to capture backward
            fw=1.: how many seconds to capture forward.
            thresh=2: min peak height=max(ts)/thresh.
            d=2.5: number of secs separating each rep
        Returns (segs, reps)
            seg: list of Array [start, end] indices for segmenting time series.
            reps: list of segmented time series
    """
    # set coordinate origin to zero
    ts1 -= min(ts1)
    ts2 -= min(ts2)  
    defaultKwargs = {'rw': 2, 'fw': 2,
                     'thresh': 20, 'd': 1.5,
                     'plot': True, 'smooth': 5,
                     'plot_peaks': False,
                     'expected_reps': 3}
    kwargs = {**defaultKwargs, **kwargs}
    if kwargs['smooth']:
        ts1 = smoothen(ts1, n=smooth)
        ts2 = smoothen(ts2, n=smooth)
    # peaks = argrelextrema(ts, np.less, order=fps*kwargs['d'])[0]
    ts1_copy = flip_axis(deepcopy(ts1))
    ts2_copy = flip_axis(deepcopy(ts2))
    peaks1, _ = find_peaks(ts1_copy, height=max(ts1_copy)/1.6,
                            distance=fps1*kwargs['d'])
    peaks2, _ = find_peaks(ts2_copy, height=max(ts2_copy)/1.6,
                            distance=fps2*kwargs['d'])

    # get bounding window with ts1
    forward = fps1 * kwargs['fw']
    rewind = fps1 * kwargs['rw']
    bounds1 = []
    for peak in peaks1:
        left_window = ts1[max(peak-rewind, 0):peak]
        rw = np.argmax(left_window)
        left_bound = peak - (len(left_window)-rw)
        while left_bound > 0 and ts1[left_bound-1] > ts1[left_bound]:
            left_bound -= 1
        right_window = ts1[peak:peak+forward]
        fw = np.argmax(right_window)
        right_bound = peak + fw
        while (right_bound+1) < len(ts1) and (ts1[right_bound+1] > ts1[right_bound]):
            right_bound += 1
        bounds1.append([peak-left_bound, right_bound-peak])
    bounds1 = np.array(bounds1, dtype='i')
    bounds2 = np.round(fps2 * (bounds1/fps1)).astype('int')
    segs1, reps1 = get_segments(ts1,peaks1,bounds1, **kwargs)
    segs2, reps2 = get_segments(ts2,peaks2,bounds2, **kwargs)

    return segs1, reps1, segs2, reps2

def get_segments(ts, peaks, bounds,**kwargs):
    if kwargs['plot_peaks']:
        plt.figure(figsize=(10, 3), dpi=100)
        plt.plot(ts)
        plt.plot(peaks, ts[peaks], 'x')
        for i, bound in enumerate(bounds):
            plt.vlines(peaks[i]-bound[0], 1, max(ts), color='green', linestyles='--')
            plt.vlines(peaks[i]+bound[1], 1, max(ts), color='red', linestyles='--')
        sns.despine()
        plt.show()
    
    thresh = max(ts)/2
    reps = []
    segs = []
    for i, bound in enumerate(bounds):
        b1 = max(peaks[i]-bound[0], 0)
        b2 = peaks[i]+bound[1]
        rep = ts[b1:b2]
        if len(rep) == 0:
            continue
        below_threshold = min(rep) < thresh
        if below_threshold:
            reps.append(rep)
            segs.append([b1,b2])
                
    while len(reps) > kwargs['expected_reps']:
        thresh -= 1
        reps = []
        segs = []
        for i, bound in enumerate(bounds):
            b1 = max(peaks[i]-bound[0], 0)
            b2 = peaks[i]+bound[1]
            rep = ts[b1:b2]
            if len(rep) == 0:
                continue
            below_threshold = min(rep) < thresh
            if below_threshold:
                reps.append(rep)
                segs.append([b1,b2])
        
    if kwargs['plot']:
        fig, axs = plt.subplots(1, len(reps),
                                figsize=(12, 4) or kwargs['figsize'],
                                dpi=100, sharey='row')
        for i in range(len(reps)):
            ts_subplots(axs[i], reps[i], title=f'Repetition {i+1}',
                   mode='single', legend=False, titlepad=10.)
        axs[0].set_ylabel('Position')
        fig.supxlabel('Frame')
        sns.despine()
        plt.show()
        
    return segs, reps

def add_video_data(P, df, pe, participant_idxs):
    i = 0
    for row in tqdm(df.itertuples()):
        p_id = participant_idxs[row.id]
        task = row.task
        hip = flip_axis(np.array(row.MidHip), 1)
        P[p_id][task]['hip'][pe] = hip
        knee = flip_axis(np.array(row.RKnee), 1)
        P[p_id][task]['knee'][pe] = knee
        ankle = flip_axis(np.array(row.RAnkle), 1)
        P[p_id][task]['ankle'][pe] = ankle
        toe = flip_axis(np.array(row.RSmallToe), 1)
        P[p_id][task]['toe'][pe] = toe
        # knee_angles = get_angle(hip, knee, ankle)
        # P[p_id][task]['knee rom'][pe] = knee_angles
        # ankle_angles = get_angle(knee, ankle, toe)
        # P[p_id][task]['ankle rom'][pe] = ankle_angles
        i += 1
    return P

def ts_jump_height(ts, fps=100, base='mean_rest'):
    if base=='mean_rest':
        starting_pos = np.mean(ts[:int(0.5*fps)])
    else:
        starting_pos = base
    jump_height = np.round(
        (max(ts) - starting_pos)/10, 2)
    return jump_height

def force_flight_time(subjects, thresh=40, plot=False):
    force_ft = {'ul': [], 'bl': []}
    for s, subject in enumerate(subjects):
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(10, 5),
                                    dpi=100, sharex='col')
        for i, task in enumerate(['bl', 'ul']):
            F = subject[task]['force']
            f1 = F[1] + F[3]
            f2 = F[2] + F[4]
            f = f1 if f1.max() > f2.max() else f2
            force = f - f.min()
            if thresh == 'auto':
                thresh = force[:1000].mean()/10
            toe = F = subject[task]['toe']['coda'][:,2]
            toe = resample(toe, len(f))
            bases = np.where(force<=thresh)[0]
            force_diffs = np.diff(bases, prepend=0)
            second_ = np.where(force_diffs>100)[0]
            second_diffs = np.diff(second_)
            last_jump_time = len(force_diffs) - second_diffs.sum()
            T = np.hstack([second_diffs, last_jump_time])
            times = []
            for t in T:
                if t >= 100:
                    times.append(t)
            if len(times) > 3:
                times = times[1:]
            force_ft[task].append(times)
            if plot:
                axs[i,0].plot(force, label='GRF (N)' if i==0 else '')
                axs[i,0].plot(toe, label='toe position (mm)' if i==0 else '')
                axs[i,0].hlines(thresh, 0, len(f), color='red',
                         label='force threshold' if i==0 else '')
                axs[0,0].set_title('ground reaction forces')
                axs[i,1].plot(force_diffs)
                text_pos = 50
                for j in range(len(times)):
                    tex = f'${np.round(times[j]/1000, 2)}$ $s$'
                    axs[i,1].text(text_pos, 300, tex, color='red')
                    text_pos += times[j]
                axs[0,1].set_title('off-plate time diffs')
        if plot:
            fig.suptitle(f'Participant {s+3}')
            fig.supxlabel('Frame')
            sns.despine()
            fig.legend()
            fig.tight_layout()
            plt.show()
    return force_ft

def grf_time(subjects, thresh=40, plot=True):
    force_ft = {'ul': [], 'bl': []}
    for s, subject in enumerate(subjects):
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(10, 5),
                                    dpi=100, sharex='col')
        for i, task in enumerate(['bl', 'ul']):
            F = subject[task]['force']
            f1 = F[1] + F[3]
            f2 = F[2] + F[4]
            f = f1 if f1.max() > f2.max() else f2
            force = f - f.min()
            if thresh == 'auto':
                lim = 10 if task == 'bl' else 20
                thresh = force[:1000].mean()/lim
                bases = np.where(force<=thresh)[0]
                force_bases = force[bases]
                thresh = max(force_bases[50:150]) + 5
            toe = F = subject[task]['toe']['coda'][:,2]
            toe = resample(toe, len(f))
            bases = np.where(force<=thresh)[0]
            force_diffs = np.diff(bases, prepend=0)
            second_ = np.where(force_diffs>100)[0]
            second_diffs = np.diff(second_)
            last_jump_time = len(force_diffs) - second_diffs.sum()
            T = np.hstack([second_diffs, last_jump_time])
            times = []
            for t in T:
                if t >= 100:
                    times.append(t)
            if len(times) > 3:
                times = times[1:]
            force_ft[task].append(times)
            if plot:
                axs[i,0].plot(force, label='GRF (N)' if i==0 else '')
                axs[i,0].plot(toe, label='toe position (mm)' if i==0 else '')
                axs[i,0].hlines(thresh, 0, len(f), color='red',
                         label='force threshold' if i==0 else '')
                axs[0,0].set_title('ground reaction forces')
                axs[i,1].plot(force_diffs)
                text_pos = 50
                for j in range(len(times)):
                    tex = f'${np.round(times[j]/1000, 2)}$ $s$'
                    axs[i,1].text(text_pos, 300, tex, color='red')
                    text_pos += times[j]
                axs[0,1].set_title('off-plate time diffs')
        if plot:
            fig.suptitle(f'Participant {s+3}')
            fig.supxlabel('Frame')
            sns.despine()
            fig.legend()
            fig.tight_layout()
            plt.show()
    return force_ft

def segment_squat(ts, segments, **kwargs):
    """
        Inputs:
            ts: input time series
            segments: a list indicating the boundaries of the reps
        kwargs:
            expected_reps: how many reps are expected
            plot: boolean
            plot_peaks: boolean (visualize segmentation)
        Returns (reps)
            reps: list of segmented and synchronized repetitions
    """
    ts = np.array(ts)
    ts -= ts.min() # set coordinate origin to zero
    defaultKwargs = {'plot_peaks': False, 'plot': False,
                     'expected_reps': 3, 'rw': 1, 'fw': 1}
    kwargs = {**defaultKwargs, **kwargs}

    reps = []
    for i in range(len(segments)-1):
        rep = ts[segments[i]:segments[i+1]]
        rep -= rep.min() # set rep minimum to zero
        reps.append(rep)

    if kwargs['plot_peaks']:
        plt.figure(figsize=(10, 3), dpi=100)
        plt.plot(ts)
        for b in segments:
            plt.vlines(b, 1, max(ts), color='green', linestyles='--')
        sns.despine()
        plt.show()
        
    if kwargs['plot']:
        fig, axs = plt.subplots(1, len(reps),
                                figsize=(12, 4) or kwargs['figsize'],
                                dpi=100, sharey='row')
        for i in range(len(reps)):
            ts_subplots(axs[i], reps[i], title=f'Repetition {i+1}',
                   mode='single', legend=False, titlepad=10.)
        axs[0].set_ylabel('Position')
        fig.supxlabel('Frame')
        sns.despine()
        plt.show()
    return reps

def plot_fit(fig, x, y, **kwargs):
    defaultKwargs = {'figsize':(10,10), 'dpi':100, 'marker': 'o',
        'linestyle': '-', 'c': 'blue', 'xlabel': '', 'ylabel': '',
        'label': ''}
    kwargs = {**defaultKwargs, **kwargs}
    #find line of best fit
    a, b = np.polyfit(x, y, 1)

    #add points to plot
    plt.scatter(x, y, marker=kwargs['marker'], c=kwargs['c'],
        label=kwargs['label'], s=20)

    #add line of best fit to plot
    plt.plot(x, a*x+b, linestyle=kwargs['linestyle'],
        c=kwargs['c'])
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])

    return fig

def reverse_minmax(a, b, zero_min=False):
    if zero_min:
        a -= min(a)
        b -= min(b)
    _, a_factor = scale_filter(a)
    b = resample(b, len(a))
    b, _ = scale_filter(b)
    b /= a_factor
    
    return a, b

def smooth_by_resampling(ts, window_size=4):
    return resample(ts[::window_size], len(ts))

def velocity(ts, fps, frac=0.1, upsample=False):
    window_size = int(fps*frac)
    vel = np.diff(ts[::window_size])/window_size
    if upsample:
        vel = resample(vel, len(ts))
    return vel

def segment_by_velocity(ts, order):
    velocity = abs(np.gradient(ts))
    thresh = velocity[:20].mean() + 2
    peaks, _ = find_peaks(velocity, height=thresh, distance=order)
    print(peaks)
    return velocity, peaks

def fix_zeros(vec):
    zero_points = np.where(vec==0)[0]
    for point in zero_points:
        vec[point] = vec[point-1]
    return vec

def smooth_array(arr, axs, smooth_fn='savgol', fill_zeros=False):
    if fix_zeros:
        for ax in axs:
           arr[:, ax] = fix_zeros(arr[:, ax]) 
    if smooth_fn == 'savgol':
        for ax in axs:
            arr[:, ax] = savgol_filter(arr[:, ax], 21, 2)
    else:
        for ax in axs:
            arr[:, ax] = smooth_by_resampling(arr[:, ax])

    return arr

def zero_array(arr):
    if arr.ndim == 1:
        return arr - min(arr)
    arr = np.array(arr)
    for i in range(arr.shape[1]):
        arr[:,i] -= min(arr[:,i])
    return arr


def seg_by_matching(ts, template, plot=True):
    w = len(template)
    template, _ = scale_filter(template)
    corrs = []
    for i in range(len(ts)-w):
        this_chunk, _ = scale_filter(ts[i:i+w])
        # Y and Z are numpy arrays or lists of variables 
        corr = stats.pearsonr(template, this_chunk)
        corrs = [*corrs, corr.statistic]
    corrs = np.array(corrs)
    # min_corr = candidates.min()
    # max_corr = candidates.max()
    # idxs = np.argsort(candidates)
    # thresh = abs(max_corr - min_corr)/2
    corr_peaks, _ = find_peaks(corrs, distance=2000)
    print(corr_peaks)
    if plot:
        plt.figure(figsize=(12,3), dpi=100)
        temp_ts, _ = scale_filter(ts)
        plt.plot(temp_ts, label='ts')
        plt.plot(corrs, label='corrs')
        plt.plot(corr_peaks, corrs[corr_peaks], 'x')
        # suggested_reps = np.where(candidates<thresh)[0]
        # suggested_reps_diffs = np.diff(suggested_reps, prepend=0)
        # starts = np.where(suggested_reps_diffs>=max(suggested_reps_diffs/2))[0]
        # reps = suggested_reps[starts]
        # plt.plot(reps, ts[reps], 'x', c='red')
        plt.legend()
        plt.show()

    # return reps

def accel(x, fps):
    return np.gradient(np.gradient(x[2*fps:-2*fps]))

def balance_score(x, y, z=None):
    stdX = np.std(accel(x - min(x), fps=100))
    stdY = np.std(accel(y - min(y), fps=100))
    stdZ = np.std(accel(z - min(z), fps=100)) if z is not None else 0

    return stdX + stdY + stdZ

def time_from_velocity(vel, h=None, d=None):
    peaks, _ = find_peaks(vel, height=h, distance=d)
    if len(peaks) > 3:
        peaks = peaks[1:]
        # return time_from_velocity(vel, h, d+1)
    times = np.round(np.diff(peaks)/100, 2)

    return times, peaks

def rjt_flight_times(ts, thresh, fps):
    above_threshold = np.where(ts>thresh)[0]
    threshhold_diffs = np.diff(above_threshold)
    flight = np.where(threshhold_diffs>2)[0]
    flight_diffs = np.diff(flight, prepend=0)
    # FT = diffs[diffs>int(fps/10)]
    # FT = np.round(FT[FT < fps]/fps, 2)
    FT = np.round(flight_diffs/fps, 2)
    return FT

def rjt_contact_times(ts, peaks, thresh, fps):
    contact_times = []
    for i in range(len(peaks)-1):
        this_selection = ts[peaks[i]:peaks[i+1]]
        contact = np.where(this_selection<thresh)[0]
        contact_times.append(len(contact)/fps)
    contact_times = np.round(np.array(contact_times), 2)
    return contact_times

def handle_zeros_nans(arr):
    arr = np.where(arr!=0, arr, np.nan)
    arr = pd.DataFrame.from_records(arr).interpolate().to_numpy()
    return arr[:, :2]

def intersection_angle(line1, line2):
    np.seterr(divide='ignore', invalid='ignore')
    vec11, vec12 = [handle_zeros_nans(l) for l in line1]
    vec21, vec22 = [handle_zeros_nans(l) for l in line2]
    # m = y2-y1 / x2-x1
    m1 = (vec12[:,1]-vec11[:,1])/(vec12[:,0]-vec11[:,0])
    m2 = (vec22[:,1]-vec21[:,1])/(vec22[:,0]-vec21[:,0])
    theta = np.arctan(abs((m2-m1)/(1+(m1*m2))))
    theta = pd.Series(theta).interpolate() # handle nans
    theta = np.degrees(theta.to_numpy())

    return theta
        
def read_list(list_name):
    # for reading also binary mode is important
    with open(f'./data/pickle/{list_name}.pkl', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def get_ratings(reps, rater):
    L = reps.size + 1 # all reps
    return pd.DataFrame({
        'rep': np.arange(1, L),
        'score': reps,
        'rater': [rater for i in range(1,L)]
    })

def plot_ba(x, y, ax, title='', label_y=True):
    sns.set_style('white')
    sm.graphics.mean_diff_plot(x, y, ax=ax, scatter_kwds={ 's':3, 'alpha':0.9})
    sns.despine()
    ax.set_title(title)
    ax.set_xlabel('')
    if label_y == False:
        ax.set_ylabel('')

def fill_nan(arr):
    arr = np.where(arr==0, np.nan, arr)
    interp = np.array(pd.DataFrame(arr).interpolate())
    return interp