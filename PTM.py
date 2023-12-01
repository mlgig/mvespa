import numpy as np
import pandas as pd
from scipy.constants import g
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_flight_time, get_reps, read_list, euclidean
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ptmg = np.array([3.79956525, 4.03131528, 4.02184535, 3.63973693, 3.54615941, 3.38307119, 3.60175415, 3.69389473, 4.26698498, 4.11536437, 4.24260082, 3.85797096, 3.84339437, 3.41462725, 3.01710485, 3.70168633])
# ptmh = np.array([3.79956525, 4.03131528, 4.02184535, 3.63973693, 3.54615941, 3.38307119, 3.60175415, 3.69389473, 4.26698498, 4.11536437, 4.24260082, 3.85797096, 3.84339437, 3.41462725, 3.01710485, 3.70168633])
def barbell():
    return np.array([3.839590444, 3.652597403, 3.839590444, 3.652597403, 3.403933434, 3.098320022, 3.579382755, 3.537735849, 3.90625, 4.012125535, 3.90625, 3.523332289, 3.523332289, 3.390596745, 3.482972136, 3.668079557])

def mm_per_px(ts, fps, label='', verbose=False, plot=False):
    rest = ts[:10].mean()
    rest_ratio = rest/max(ts)
    jump_times, _ = get_flight_time(ts, fps, d=5, rel_height=rest_ratio)
    t = np.round(jump_times[0]/3, 1)
    t_display = np.round(t,2)
    def obj(x, a, b, c):
        return a*x + 0.5*b*x**2 + c
    # ts = savgol_filter(ts, 5, 2)
    peaks, _ = find_peaks(ts, distance=fps*5)
    nxt_ms = peaks[0] + int(fps*t)
    prev_ms = peaks[0] - int(fps*t)
    y = ts[prev_ms:nxt_ms+1]
    x = np.arange(len(y))
    popt, _ = curve_fit(obj, x, y)
    a,b,c = popt
    y_line = obj(x, a, b, c)
    new_peak = np.argmax(y_line)
    peak_dist = peaks[0] - new_peak
    new_nxt_ms = new_peak + (nxt_ms-peaks[0])
    if new_nxt_ms >= len(y_line):
        new_nxt_ms = len(y_line)-1
    mm_per_px = (500* g* (t**2))/(ts[peaks[0]] - ts[nxt_ms])
    if verbose:
        print(f'1 {label} px equals {np.round(mm_per_px,1)} mm')
    if plot:
        plt.figure(figsize=(5,4), dpi=100)
        plt.plot(ts, ':', label='hip position')
        plt.plot(x+peak_dist, y_line, c='red', label='fitted curve')
        plt.plot(peaks[0], ts[peaks[0]], 'o', c='black')
        plt.text(peaks[0]+4, ts[peaks[0]]-4, f't = 0s; $d_0 \\approx$ {np.round(ts[peaks[0]],2)} px')
        plt.plot(nxt_ms, ts[nxt_ms], 'o', c='black')
        plt.text(nxt_ms+2, ts[nxt_ms]-0.4, f't = {t_display}s; $d_T \\approx$ {np.round(ts[nxt_ms],2)} px')
        plt.text(2, ts[:10].mean()+20, f'$1 px \\approxeq {np.round(mm_per_px,1)} mm$\nin this example.')
        plt.title(label)
        plt.xlabel('Frame number')
        plt.ylabel('hip vertical displacement (pixels)')
        plt.legend(loc='upper left')
        sns.despine()
        plt.savefig('mm_to_px.pdf')
        plt.show()
    return mm_per_px

def gravity(visualize=False):
    # Always uses CMJ data
    P = read_list('cmj_data')
    ptm = []
    for a in range(len(P)):
        p = P[a]
        _, op_reps = get_reps(p['bl']['hip']['op'][:,1],
                            fps=30, plot=False, d=5)
        mm_px_list = []
        for i, rep in enumerate(op_reps):
            mmpx = mm_per_px(rep, fps=30, verbose=False,
                plot=True if (visualize and a==0 and i==0) else False)
            mm_px_list.append(mmpx)
        ave_mm_px = mm_px_list[0]
        ptm.append(ave_mm_px)
    return np.array(ptm)

def get_pixel_height(subject):
    tibia_length = euclidean(subject.RHeel[0][:2], subject.RKnee[0][:2])
    femur_length = euclidean(subject.RHip[0][:2], subject.RKnee[0][:2])
    trunk_length = euclidean(subject.MidHip[0][:2], subject.Neck[0][:2])
    trunk_to_heel = trunk_length+femur_length+tibia_length
    # estimate trunk_to_heel to be 6/8 of total height
    pixel_height = (8 * trunk_to_heel)/6
    return pixel_height

def height(task, start=0, end=16, skip=1):
    df = pd.read_pickle(f"./data/pickle/{task}_df.pkl")
    metre_heights = np.array([1.84, 1.5, 1.725, 1.73, 1.635, 1.67, 1.71, 1.65, 1.8, 1.88, 1.62, 1.85, 1.915, 1.78, 1.84, 1.76])
    pixel_heights = []
    for i in range(start, end, skip):
        row = df.iloc[i]
        pixel_heights.append(get_pixel_height(row))  
    return 1000*metre_heights/np.array(pixel_heights)



