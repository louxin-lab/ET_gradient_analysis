# %%
import os
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import t as tdist
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

def sparsity(array, thres=0.9):
    non_zero = array.ravel()[np.flatnonzero(array)]
    sorted = np.sort(non_zero) 
    nth = sorted[int(len(sorted)*thres)]
    array[array<nth] = 0
    return array

def row_sparsity(array, thres=0.9):
    tmp_array = np.zeros_like(array)

    for i in range(array.shape[0]):
        tmp = copy.deepcopy(array[i])
        tmp_array[i] = sparsity(tmp, thres)
    return tmp_array

def fdr(array, alpha=0.05, is_r=True, n=None):
    # input pearson r array or p array
    shape = array.shape
    array = array.flatten()

    if is_r:
        if n is None:
            raise ValueError('Please input n if array is pearsonr array')
        else:
            r = array
            df = n - 2
            t = r * np.sqrt(df/(1 - r**2))
            # 2 tail
            p = 2*tdist.sf(t, df)
            array = p
    qarray = fdrcorrection(array, alpha)[1]
    array = np.reshape(qarray, shape)
    return array


def binary_array(array):
    array[array!=0] = 1 #
    return array

def min_max(array): 
    _min = np.nanmin(array)
    _max = np.nanmax(array)
    array = (array-_min)/(_max-_min)
    array = np.nan_to_num(array, nan=0, posinf=1, neginf=0) 
    return array

def z_score(array): 
    _mean = np.nanmean(array, axis=None)
    _std = np.nanstd(array,axis=None)
    array = (array - _mean)/_std
    array = np.nan_to_num(array, nan=-4*_std, posinf=4*_std, neginf=-4*_std)
    return array

def min_max_4D(array):
    array = np.array(array)
    new_array = np.zeros_like(array)
    for tmp, i in zip(array, range(array.shape[0])):
        _min = np.min(tmp, axis=None)
        _max = np.max(tmp,axis=None)
        new_array[i,:,:] = (tmp - _min)/(_max - _min)
    return new_array

def z_score_4D(array):
    array = np.array(array)
    new_array = np.zeros_like(array)
    for tmp, i in zip(array, range(array.shape[0])):
        _mean = np.mean(tmp, axis=None)
        _std = np.std(tmp,axis=None)
        new_array[i,:,:] = (tmp - _mean)/_std
    return new_array

def findwalks_v2(array, n_step=None, normalize=z_score):
    """
    based on Sepulcre-Bernad, Jorge,M.D. matlab code
    """
    array = np.array(array, dtype=np.float32)
    array = binary_array(array)

    if n_step is None:
        n_step = array.shape[0]
    n = np.shape(array)[0]
    wq = np.zeros((n_step,n,n))
    array = normalize(array)
    np.fill_diagonal(array, 0)
    
    array2 = copy.deepcopy(array)
    wq[0,:,:] = array
    for q in range(1, n_step):
        array2 = array2 @ array
        # Normalize array2 to avoid path count explode
        array2 = normalize(array2)
        # diagonal to 0 according to SSS
        np.fill_diagonal(array2, 0)
        '''
        原版的是min-max
        原版是先normalize，在进行置零，这里倒过来也是可以的        
        '''
        wq[q,:,:] = array2
    wlq = np.sum(np.sum(wq, axis=-1), axis=-1)
    twalk = np.sum(wlq)
    return wq, twalk, wlq

def find_stable(wq, seed, thres, patience, step=None, out_dir=None):
    plot_array = []
    x = []
    y = []
    last_pointer = 0
    last = wq[last_pointer, seed, :]
    plot_array.append(last)
    count = 0
    if step is None:
        step = wq.shape[0]

    for i in range(1, step):
        to = wq[i, seed, :]
        dist = np.linalg.norm(last - to)

        plot_array.append(to)
        x.append(i)
        y.append(dist)

        if dist < thres:
            count += 1
            if count >= patience:
                break
        else:
            count = 0
            last_pointer = i
            last = wq[last_pointer, seed, :]

    if last_pointer + patience > step:
        last_pointer = -1

    sns.heatmap(np.array(plot_array).T, square=False)
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, f'stepcount_{seed}_{thres}_{patience}.png'))
    plt.close()
    sns.lineplot(x=x, y=y)
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, f'dist_{seed}_{thres}_{patience}.png'))
    plt.close()
    return last_pointer

def find_stable_corr(wq, seed, thres, patience, step=None, out_dir=None):
    plot_array = []
    x = []
    y = []
    last_pointer = 0
    last = wq[last_pointer, seed, :]
    plot_array.append(last)
    count = 0
    if step is None:
        step = wq.shape[0]

    for i in range(1, step):
        to = wq[i, seed, :]
        
        r, p = pearsonr(last, to)

        plot_array.append(to)
        x.append(i)
        y.append(r)

        if r > thres:
            count += 1
            if count >= patience:
                break
        else:
            count = 0
            last_pointer = i
            last = wq[last_pointer, seed, :]

    if last_pointer + patience > step:
        last_pointer = -1

    sns.heatmap(np.array(plot_array).T, square=False)
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, f'stepcount_{seed}_{thres}_{patience}.png'))
    plt.close()
    sns.lineplot(x=x, y=y)
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, f'dist_{seed}_{thres}_{patience}.png'))
    plt.close()
    return last_pointer

def dist_lineplot(wq, seed, out_dir=None):
    x = []
    y = []

    for i in range(1, wq.shape[0]):
        last = wq[i-1, seed, :]
        to = wq[i, seed, :]
        dist = np.linalg.norm(last - to)

        x.append(i)
        y.append(dist)
    sns.lineplot(x=x, y=y)
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, f'dist_{seed}.png'))
        plt.close()
    else:
        plt.show()