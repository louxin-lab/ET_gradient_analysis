import os
from datasets import mask
from brainspace.gradient import GradientMaps
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind

def load_arrays(in_dir):
    files = os.listdir(in_dir)
    arrays = []
    for f in files:
        if f.startswith('z') and f.endswith('txt'):
            array = np.loadtxt(os.path.join(in_dir, f))
            arrays.append(array)
    return arrays, files

def generate_template_gradientmap(in_dir, n_components, kernel, sparsity):
    arrays, _ = load_arrays(in_dir)
    temp_array = np.mean(arrays, axis=0)

    gm_temp = GradientMaps(n_components=n_components, kernel=kernel)
    gm_temp.fit(temp_array, sparsity=sparsity)

    return gm_temp

def get_aligned_gradientmaps(in_dir, n_components, approach, kernel, alignment, sparsity, gm_temp):
    arrays, files = load_arrays(in_dir)
    temp_gradients = gm_temp.gradients_

    gm = GradientMaps(n_components=n_components, approach=approach, kernel=kernel,  alignment=alignment)
    gm.fit(arrays, sparsity=sparsity, reference=temp_gradients)

    return gm, files

def save_gradient(out_dir, subject_name, gradient, save_csv, save_nii, mask=None):
    basename = os.path.splitext(os.path.basename(subject_name))[0]
    value_dict = dict(zip(range(1, np.shape(gradient)[0]+1), gradient))
    if save_csv:
        df = pd.DataFrame.from_dict(value_dict, orient='index', columns=['gradient'])
        df.to_csv(os.path.join(out_dir, basename+'.csv'))
    if save_nii:
        if mask is not None:
            mask.save_values(value_dict, out_path=os.path.join(out_dir, basename+'.nii'))
        else:
            raise ValueError('$mask$ is needed.')

def save_gradients(out_dir, subject_names, subjects_gradients, nth_gradient, save_csv, save_nii, mask):
    for subject_name, gradients in zip(subject_names, subjects_gradients):
        save_gradient(out_dir, subject_name, gradients[:, nth_gradient], save_csv=save_csv, save_nii=save_nii, mask=mask)

def load_gradient_csv(in_dir, subject_name):
    basename = os.path.splitext(os.path.basename(subject_name))[0]
    df = pd.read_csv(os.path.join(in_dir, basename)+'.csv', index_col=0)
    return df

def load_gradient_csvs(in_dir, subject_names):
    dfs = []
    for subject_name in subject_names:
        df = load_gradient_csv(in_dir, subject_name)
        dfs.append(df)
    return dfs

def reverse_gradient(in_dir, subject_name, save_csv, save_nii, mask):
    df = load_gradient_csv(in_dir, subject_name)
    gradient = df['gradient'].values
    gradient = -gradient
    save_gradient(in_dir, subject_name, gradient, save_csv, save_nii, mask)
