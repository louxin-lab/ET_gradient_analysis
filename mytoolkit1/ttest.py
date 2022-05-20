import csv
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
from mytoolkit2 import stepwise

from mytoolkit2 import controllability

def load_subjects_cbf(subjects, obs_name1, obs_name2):
    all_values1 = []
    all_values2 = []

    for subject in subjects:
        obs1 = subject.get_observation(obs_name1)
        obs2 = subject.get_observation(obs_name2)

        asl1 = obs1.asl
        asl2 = obs2.asl
        try:
            values1 = asl1.load_csv('roi_mean_cbf.csv')['Mean'].values
            values2 = asl2.load_csv('roi_mean_cbf.csv')['Mean'].values

            all_values1.append(values1)
            all_values2.append(values2)
        except FileNotFoundError:
            print('{}, lack CBF file'.format(subject.name))

    # transpose all_values to shape (nodal, subject)
    all_values1 = np.array(all_values1).T
    all_values2 = np.array(all_values2).T
    return all_values1, all_values2

def load_subjects_gmv(subjects, obs_name1, obs_name2):
    all_values1 = []
    all_values2 = []

    for subject in subjects:
        obs1 = subject.get_observation(obs_name1)
        obs2 = subject.get_observation(obs_name2)

        t11 = obs1.t1
        t12 = obs2.t1

        values1 = t11.load_csv('label/roi_gmv_{}.csv')['Volume'].values
        values2 = t12.load_csv('label/roi_gmv_{}.csv')['Volume'].values

        all_values1.append(values1)
        all_values2.append(values2)

    # transpose all_values to shape (nodal, subject)
    all_values1 = np.array(all_values1).T
    all_values2 = np.array(all_values2).T
    return all_values1, all_values2

def load_subjects_global_metric(subjects, obs_name, mat_name, metric_name, threshold_type, threshold_value):
    metrics = []
    for subject in subjects:
        # dti = subject.get_observation(obs_name).dti #0803添加
        bold = subject.get_observation(obs_name).bold

        # metric = dti.get_global_metric(metric_name, mat_name, threshold_type, threshold_value=threshold_value) #0803添加
        metric = bold.get_global_metric(metric_name, mat_name, threshold_type, threshold_value=threshold_value)

        metrics.append(metric)
    metrics = np.array(metrics)
    return metrics

def load_subjects_nodal_metric(subjects, obs_name1, obs_name2, mat_name,
                               metric_name, threshold_type, threshold_value):
    all_values1 = []
    all_values2 = []

    for subject in subjects:
        obs1 = subject.get_observation(obs_name1)
        obs2 = subject.get_observation(obs_name2)

        bold1 = obs1.brant2
        bold2 = obs2.brant2

        values1 = bold1.get_nodal_metric(metric_name, mat_name=mat_name,
                                        threshold_type=threshold_type, threshold_value=threshold_value)
        values2 = bold2.get_nodal_metric(metric_name, mat_name=mat_name,
                                        threshold_type=threshold_type, threshold_value=threshold_value)

        all_values1.append(values1)
        all_values2.append(values2)

    # transpose all_values to shape (nodal, subject)
    all_values1 = np.array(all_values1).T
    all_values2 = np.array(all_values2).T
    return all_values1, all_values2

def load_subjects_ave_controllablity(subjects, obs_name1, obs_name2, filename):
    all_values1 = []
    all_values2 = []

    for subject in subjects:
        obs1 = subject.get_observation(obs_name1)
        obs2 = subject.get_observation(obs_name2)

        bold1 = obs1.bold
        bold2 = obs2.bold

        A = np.loadtxt(bold1.build_path(filename))
        #A[A<0] = 0
        A = stepwise.sparsity(A, 0.9)
        A[A>0] = 1

        B = np.loadtxt(bold2.build_path(filename))
        #B[B<0] = 0
        B = stepwise.sparsity(A, 0.9)
        B[B>0] = 1

        values1 = controllability.ave_controllablity(A)
        values2 = controllability.ave_controllablity(B)

        all_values1.append(values1)
        all_values2.append(values2)

    # transpose all_values to shape (nodal, subject)
    all_values1 = np.array(all_values1).T
    all_values2 = np.array(all_values2).T
    return all_values1, all_values2

def load_clinical(subjects, obs_name, metric_name):
    metrics = []
    for subject in subjects:
        obs = subject.get_observation(obs_name)
        if metric_name == 'handtremor':
            metrics.append(obs.args['handtremor'])
        elif metric_name == 'CRST_A':
            metrics.append(obs.args['CRSTA_total'])
        elif metric_name == 'CRST_B':
            metrics.append(obs.args['CRST b_total'])
        elif metric_name == 'CRST_C':
            metrics.append(obs.args['CRST C'])
        elif metric_name == 'CRST_TOTAL':
            metrics.append(obs.args['CRST TOTAL'])
    metrics = np.array(metrics)
    return metrics

def save_nodal_ttest(all_values1, all_values2,
                     t1_mask, out_csv_path, out_nii_path):
    i = 1
    t_dict = {}
    with open(out_csv_path, 'w', newline='') as file:
        
        fieldnames = ['ID', 't-value', 'p-value']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for values1, values2 in zip(all_values1, all_values2):
            t, p = ttest_rel(values1, values2)
            writer.writerow({'ID': i, 't-value': t, 'p-value': p})
            t_dict[i] = t
            i += 1
    t1_mask.save_values(t_dict, out_nii_path)

# graph metric ttest
def graph_metric_ttest(subjects, t1_mask,
                        out_csv_path_ske, out_nii_path_ske,
                        obs_name1='base', obs_name2='360d',
                        threshold_type='intensity',
                        threshold_value=0.006):
    metrics = ['neighbor_degree',
                'betweenness_centrality', 'degree',
                'fault_tolerance', 'shortest_path_length',
                'global_efficiency', 'clustering_coefficient',
                'local_efficiency', 'vulnerability']

    for metric in metrics:
        out_csv_path = out_csv_path_ske.format(metric)
        out_nii_path = out_nii_path_ske.format(metric)
        all_values1, all_values2 = load_subjects_nodal_metric(subjects, obs_name1, obs_name2,
                                                              metric, threshold_type, threshold_value)
        save_nodal_ttest(all_values1, all_values2, t1_mask,
                        out_csv_path, out_nii_path)
# GMV ttest
def gmv_ttest(subjects, t1_mask, out_csv_path, out_nii_path, 
                obs_name1='base', obs_name2='360d', threshold_value=0.006):
    all_values1, all_values2 = load_subjects_gmv(subjects, obs_name1, obs_name2)
    save_nodal_ttest(all_values1, all_values2, t1_mask,
                    out_csv_path, out_nii_path)

def cbf_ttest(subjects, t1_mask, out_csv_path, out_nii_path,
            obs_name1='base', obs_name2='360d', threshold_value=0.006):
    all_values1, all_values2 = load_subjects_cbf(subjects, obs_name1, obs_name2)
    save_nodal_ttest(all_values1, all_values2, t1_mask,
                    out_csv_path, out_nii_path)

def load_subjects_surgery_info(subjects, obs_name, info_name):
    infos = []
    for subject in subjects:
        obs = subject.get_observation(obs_name)
        value = float(obs.args[info_name])
        infos.append(value)
    infos = np.array(infos)
    return infos

def ind_cohen_d(x1, x2, axis=0):
    #https://www.datanovia.com/en/lessons/t-test-effect-size-using-cohens-d-measure/
    #https://mlln.cn/2020/12/23/%E5%9D%87%E5%80%BC%E5%B7%AE%E5%BC%82%E7%9A%84%E6%95%88%E5%BA%94%E9%87%8F%E5%9C%A8%E7%BA%BF%E8%AE%A1%E7%AE%97%E5%99%A8/
    x1 = np.array(x1)
    x2 = np.array(x2)

    if len(x1.shape) == 1 and len(x2.shape) == 1:
        axis = None
        n1 = x1.shape[0]
        n2 = x2.shape[0]
    else:
        n1 = x1.shape[axis]
        n2 = x2.shape[axis]
    m1 = np.mean(x1, axis=axis)
    s1 = np.std(x1, axis=axis)
    
    m2 = np.mean(x2, axis=axis)
    s2 = np.std(x2, axis=axis)
    
    s = np.sqrt(((n1-1)*(s1**2)+(n2-1)*(s2**2))/(n1+n2-2))
    d = (m1 - m2) / s

    return d

def rel_cohen_d(x1, x2, axis=0):
    #https://www.real-statistics.com/students-t-distribution/paired-sample-t-test/cohens-d-paired-samples/
    x1 = np.array(x1)
    x2 = np.array(x2)

    if len(x1.shape) == 1 and len(x2.shape) == 1:
        axis = None

    x_delta = x1 - x2
    
    m = np.mean(x_delta, axis=axis)
    s = np.std(x_delta, axis=axis)
    d = m/s
    return d


from scipy.stats import pearsonr
def rel_cohen_drm(x1, x2, axis=0):
    #https://www.real-statistics.com/students-t-distribution/paired-sample-t-test/cohens-d-paired-samples/
    m1 = np.mean(x1, axis=axis)
    s1 = np.std(x1, axis=axis)
    m2 = np.mean(x2, axis=axis)
    s2 = np.std(x2, axis=axis)
    r, _ = pearsonr(x1, x2, axis=axis)
    sz = np.sqrt(s1**2+s2**2-2*r*s1*s2)
    srm = sz / np.sqrt(2*(1-r))
    d = (m1 - m2) / srm
    return d