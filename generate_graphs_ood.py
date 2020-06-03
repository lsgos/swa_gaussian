import numpy as np
import os
import re
import pandas as pd
import glob
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score


def process_dirname(name):
    match = re.match("([a-zA-Z]+)(\d+)([a-zA-Z0-9]+).*", name)
    if match is None:
        raise
    model = match.group(1)
    depth = match.group(2)
    dataset = match.group(3)

    depth = int(depth)
    return model, depth, dataset


def calculate_metrics(results):
    """
    Calculate the relevant metrics from the result numpy files
    """
    preds = results["predictions"]
    targets = results["targets"]
    entropy = results["entropies"]
    p = preds[range(preds.shape[0]), targets.astype(np.int)]
    ll = np.log(p + 1e-6)
    nll = -np.mean(ll)
    acc = np.mean(preds.argmax(-1) == targets)


    # calculate expected calibration error
    # use the maximum probability as probability of positive
    # and calculate prediction against a binary correct/incorrect problem
    maxprob = preds.max(-1)
    correct = preds.argmax(-1) == targets
    prob_true, prob_pred = calibration_curve(correct, maxprob, n_bins=5) # 20 used in SWAG paper
    ece = np.mean(np.abs(prob_true - prob_pred))
    return nll, acc, ece


def get_results(result_root):
    results = {}
    results["model"] = []
    results["depth"] = []
    results["auc_diag"] = []
    results["auc_full"] = []
    results["auc_sgd"] = []
    results["dataset"] = []

    for result_dir in os.listdir(result_root):
        model, depth, dataset = process_dirname(result_dir)

        prefix = os.path.join(result_root, result_dir)
        try:
            r_full = np.load(os.path.join(prefix, "uncertainty_res_swag.npz"))
            r_diag = np.load(os.path.join(prefix, "uncertainty_res_diag.npz"))
            r_sgd  = np.load(os.path.join(prefix, "uncertainty_res_sgd.npz"))

            r_full_ood = np.load(glob.glob(os.path.join(prefix, "ood_res_swag*"))[0])
            r_diag_ood = np.load(glob.glob(os.path.join(prefix, "ood_res_diag*"))[0])
            r_sgd_ood = np.load(glob.glob(os.path.join(prefix, "ood_res_sgd*"))[0])

            targets_real = np.zeros_like(r_diag['targets'])
            targets_ood = np.ones_like(r_diag_ood['targets'])

            y = np.r_[targets_real, targets_ood]
            
            x_full = np.r_[r_full['entropies'], r_full_ood['entropies']]            
            x_diag = np.r_[r_diag['entropies'], r_diag_ood['entropies']]            
            x_sgd = np.r_[r_sgd['entropies'], r_sgd_ood['entropies']]            


        except FileNotFoundError as f:
            print("warning: no results data found for experiment {}".format(result_dir))
            continue
        except IndexError as f:
            print("warning: no results data found for experiment {}".format(result_dir))
            continue
            
        results['auc_full'].append(roc_auc_score(y, x_full))
        results['auc_diag'].append(roc_auc_score(y, x_diag))
        results['auc_sgd'].append(roc_auc_score(y, x_sgd))

        results["model"].append(model)
        results["depth"].append(depth)
        results["dataset"].append(dataset)

    return pd.DataFrame(data=results)


def get_average_results(df):
    """
    Take the dataframe, which includes multiple runs with the same dataset/depth settings, and 
    collates these into single mean/std values
    This assumes that the dataframe is all the same dataset, and doesn't bother storing it
    """
    datasets = np.unique(df.dataset.values)
    depths = np.unique(df.depth.values)
    
    results = {}
    results["depth"] = []
    results['dataset'] = []

    fields = ["auc_diag", "auc_full", "auc_sgd"]
    for f in fields:
        results[f + '_mean'] = []
        results[f + '_std'] = []

    for dataset in datasets:
        for depth in depths:
            rows = df[(df.dataset == dataset) & (df.depth == depth)]
            results['depth'].append(depth)
            results['dataset'].append(dataset)
            for f in fields:
                results[f + '_mean'].append(rows[f].mean())
                results[f + '_std'].append(rows[f].std())
    return pd.DataFrame(data=results)

if __name__ == "__main__":
    df = get_results("results")
    df = df.sort_values(by='depth')
    df = get_average_results(df)


    plt.rcParams['figure.figsize'] = 10, 10
    dataset = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']

    def create_plot(metric='nll'):
        f, ax = plt.subplots(2, 2, sharex=True)
        axes = ax.flatten()

        for a, d in zip(axes, dataset):
            ss = df.dataset.eq(d)
            depth = df[ss]['depth']

            swag = df[ss][metric + '_full_mean']
            diag = df[ss][metric + '_diag_mean']
            sgd = df[ss][metric + '_sgd_mean']

            swag_std = df[ss][metric + '_full_std']
            diag_std = df[ss][metric + '_diag_std']
            sgd_std = df[ss][metric + '_sgd_std']

            a.plot(depth, swag, label=metric + '_swag_{}'.format(d), color='b')
            a.plot(depth, diag, label=metric + '_swag_diag_{}'.format(d), color='r')
            # a.plot(depth, sgd, label=metric + '_sgd_{}'.format(d), color='g')

            a.fill_between(depth, swag - swag_std, swag + swag_std, color='b', alpha=0.4, edgecolor=None)
            a.fill_between(depth, diag - diag_std, diag + diag_std, color='r', alpha=0.4, edgecolor=None)
            # a.fill_between(depth, sgd - sgd_std, sgd + sgd_std, color='g', alpha=0.4, edgecolor=None)
            a.legend()

        f.savefig(metric + '_depth_plot.png')
    for m in ['auc']:
        create_plot(m)

    plt.show()

