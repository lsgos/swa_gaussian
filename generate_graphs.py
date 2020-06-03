import numpy as np
import os
import re
import pandas as pd
import glob
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7.0 

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
    results["nll_diag"] = []
    results["acc_diag"] = []
    results["ece_diag"] = []
    results["acc_full"] = []
    results["nll_full"] = []
    results["ece_full"] = []
    results["acc_sgd"] = []
    results["nll_sgd"] = []
    results["ece_sgd"] = []
    results["dataset"] = []

    for result_dir in os.listdir(result_root):
        model, depth, dataset = process_dirname(result_dir)

        prefix = os.path.join(result_root, result_dir)
        try:
            r_full = np.load(os.path.join(prefix, "uncertainty_res_swag.npz"))
            r_diag = np.load(os.path.join(prefix, "uncertainty_res_diag.npz"))
            r_sgd  = np.load(os.path.join(prefix, "uncertainty_res_sgd.npz"))


        except FileNotFoundError as f:
            print("warning: no results data found for experiment {}".format(result_dir))
            continue

        nll_f, acc_f, ece_f = calculate_metrics(r_full)
        nll_d, acc_d, ece_d = calculate_metrics(r_diag)
        nll_s, acc_s, ece_s = calculate_metrics(r_sgd)

        results["model"].append(model)
        results["depth"].append(depth)
        results["dataset"].append(dataset)

        results["nll_diag"].append(nll_d)
        results["acc_diag"].append(acc_d)
        results["ece_diag"].append(ece_d)

        results["nll_full"].append(nll_f)
        results["acc_full"].append(acc_f)
        results["ece_full"].append(ece_f)

        results["nll_sgd"].append(nll_s)
        results["acc_sgd"].append(acc_s)
        results["ece_sgd"].append(ece_s)


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

    fields = ["nll_diag","acc_diag","ece_diag","acc_full","nll_full","ece_full","acc_sgd","nll_sgd","ece_sgd",]
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


    palette = sns.color_palette(n_colors=3)

    dataset = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']

    def create_plot(metric='nll'):
        ylabels = {
            'nll': 'NLL',
            'acc': 'Accuracy',
            'ece': 'Expected Calibration Error'
        }
        for d in dataset:
            f, a = plt.subplots(figsize=(2.64 , 1.631 ))
            
            ss = df.dataset.eq(d)
            depth = df[ss]['depth']

            swag = df[ss][metric + '_full_mean']
            diag = df[ss][metric + '_diag_mean']
            sgd = df[ss][metric + '_sgd_mean']

            swag_std = df[ss][metric + '_full_std']
            diag_std = df[ss][metric + '_diag_std']
            sgd_std = df[ss][metric + '_sgd_std']

            a.plot(depth, swag, label='SWAG', color=palette[0])
            a.plot(depth, diag, label='SWAG-Diag'.format(d), color=palette[1])
            # a.plot(depth, sgd, label= 'SGD'.format(d), color=palette[2])
            a.set_xlabel('PreResNet Depth')
            a.set_ylabel(ylabels[metric])

            a.fill_between(depth, swag - swag_std, swag + swag_std, color=palette[0], alpha=0.4, edgecolor=None)
            a.fill_between(depth, diag - diag_std, diag + diag_std, color=palette[1], alpha=0.4, edgecolor=None)
            # a.fill_between(depth, sgd - sgd_std, sgd + sgd_std, color=palette[2], alpha=0.4, edgecolor=None)
            a.legend(frameon=False)
            sns.despine(f)
            f.savefig(os.path.join('images', d + '_' + metric + '_depth_plot.pdf'), bbox_inches='tight')
    for m in ['nll', 'acc', 'ece']:
        create_plot(m)

    plt.show()

