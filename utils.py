import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def result_stats(solutions, keyname, basename=None, other_keys=[]):
    algos = list(solutions.keys())
    total_algo = len(algos)

    values = pd.DataFrame([solutions[key][keyname]
                          for key in algos], index=algos).T
    if basename is not None:
        values_base = values.drop(basename, axis=1)
        for col in values_base.columns:
            values_base[col] /= values[basename]
    dfs = [values] if basename is None else [values, values_base]
    other_dfs = {key: pd.DataFrame(
        [solutions[algo][key] for algo in algos], index=algos).T for key in other_keys}

    ncols = 1 if basename is None else 2
    fig, ax = plt.subplots(ncols=ncols, figsize=(20, 4))
    if ncols == 1:
        ax = [ax]
    tags = [keyname, 'relative %s' % keyname]
    for i in range(ncols):
        _mean = dfs[i].mean()
        title = 'average %s-  ' % tags[i]
        for j in range(_mean.shape[0]):
            title += '%s: %.4f;  ' % (_mean.index[j], _mean.iloc[j])
        if i == 0:
            for key in other_keys:
                _mean = other_dfs[key].mean()
                title += '\naverage %s-  ' % key
                for j in range(_mean.shape[0]):
                    title += '%s: %.4f;  ' % (_mean.index[j], _mean.iloc[j])
        print(title)
        ax[i].set_title(title)
        ax[i].set_xlabel(keyname)
        ax[i].set_ylabel('case count')
        dfs[i].plot(kind='hist', ax=ax[i], bins=50, alpha=.5)
    fig.tight_layout()
    plt.show()


def comparison_table(solutions, keyname, maximize=False):
    noneq_table = pd.DataFrame(columns=list(
        solutions.keys()), index=list(solutions.keys()))
    eq_table = pd.DataFrame(columns=list(
        solutions.keys()), index=list(solutions.keys()))
    tag = 'better' if maximize else 'worse'
    tags = [tag, 'equal']

    algos = list(solutions.keys())
    total_algo = len(algos)
    values = pd.DataFrame([solutions[key][keyname]
                          for key in algos], index=algos).T
    for i in range(total_algo):
        for j in range(i, total_algo):
            eq = (values[algos[i]] == values[algos[j]]).sum()
            igtj = (values[algos[i]] > values[algos[j]]).sum()
            iltj = (values[algos[i]] < values[algos[j]]).sum()
            noneq_table[algos[i]].loc[algos[j]] = igtj
            noneq_table[algos[j]].loc[algos[i]] = iltj
            eq_table[algos[i]].loc[algos[j]] = eq
            eq_table[algos[j]].loc[algos[i]] = eq
    noneq_table.columns = [[tags[0] for _ in range(
        noneq_table.shape[1])], noneq_table.columns]
    eq_table.columns = [[tags[1]
                         for _ in range(eq_table.shape[1])], eq_table.columns]
    comb_table = pd.concat([noneq_table, eq_table], axis=1, sort=False)
    return comb_table
