import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, average_precision_score, ndcg_score
from sklearn.metrics import roc_curve
import pytrec_eval
import json
import pandas as pd


def calculate_auc_roc(Y, Y_):
    auc = roc_auc_score(Y, Y_, average='micro', multi_class="ovr")
    fpr, tpr, _ = roc_curve(Y.ravel(), Y_.ravel())
    return auc, fpr, tpr


def calculate_metrics(Y, Y_, per_instance=False, metrics=None):
    if metrics is None:
        metrics = {
            'P_2', 'P_5', 'P_10', 'recall_2', 'recall_5', 'recall_10',
            'ndcg_cut_2', 'ndcg_cut_5', 'ndcg_cut_10',
            'map_cut_2', 'map_cut_5', 'map_cut_10'
        }

    aucroc, fpr, tpr = calculate_auc_roc(Y, Y_)

    qrel = {}
    run = {}
    print(f'Building pytrec_eval input for {Y.shape[0]} instances ...')
    for i, (y, y_) in enumerate(zip(Y, Y_)):
        qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in np.where(y != 0)[0]}
        run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j, v in enumerate(y_.tolist())}

    empty_qrel_indexes = [i for i in sorted(qrel.keys()) if not qrel[i]]

    for i in sorted(empty_qrel_indexes, reverse=True):
        del qrel[f'q{i}']
        del run[f'q{i}']

    print(f'Evaluating {metrics} ...')
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    results = evaluator.evaluate(run)

    df = pd.DataFrame.from_dict(results, orient='index')
    print(f'Averaging ...')
    df_mean = pd.concat(
        [df.mean(axis=0).to_frame('mean'), pd.Series(aucroc, index=['aucroc'], name='mean').to_frame().T])

    return df if per_instance else None, df_mean, (fpr, tpr)