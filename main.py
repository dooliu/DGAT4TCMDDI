# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from dgllife.utils import Meter, EarlyStopping
from hyperopt import fmin, tpe
from shutil import copyfile
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.hyper import init_hyper_space
from utils import get_configure, mkdir_p, init_trial_path, \
    split_dataset, collate_molgraphs, load_model, predict, init_featurizer, load_dataset

from sklearn.metrics import roc_curve, auc

torch.cuda.device_count()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# precision, recall, f1, mcc, mae, rmse
metrics_map = {'SN': [], 'SP': [], 'ACC': [], 'MCC': [], 'precision': [], 'recall': [], 'r2': [],
               'f1': [], 'MAE': [], 'RMSE': [], 'AUC': []}
predict_list = []
ground_truth_list = []
loss_list = []
accurate_list = []
mae_list = []
train_score_list = []
valid_score_list = []
test_score_list = []

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()

    for batch_id, batch_data in enumerate(data_loader):
        # print(f"batchid:{batch_id}, {batch_data}")
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        logits = predict(args, model, bg)
        # logits = torch.sigmoid(logits)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        # print(f"loss value:{loss}")
        loss_list.append(loss.cpu().clone().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
        # if batch_id % args['print_every'] == 0:
        #     print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
        #         epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
    train_score = np.mean(train_meter.compute_metric(args['metric']))
    train_score_list.append(train_score)
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            logits = predict(args, model, bg)
            logits = torch.sigmoid(logits)
            # print(f"predict logit: {logits}")
            eval_meter.update(logits, labels, masks)

    return np.mean(eval_meter.compute_metric(args['metric'], reduction='mean'))

def main(args, exp_config, train_set, val_set, test_set):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })
    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()

    # Set up directory for saving results
    args = init_trial_path(args)

    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    print(f"exp_config:{exp_config}")
    model = load_model(exp_config).to(args['device'])

    loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
    # loss_criterion = nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                     weight_decay=exp_config['weight_decay'])
    stopper = EarlyStopping(patience=exp_config['patience'],
                            filename=args['trial_path'] + '/model.pth',
                            metric=args['metric'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        valid_score_list.append(val_score)
        test_score = run_an_eval_epoch(args, model, test_loader)
        test_score_list.append(test_score)
        # _, _, accurate_rate, _, _, _, _, _, mae, rmse = draw_roc_line(model, test_loader)
        # accurate_list.append(accurate_rate)
        # mae_list.append(mae)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'],
            val_score, args['metric'], stopper.best_score))
        if val_score < args['thresh_value']:
            early_stop = True
        if early_stop:
            break

    stopper.load_checkpoint(model)
    # draw a roc line
    SN, SP, accurate_rate, precision, recall, r2, f1, mcc, mae, rmse, roc_auc = cal_predict_res(model, test_loader)

    metrics_map['SN'].append(SN)
    metrics_map['SP'].append(SP)
    metrics_map['ACC'].append(accurate_rate)
    metrics_map['MCC'].append(mcc)
    metrics_map['precision'].append(precision)
    metrics_map['recall'].append(recall)
    metrics_map['r2'].append(r2)
    metrics_map['f1'].append(f1)
    metrics_map['MAE'].append(mae)
    metrics_map['RMSE'].append(rmse)
    metrics_map['AUC'].append(roc_auc)

    test_score = run_an_eval_epoch(args, model, test_loader)
    print('test {} {:.4f}'.format(args['metric'], test_score))

    with open(args['trial_path'] + '/eval.txt', 'w') as f:
        f.write('Best val {}: {}\n'.format(args['metric'], stopper.best_score))
        f.write('Test {}: {}\n'.format(args['metric'], test_score))

    with open(args['trial_path'] + '/configure.json', 'w') as f:
        json.dump(exp_config, f, indent=2)

    return args['trial_path'], stopper.best_score

def cal_predict_res(model, data_loader):
    # get predict value
    model.eval()

    real_y = []
    predict_y = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            logits = predict(args, model, bg)
            predict_y.extend(torch.sigmoid(logits).cpu().clone().detach().numpy())
            real_y.extend(labels.cpu().clone().detach().numpy())

    predict_y = np.array(predict_y)
    predict_list.extend(predict_y[:, 0])
    predict_y[predict_y < 0.5] = 0
    predict_y[predict_y >= 0.5] = 1
    predict_y = predict_y[:, 0]

    real_y = np.array(real_y)
    real_y = real_y[:, 0]
    ground_truth_list.extend(real_y)
    fpr, tpr, thresholds = roc_curve(real_y, predict_y)

    roc_auc = auc(fpr, tpr)

    from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, mean_absolute_error, \
        matthews_corrcoef, roc_auc_score, r2_score, mean_squared_error, confusion_matrix

    tn, fp, fn, tp = confusion_matrix(real_y, predict_y).ravel()
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)
    # 精准度，预测为阳性的样本中正确分类的比例
    precision = precision_score(real_y, predict_y)
    # 准确率
    accurate_rate = accuracy_score(real_y, predict_y)
    # recall tp / (tp + fn)
    recall = recall_score(real_y, predict_y)
    f1 = f1_score(real_y, predict_y)
    mae = mean_absolute_error(real_y, predict_y)
    rmse = mean_squared_error(real_y, predict_y)
    # MCC
    mcc = matthews_corrcoef(real_y, predict_y)
    # r2 score
    r2 = r2_score(real_y, predict_y)
    rocaucscore = roc_auc_score(real_y, predict_y)
    print(f"accurate rate:{accurate_rate}, roc_auc:{roc_auc}, roc_auc_score-{rocaucscore}")
    print(f"sklearn calculate results: SN-{SN}, SP-{SP}, precision-{precision}, recall-{recall}, f1_score-{f1}, MCC-{mcc}, "
          f"mae-{mae}, r2_score-{r2}, rmse-{rmse}")
    return SN, SP, accurate_rate, precision, recall, r2, f1, mcc, mae, rmse, roc_auc


def bayesian_optimization(args, train_set, val_set, test_set):
    # Run grid search
    results = []

    candidate_hypers = init_hyper_space(args['model'])

    def objective(hyperparams):
        configure = deepcopy(args)
        trial_path, val_metric = main(configure, hyperparams, train_set, val_set, test_set)

        if args['metric'] in ['roc_auc_score', 'pr_auc_score']:
            # Maximize ROCAUC is equivalent to minimize the negative of it
            val_metric_to_minimize = -1 * val_metric
        else:
            val_metric_to_minimize = val_metric

        results.append((trial_path, val_metric_to_minimize))

        return val_metric_to_minimize

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=args['num_evals'])
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric = results[0]

    return best_trial_path


if __name__ == '__main__':
    from argparse import ArgumentParser

    seed = random.randint(100000, 10000000)
    # seed = 5149199
    print(f"本轮的随机种子为：{seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = ArgumentParser('Multi-label Binary Classification')
    parser.add_argument('-c', '--csv-path', type=str, required=True,
                        help='Path to a csv file for loading a dataset')
    parser.add_argument('-sc1', '--smiles-column-1', type=str, required=True,
                        help='Header for the SMILES column in the CSV file')
    parser.add_argument('-sc2', '--smiles-column-2', type=str, required=True,
                        help='Header for the SMILES column in the CSV file')
    parser.add_argument('-t', '--task-names', default=None, type=str,
                        help='Header for the tasks to model. If None, we will model '
                             'all the columns except for the smiles_column in the CSV file. '
                             '(default: None)')
    parser.add_argument('-s', '--split',
                        choices=['scaffold_decompose', 'scaffold_smiles', 'random', 'ordinal', 'cross'],
                        default='scaffold_smiles',
                        help='Dataset splitting method (default: scaffold_smiles). For scaffold '
                             'split based on rdkit.Chem.AllChem.MurckoDecompose, '
                             'use scaffold_decompose. For scaffold split based on '
                             'rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles, '
                             'use scaffold_smiles.')
    parser.add_argument('-scl', '--cross-split-folds', default=3, type=int,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 3)')
    parser.add_argument('-sr', '--split-ratio', default='0.6,0.1,0.3', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-th', '--thresh-value', default=0.05, type=float,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.05)')
    parser.add_argument('-me', '--metric', choices=['mae', 'roc_auc_score', 'pr_auc_score'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-mo', '--model', choices=['DGAT'],
                        default='DGAT', help='Model to use (default: DGAT)')
    parser.add_argument('-a', '--atom-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for atoms (default: canonical)')
    parser.add_argument('-b', '--bond-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for bonds (default: canonical)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs allowed for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-nw', '--num-workers', type=int, default=1,
                        help='Number of processes for data loading (default: 1)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-p', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    parser.add_argument('-ne', '--num-evals', type=int, default=None,
                        help='Number of trials for hyperparameter search (default: None)')
    parser.add_argument('-re', '--run-evals', type=int, default=1,
                        help='Number of re-run times, for stable metrics (default: 1)')
    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    if args['task_names'] is not None:
        args['task_names'] = args['task_names'].split(',')

    args = init_featurizer(args)
    df = pd.read_csv(args['csv_path'])
    mkdir_p(args['result_path'])
    dataset = load_dataset(args, df)
    args['n_tasks'] = dataset.n_tasks
    begin_time = time.time()
    for _ in range(args['run_evals']):
        cross_flag = True if args['split'] == 'cross' else False
        data_tuple = []
        if cross_flag:
            pre_data_tuple = split_dataset(args, dataset)
            for train_set, val_set in pre_data_tuple:
                test_set = val_set
                data_tuple.append((train_set, val_set, test_set))
        else:
            train_set, val_set, test_set = split_dataset(args, dataset)
            data_tuple.append((train_set, val_set, test_set))
        for train_set, val_set, test_set in data_tuple:
        # data_tuple = split_dataset(args, dataset)
        # for train_set, val_set in data_tuple:
            # test_set = val_set
            if args['num_evals'] is not None:
                assert args['num_evals'] > 0, 'Expect the number of hyperparameter search trials to ' \
                                              'be greater than 0, got {:d}'.format(args['num_evals'])
                print('Start hyperparameter search with Bayesian '
                      'optimization for {:d} trials'.format(args['num_evals']))
                trial_path = bayesian_optimization(args, train_set, val_set, test_set)
            else:
                print('Use the manually specified hyperparameters')
                exp_config = get_configure(args['model'])
                main(args, exp_config, train_set, val_set, test_set)
                trial_path = args['result_path'] + '/1'
    print(f"accurate list:{metrics_map['ACC']}, SN list:{metrics_map['SN']}, SP list:{metrics_map['SP']}, "
          f"MCC list: {metrics_map['MCC']}, precision:{metrics_map['precision']}, recall:{metrics_map['recall']}, "
          f"r2：{metrics_map['r2']}, f1:{metrics_map['f1']}, MAE:{metrics_map['MAE']}, RMSE:{metrics_map['RMSE']}, "
          f"AUC:{metrics_map['AUC']}")

    folds = args['cross_split_folds']
    acc_list = metrics_map['ACC']
    sn_list = metrics_map['SN']
    sp_list = metrics_map['SP']
    mcc_list = metrics_map['MCC']
    precision_list = metrics_map['precision']
    recall_list = metrics_map['recall']
    r2_list = metrics_map['r2']
    f1_list = metrics_map['f1']
    auc_list = metrics_map['AUC']

    print(
        f"SN:{np.mean(sn_list)} +- {np.std(sn_list)} , SP:{np.mean(sp_list)} +- {np.std(sp_list)}, "
        f"ACC:{np.mean(acc_list)} +- {np.std(acc_list)}, MCC:{np.mean(mcc_list)} +- {np.std(mcc_list)}, "
        f"precision:{np.mean(precision_list)} +- {np.std(precision_list)}, recall:{np.mean(recall_list)} +- {np.std(recall_list)}, "
        f"r2:{np.mean(r2_list)} +- {np.std(r2_list)}, f1:{np.mean(f1_list)} +- {np.std(f1_list)}, "
        f"MAE:{np.mean(metrics_map['MAE'])}, RMSE:{np.mean(metrics_map['RMSE'])}, "
        f"AUC:{np.mean(auc_list)} +- {np.std(auc_list)}")

    copyfile(trial_path + '/model.pth', args['result_path'] + '/model.pth')
    copyfile(trial_path + '/configure.json', args['result_path'] + '/configure.json')
    copyfile(trial_path + '/eval.txt', args['result_path'] + '/eval.txt')
    print(f"time consume:{time.time() - begin_time}")