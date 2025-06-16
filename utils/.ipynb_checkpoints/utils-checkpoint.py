#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import datetime
import os

def save_result(data, ylabel, args):
    data = {'base' :data}

    path = './output/{}'.format(args.noniid_case)
    if args.dataset == 'emnist':
        dataset_label = f"{args.dataset}-{args.emnist_type}"
    else:
        dataset_label = args.dataset
    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}_{}_pre_{}_B_{}_KD_{}_Pow_{}_T_{}.txt'.format(dataset_label, args.algorithm, args.model,
                                                                ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"),args.frac, args.num_users,args.first_stage_bound,args.KD_buffer_bound,args.KD_alpha,args.Dynamic_KD_power,args.temperature)
    else:
        path += '/{}'.format(args.data_beta)
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}_{}_pre_{}_B_{}_KD_{}_Pow_{}_T_{}.txt'.format(dataset_label, args.algorithm, args.model,
                                                                ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"),args.frac, args.num_users,args.first_stage_bound,args.KD_buffer_bound,args.KD_alpha,args.Dynamic_KD_power,args.temperature)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path,file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()




def save_model(data, ylabel, args):

    path = './output/{}'.format(args.noniid_case)
    if args.dataset == 'emnist':
        dataset_label = f"{args.dataset}-{args.emnist_type}"
    else:
        dataset_label = args.dataset
    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}_{}_pre_{}_B_{}_KD_{}_Pow_{}_T_{}.txt'.format(dataset_label, args.algorithm, args.model,
                                                                ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"),args.frac, args.num_users,args.first_stage_bound,args.KD_buffer_bound,args.KD_alpha,args.Dynamic_KD_power,args.temperature)
    else:
        path += '/{}'.format(args.data_beta)
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}_{}_pre_{}_B_{}_KD_{}_Pow_{}_T_{}.txt'.format(dataset_label, args.algorithm, args.model,
                                                                ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"),args.frac, args.num_users,args.first_stage_bound,args.KD_buffer_bound,args.KD_alpha,args.Dynamic_KD_power,args.temperature)

    if not os.path.exists(path):
        os.makedirs(path)

    # with open(os.path.join(path,file), 'a') as f:
    #     for label in data:
    #         f.write(label)
    #         f.write(' ')
    #         for item in data[label]:
    #             item1 = str(item)
    #             f.write(item1)
    #             f.write(' ')
    #         f.write('\n')
    torch.save(data, os.path.join(path,file))
    print('save finished')
    # f.close()
