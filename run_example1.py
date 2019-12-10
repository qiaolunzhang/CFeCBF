#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/12/18

@author: Maurizio Ferrari Dacrema
"""

from GraphBased.RP3betaRecommender import RP3betaRecommender
from FeatureWeighting.Cython.CFW_D_Similarity_Cython import CFW_D_Similarity_Cython, EvaluatorCFW_D_wrapper

from Base.Evaluation.Evaluator import EvaluatorHoldout

from Data_manager.Movielens_20m.Movielens20MReader import Movielens20MReader
from Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold


import numpy as np
import os
from datetime import datetime
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import scipy.sparse as sps


def rowSplit(rowString, numColumns):
    split = rowString.split(",")
    split[numColumns-1] = split[numColumns-1].replace("\n", "")
    # change the type of data
    for i in range(numColumns-1):
        split[i] = int(split[i])
    split[numColumns-1] = float(split[numColumns-1])
    result = tuple(split)
    return result

def get_URM(URM_path):
    with open(URM_path, 'r') as URM_file:
        URM_file.seek(1)
        URM_tuples = []
        index = 0
        for line in URM_file:
            if index:
                URM_tuples.append(rowSplit(line, 3))
            index = index + 1
        print("Print out the first tripple: ")
        print(URM_tuples[0])

        userList, itemList, ratingList = zip(*URM_tuples)

        userList = list(userList)
        itemList = list(itemList)
        ratingList = list(ratingList)

        URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
        URM_all = URM_all.tocsr()
    return URM_all


def rowSplit_ICM(rowString):
    split = rowString.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])  # tag is a string, not a float like the rating

    result = tuple(split)

    return result


def get_ICM(ICM_path, URM_all):
    ICM_sub_class_file.seek(0)
    ICM_sub_class_tuples = []

    index_line = 0
    for line in ICM_sub_class_file:
        if index_line:
            ICM_sub_class_tuples.append(rowSplit_ICM(line))
        index_line = index_line + 1

    itemList_icm_sub_class, featureList_icm_sub_class, dataList_icm_sub_class = zip(*ICM_sub_class_tuples)


    n_items = URM_all.shape[1]
    n_features = max(featureList_icm_sub_class) + 1

    ICM_shape = (n_items, n_features)

    ones = np.ones(len(dataList_icm_sub_class))
    ICM_all = sps.coo_matrix((ones, (itemList_icm_sub_class, featureList_icm_sub_class)), shape=ICM_shape)
    ICM_all = ICM_all.tocsr()
    return ICM_all, n_items, n_features



def test_save(recommender, data_file_path):
    test_path = data_file_path
    test_num = open(test_path, 'r')
    test_tuples = []
    idx = 0
    for aline in test_num:
        if idx:
            test_tuples.append(aline.replace("\n", ""))
        idx = idx + 1

    result = {}
    label = 0
    for user_idx in test_tuples:
        # for user_id in range(n_users_to_test):
        user_idx = int(user_idx)
        rate = recommender.recommend(user_idx, cutoff=10)
        if user_idx % 3000 == 0:
            label = label + 1
            print('---------------{}0----------------'.format(label))
        result[user_idx] = rate
    return result


def create_csv(results, result_name, results_dir='./res/', ):

    csv_fname = result_name + datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:
        # masterList, temp = [], []
        # writer = csv.writer(f,delimiter=',')
        # for key, value in results.items():
        #     temp = [key, value]
        #     masterList.append(temp)
        #     writer.writerows(masterList)
        f.write('user_id,item_list\n')

        for key, value in results.items():
            nor_v = ""
            for i in value:
                nor_v = nor_v + str(i) + " "
            f.write(str(key) + ',' + nor_v + '\n')


def get_precision(learning_rate, num_epoch, URM_train, URM_test):
    recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False)

    recommender.fit(epochs=num_epoch, batch_size=1, sgd_mode='sgd', learning_rate=learning_rate, positive_threshold_BPR=1)

    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results_dict, results_run_string = evaluator_validation.evaluateRecommender(recommender)
    return results_dict[10]['PRECISION']


data_file_path = "./data"
URM_path = data_file_path + "/data_train.csv"
test_path = data_file_path + "/data_target_users_test.csv"
URM_all = get_URM(URM_path)


ICM_asset_path = "data/data_ICM_asset.csv"
ICM_asset_file = open(ICM_asset_path, 'r')

ICM_price_path = "data/data_ICM_price.csv"
ICM_price_file = open(ICM_price_path, 'r')

ICM_sub_class = "data/data_ICM_sub_class.csv"
ICM_sub_class_file = open(ICM_sub_class, 'r')


ICM_all, n_items, n_features = get_ICM(ICM_sub_class, URM_all)
print("Number of items is ", str(n_items))
print("n_features is ", str(n_features))




# Selecting a dataset
dataReader = Movielens20MReader()

# Splitting the dataset. This split will produce a warm item split
# To replicate the original experimens use the dataset accessible here with a cold item split:
# https://mmprj.github.io/mtrm_dataset/index
dataSplitter = DataSplitter_Warm_k_fold(dataReader)
dataSplitter.load_data()

# Each URM is a scipy.sparse matrix of shape |users|x|items|
URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

# The ICM is a scipy.sparse matrix of shape |items|x|features|
ICM = dataSplitter.get_ICM_from_name("ICM_genre")

# This contains the items to be ignored during the evaluation step
# In a cold items setting this should contain the indices of the warm items
ignore_items = []

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5], ignore_items=ignore_items)
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5], ignore_items=ignore_items)

# This is used by the ML model of CFeCBF to perform early stopping and may be omitted.
# ICM_target allows to set a different ICM for this validation step, providing flexibility in including
# features present in either validation or test but not in train
evaluator_validation_earlystopping = EvaluatorCFW_D_wrapper(evaluator_validation, ICM_target=ICM, model_to_use="last")



# We compute the similarity matrix resulting from a RP3beta recommender
# Note that we have not included the code for parameter tuning, which should be done

cf_parameters = {'topK': 500,
                 'alpha': 0.9,
                 'beta': 0.7,
                 'normalize_similarity': True}

recommender_collaborative = RP3betaRecommender(URM_train)
recommender_collaborative.fit(**cf_parameters)

result_dict, result_string = evaluator_test.evaluateRecommender(recommender_collaborative)
print("CF recommendation quality is: {}".format(result_string))


# We get the similarity matrix
# The similarity is a scipy.sparse matrix of shape |items|x|items|
similarity_collaborative = recommender_collaborative.W_sparse.copy()




# We instance and fit the feature weighting algorithm, it takes as input:
# - The train URM
# - The ICM
# - The collaborative similarity matrix
# Note that we have not included the code for parameter tuning, which should be done as those are just default parameters

fw_parameters =  {'epochs': 200,
                  'learning_rate': 0.0001,
                  'sgd_mode': 'adam',
                  'add_zeros_quota': 1.0,
                  'l1_reg': 0.01,
                  'l2_reg': 0.001,
                  'topK': 100,
                  'use_dropout': True,
                  'dropout_perc': 0.7,
                  'initialization_mode_D': 'random',
                  'positive_only_D': False,
                  'normalize_similarity': True}

recommender_fw = CFW_D_Similarity_Cython(URM_train, ICM, similarity_collaborative)
recommender_fw.fit(**fw_parameters,
                   evaluator_object=evaluator_validation_earlystopping,
                   stop_on_validation=True,
                   validation_every_n = 5,
                   lower_validations_allowed=10,
                   validation_metric="MAP")


result_dict, result_string = evaluator_test.evaluateRecommender(recommender_fw)
print("CFeCBF recommendation quality is: {}".format(result_string))
