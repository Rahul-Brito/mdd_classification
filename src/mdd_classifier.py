import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.pyplot import subplots, cm

import os
from glob import glob
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from b2aiprep.prepare.dataset import BIDSDataset as bids
from b2aiprep.prepare.dataset import VBAIDataset as bext

from senselab.audio.data_structures.audio import Audio

import matplotlib.pyplot as plt

import IPython.display as Ipd
import pdb

import torch

dataset = bids('../bridge2ai-voice-corpus-3/corpus-3_bids/')



def load_features(feat_files, feat_type):
    
    features_list= []

    
    for tup, paths in feat_files.items():
        dict_list = []
        for p in paths:
            f = torch.load(p) 
            if feat_type == 'opensmile':
                f['task'] = str(p).split('task-')[1].split(f'_{feat_type}')[0]
                dict_list.append(f)
            elif feat_type == 'melfilterbank':
                mean_feat = pd.DataFrame(f['mel_filter_bank'].mean(axis=1)).T
                mean_feat['task'] = str(p).split('task-')[1].split(f'_{feat_type}')[0]
                dict_list.append(mean_feat)
            elif feat_type == 'specgram':
                mean_feat = pd.DataFrame(f['spectrogram'].mean(axis=1)).T
                mean_feat['task'] = str(p).split('task-')[1].split(f'_{feat_type}')[0]
                dict_list.append(mean_feat)
            elif feat_type == 'mfcc':
                mean_feat = pd.DataFrame(f['mfcc'].mean(axis=1)).T
                mean_feat['task'] = str(p).split('task-')[1].split(f'_{feat_type}')[0]
                dict_list.append(mean_feat)
            elif feat_type == 'speaker_embedding':
                mean_feat = pd.DataFrame(f).T
                mean_feat['task'] = str(p).split('task-')[1].split(f'_{feat_type}')[0]
                dict_list.append(mean_feat)  

        if feat_type == 'opensmile':
            df = pd.DataFrame.from_dict(dict_list, orient='columns')
        elif feat_type == 'melfilterbank' or 'specgram' or 'mfcc' or 'speaker_embedding':
            df = pd.concat(dict_list)
        df['record_id'] = [tup[0]] * df.shape[0]
        df['session_id'] = [tup[1]] * df.shape[0]
        features_list.append(df)
    features = pd.concat(features_list).reset_index(drop=True)
    
    return features




def feature_selection(features, scale = True, variance_cutoff = 0.85, method = 'pca'):
    
    #pdb.set_trace()
    if scale:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.drop(columns=['record_id','session_id','task']))
        #features_scaled = scaler.fit_transform(features)

    if method == 'pca':
        pca = PCA()
        pca.fit(features_scaled)

        reduced = pca.fit_transform(features_scaled)
        reduced = pd.DataFrame(reduced[:,np.cumsum(pca.explained_variance_ratio_) < variance_cutoff])

        reduced[['record_id','session_id','task']] = features[['record_id','session_id','task']]
    else:
        reduced = pd.DataFrame(features_scaled)
        reduced[['record_id','session_id','task']] = features[['record_id','session_id','task']]
    
    return reduced



def stratified_train_test_split(X, y, groups, split):
    
    #train/test split. Hold out test set for after k-fold CV training
    #X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, test_size=0.2, random_state=0)
    #using stratified to ensure no participant IDs are in both train and test
    #use 5-fold to mimick 80/20 train/test split
    
    if split:
        nested_sgk = StratifiedGroupKFold(n_splits=5) #80/20 split
        train_idx, test_idx = [(tr, tt) for tr, tt in nested_sgk.split(X, y, groups=groups)][0] #grab indecies from data for train and test

        X_train_raw = X[train_idx, :]
        y_train = y[train_idx]
        groups_train = groups[train_idx]

        X_test_raw = X[test_idx, :]
        y_test = y[test_idx]
        groups_test = groups[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw) 
        X_test = scaler.transform(X_test_raw)


        ### drop columns with zero variance here

        #test to confirm proper group splitting
        if [g for g in groups_test if g in groups_train]:
            raise ValueError('subjects from train set are in test test')
    else:
        X_train = X
        y_train = y
        groups_train = groups
        X_test = 0
        y_test = 0
        groups_test = 0

#     stratified_train_test_dict = {'X_train': X_train,
#                                   'y_train': y_train,
#                                   'groups_train': groups_train,
#                                   'X_test': X_test,
#                                   'y_test': y_test,
#                                   'groups_test': groups_test}
    
    return X_train, y_train, groups_train, X_test, y_test, groups_test
    #return stratified_train_test_dict

    
    
    
    
def calculate_SVM_scores(model, X_train, y_train, groups_train, X_test, y_test, groups_test, scoring, split):

#     X_train = stratified_train_test_dict['X_train']
#     y_train = stratified_train_test_dict['y_train']
#     groups_train = stratified_train_test_dict['groups_train']
    
#     X_test = stratified_train_test_dict['X_test']
#     y_test = stratified_train_test_dict['y_test']
#     groups_test = stratified_train_test_dict['groups_test']
    
    # SVM grid search
    #kernel = 'linear'
    
        #kernel = 'rbf'
    if type(model) == SVC:
        #if kernel == 'rbf':
        param_grid_rad = { 
            'C': np.logspace(-1, 2, 4),
            'gamma': np.logspace(-3, 1, 5)}    
        #elif kernel == 'linear':
        #    param_grid_rad = { 
        #        'C': np.logspace(-1, 2, 4)}
    elif type(model) == LogisticRegression:
         param_grid_rad = { 
            'C': np.logspace(-1, 2, 4)}
    elif type(model) == RandomForestClassifier:
        param_grid_rad = {
            'n_estimators': [25, 50, 100],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [3, 6, 9, 12, None],
        }

        
    #tried to search kernel space too
    # param_grid_rad = {
    #     'kernel':['linear','rbf'],  
    #     'C': np.logspace(-1, 2, 4),
    #     'gamma': np.logspace(-3, 1, 5)}    

    
        #weighted accuracy
    #search over number of PCs to try
    #CLAC data jim glass
    
    k = 10                                             # 10-fold CV for within-train-set hyperparameter/parameter training
    sgkf = StratifiedGroupKFold(n_splits=k)  
    
    search = GridSearchCV(estimator = model, param_grid = param_grid_rad, scoring = scoring, cv = sgkf) #verbose=3)
    tuned_svm = search.fit(X_train, y_train, groups=groups_train)

    if split:
        scores = tuned_svm.score(X_test, y_test)
    else:
        scores = tuned_svm.best_score_

    output = {'scores': scores,
              'X_train': X_train,
              'y_train': y_train,
              'groups_train': groups_train,
              'X_test': X_test,
              'y_test': y_test,
              'groups_test': groups_test}

    return output



def generate_X_y_groups(full_df, X_raw):
    X = X_raw.to_numpy() #normalized to PCA but added back in gender and age which need normalization.
    y = full_df.mdd.to_numpy()
    groups = full_df.record_id.to_numpy()
    
    return X, y, groups





def fit_by_feature_category(model, feature_files, task, feature_categories, diagnosis_df, scoring, split):
    all_features_dict = {}
    per_feature_type_scores = {}
    
    all_stratified_train_test_dict = {}

    for feat_type in feature_categories:
        #load features for specific task
        feat_files = {tup: [p for p in paths if task in str(p) if feat_type in str(p)] for tup, paths in feature_files.items()}
        feat_files = {tup:paths for tup, paths in feat_files.items() if paths} #drop any empty path lists (e.g. for a bad session)

        features = load_features(feat_files, feat_type) #load features from the feature files
        
        #if pca:
        #    reduced = feature_selection(features) #select features
        #else:
        #    reduced = feature_selection(features, method='skip')

        all_features_dict[feat_type] = features

        
        
        ##Data prep
        #drop unused columns (keeping age and gender with the features), convert to numpy

        #reduced = feature_selection(all_features_df) #select features

        #full_df = prepare_acoustic_fulldf(diagnosis_df, reduced)
        full_df = prepare_acoustic_fulldf(diagnosis_df, features)
        
        
        reduced = feature_selection(full_df.drop(columns= ['mdd','phq9','gad7'])) #select features
        X_raw = reduced.drop(columns=['record_id','session_id','task'])
        
        #full_df = reduced

        ##Data prep
        #drop unused columns (keeping age and gender with the features), convert to numpy
        #X_raw = full_df.drop(columns= ['record_id','session_id','mdd','phq9','gad7','task'])
        X, y, groups = generate_X_y_groups(full_df, X_raw)
        X_train, y_train, groups_train, X_test, y_test, groups_test = stratified_train_test_split(X, y, groups, split)

        if split:
            scores = calculate_SVM_scores(model, X_train, y_train, groups_train, X_test, y_test, groups_test, scoring, split)
        else:
            scores = calculate_SVM_scores(model, X_train, y_train, groups_train, X_train, y_train, groups_train, scoring, split)

        per_feature_type_scores[feat_type] = scores['scores']
        
        #all_stratified_train_test_dict[feat_type] = stratified_train_test_dict
    return per_feature_type_scores, all_features_dict#, all_stratified_train_test_dict



def prepare_acoustic_fulldf(diagnosis_df, reduced):
    full_df = diagnosis_df.merge(reduced, how='inner', on=['record_id','session_id'])
    full_df.columns = full_df.columns.astype(str)
    full_df.gender = full_df.gender.map({'Female gender identity':0, 'Male gender identity':1, 'Non-binary or genderqueer gender identity': 2})
    
    return full_df





def run_pipeline(model, feature_files, task, feature_categories, diagnosis_df, scoring, split):
    
    ##### Each feature type
    # with MFCC and spectrograms meaned per channel   
    output, all_features_dict = fit_by_feature_category(model, feature_files, task, feature_categories, diagnosis_df, scoring, split)

    
    #### all acoustic features
    tmp_full_df = pd.concat(all_features_dict.values(), axis=1)#.drop(columns=['record_id', 'session_id', 'task'])

    tmp = pd.concat(all_features_dict.values(), axis=1).drop(columns=['record_id', 'session_id', 'task'])
    all_features_df = pd.concat([tmp_full_df[['record_id', 'session_id', 'task']].loc[:,~tmp_full_df[['record_id', 'session_id', 'task']].columns.duplicated()].copy(), tmp], axis=1)
    all_features_df.columns = all_features_df.columns.astype(str)
    
    ##Data prep
    #drop unused columns (keeping age and gender with the features), convert to numpy
    
    #reduced = feature_selection(all_features_df) #select features

    #full_df = prepare_acoustic_fulldf(diagnosis_df, reduced)
    full_df = prepare_acoustic_fulldf(diagnosis_df, all_features_df)
    
    reduced = feature_selection(full_df.drop(columns= ['mdd','phq9','gad7'])) #select features
    X_raw = reduced.drop(columns=['record_id','session_id','task'])
    
    #X_raw = full_df.drop(columns= ['record_id','session_id','mdd','phq9','gad7','task'])
    X, y, groups = generate_X_y_groups(full_df, X_raw)
    X_train, y_train, groups_train, X_test, y_test, groups_test = stratified_train_test_split(X, y, groups, split)
    
    
    if split:
        all_acoustic_features_score = calculate_SVM_scores(model, X_train, y_train, groups_train, X_test, y_test, groups_test, scoring, split)['scores']
    else:
        all_acoustic_features_score = calculate_SVM_scores(model, X_train, y_train, groups_train, X_train, y_train, groups_train, scoring, split)['scores']




    ####PHQ9/GAD7 test

    # -----Model fit
    ##Data prep
    #drop unused columns (keeping age and gender with the features)
    X_raw = full_df[['phq9','gad7', 'age', 'gender']]
    #X_raw = full_df[['phq9','gad7']]

    ##Data prep
    #drop unused columns (keeping age and gender with the features), convert to numpy
    X, y, groups = generate_X_y_groups(full_df, X_raw)
    X_train, y_train, groups_train, X_test, y_test, groups_test = stratified_train_test_split(X, y, groups, split)
    #phq9_gad7_score = calculate_SVM_scores(X_train, y_train, groups_train, X_test, y_test, groups_test, scoring)['scores']
    
    if split:
        phq9_gad7_score = calculate_SVM_scores(model, X_train, y_train, groups_train, X_test, y_test, groups_test, scoring, split)['scores']
    else:
        phq9_gad7_score = calculate_SVM_scores(model, X_train, y_train, groups_train, X_train, y_train, groups_train, scoring, split)['scores']
        
    
    
    
    
    ####Acoustic features + PHQ9/GAD7 test

    # -----Model fit
    ##Data prep
    #drop unused columns (keeping age and gender with the features)

    reduced = feature_selection(full_df.drop(columns= ['mdd']))
    X_raw = reduced.drop(columns=['record_id','session_id','task'])
    #X_raw = full_df[['phq9','gad7']]

    ##Data prep
    #drop unused columns (keeping age and gender with the features), convert to numpy
    X, y, groups = generate_X_y_groups(full_df, X_raw)
    X_train, y_train, groups_train, X_test, y_test, groups_test = stratified_train_test_split(X, y, groups, split)
    #phq9_gad7_score = calculate_SVM_scores(X_train, y_train, groups_train, X_test, y_test, groups_test, scoring)['scores']
    
    if split:
        acc_quest_score = calculate_SVM_scores(model, X_train, y_train, groups_train, X_test, y_test, groups_test, scoring, split)['scores']
    else:
        acc_quest_score = calculate_SVM_scores(model, X_train, y_train, groups_train, X_train, y_train, groups_train, scoring, split)['scores']


    
    
    
    
    output['all_acoustic_features_score'] =  all_acoustic_features_score
    output['phq9_gad7_score'] =  phq9_gad7_score
    output['acc_quest_score'] = acc_quest_score
    
    #stratified_splits = {'per_feature_type': all_stratified_train_test_dict,
    #                     'all_acoustic_features': stratified_train_test_dict_all_feat,
    #                     'phq9_gad7': stratified_train_test_dict_mhscores}
    
    return output#, stratified_splits
