import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from table_evaluator import TableEvaluator
from ctgan import CTGAN

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import csv

import os.path

def load_adult() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    path_test =  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df_train = pd.read_csv(path, names=names, index_col=False)
    df_test = pd.read_csv(path_test, names=names, index_col=False)[1:]

    df = pd.concat([df_train, df_test])
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    df["label"].replace({'<=50K.': '<=50K', '>50K.': '>50K'}, inplace=True)

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    for row in replace:
        df = df.replace(row, range(len(row)))

    df = pd.DataFrame(MinMaxScaler().fit_transform(df),
                      index=df.index, columns=df.columns)

    return df


discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'label'
]

# Node order contains the order in which to generate the data, starting with the root nodes
node_order = [['race','age','sex','native-country'],['marital-status'],['education'],['occupation','hours-per-week','workclass','relationship'],['label']]
node_order_nl = ['race','age','sex','native-country','marital-status','education','occupation','hours-per-week','workclass','relationship','label']

# List of connections; key is receiving node
node_connections_normal = {'label':['occupation','race','hours-per-week','age','marital-status','education','sex','workclass','native-country','relatinship'],
                    'occupation':['race','age','sex','marital-status','education'],
                    'hours-per-week':['race','age','marital-status','native-country','education','sex'],
                    'workclass':['age','marital-status','sex','education','native-country'],
                    'relationship':['marital-status','education','age','sex','native-country'],
                    'education':['race','age','marital-status','sex','native-country'],
                    'marital-status':['race','age','sex','native-country']
                    }

'''
Connections are removed according to the privacy criterion
'''
node_connections_FTU = {'label':['occupation','race','hours-per-week','age','marital-status','education','workclass','native-country','relationship'],
                    'occupation':['race','age','sex','marital-status','education'],
                    'hours-per-week':['race','age','marital-status','native-country','education','sex'],
                    'workclass':['age','marital-status','sex','education','native-country'],
                    'relationship':['marital-status','education','age','sex','native-country'],
                    'education':['race','age','marital-status','sex','native-country'],
                    'marital-status':['race','age','sex','native-country']
                    }

node_connections_DP = {'label':['race','age','native-country'],
                    'occupation':['race','age','sex','marital-status','education'],
                    'hours-per-week':['race','age','marital-status','native-country','education','sex'],
                    'workclass':['age','marital-status','sex','education','native-country'],
                    'relationship':['marital-status','education','age','sex','native-country'],
                    'education':['race','age','marital-status','sex','native-country'],
                    'marital-status':['race','age','sex','native-country']
                    }

node_connections_CF = {'label':['occupation','race','hours-per-week','age','education','workclass','native-country',],
                    'occupation':['race','age','sex','marital-status','education'],
                    'hours-per-week':['race','age','marital-status','native-country','education','sex'],
                    'workclass':['age','marital-status','sex','education','native-country'],
                    'relationship':['marital-status','education','age','sex','native-country'],
                    'education':['race','age','marital-status','sex','native-country'],
                    'marital-status':['race','age','sex','native-country']
                    }


@ignore_warnings(category=ConvergenceWarning)
def generate_data(df, mode, load_model=False):
    ctgan = CTGAN(epochs=10)
    # How much more data the synthetic dataset should contain that the OG data (This is to ensure we can
    # take a sample that looks like the original data)
    factor = 50
    
    # Define the privacy measure
    if mode == 'FTU':
        node_connections = node_connections_FTU
    elif mode == 'DP':
        node_connections = node_connections_DP
    elif mode == 'CF':
        node_connections = node_connections_CF
    else:
        print('Mode is not correct!')
    
    model_name = 'CTGANrootnodes' + str(mode) + '.pkl'
    path = 'Models/' + model_name
    if os.path.isfile(path) and load_model:
        ctgan = ctgan.load(path)
        print('model succesfully loaded!')
    else:
        print('training model ...')
        # DF to fit the first model on
        start_df = df[['race','age','sex','native-country']]
        temp_discrete = ['race','age','sex','native-country']

        ctgan.fit(start_df, temp_discrete)
        ctgan.save('Models/'+model_name)
    
    synth_df = ctgan.sample(factor * len(df.index))
    
    # Iteratively generate the data
    for node in node_order_nl:
         # If the node has not been generated yet
        if node not in synth_df.columns:
            # Grab the old data
            empty_df = df[[node]]

            # Grab the attributes that need to be looked at when generating data
            if node in node_connections.keys():
                attributes = node_connections[node]
            else:
                attributes = []
                for n in node_order_nl:
                    attributes.append(n)
                    if n == node:
                        break
                            
            model_name = 'CTGAN' + str(node) + str(mode) + '.pkl'
            path = 'Models/' + model_name
            if os.path.isfile(path) and load_model:
                ctgan = ctgan.load(path)
                generated_data = ctgan.sample(len(synth_df.index))
                print(f'model for node {node} succesfully loaded!')
            else:
                print(f'Training model for node {node} ...')
                # Grab the attributes from the final df
                gen_df = synth_df.loc[:,synth_df.columns.isin(attributes)]

                # Add the old attribute to the current dataframe
                at = df[attributes]
                empty_df = empty_df.join(at)

                temp_discrete = []
                for d in discrete_columns:
                    if d in gen_df.columns:
                        temp_discrete.append(d)

                ctgan.fit(empty_df, temp_discrete)

                model_name = str(node) + str(mode)
                ctgan.save('Models/CTGAN' + model_name + '.pkl')
                generated_data = ctgan.sample(len(synth_df.index))
                
            # Add the generated data to the output
            for attribute in attributes + [node]:
                if attribute not in synth_df.columns:
                    synth_df[attribute] = generated_data[attribute].values
            #print('Finished node',node,'for',mode)
    # Finally, we have to manually add the label
    return synth_df

    def get_metrics(mode, df, synthetic):

    # Split the data into train, test
    traindf, testdf = train_test_split(df, test_size=0.3)
    
    X_train = traindf.loc[:, traindf.columns != 'label']
    y_train = traindf['label']
    X_test = testdf.loc[:, testdf.columns != 'label']
    y_test = testdf['label']

    clf_df = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                     learning_rate='constant', learning_rate_init=0.001).fit(X_train, y_train)
    '''
    SYNTHETIC DATASET
    '''
    # Make sure the data is representative of the original dataset
    synthetic_balanced_1 = synthetic[synthetic.label == 1].sample(22654)
    synthetic_balanced_0 = synthetic[synthetic.label == 0].sample(7508)
    synthetic_balanced = synthetic_balanced_1.append(synthetic_balanced_0)

    # Split the data into train,test
    X_syn = synthetic_balanced.loc[:, synthetic_balanced.columns != 'label']
    y_syn = synthetic_balanced['label']

    y_pred_syn = clf_df.predict(X_syn)

    synthetic_pos = synthetic.assign(sex=0)
    synthetic_neg = synthetic.assign(sex=1)
    
    x_pos_syn = synthetic_balanced[synthetic_balanced['sex'] == 0].drop(['label'], axis = 1)[:7508]
    x_neg_syn = synthetic_balanced[synthetic_balanced['sex'] == 1].drop(['label'], axis = 1)[:7508]
    
    pos = clf_df.predict(synthetic_pos.drop('label',axis=1))
    neg = clf_df.predict(synthetic_neg.drop('label',axis=1)) 

    pred_pos_syn = clf_df.predict(x_pos_syn)
    pred_neg_syn = clf_df.predict(x_neg_syn)
    
    FTU = np.abs(np.mean(pos-neg))
    DP = np.mean(pred_pos_syn)-np.mean(pred_neg_syn)
    prec_score = precision_score(y_syn, y_pred_syn, average='binary')
    rec_score = recall_score(y_syn, y_pred_syn, average='binary')
    auroc = roc_auc_score(y_syn, y_pred_syn)

    # Print the obtained statistics
    print('Statistics for dataset for mode:', mode)
    print('Precision:', prec_score)
    print('Recall:', rec_score)
    print('AUROC:', auroc)
    print('FTU:', FTU)
    print('DP:', DP)

    return {'precision': prec_score, 'recall': rec_score, 'auroc': auroc,'dp': DP, 'ftu': FTU}

def run_experiment_CTGAN(mode):
    # Generate the synthetic a data
    df = load_adult()
    synthetic = generate_data(df, mode)

    return get_metrics(mode, df, synthetic)

final_FTU = {'precision': [],
'recall': [],
'auroc': [],
'dp': [],
'ftu': []}

final_DP = {'precision': [],
'recall': [],
'auroc': [],
'dp': [],
'ftu': []}

final_CF = {'precision': [],
'recall': [],
'auroc': [],
'dp': [],
'ftu': []}

def multi_run_ctgan(mode, RUNS=10):

    for r in range(RUNS):
        print(f"Run {r}: ")
        res = run_experiment_CTGAN(mode)

        if mode =='FTU':
            final_FTU['precision'].append(res['precision'])
            final_FTU['recall'].append(res['recall'])
            final_FTU['auroc'].append(res['auroc'])
            final_FTU['dp'].append(res['dp'])
            final_FTU['ftu'].append(res['ftu'])

            print(final_FTU)

        if mode =='DP':
            final_DP['precision'].append(res['precision'])
            final_DP['recall'].append(res['recall'])
            final_DP['auroc'].append(res['auroc'])
            final_DP['dp'].append(res['dp'])
            final_DP['ftu'].append(res['ftu'])

            print(final_DP)
        if mode =='CF':
            final_CF['precision'].append(res['precision'])
            final_CF['recall'].append(res['recall'])
            final_CF['auroc'].append(res['auroc'])
            final_CF['dp'].append(res['dp'])
            final_CF['ftu'].append(res['ftu'])

            print(final_CF)

    if mode =='FTU':
        # Print the obtained statistics
        print('Statistics for dataset for mode:', mode)
        print('Precision:', np.mean(final_FTU['precision']), np.std(final_FTU['precision']))
        print('Recall:', np.mean(final_FTU['recall']), np.std(final_FTU['recall']))
        print('AUROC:', np.mean(final_FTU['auroc']), np.std(final_FTU['auroc']))
        print('FTU:', np.mean(final_FTU['ftu']), np.std(final_FTU['ftu']))
        print('DP:', np.mean(final_FTU['dp']), np.std(final_FTU['dp']))
    if mode =='DP':
        # Print the obtained statistics
        print('Statistics for dataset for mode:', mode)
        print('Precision:', np.mean(final_DP['precision']), np.std(final_DP['precision']))
        print('Recall:', np.mean(final_DP['recall']), np.std(final_DP['recall']))
        print('AUROC:', np.mean(final_DP['auroc']), np.std(final_DP['auroc']))
        print('FTU:', np.mean(final_DP['ftu']), np.std(final_DP['ftu']))
        print('DP:', np.mean(final_DP['dp']), np.std(final_DP['dp']))
    if mode =='CF':
        # Print the obtained statistics
        print('Statistics for dataset for mode:', mode)
        print('Precision:', np.mean(final_CF['precision']), np.std(final_CF['precision']))
        print('Recall:', np.mean(final_CF['recall']), np.std(final_CF['recall']))
        print('AUROC:', np.mean(final_CF['auroc']), np.std(final_CF['auroc']))
        print('FTU:', np.mean(final_CF['ftu']), np.std(final_CF['ftu']))
        print('DP:', np.mean(final_CF['dp']), np.std(final_CF['dp']))

multi_run_ctgan('FTU', RUNS=10)
multi_run_ctgan('DP', RUNS=10)
multi_run_ctgan('CF', RUNS=10)