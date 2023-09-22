import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

import pickle
import sys

# from fairGAN import Medgan

# sys.path.insert(0, '../metrics')
# from combined import compute_metrics
# sys.path.insert(0, '../fairGAN_code')

# global params for new_view stats function
params = dict()
params["iterations"] = 2000
params["h_dim"] = 200
params["z_dim"] = 10
params["mb_size"] = 128
params["lambda_gp"] = 10
params["d_updates"] = 10
params['lambda'] = 0

def train_fairgan(datapath):

    #data = np.load(datapath, allow_pickle = True)
    inputDim = data.shape[1]-1
    inputNum = data.shape[0]
    tf.reset_default_graph()
    
    mg = Medgan(dataType='count',
                inputDim=inputDim,
                embeddingDim=128,
                randomDim=128,
                generatorDims=(128,128),
                discriminatorDims=(256,128),
                compressDims=(),
                decompressDims=(),
                bnDecay=0.99,
                l2scale=0.001)

    model_file = ''
    out_file = 'fair'
    batch_size = 32
    
    mg.train(dataPath=datapath,
             modelPath=model_file,
             outPath=out_file,
             pretrainEpochs=200,
             nEpochs=50,
             discriminatorTrainPeriod=2,
             generatorTrainPeriod=1,
             pretrainBatchSize=batch_size,
             batchSize=batch_size,
             # protected = [6],
             saveMaxKeep=0)
    tf.reset_default_graph()
    
    return mg.generateData(nSamples=inputNum,
                        modelFile='fair-49',
                        batchSize=batch_size,
                        outFile=out_file)

def train_model(data, bias_dict = {}, surrogate = False):
    dm = DataModule(data.values)
    data_tensor = dm.setup()

    #dm = SyntheticDataModule()
    #data_tensor = dm.setup()
    #activation_layer = nn.ReLU(inplace=True) #nn.LeakyReLU(0.2, inplace=True)

    # Causal GAN
    #%% Import functions

    params = dict()
    params["iterations"] = 2000
    params["h_dim"] = 200
    params["z_dim"] = 10
    params["mb_size"] = 128
    params["lambda_gp"] = 10
    params["d_updates"] = 10

    max_epochs = (10 + 1) * 25 
    number_of_gpus = 0

    # Remove all the education level edges.#5
    biased_list =[[1, 7],  [7, 10], 
                  [8, 10], 
                  [2, 13], [9, 5], [9, 10], 
                  [7, 8], 
                  [12, 3], # Removed edge between age and ethnicity.
                  [9, 4], 
                  [8, 3], 
                  [6, 15], # Remove this for training purposes. 
                  [7, 11], [7, 15], [13, 3], [13, 14], [10, 2], [2, 14], 
                  [5, 3], [7, 2], [9, 15], [8, 2], [14, 3], [14, 15], [4, 3], [8, 15], 
                  [13, 11], [9, 12], [8, 9],
                 [6,9] # This is the edge from ethnicity to employed
                 ]
    
    
    biased_list =[[1, 7],  [7, 10], 
                  [8, 10], 
                  [2, 13],  [9, 10], 
                  [7, 8], 
                  [12, 3], # Removed edge between age and ethnicity.
                  [9, 4], 
                  [8, 3], 
                  [6, 15], # Remove this for training purposes. 
                  [7, 11], [7, 15], [13, 3], [13, 14], [10, 2], [2, 14], [7, 2], [9, 15], [8, 2], [14, 3], [14, 15], [4, 3], [8, 15], 
                  [13, 11], [9, 12], [8, 9],
                 [6,9] # This is the edge from ethnicity to employed
                 ]
    
    
    # model initialisation and train
    model = causal_gan(dm, dag_seed = biased_list,
               h_dim=200,
               lr=1e-3,
               batch_size=64,
               lambda_privacy=0,
               lambda_gp=10,
               d_updates=10,
               causal=True,
               alpha=2,
               rho=2,
               weight_decay=1e-2,
               grad_dag_loss=False,
               l1_g=0,
               l1_W=1e-4,
               p_gen=-0.2,
               use_mask=True,
              )
    
    print(model.hparams)
    trainer = pl.Trainer(gpus=number_of_gpus, max_epochs=max_epochs, progress_bar_refresh_rate=1, profiler = False)
    model.set_val_data(data_tensor)

    print("Training")
    trainer.fit(model, dm)
    synth_data = model.gen_synthetic(data_tensor, gen_order = model.get_gen_order(), biased_edges = bias_dict, surrogate = surrogate).detach().numpy()
    print(synth_data.shape)
    
    return synth_data

def view_stats_new(method_list, input_data, orig_data = [], protected = '', skip_synth = False, protected_idx = -1, runs=10, bias_dict = {}, remove_protected = False, surrogate = False):

    summary = ''
    samples = 5000
    
    # Note that for gender 0 is female, and 1 is male
    
    if not remove_protected:
        x_pos = orig_data[orig_data[p_attr] == 0].drop(['approved'], axis = 1)[:samples]
        y_pos = orig_data[orig_data[p_attr] == 0]['approved'][:samples]
        x_neg = orig_data[orig_data[p_attr] == 1].drop(['approved'], axis = 1)[:samples]
        y_neg = orig_data[orig_data[p_attr] == 1]['approved'][:samples]
        print(len(x_pos), len(y_pos), len(x_neg), len(y_neg))
    else:
        input_data = input_data.drop([protected], axis = 1)
        x_pos = orig_data[orig_data[p_attr] == 0].drop(['approved', protected], axis = 1)[:samples]
        y_pos = orig_data[orig_data[p_attr] == 0]['approved'][:samples]
        x_neg = orig_data[orig_data[p_attr] == 1].drop(['approved', protected], axis = 1)[:samples]
        y_neg = orig_data[orig_data[p_attr] == 1]['approved'][:samples]
        print(len(x_pos), len(y_pos), len(x_neg), len(y_neg))
        
    X_unbiased = pd.concat([x_pos, x_neg],axis=0).copy()
    y_unbiased  = pd.concat([y_pos, y_neg],axis=0).copy()
    
    for method in method_list:
        
        params['gen_model_name'] = method.replace('-pr', '')
        
        if method == 'adsgan':
            params['lambda'] = 0
        else:
            params['lambda'] = 1
            
            
        if method == 'vae':
            params["iterations"] = 1000
        else:
            params["iterations"] = 2000
        err = []
        feat_importance = []
        recall_ratio = []
        
        mutual_info = []
        precision = []
        recall = []
        density =[]
        coverage = []
        roc = []
        
        for i in range(runs):
            
            if skip_synth:
                synth_data = input_data.values
            else:
                if method == 'fairgan':
                    # Need to swap 0 column with protected idx.
                    temp = input_data.copy()
                    popped = temp.pop('ethnicity')
                    temp.insert(0, 'ethnicity', popped)
                    
                    pickle.dump(temp.values, open( "adult.npy", "wb" ) )
                    synth_data, synth_data_z = train_fairgan('adult.npy')
            
                    # Have to swap columns back like so.... x[:,[2,1]] = x[:,[1,2]]
                    #synth_data[:,[0,6]] = synth_data[:,[6,0]]
                    
                    print("synth before:", synth_data.shape, synth_data_z.shape)
                    synth_data = np.insert(synth_data, 6, synth_data_z, axis=1)
                    print("synth after:", synth_data.shape)

                elif method == 'adsgan' or method == 'adsgan-pr':
                    synth_data = adsgan(input_data, params)
                elif method == 'gan' or method == 'wgan' or method == 'gan-pr' or method == 'wgan-pr':
                    synth_data = gan(input_data, params)
                elif method == 'vae':
                    synth_data = vae(input_data, params)
                else:
                    synth_data = train_model(input_data, bias_dict, surrogate = surrogate)

                                                     
            # This step is to ensure at least one sample there.
            pos_sample = input_data[input_data.approved == 0].iloc[0].values
            neg_sample = input_data[input_data.approved == 1].iloc[0].values
            synth_data = np.concatenate([synth_data, [pos_sample], [neg_sample]], axis = 0)        
            X = synth_data[:,:-1]

            #if remove_protected: 
            #    X = np.delete(synth_data, protected_idx, axis = 1)[:,:-1]
            
            y = np.round(synth_data[:, -1])

            mlp = MLPClassifier(random_state = i, max_iter = 100).fit(X, y)
            #mlp = LogisticRegression(random_state = i, max_iter = 100).fit(X, y)
                
            for X_unbiased, y_unbiased, _label in zip([x_pos, x_neg, pd.concat([x_pos, x_neg],axis=0).copy()], 
                                                      [y_pos, y_neg, pd.concat([y_pos, y_neg],axis=0).copy()],
                                                      ['pos', 'neg', 'both']):
                
                print("LOGGGING", len(X_unbiased), len(y_unbiased))
                if not remove_protected:
                    def compute_FTU(x):
                        x[p_attr] = 0
                        neg = mlp.predict(x)
                        x[p_attr] = 1
                        pos = mlp.predict(x)
                        return pos-neg
                    if _label == 'pos':
                        FTU = compute_FTU(x_pos)
                    elif _label == 'neg':
                        FTU = compute_FTU(x_neg)
                    else:
                        x_all = pd.concat([x_pos, x_neg],axis=0)
                        FTU = compute_FTU(x_all) 
                else:
                    FTU = 0 # by definition
                #print('FTU', FTU)
                pred_pos = mlp.predict(x_pos)
                pred_neg = mlp.predict(x_neg)
                if _label == 'pos':
                    DP = np.mean(pred_pos)
                elif _label =='neg':
                    DP = np.mean(pred_neg)
                else:
                    DP = np.mean(pred_pos)-np.mean(pred_neg)
                #print('DP', DP)

                CM = confusion_matrix(y_pos, mlp.predict(x_pos))
                TN = CM[0][0]
                FN = CM[1][0]
                TP = CM[1][1]
                FP = CM[0][1]

                tpr_pos = CM[1][1]/(CM[0][0]+CM[0][1])
                CM = confusion_matrix(y_neg, mlp.predict(x_neg))
                tpr_neg = CM[1][1]/(CM[1][1]+CM[0][1])

                roc.append(roc_auc_score(y_unbiased, mlp.predict_proba(X_unbiased)[:,1]))

                print(tpr_pos), print(tpr_neg)
                err.append(DP) #tpr_pos - tpr_neg)

                if True:
                    feat_importance.append(np.mean(FTU)) # BB 06/05 - just checking whether this leads to decent results.
                elif protected_idx >= 0:
                    feat_importance.append(mlp.coef_[0][protected_idx])

                else:
                    feat_importance.append(-1)

                print("Feature Importance = ", feat_importance)

                mutual_info.append(-1)


                if remove_protected:
                    results = compute_metrics(orig_data.drop([protected], axis = 1), synth_data,  which_metric = [['PRDC']], 
                                           wd_params = {},model = None,verbose = True)
                else:
                    #results = compute_metrics(orig_data.drop([protected], axis=1), np.delete(synth_data, protected_idx, 1),  which_metric = [['PRDC']], 
                    #                       wd_params = {},model = None,verbose = True)
                    
                    if _label == 'pos':
                        results = compute_metrics(pd.concat([X_unbiased, y_unbiased],axis=1), synth_data[synth_data[:,protected_idx].astype(bool)],  which_metric = [['PRDC']], 
                                           wd_params = {},model = None,verbose = True)
                    elif _label == 'neg':
                        print("Computing neg")
                        print( synth_data[1-synth_data[:,protected_idx].astype(bool)].shape)
                        results = compute_metrics(pd.concat([X_unbiased, y_unbiased],axis=1), synth_data[1-synth_data[:,protected_idx].astype(bool)],  which_metric = [['PRDC']], 
                                           wd_params = {},model = None,verbose = True)
                    else:
                        results = compute_metrics(pd.concat([X_unbiased, y_unbiased],axis=1), synth_data,  which_metric = [['PRDC']], 
                                           wd_params = {},model = None,verbose = True)
                precision.append(results['precision'])
                recall.append(results['recall'])
                density.append(results['density'])
                coverage.append(results['coverage'])

                # Writing to file
                with open("plots_surrogate_confounder_" + _label + '.csv', "a") as log:
                    # Writing data to a file
                    log.write(method +"," + str(bias) + "," + str(results['precision']) + ',' + str(results['recall']) + ',' + str(results['density']) + \
                              ',' + str(results['coverage']) + ',' + str(err[-1]) + ',' + str(feat_importance[-1]) + ',' + str(roc[-1]) + '\n')

        if skip_synth:
            print("no_synth", round(np.mean(err),3), round(np.std(err),3), 
              round(np.mean(feat_importance),3), round(np.std(feat_importance),3),
              round(np.mean(mutual_info),3), round(np.std(mutual_info),3)
             )
            break
        else:
            print(method, round(np.mean(err),3), round(np.std(err),3), 
              round(np.mean(feat_importance),3), round(np.std(feat_importance),3),
              round(np.mean(mutual_info),3), round(np.std(mutual_info),3)
             )

        #summary+= method + '&$' + str(round(np.mean(precision),3)) + '\pm' + str(round(np.std(precision),3)) + '$&$' + str(round(np.mean(recall),3)) + '\pm' + str(round(np.std(recall),3)) + \
        #         '$&$' + str(round(np.mean(density),3)) + '\pm' + str(round(np.std(density),3)) + '$&$' + str(round(np.mean(coverage),3)) + '\pm' + str(round(np.std(coverage),3)) + \
        #         '$&$' + str(round(np.mean(err),3)) + '\pm' + str(round(np.std(err),3)) + '$&$' + str(round(np.mean(mutual_info),3)) + '\pm' + str(round(np.std(mutual_info),3)) + '$\\\\\n'
        
        
        summary+= method + '&$' + str(round(np.mean(precision),3)) + '\pm' + str(round(np.std(precision),3)) + '$&$' + str(round(np.mean(recall),3)) + '\pm' + str(round(np.std(recall),3)) + \
                 '$&$' + str(round(np.mean(err),3)) + '\pm' + str(round(np.std(err),3)) + '$&$' + str(round(np.mean(feat_importance),3)) + '\pm' + str(round(np.std(feat_importance),3)) + \
                 '$&$' + str(round(np.mean(roc),3)) + '\pm' + str(round(np.std(roc),3)) +'$\\\\\n'
        
        print(summary)


def label_fix(label):
    if label==' <=50K':
        return 0
    else:
        return 1

def process_data(datapath, args):
    data = pd.read_csv(datapath, column_names=args.column_names)

    data['income_bracket'] = data['income_bracket'].apply(label_fix)

    gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
    relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
    education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
    workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
    native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    feat_cols = [gender, occupation, marital_status, relationship, education, workclass, native_country,
                age, education_num, capital_gain, capital_loss, hours_per_week]

    print("Base MI RACE;INCOME = ", mutual_info_classif(np.expand_dims(data['income_bracket'], -1), data['race'], discrete_features = True))
    print("Base MI Gender;INCOME = ", mutual_info_classif(np.expand_dims(data['income_bracket'], -1), data['gender'], discrete_features = True))

    for feat in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']:
        data[feat] = preprocessing.LabelEncoder().fit_transform(data[feat])

    data.drop(['capital_gain', 'capital_loss', 'education'], axis = 1, inplace = True)
    #np.save('adult', data.values, allow_pickle = True)

    scaler = MinMaxScaler()
    scaler.fit(data)
    data[data.columns] = scaler.fit_transform(data)
    pickle.dump(data.values, open( "adult.npy", "wb" ) )

    return data

def bias_data(datapath, column_names):

    data = pd.read_csv(datapath, header=None,  names=column_names)
    data.reset_index(drop=True, inplace=True) 
    data = data.dropna(how = 'all')
    data = data[data.age != '?']
    data.reset_index(drop=True, inplace = True)

    for feat in ['male', 'married','bankcustomer', 'educationlevel', 'ethnicity','priordefault', 'employed', 'driverslicense', 'citizen', 'zip', 'approved']:
        data[feat] = preprocessing.LabelEncoder().fit_transform(data[feat])

    data['age'] = pd.to_numeric(data['age'],errors='coerce')

    # binarize the protected variable
    data.loc[data['ethnicity'] <= 4, 'ethnicity'] = 0
    data.loc[data['ethnicity'] > 4, 'ethnicity']= 1
    data.loc[data['ethnicity'] == 1 , 'employed'] =  1

    biased_data = data.copy()
    biased_data.loc[biased_data['ethnicity'] == 1, 'approved'] = 0

    thresh = 0.8

    scaler = MinMaxScaler()
    scaler.fit(data)
    data[data.columns] = scaler.fit_transform(data)
    biased_data[biased_data.columns] = scaler.transform(biased_data)

    return data, biased_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="data/adult_fairgan.data")
    parser.add_argument("--credit", type=bool, default=False)
    parser.add_argument("--adult", type=bool, default=False)
    
    datapath_credit = "../data/crx.data"
    datapath_adult = "../data/adult.data"

    args = parser.parse_args()
    
    if args.credit:
        column_names_credit = ['male', 'age', 'debt', 'married', 'bankcustomer', 'educationlevel', 'ethnicity', 'yearsemployed',
                    'priordefault', 'employed', 'creditscore', 'driverslicense', 'citizen', 'zip', 'income', 'approved']
        protected_idx = 6
        p_attr = 'ethnicity'

        data, biased_data = bias_data(datapath_credit, column_names)

        view_stats_new(['fairgan'], biased_data, protected = p_attr, remove_protected = False,
                orig_data = data ,protected_idx = protected_idx, bias_dict ={})

    # train model for adult dataset
    if args.adult:
        column_names_adult = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship',
                        'race','gender','capital_gain','capital_loss','hours_per-week','native_country','income_bracket']

        data_adult = process_data(datapath_adult, column_names_adult)

        train_fairgan()
        view_stats_new(['fairgan'], biased_data, protected = p_attr, remove_protected = False,
                orig_data = data ,protected_idx = protected_idx, bias_dict ={})

    