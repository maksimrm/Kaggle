
import pandas as pd
import numpy as np
import random

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

from sklearn import tree, metrics
from sklearn import ensemble
from sklearn import neighbors
import xgboost as xgb
import random
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier

real_columns = ['Age', 'Fare', "number_of_cabin", "ticket"]
cat_columns = ["Embarked","SibSp", "Parch", 'Pclass','Sex', "Deck", "first_name", "second_name", "title"]


class xgb_keras:
    def __init__(self, n_est = 10, params = None, epoch = 20, lr = 0.01):
        self.n_est = n_est
        if params:
            self.layers = params["layers"]
            self.optimizer = params["optimizer"]
            self.metrics = params["metrics"]
            self.epoch = params["epoch"]
            self.lr = params["lr"]
            self.loss = params["loss"]
        else :
            
            self.layers = [[10, "sigmoid"], [1, "sigmoid"]]
            self.optimizer = 'rmsprop'
            self.metrics = ['mae', 'mse']
            self.loss = 'mse'
            self.epoch = epoch
            self.lr = 0.01
        self.models = [self.make_model() for _ in range(self.n_est)]
    
    def make_model(self):
        
        model = keras.Sequential([
                keras.layers.Dense(layer[0], activation = layer[1]) 
                for layer in self.layers
            ])
        
        opt = keras.optimizers.RMSprop(lr = self.lr)
        model.compile(loss=self.loss,
                optimizer=opt,
                metrics=self.metrics)
        return model
        
    def fit(self, X, y):
        pred = y.copy()
        pred = np.array(pred, dtype = float)
        self.models[0].fit(X, y, epochs=self.epoch)
        pred = (self.models[0].predict(X)).reshape(X.shape[0]) - pred
        for i in range(1,self.n_est):
            self.models[i].fit(X, -pred, epochs=self.epoch)
            pred += (self.models[i].predict(X)).reshape(X.shape[0])
        
    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for i in range(self.n_est):
            pred += (self.models[i].predict(X)).reshape(X.shape[0])
            
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return np.array(pred, dtype = int)
    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])
        for i in range(self.n_est):
            pred += (self.models[i].predict(X)).reshape(X.shape[0])
        return np.array(pred, dtype = int)


params_catboost = [
    {"iterations" : 1200,
        'bagging_temperature': 8.45055827317225,
   'depth': 4,
   'l2_leaf_reg': 48.2436874528427,
   'learning_rate': 0.01420252223624052,
   'max_ctr_complexity': 14,
   'model_size_reg': 0.11898522787314908,
   'random_seed': 4228.0,
   'random_strength': 9.37100876430796},



{'bagging_temperature': 9.670298390136766,
   'depth': 4,
   'iterations': 800,
   'l2_leaf_reg': 71.48188455144279,
   'learning_rate': 0.03491666834740382,
   'max_ctr_complexity': 6,
   'model_size_reg': 9.762981803214656,
   'random_strength': 0.07224024949385273},


    
 {'bagging_temperature': 9.61328418703806,
   'depth': 4,
   'iterations': 2700,
   'l2_leaf_reg': 0.5910213888215504,
   'learning_rate': 0.001394419032445593,
   'max_ctr_complexity': 5,
   'model_size_reg': 5.031952491438889,
   'random_strength': 0.2587322665174468},
    
    
{'bagging_temperature': 9.780791797560955,
   'depth': 4,
   'iterations': 801,
   'l2_leaf_reg': 99.78797918719273,
   'learning_rate': 0.016712102039935662,
   'max_ctr_complexity': 7,
   'model_size_reg': 8.953575391415118,
   'random_strength': 6.031314217877166},
    
 {'bagging_temperature': 0.576199674896426,
   'depth': 4,
   'iterations': 1518,
   'l2_leaf_reg': 0.40461232088245114,
   'learning_rate': 0.004132045878305208,
   'max_ctr_complexity': 7,
   'model_size_reg': 0.18443448881373273,
   'random_strength': 9.559926698716335},
    
 {'bagging_temperature': 2.2834589965948426,
   'depth': 4,
   'iterations': 800,
   'l2_leaf_reg': 5.5489130567391545,
   'learning_rate': 0.016589013348662508,
   'max_ctr_complexity': 4,
   'model_size_reg': 0.7354861485756213,
   'random_strength': 2.595331192378417},
    
    
 {'bagging_temperature': 0.6587611244407032,
   'depth': 4,
   'iterations': 120,
   'l2_leaf_reg': 0.4162267739371724,
   'learning_rate': 0.04885607038603426,
   'max_ctr_complexity': 6,
   'model_size_reg': 0.4132388508571041,
   'random_strength': 9.39671242089991},
    
{'bagging_temperature': 3.5160293054817138,
   'depth': 4,
   'iterations': 250,
   'l2_leaf_reg': 5.2375693076671626,
   'learning_rate': 0.020639007895187373,
   'max_ctr_complexity': 4,
   'model_size_reg': 1.5625090612857475,
   'random_strength': 9.976419813532347},
    
    
    
{'bagging_temperature': 1.8135304834257837,
   'depth': 4,
   'iterations': 840,
   'l2_leaf_reg': 75.97812547264955,
   'learning_rate': 0.014818250003392812,
   'max_ctr_complexity': 4,
   'model_size_reg': 0.14866557227404564,
   'random_strength': 9.904832257732568},


]




class ensemble_for_titanic:
    def __init__(self, n_trees = 10, n_neural = 11, n_xgb = 10, n_neigh = 0, n_grad_boost = 0, n_catboost = 10):
        self.n_trees = n_trees
        self.n_neural = n_neural
        self.n_xgb = n_xgb
        self.n_neigh = n_neigh
        self.n_grad_boost = n_grad_boost
        self.n_catboost = n_catboost
        self.params_tree = {
            "n_est" : [50, 250],
            "max_depth" : [5, 25],
            "size" : [1, 1],
            "f":[1, 1]
        }
        
        self.params_net = {
            "epoches" : [10, 40],
            "size" : [0.9, 1],
            "f":[0.9, 1],
            "n_est" : [10, 30],
            "lr" :[0.008, 0.012]
        }
        self.params_xgb = {
            "lr" : [0.85, 0.95],
            "n_est" : [50, 250],
            "max_depth" : [5, 25],
            "size" : [1, 1],
            "f":[1, 1]
        }
        
        self.params_neigh = {
            "k" : [1, 10],
            "size" : [0.8, 1],
            "f":[0.8, 1]
        }
        
        self.params_grad_boost = {
            "lr" : [0.2, 0.4],
            "n_est" : [100, 250],
            "size" : [1, 1],
            "f":[1, 1]
        }
        
        
        
        
    
    def get_param(self, params, tp = None):
        
        res = params[0] + random.random()*(params[1] - params[0])
        if tp:
            res = int(res)
        return res
    
    def get_idx(self,params, size):
        idx = np.linspace(0, size - 1, size, dtype = int)
        random.shuffle(idx)
        bound = self.get_param(params)
        idx = idx[:int(bound*size)]
        return idx
    
    def fit(self, X, y, X_cat):
        self.models = []
        self.idx = []
        anses_for_train = pd.DataFrame()
        probs_for_train = pd.DataFrame()
        for i in range(self.n_trees):
            
            n_est = self.get_param(self.params_tree["n_est"] , tp = "int")
            depth = self.get_param(self.params_tree["max_depth"] , tp = "int")
                        
            idx = self.get_idx(self.params_tree["size"], X.shape[0])           
            idx2 = self.get_idx(self.params_tree["f"], X.shape[1])
            self.idx.append(idx2)
                      
            rf_classifier_low_depth = ensemble.RandomForestClassifier(n_estimators = n_est, max_depth = depth)
           
            rf_classifier_low_depth.fit(X[idx,:][:, idx2], y[idx])
            self.models.append(rf_classifier_low_depth)
            anses_for_train[f"{i} Random Forest"] = rf_classifier_low_depth.predict(X[:, idx2])
            probs_for_train[f"{i} Random Forest"] = rf_classifier_low_depth.predict_proba(X[:, idx2])[:,1]


            
        
        for i in range(self.n_neural):
            
            epoch = self.get_param(self.params_net["epoches"] , tp = "int")
            n_est = self.get_param(self.params_net["n_est"] , tp = "int")
            lr = self.get_param(self.params_net["lr"])
            
            model = xgb_keras(n_est = n_est, epoch = epoch, lr = lr)
            
            idx = self.get_idx(self.params_net["size"], X.shape[0])           
            idx2 = self.get_idx(self.params_net["f"], X.shape[1])
            self.idx.append(idx2)
            
            model.fit(X[idx,:][:, idx2], y[idx])
            self.models.append(model)
            anses_for_train[f"{i} neural net"] = model.predict(X[:, idx2])
            probs_for_train[f"{i} neural net"] = model.predict_proba(X[:, idx2])
            
        for i in range(self.n_xgb):
            lr = self.get_param(self.params_xgb["lr"])
            n_est = self.get_param(self.params_xgb["n_est"] , tp = "int")
            depth = self.get_param(self.params_xgb["max_depth"] , tp = "int")
            
            idx = self.get_idx(self.params_xgb["size"], X.shape[0])           
            idx2 = self.get_idx(self.params_xgb["f"], X.shape[1])
            self.idx.append(idx2)
                
            model = xgb.XGBClassifier(learning_rate=lr, max_depth=depth, n_estimators=n_est, min_child_weight=3)
            model.fit(X[idx,:][:, idx2], y[idx])
            self.models.append(model)
            
            anses_for_train[f"{i} xgb"] = model.predict(X[:, idx2])
            probs_for_train[f"{i} xgb"] = model.predict_proba(X[:, idx2])[:,1]
        
        for i in range(self.n_neigh):
            
            idx = self.get_idx(self.params_neigh["size"], X.shape[0])           
            idx2 = self.get_idx(self.params_neigh["f"], X.shape[1])
            self.idx.append(idx2)
            k = self.get_param(self.params_neigh["k"] , tp = "int")            
            model = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
            model.fit(X[idx,:][:, idx2], y[idx])
            self.models.append(model)
            
            anses_for_train[f"{i} neigb"] = model.predict(X[:, idx2])
            probs_for_train[f"{i} neigh"] = model.predict_proba(X[:, idx2])[:,1]
            
            
        for i in range(self.n_grad_boost):
            lr = self.get_param(self.params_grad_boost["lr"])
            n_est = self.get_param(self.params_grad_boost["n_est"] , tp = "int")
            
            idx = self.get_idx(self.params_grad_boost["size"], X.shape[0])           
            idx2 = self.get_idx(self.params_grad_boost["f"], X.shape[1])
            self.idx.append(idx2)
            
            model = GradientBoostingClassifier(learning_rate=lr, n_estimators=n_est)
            
            model.fit(X[idx,:][:, idx2], y[idx])
            self.models.append(model)
            
            anses_for_train[f"{i} grad boost"] = model.predict(X[:, idx2])
            probs_for_train[f"{i} grad boost"] = model.predict_proba(X[:, idx2])[:,1]
            
        for i in range(self.n_catboost):
            
            param = params_catboost[i % len(params_catboost)]
            param["random_seed"] = 422
            param["class_weights"] = [1, 0.8]
              
            param['iterations'] = int(param['iterations'])
            param['depth'] = int(param['depth'])
            param['max_ctr_complexity'] = int(param['max_ctr_complexity'])
            
            model = CatBoostClassifier(**param)
            model.fit(X_cat[cat_columns + real_columns], y, 
                  cat_features=cat_columns
            )
            self.models.append(model)
            
            anses_for_train[f"{i} cat boost"] = model.predict(X_cat[cat_columns + real_columns])
            probs_for_train[f"{i} cat boost"] = model.predict_proba(X_cat[cat_columns + real_columns])[:,1]
            
            
        return anses_for_train, probs_for_train
            
            
    def predict(self, X, X_cat):
        prediction = []
        anses_for_test = pd.DataFrame()
        probs_for_test = pd.DataFrame()
        for i in range(self.n_trees):
            X_predicted = self.models[i].predict(X[:,self.idx[i]])
            probs_for_test[f"{i} Random Forest"] = self.models[i].predict_proba(X[:,self.idx[i]])[:, 1]
            anses_for_test[f"{i} Random Forest"]= X_predicted
    
        for i in range(self.n_neural):
            ans = self.models[i + self.n_trees].predict(X[:,self.idx[i+self.n_trees]])
            probs_for_test[f"{i} neural net"] = self.models[i + self.n_trees].predict_proba(X[:,self.idx[i+self.n_trees]])
            anses_for_test[f"{i} neural net"]= ans
            
        for i in range(self.n_xgb):
            
            ans = self.models[i + self.n_trees + self.n_neural].predict(X[:,self.idx[i+self.n_trees+self.n_neural]])
            probs_for_test[f"{i} xgb"] =self.models[i + self.n_trees + self.n_neural].predict_proba(X[:,self.idx[i+self.n_trees+self.n_neural]])[:, 1]
            anses_for_test[f"{i} xgb"]= ans
            
        for i in range(self.n_neigh):
            
            ans = self.models[i + self.n_trees + self.n_neural + self.n_xgb].predict(X[:,self.idx[i+self.n_trees+self.n_neural+self.n_xgb]])
            probs_for_test[f"{i} neigb"] =self.models[i + self.n_trees + self.n_neural + self.n_xgb].predict_proba(X[:,self.idx[i+self.n_trees+self.n_neural+self.n_xgb]])[:, 1]
            anses_for_test[f"{i} neigb"]= ans
            
        for i in range(self.n_grad_boost):
            
            ans = self.models[i + self.n_trees + self.n_neural + self.n_xgb + self.n_neigh].predict(X[:,self.idx[i+self.n_trees+self.n_neural+self.n_xgb + self.n_neigh]])
            probs_for_test[f"{i} grad boost"] =self.models[i + self.n_trees + self.n_neural + self.n_xgb + self.n_neigh].predict_proba(X[:,self.idx[i+self.n_trees+self.n_neural+self.n_xgb + self.n_neigh]])[:, 1]
            anses_for_test[f"{i} grad boost"]= ans
            
            
        for i in range(self.n_catboost):
            
            ans = self.models[i + self.n_trees + self.n_neural + self.n_xgb + self.n_neigh  + self.n_grad_boost].predict(X_cat)
            probs_for_test[f"{i} cat boost"] = self.models[i + self.n_trees + self.n_neural + self.n_xgb + self.n_neigh  + self.n_grad_boost].predict_proba(X_cat)[:, 1]
            anses_for_test[f"{i} cat boost"] = ans
            
        return anses_for_test, probs_for_test
    
    
    def make_prediction(self, prediction):
        final = np.zeros(len(prediction[0]))
        prediction = np.array(prediction)
        size = len(prediction[0])
        prediction = prediction.T
        for i in range(size):
            if (prediction[i].mean() > 0.5):
                final[i] = 1
            
            
        
        return final