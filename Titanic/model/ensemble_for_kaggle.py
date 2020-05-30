import random
from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.cluster import MeanShift


class ensemble:
    def __init__(self, models_dict, tp , params = None, pbounds = None, names_d = None):
        self.models = []
        self.tp = []
        self.models_names = models_dict.keys()
        self.ns = np.array(list(models_dict.values()))
        self.names = []
        for k, key in enumerate(models_dict.keys()):
            for i in range(models_dict[key]):
                self.tp.append(tp[key])
                if params and key in params.keys():
                    self.models.append(key(**params[key][i % len(params[key])]))
                elif pbounds and key in pbounds.keys():
                    param = {}
                    for p in pbounds[key].keys():
                        param[p] = self.get_param(pbounds[key][p])
                    self.models.append(key(**param))
                else:
                    self.models.append(key())
                if names_d and key in names_d.keys():
                    self.names.append(names_d[key] + " " + str(i))
                else:
                    self.names.append(str(k) + " : " + str(i))
                    
                    
    
    
    def add_model(self, model, n, tp,params = None, pbounds = None, names_d = None):
        for i in range(n):
            self.tp.append(tp)
                
            if params :
                self.models.append(model(**params[i % len(params[key])]))
            elif pbounds:
                param = {}
                for p in pbounds.keys():
                    param[p] = self.get_param(pbounds[p])
                self.models.append(model(**param))
            else:
                self.models.append(model())
             
                    
    def get_param(self, params):
        res = params[0] + random.random()*(params[1] - params[0])
        if (type(params[0]) == int) or (type(params[1]) == int):
            res = int(res)
        return res
    
    def get_intersection(self, fist, second):
        final = []
        for a in first:
            if a in second:
                final.append(a)
        return final
    
    def fit(self, X_num, y,X_cat = None, cat_columns = None, f_dict = None):
        self.f_dict = f_dict
        for i in tqdm(range(len(self.models))):
            k = 0
            while i < self.ns[:k].sum():
                k+= 1
                
            k -= 1
            if self.tp[i] == "cat":
                if  self.f_dict and self.models_names[k] in f_dict.keys():
                    self.models[i].fit(X_cat[f_dict[self.models_names[k]]], y, 
                                       cat_features=self.get_intersection(cat_columns, f_dict[self.models_names[k]]))
                else:
                    self.models[i].fit(X_cat, y, 
                                       cat_features=cat_columns)
            else:
                if  self.f_dict and self.models_names[k] in f_dict.keys():
                    self.models[i].fit(X_num[f_dict[self.models_names[k]]], y)
                else:
                    print(i, self.models[i])
                    print(X_num, y)
                    self.models[i].fit(X_num, y)
               
            
            
           
        
    def predict(self, X_num, X_cat = None):
        ans = pd.DataFrame()
        
        for i in tqdm(range(len(self.models))):
            k = 0
            while i < self.ns[:k].sum():
                k+= 1
                
            k -= 1
            
            if self.tp[i] == "cat":
                if  self.f_dict and self.models_names[k] in self.f_dict.keys():
                    ans[self.names[i]] = self.models[i].predict(X_cat[self.f_dict[self.models_names[k]]])
                else:
                    ans[self.names[i]] = self.models[i].predict(X_cat)
            else:
                if  self.f_dict and self.models_names[k] in self.f_dict.keys():
                    ans[self.names[i]] = self.models[i].predict(X_num[self.f_dict[self.models_names[k]]])
                else:
                    ans[self.names[i]] = self.models[i].predict(X_num)
        self.ans = ans
        return ans.mean(axis = 1)
    
    
    def predict_proba(self, X_num, X_cat = None, num_classes = 2):
        probs = []
        
        for _ in range(num_classes):
            probs.append(pd.DataFrame())
        for i in tqdm(range(len(self.models))):
            k = 0
            while i < self.ns[:k].sum():
                k+= 1

            k -= 1

            if self.tp[i] == "cat":
                if  self.f_dict and self.models_names[k] in self.f_dict.keys():
                    proba = self.models[i].predict_proba(X_cat[self.f_dict[self.models_names[k]]])
                   
                else:
                    proba =  self.models[i].predict_proba(X_cat)
                    
            else:
                if  self.f_dict and self.models_names[k] in self.f_dict.keys():
                    proba =  self.models[i].predict_proba(X_num[self.f_dict[self.models_names[k]]])
                    
                else:
                    proba = self.models[i].predict_proba(X_num)

            for j in range(num_classes):
                probs[j][self.names[i]] = proba[:, j]
             
        
        self.probs = probs
        ans = []
        for item in probs:
            ans.append(self.get_final_ans(item))
        ans = np.array(ans).T
        return ans
            
        
        
        
        
    def get_final_ans(self, X_test_proba, h = 0.3):
        ans = np.zeros(X_test_proba.shape[0])
        for i, pred in enumerate(X_test_proba.values):
            ms = MeanShift(bandwidth=h)
            sam = pred.reshape(-1, 1)
            ms.fit(sam)
            a = ms.predict(sam)
            unique, counts = np.unique(a, return_counts=True)
            cluster = unique[np.where(counts == counts.max())[0][0]]
            ans[i] = pred[a == cluster].mean()
        return ans
            