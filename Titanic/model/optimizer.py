from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from tqdm import tqdm

from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import copy

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import random





class bayes_opt:
    
    def __init__(self, model,  X_train, y_train, verbose=0, cv = None, pbounds = None, init_params =None, tp = None, n_iter = 5, init_points = 1 , cat_columns = None, av_params = None):
        if av_params and model in av_params.keys():
            
            self.av_params = av_params[model]
        else :
            self.av_params = None
        self.model = model
        self.X_train =  X_train
        self.y_train = y_train
        self.cv = cv
        self.tp = tp
        self.pbounds = pbounds
        self.init_params = init_params
        optimizer = BayesianOptimization(self.evaluate_model, pbounds, random_state=4)
        self.cat_columns = cat_columns
        logger = JSONLogger(path="./logs.json")
        optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        self.opt = optimizer
        self.results = optimizer.res

        # optimized params y
        


    def get_model(self, **params):
        print(params)
        return self.model(**params)

    def validate(self, model, X_train, y_train, verbose=0, method = "accuracy"):
        
        if self.cv:
            kfold = KFold(self.cv, True, 1)
            current_score = 0
            for train, test in kfold.split(X_train):
                model1 = copy.deepcopy(model)
                if self.cat_columns:
                    model1.fit(X_train.iloc[train], y_train[train], cat_features = self.cat_columns
                    )
                else:
                    model1.fit(X_train.iloc[train], y_train[train]
                    )
                print(model1.get_params())
                if method == "accuracy":
                    probs = model1.predict(X_train.iloc[test])
                    current_score += accuracy_score(probs, y_train[test])
               
                    
            if verbose:
                print(current_score/self.cv)
            current_score /= self.cv
        else:
            X, X_val, y, y_val = tarin_test_split(X_train, y_train)
            model1 = copy.deepcopy(model)
            model1.fit(X, y)
            print(model1.get_params())
            if method == "accuracy":
                probs = model1.predict(X_val)
                current_score = accuracy_score(probs, y_val)
            
        return current_score

    def evaluate_model(self, **params):
        if self.init_params:
            params_init = init_params
        else:
            params_init = self.random_choice(self.pbounds)
        params_init.update(params)
        
        for key in self.tp:
            params_init[key] = int(params_init[key])
        if self.av_params :
            for key in self.av_params.keys():
                params_init[key] = self.av_params[key]
        model = self.get_model(**params_init)
        current_score = self.validate(model, self.X_train, self.y_train, verbose=0)
        return current_score

    def random_choice(self, pbounds):
        param = {}
        for key in pbounds.keys():
            param[key] = self.get_param(pbounds[key])
            
        return param
            
            
            
    def get_param(self, params):
        res = params[0] + random.random()*(params[1] - params[0])
        if (type(params[0]) == int) or (type(params[1]) == int):
            res = int(res)
        return res
