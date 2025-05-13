import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,train_test_split
import pandas as pd
import numpy as np

data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name="target")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rf=RandomForestClassifier(random_state=42)

param_grid={
    "n_estimators":[10,50,100],
    "max_depth":[None,10,20,30]
}
# apply grid search cv 
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)

# without using mlflow
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# using mlflow
mlflow.set_experiment("breast_cancer-rf-gridsearch")
with mlflow.start_run() as parant:
    grid_search.fit(X_train,y_train)
    # log all child runs
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy",grid_search.cv_results_["mean_test_score"][i])
            
    # displaying best params and the best score
    best_params=grid_search.best_params_
    best_score=grid_search.best_score_
    #Log params and score
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy",best_score)
    
    # log training data
    train_df=X_train.copy()
    train_df["target"]=y_train
    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")
    
    # log test data
    test_df=X_test.copy()
    test_df["target"]=y_test
    
    test_df=mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"testing")
    
    # log source code
    mlflow.log_artifact(__file__)
    
    # log best model
    mlflow.sklearn.log_model(grid_search.best_estimator_,"random forest")
    
    # set tags
    
    mlflow.set_tag("author","rohan")
    print(best_params)
    print(best_score)
    