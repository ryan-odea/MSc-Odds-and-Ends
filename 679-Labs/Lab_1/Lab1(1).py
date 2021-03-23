import h2o
import pandas as pd
import numpy as np
import os

h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
h2o.remove_all()                          #clean slate, in case cluster was already running

help(h2o)

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

help(H2OGeneralizedLinearEstimator)

help(h2o.import_file)


covtype_df = h2o.import_file(os.path.realpath("./prostate.csv"))

train, valid, test = covtype_df.split_frame([0.7, 0.15], seed=1234)


#Prepare predictors and response columns
covtype_X = ["AGE", "RACE", "PSA", "GLEASON"]     #last column is Cover_Type, our desired response variable 
covtype_y = covtype_df.col_names[7]   
     
glm_gauss_v1 = H2OGeneralizedLinearEstimator(
                    model_id='glm_v1',           
                    family='gaussian',
                    solver='L_BFGS',Lambda=0)

glm_gauss_v1.train(covtype_X, covtype_y,
                training_frame=train, 
                validation_frame=valid
                 )

glm_gauss_v1.coef()
glm_gauss_v1.confusion_matrix(valid)

# Cross validation

glm_gauss_v2 = H2OGeneralizedLinearEstimator(
                    model_id='glm_v2',           
                    family='gaussian',
                    nfolds=10,
                    solver='L_BFGS',Lambda=0)
glm_gauss_v2.train(covtype_X, covtype_y,
                training_frame=train, 
                validation_frame=valid
                 )
                 
# Binomial example

covtype_X2 = ["AGE", "RACE", "PSA", "GLEASON"]     #last column is Cover_Type, our desired response variable 
covtype_y2 = "CAPSULE" 

glm_bin_v1 = H2OGeneralizedLinearEstimator(
                    model_id='glm_bin_v1',           
                    family='binomial',
                    nfolds=10,
                    solver='L_BFGS',Lambda=0)
glm_bin_v1.train(covtype_X2, covtype_y2,
                training_frame=train, 
                validation_frame=valid
                 )
                 
glm_bin_v1.accuracy(valid=True)

