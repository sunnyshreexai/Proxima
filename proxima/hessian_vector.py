# -*- coding: utf-8 -*-
#Final Influence - 3
"""Do inverse hessian-vector-product.
"""
import pdb
import time
import numpy as np
import pandas as pd  

#file imports
from grad_utils import *
from hinge import *

#from grad_utils import grad_logloss_theta_lr, hessian_logloss_theta_lr, hessian_vector_product_lr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
# Tensorflow import
import tensorflow as tf
# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions

#loading dataset
def dataset_method():
  dataset = helpers.load_adult_income_dataset()
  return dataset

def inverse_hvp_lissa(x_train,y_train,y_pred,v,
    batch_size=100,repeat=10,max_recursion_depth=10,
    l2_norm=0.01,tol=1e-6,hessian_free=True):
    """Get inverse hessian-vector-products H^-1 * v with stochastic esimation:
    linear (time) stochastic second-order algorithm, LISSA
    this method is suitable for the large dataset and useful for broad algorithms.
    Refers to 
    `Second-order Stochastic Optimization for Machine Learning in Linear Time 2017 JMLR` .
    """
    start_time = time.time()
    inverse_hvp = None
    for r in range(repeat):
        # initialize H_0 ^ -1 * v = v begin with each repeat.
        current_estimate = v
        for j in range(max_recursion_depth):
            batch_idx = np.random.choice(np.arange(x_train.shape[0]),size=batch_size)
            if hessian_free:
                hessian_vector_val = hessian_vector_product_lr(y_train[batch_idx],
                    y_pred[batch_idx],x_train[batch_idx],
                    current_estimate,l2_norm)
            else:
                hessian_matrix = hessian_logloss_theta_lr(y_train[batch_idx],
                    y_pred[batch_idx],x_train[batch_idx],l2_norm)
                hessian_vector_val = np.dot(current_estimate,hessian_matrix)
            
            current_estimate_new = v + current_estimate - hessian_vector_val

            diffs = np.linalg.norm(current_estimate_new) - np.linalg.norm(current_estimate)
            diffs = diffs / np.linalg.norm(current_estimate)
            if diffs <= tol:
                current_estimate = current_estimate_new
                print("Break in depth {}".format(str(j)))
                break
            current_estimate = current_estimate_new

        print("Repeat at {} times: norm is {:.2f}".format(r, 
                np.linalg.norm(current_estimate)))

        if inverse_hvp is None:
            inverse_hvp = current_estimate
        else:
            inverse_hvp = inverse_hvp + current_estimate

    # average
    inverse_hvp = inverse_hvp / float(repeat)
    print("Inverse HVP took {:.1f} sec".format(time.time() - start_time))
    return inverse_hvp

def inverse_hvp_lr_newtonCG(x_train,y_train,y_pred,v,C=0.01,hessian_free=True,tol=1e-5,has_l2=True,M=None,scale_factor=1.0):
    """Get inverse hessian-vector-products H^-1 * v, this method is not suitable for
    the large dataset.
    Args:
        x_train, y_train: training data used for computing the hessian, e.g. x_train: [None,n]
        y_pred: predictions made on x_train, e.g. [None,]
        v: value vector, e.g. [n,]
        hessian_free: bool, `True` means use implicit hessian-vector-product to avoid
            building hessian directly, `False` will build hessian.
            hessian free will save memory while be slower in computation, vice versa.
            such that set `True` when cope with large dataset, and set `False` with 
            relatively small dataset.
    Return:
        H^-1 * v: shape [None,]
    """
    if not hessian_free:
        hessian_matrix = hessian_logloss_theta_lr(y_train,y_pred,x_train,C,has_l2,scale_factor)

    # build functions for newton-cg optimization
    def fmin_loss_fn(x):
        """Objective function for newton-cg.
        H^-1 * v = argmin_t {0.5 * t^T * H * t - v^T * t}
        """
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor) # [n,]
        else:
            hessian_vec_val = np.dot(x,hessian_matrix) # [n,]
        obj = 0.5 * np.dot(hessian_vec_val,x) - \
                    np.dot(x, v)

        return obj

    def fmin_grad_fn(x):
        """Gradient of the objective function w.r.t t:
        grad(obj) = H * t - v
        """
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor)
        else:
            hessian_vec_val = np.dot(x,hessian_matrix) # [n,]

        grad = hessian_vec_val - v

        return grad

    def get_fmin_hvp(x,p):
        # get H * p
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,p,C,has_l2,scale_factor)
        else:
            hessian_vec_val = np.dot(p,hessian_matrix)

        return hessian_vec_val

    def get_cg_callback(verbose):
        def fmin_loss_split(x):
            if hessian_free:
                hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor)
            else:
                hessian_vec_val = np.dot(x,hessian_matrix)

            loss_1 = 0.5 * np.dot(hessian_vec_val,x)
            loss_2 = - np.dot(v, x)
            return loss_1, loss_2

        def cg_callback(x):

            if verbose:
                print("Function value:", fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print("Split function value: {}, {}".format(quad, lin))
                # print("Predicted loss diff on train_idx {}: {}".format(idx_to_remove, predicted_loss_diff))

        return cg_callback

    start_time = time.time()
    cg_callback = get_cg_callback(verbose=True)
    fmin_results = fmin_ncg(f=fmin_loss_fn,
                           x0=v,
                           fprime=fmin_grad_fn,
                           fhess_p=get_fmin_hvp,
                           callback=cg_callback,
                           avextol=tol,
                           maxiter=100,
                           preconditioner=M)

    print("implicit hessian-vector products mean:",fmin_results.mean())
    print("implicit hessian-vector products norm:",np.linalg.norm(fmin_results))
    print("Inverse HVP took {:.1f} sec".format(time.time() - start_time))
    return fmin_results

def categorical_encoding(df, categorical_cloumns, encoding_method):
    
    if encoding_method == 'label':
        print('You choose label encoding for your categorical features')
        encoder = LabelEncoder()
        encoded = df[categorical_cloumns].apply(encoder.fit_transform)
        return encoded
    
    elif encoding_method == 'one-hot':
        print('You choose one-hot encoding for your categorical features') 
        encoded = pd.DataFrame()
        for feature in categorical_cloumns:
            dummies = pd.get_dummies(df[feature], prefix=feature, drop_first = True)
            encoded = pd.concat([encoded, dummies], axis=1)
        return encoded
    
def data_preprocessing(df, features, target, encoding_method, test_size, random_state):
    y = df[target]
    
    X = df[features]
    
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) != 0 :
        encoded = categorical_encoding(X, categorical_cloumns=categorical_columns, encoding_method=encoding_method)
        X = X.drop(columns=categorical_columns, axis=1)
        X = pd.concat([X, encoded], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler=MinMaxScaler()
    X_train= pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    return X_train, X_test, y_train, y_test