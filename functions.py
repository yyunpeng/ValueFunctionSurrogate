import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import TruncatedNormal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
import time
import yfinance as yf
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import load_model
import os
from functools import partial
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import shap
from alpha_vantage.timeseries import TimeSeries
from sklearn.linear_model import Lasso
from tensorflow.keras.layers import Lambda
import itertools
from tensorflow.keras.constraints import Constraint
from datetime import datetime



T=3

def print_version():
    print('version', 16.4)

def VARMA_sim1(current_d_ln_x, last_d_ln_x, Sigma, numSim, T, mu, Phi1, Phi2):

    d_ln_S1 = np.zeros((numSim, T+5))
    d_ln_S2 = np.zeros((numSim, T+5))
    d_ln_S3 = np.zeros((numSim, T+5))
    d_ln_L  = np.zeros((numSim, T+5))
    noise   = np.zeros((numSim, T+5, 4))

    for path in range(numSim):
        current = current_d_ln_x.copy()
        last = last_d_ln_x.copy()
        d_ln_S1[path, 0] = current[0]
        d_ln_S2[path, 0] = current[1]
        d_ln_S3[path, 0] = current[2]
        d_ln_L[path, 0] = current[3]
        noise[path, 0] = np.random.normal(0, 1, 4)

        for t in range(1, T+5):
            eps = np.random.multivariate_normal(mean=np.zeros(4), cov=Sigma)
            noise[path, t] = eps
            dx_current = current - mu
            dx_last = last - mu

            full_dln = mu + Phi1 @ dx_current + Phi2 @ dx_last + eps
            d_ln_S1[path, t] = full_dln[0]
            d_ln_S2[path, t] = full_dln[1]
            d_ln_S3[path, t] = full_dln[2]
            d_ln_L[path, t]  = full_dln[3]

            last = current
            current = full_dln

    return noise, d_ln_S1, d_ln_S2, d_ln_S3, d_ln_L


def lasso_VAR_centered(p, data, alpha=0.01):

    Y = data.values
    T, K = Y.shape

    mu = Y.mean(axis=0).reshape(K, 1)

    Y_demeaned = Y - mu.T

    X, Y_target = [], []
    for t in range(p, T):
        lagged = Y_demeaned[t - p:t][::-1].flatten() 
        X.append(lagged)
        Y_target.append(Y_demeaned[t])  # Y_t - mu

    X = np.array(X)  
    Y_target = np.array(Y_target) 

    Phi_list = []
    predictions = []

    for k in range(K):
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        lasso.fit(X, Y_target[:, k])
        predictions.append(lasso.predict(X))

        coefs = lasso.coef_.reshape(p, K).T 
        Phi_list.append(coefs)

    Phi_matrices = []
    for lag in range(p):
        Phi_lag = np.column_stack([Phi_list[k][:, lag] for k in range(K)])
        Phi_matrices.append(Phi_lag)

    Y_hat = np.column_stack(predictions)
    residuals = Y_target - Y_hat
    Sigma = np.cov(residuals.T)

    return Phi_matrices, mu, Sigma


def generate_training_data(
    d_ln_S1_vali, d_ln_S2_vali, d_ln_S3_vali, d_ln_L_vali,
    ln_L_vali, C_range, numTrain, T
):
    def get_range(vali_array):
        # shape: (numSim, T+1), exclude t=0
        return np.min(vali_array[:, 1:]), np.max(vali_array[:, 1:])

    def get_ln_range(vali_array):
        return np.min(vali_array[:, 1:]), np.max(vali_array[:, 1:])  # exclude t=0

    d_ln_S1_range = get_range(d_ln_S1_vali)
    d_ln_S2_range = get_range(d_ln_S2_vali)
    d_ln_S3_range = get_range(d_ln_S3_vali)
    d_ln_L_range  = get_range(d_ln_L_vali)
    ln_L_range    = get_ln_range(ln_L_vali)
    c_min, c_max  = C_range

    sobol = qmc.Sobol(d=10, scramble=True)
    sobol_samples = sobol.random(n=numTrain)

    d_ln_S1t_train        = sobol_samples[:, 0] * (d_ln_S1_range[1] - d_ln_S1_range[0]) + d_ln_S1_range[0]
    d_ln_S1t_minus1_train = sobol_samples[:, 1] * (d_ln_S1_range[1] - d_ln_S1_range[0]) + d_ln_S1_range[0]

    d_ln_S2t_train        = sobol_samples[:, 2] * (d_ln_S2_range[1] - d_ln_S2_range[0]) + d_ln_S2_range[0]
    d_ln_S2t_minus1_train = sobol_samples[:, 3] * (d_ln_S2_range[1] - d_ln_S2_range[0]) + d_ln_S2_range[0]

    d_ln_S3t_train        = sobol_samples[:, 4] * (d_ln_S3_range[1] - d_ln_S3_range[0]) + d_ln_S3_range[0]
    d_ln_S3t_minus1_train = sobol_samples[:, 5] * (d_ln_S3_range[1] - d_ln_S3_range[0]) + d_ln_S3_range[0]

    d_ln_Lt_train         = sobol_samples[:, 6] * (d_ln_L_range[1] - d_ln_L_range[0]) + d_ln_L_range[0]
    d_ln_Lt_minus1_train  = sobol_samples[:, 7] * (d_ln_L_range[1] - d_ln_L_range[0]) + d_ln_L_range[0]

    ln_Lt_train           = sobol_samples[:, 8] * (ln_L_range[1] - ln_L_range[0]) + ln_L_range[0]
    c_train               = sobol_samples[:, 9] * (c_max - c_min) + c_min

    return d_ln_S1t_train, d_ln_S1t_minus1_train, \
           d_ln_S2t_train, d_ln_S2t_minus1_train, \
           d_ln_S3t_train, d_ln_S3t_minus1_train, \
           d_ln_Lt_train,  d_ln_Lt_minus1_train, \
           ln_Lt_train, c_train


def U(x,gamma):
    return 1/gamma * np.sign(x) * (np.abs(x)) ** gamma


def V_T(C_T,gamma):
    return U(C_T,gamma)

def g(
      d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
      d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
      ln_Lt, C_t, 
      u, mu, Phi1, Phi2, p, quantizer, t
      ):
        
    d_ln_Xt_vec = np.array([d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt]) 
    d_ln_Xt_minus1_vec = np.array([d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1]) 
    
    d_ln_S1t_plus1 = mu[0] + np.dot(Phi1, d_ln_Xt_vec-mu)[0] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[0] + quantizer[t+1][:, 0]
    d_ln_S2t_plus1 = mu[1] + np.dot(Phi1, d_ln_Xt_vec-mu)[1] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[1] + quantizer[t+1][:, 1]
    d_ln_S3t_plus1 = mu[2] + np.dot(Phi1, d_ln_Xt_vec-mu)[2] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[2] + quantizer[t+1][:, 2]
    d_ln_Lt_plus1  = mu[3] + np.dot(Phi1, d_ln_Xt_vec-mu)[3] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[3] + quantizer[t+1][:, 3]
    
    d_ln_S1t = d_ln_S1t
    d_ln_S2t = d_ln_S2t
    d_ln_S3t = d_ln_S3t
    d_ln_Lt = d_ln_Lt
    
    ln_Lt_plus1 = ln_Lt + d_ln_Lt_plus1
    
    u3 = 1-u[1]-u[2]    
    C_t_plus1 = (u[1]*np.exp(d_ln_S1t_plus1) + u[2]*np.exp(d_ln_S2t_plus1)  + u3*np.exp(d_ln_S3t_plus1) ) * (C_t + p - u[0] * C_t) - np.exp(ln_Lt_plus1)
    
    return d_ln_S1t_plus1,d_ln_S2t_plus1,d_ln_S3t_plus1,d_ln_Lt_plus1,\
            d_ln_S1t,d_ln_S2t,d_ln_S3t,d_ln_Lt,\
            ln_Lt_plus1, C_t_plus1

def V_T_minus1(   
                d_ln_S1T_minus1, d_ln_S2T_minus1, d_ln_S3T_minus1, d_ln_LT_minus1,
                d_ln_S1T_minus2, d_ln_S2T_minus2, d_ln_S3T_minus2, d_ln_LT_minus2,
                ln_LT_minus1, C_T_minus1, 
                u, mu, Phi1, Phi2, p, 
                weights, gamma, v, quantizer, t=T-1
                ):
    
    d_ln_S1T, d_ln_S2T, d_ln_S3T, d_ln_LT, \
    d_ln_S1T_minus1, d_ln_S2T_minus1, d_ln_S3T_minus1, d_ln_LT_minus1, \
    ln_LT, C_T \
    = g(
        d_ln_S1T_minus1,d_ln_S2T_minus1,d_ln_S3T_minus1,d_ln_LT_minus1,
        d_ln_S1T_minus2, d_ln_S2T_minus2, d_ln_S3T_minus2, d_ln_LT_minus2,
        ln_LT_minus1, C_T_minus1, 
        u, mu, Phi1, Phi2, p, quantizer, t=T-1
        )
    E_V_T = np.sum( weights * V_T(C_T, gamma) )
    
    V_T_minus1 = U( u[0] *C_T_minus1, gamma) + v * E_V_T

    return V_T_minus1

def concave_nondecreasing_activation(x):  
    return -np.log1p(np.exp(-x))  

def V_theta_t_plus1(
    d_ln_S1t_plus1, d_ln_S2t_plus1, d_ln_S3t_plus1, d_ln_Lt_plus1,
    d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
    ln_Lt_plus1, c1,
    weights, input_scaler):

    inputdata = np.concatenate([
        d_ln_S1t_plus1.reshape(-1, 1), d_ln_S2t_plus1.reshape(-1, 1),
        d_ln_S3t_plus1.reshape(-1, 1), d_ln_Lt_plus1.reshape(-1, 1),
        d_ln_S1t.reshape(-1, 1), d_ln_S2t.reshape(-1, 1),
        d_ln_S3t.reshape(-1, 1), d_ln_Lt.reshape(-1, 1),
        ln_Lt_plus1.reshape(-1, 1), c1.reshape(-1, 1)
    ], axis=1)

    input_scaled = input_scaler.transform(inputdata)
    input_others = input_scaled[:, :-1]
    input_C = input_scaled[:, -1:]

    w1, b1, w2, b2, w3, b3, w4, b4, wC, bC, w_merge, b_merge, w_out, b_out = weights

    def tanh(x): 
        return np.tanh(x)

    layer = tanh(input_others @ w1 + b1)
    layer = tanh(layer @ w2 + b2)
    layer = tanh(layer @ w3 + b3)
    layer = tanh(layer @ w4 + b4)

    C_layer = concave_nondecreasing_activation(input_C @ wC + bC)

    merged = np.concatenate([layer, C_layer], axis=1)
    merged = concave_nondecreasing_activation(merged @ w_merge + b_merge)

    output = merged @ w_out + b_out
    value_func = output[:, 0:1] 

    return value_func 


def V_t(
        d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
        d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
        ln_Lt, C_t, 
        u, mu, Phi1, Phi2, p,
        NN_weights, inputscaler, quantizer, t, 
        weights, gamma, v
    ):

    numWeights = len(quantizer[0])  # number of quantized samples

    d_ln_S1t_plus1, d_ln_S2t_plus1, d_ln_S3t_plus1, d_ln_Lt_plus1, \
    d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt, \
    ln_Lt_plus1, C_t_plus1 = g(
        d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
        d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
        ln_Lt, C_t, u, mu, Phi1, Phi2, p, quantizer, t
    )

    approx_V_t_plus1 = V_theta_t_plus1(
        d_ln_S1t_plus1,
        d_ln_S2t_plus1,
        d_ln_S3t_plus1,
        d_ln_Lt_plus1,
        np.full((numWeights, 1), float(d_ln_S1t)),
        np.full((numWeights, 1), float(d_ln_S2t)),
        np.full((numWeights, 1), float(d_ln_S3t)),
        np.full((numWeights, 1), float(d_ln_Lt)),
        ln_Lt_plus1,
        C_t_plus1,
        NN_weights,
        inputscaler
    )
    E_V_t_plus1 = np.sum(approx_V_t_plus1[:, 0] * weights)

    V_t_val = U(u[0] * C_t, gamma) + v * E_V_t_plus1

    return V_t_val


def optimize_u_general(
    t, quantize_weights, g, j_v_t,
    d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
    d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
    ln_Lt, C_t,
    mu, Phi1, Phi2, p, quantizer,
    stock_lo, stock_up, cbond_lo, cbond_up,
    div_lo, div_up,
    best_u12=None
):

    if best_u12 is None:
        best_u12 = [(stock_lo + stock_up) / 2, (cbond_lo + cbond_up) / 2]

    def inner_objective(u12):
        u1, u2 = u12
        u = np.array([0.0, u1, u2])
        return -1*j_v_t(u)

    res_u12 = minimize(
        inner_objective,
        x0=best_u12,
        method='L-BFGS-B',
        bounds=[(stock_lo, stock_up), (cbond_lo, cbond_up)]
    )

    if not res_u12.success:
        print(f"[Warning] u12 optimization failed at t={t}: {res_u12.message}")

    u1_star, u2_star = res_u12.x
    best_u12[:] = [u1_star, u2_star]

    def outer_objective(u0):
        u = [u0, u1_star, u2_star]
        return -1*j_v_t(u)

    res_u0 = minimize_scalar(
        outer_objective,
        bounds=(div_lo, div_up),
        method='bounded',
        options={'xatol': 1e-6}
    )

    if not res_u0.success:
        print(f"[Warning] u0 optimization failed at t={t}: {res_u0.message}")

    u0_star = res_u0.x
    jv_max = -1*res_u0.fun
    u_star = np.array([u0_star, u1_star, u2_star])

    return u_star, jv_max, best_u12


metlife_dividends = [0.001791, 0.001831, 0.001848, 0.001818]
metlife_stock = [0.0047, 0.0047, 0.0047, 0.0047]
metlife_cbond  = [0.6233, 0.6291, 0.6398, 0.6340]
prudential_dividends = [0.001556, 0.001568, 0.001659, 0.001537]
prudential_stock = [0.0252, 0.0284, 0.0252, 0.0274]
prudential_cbond  = [0.5975, 0.6091, 0.6367, 0.6386]

        
def IndividualTest(c0, #gamma, 
                   path, T, 
                   ln_L_0,
                   d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                   d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1, 
                   quantizer, p, weights, gamma, v, mu,  Phi1, Phi2, 
                   div_lo, div_up, stock_lo, stock_up, cbond_lo, cbond_up, 
                   input_scaler_valuefun, V_hat_theta
                   ):
    
    def expt_utility_objective(u):
        d_ln_S1tp1, d_ln_S2tp1, d_ln_S3tp1, d_ln_Ltp1, \
        d_ln_S1t_, d_ln_S2t_, d_ln_S3t_, d_ln_Lt_, \
        ln_Ltp1, C_tp1 = g(
            d_ln_S1t[path, t], d_ln_S2t[path, t], d_ln_S3t[path, t], d_ln_Lt[path, t],
            d_ln_S1t_minus1[path, t], d_ln_S2t_minus1[path, t], d_ln_S3t_minus1[path, t], d_ln_Lt_minus1[path, t],
            ln_L_t, samples[5][4][t], 
            u, mu, Phi1, Phi2, p, quantizer, t
        )
        U_C_tp1 = U(C_tp1,gamma)
        expected_U_C_tp1 = np.sum(weights*U_C_tp1)
        div_util = U(u[0] * samples[5][4][t], gamma)
        return - ( div_util + v * expected_U_C_tp1 ) 
    def mean_variance(u):
        d_ln_S1tp1, d_ln_S2tp1, d_ln_S3tp1, d_ln_Ltp1, \
        d_ln_S1t_, d_ln_S2t_, d_ln_S3t_, d_ln_Lt_, \
        ln_Ltp1, C_tp1 = g(
            d_ln_S1t[path, t], d_ln_S2t[path, t], d_ln_S3t[path, t], d_ln_Lt[path, t],
            d_ln_S1t_minus1[path, t], d_ln_S2t_minus1[path, t], d_ln_S3t_minus1[path, t], d_ln_Lt_minus1[path, t],
            ln_L_t, samples[5][4][t], 
            u, mu, Phi1, Phi2, p, quantizer, t
        )
        U_C_tp1 = U(C_tp1,gamma)
        expected_U_C_tp1 = np.sum(weights*U_C_tp1)
        var_U_C_tp1 = np.sum(weights*U_C_tp1**2) - np.sum(weights*U_C_tp1)**2
        div_util = U(u[0] * samples[5][4][t], gamma)
        return - div_util - v * expected_U_C_tp1 + 0.2*var_U_C_tp1
        
    samples = np.ones((6, 7, T+2))  
    
    samples[:,4,0:T+2] = c0
    # index 4 means portfolio value
    
    for t in range(0, T):
        
        ln_L_t = ((ln_L_0) + sum((d_ln_Lt[path, j]) for j in range(0, t)))
        ln_L_t_plus1 = ((ln_L_0) + sum((d_ln_Lt[path, j]) for j in range(0, t+1)))
        
        R1_t_plus1 = np.exp((d_ln_S1t[path, t+1]))
        R2_t_plus1 = np.exp((d_ln_S2t[path, t+1]))
        R3_t_plus1 = np.exp((d_ln_S3t[path, t+1]))

        best_u12 = [(stock_lo + stock_up) / 2, (cbond_lo + cbond_up) / 2]  # warm-start seed (mutable list)
        
        if t < T-1:
            def g_i(u): 
                V = V_t(
                        d_ln_S1t[path, t], d_ln_S2t[path, t], d_ln_S3t[path, t], d_ln_Lt[path, t],         
                        d_ln_S1t_minus1[path, t], d_ln_S2t_minus1[path, t], d_ln_S3t_minus1[path, t], d_ln_Lt_minus1[path, t], 
                        ln_L_t, samples[0][4][t], 
                        u, mu,  Phi1, Phi2, p, 
                        V_hat_theta[t+1].get_weights(), input_scaler_valuefun, quantizer, t,
                        weights, gamma, v
                        )
                return V 
        else:
            def g_i(u): 
                V = V_T_minus1(
                                d_ln_S1t[path, t], d_ln_S2t[path, t], d_ln_S3t[path, t], d_ln_Lt[path, t],         
                                d_ln_S1t_minus1[path, t], d_ln_S2t_minus1[path, t], d_ln_S3t_minus1[path, t], d_ln_Lt_minus1[path, t], 
                               ln_L_t, samples[0][4][t], 
                               u, mu,  Phi1, Phi2, p, 
                               weights, gamma, v, 
                               quantizer
                               )         
                return V 
            
        u_star, jv_min, best_u12 = optimize_u_general(
            t=t,
            quantize_weights=weights,
            g=g, 
            j_v_t=g_i,
            d_ln_S1t=d_ln_S1t[path, t], d_ln_S2t=d_ln_S2t[path, t], d_ln_S3t=d_ln_S3t[path, t], d_ln_Lt=d_ln_Lt[path, t],
            d_ln_S1t_minus1=d_ln_S1t_minus1[path, t], d_ln_S2t_minus1=d_ln_S2t_minus1[path, t], 
            d_ln_S3t_minus1=d_ln_S3t_minus1[path, t], d_ln_Lt_minus1=d_ln_Lt_minus1[path, t],
            ln_Lt=ln_L_t, C_t=samples[0, 4, t],
            mu=mu, Phi1=Phi1, Phi2=Phi2, p=p, quantizer=quantizer,
            stock_lo=stock_lo, stock_up=stock_up,
            cbond_lo=cbond_lo, cbond_up=cbond_up,
            div_lo=div_lo, div_up=div_up,
            best_u12=None)

        u_hat = u_star
        samples[0][0][t] = u_hat[0]
        samples[0][1][t] = u_hat[1]
        samples[0][2][t] = u_hat[2]
        samples[0][3][t] = 1-u_hat[1]-u_hat[2]
        samples[0][4][t+1] = \
            (samples[0][1][t] * R1_t_plus1 +
              samples[0][2][t] * R2_t_plus1 +
              samples[0][3][t] * R3_t_plus1)*\
            (samples[0][4][t] +p - samples[0][4][t]*samples[0][0][t]) - np.exp(ln_L_t_plus1 )

        quarter = t // 3  

        samples[1][0][t] = metlife_dividends[quarter]
        samples[1][1][t] = metlife_stock[quarter]
        samples[1][2][t] = metlife_cbond[quarter]
        samples[1][3][t] = 1 - metlife_stock[quarter] - metlife_cbond[quarter]
        samples[1][4][t+1] = \
            (samples[1][1][t] * R1_t_plus1 +
             samples[1][2][t] * R2_t_plus1 +
             samples[1][3][t] * R3_t_plus1) *\
            (samples[1][4][t] +p - samples[1][4][t]*samples[1][0][t]) - np.exp(ln_L_t_plus1 )
                    
        samples[2][0][t] = prudential_dividends[quarter]
        samples[2][1][t] = prudential_stock[quarter]
        samples[2][2][t] = prudential_cbond[quarter]
        samples[2][3][t] = 1 - prudential_stock[quarter] - prudential_cbond[quarter]
        samples[2][4][t+1] = \
            (samples[2][1][t] * R1_t_plus1 +
             samples[2][2][t] * R2_t_plus1 +
             samples[2][3][t] * R3_t_plus1) *\
            (samples[2][4][t] +p - samples[2][4][t]*samples[2][0][t]) -  np.exp(ln_L_t_plus1 )
                    
        samples[3][0][t] = 0.005512
        samples[3][1][t] = 1-0.6357215606132216-0.3328385134100637
        samples[3][2][t] = 0.6357215606132216
        samples[3][3][t] = 0.3328385134100637
        samples[3][4][t+1] = \
            (samples[3][1][t] * R1_t_plus1 +
             samples[3][2][t] * R2_t_plus1 +
             samples[3][3][t] * R3_t_plus1) *\
            (samples[3][4][t] +p - samples[3][4][t]*samples[3][0][t]) -  np.exp(ln_L_t_plus1 )

        initial_guess = [(div_lo+div_up)/2, (stock_lo+stock_up)/2, (cbond_lo+cbond_up)/2]  
        bounds = [(div_lo, div_up), (stock_lo, stock_up), (cbond_lo, cbond_up)]
        result_mean_variance = minimize(mean_variance, initial_guess, bounds=bounds, method='L-BFGS-B')
        samples[4][0][t] = result_mean_variance.x[0]
        samples[4][1][t] = result_mean_variance.x[1]
        samples[4][2][t] = result_mean_variance.x[2]
        samples[4][3][t] = 1 - result_mean_variance.x[1] - result_mean_variance.x[2]
        samples[4][4][t+1] = \
            (samples[4][1][t] * R1_t_plus1 +
             samples[4][2][t] * R2_t_plus1 +
             samples[4][3][t] * R3_t_plus1) *\
            (samples[4][4][t] +p - samples[4][4][t]*samples[4][0][t]) -  np.exp(ln_L_t_plus1)

        result_expt_utility_objective = minimize(expt_utility_objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        samples[5][0][t] = result_expt_utility_objective.x[0]
        samples[5][1][t] = result_expt_utility_objective.x[1]
        samples[5][2][t] = result_expt_utility_objective.x[2]
        samples[5][3][t] = 1 - result_expt_utility_objective.x[1] - result_expt_utility_objective.x[2]
        samples[5][4][t+1] = \
            (samples[5][1][t] * R1_t_plus1 +
             samples[5][2][t] * R2_t_plus1 +
             samples[5][3][t] * R3_t_plus1) *\
            (samples[5][4][t] +p - samples[5][4][t]*samples[5][0][t]) -  np.exp(ln_L_t_plus1)
        
        loss_count = 0
        minimum_capital = c0 - sum(
                                np.exp(ln_L_0 + sum(
                                    d_ln_Lt[path, t] for j in range(1, t))
                                    ) for t in range(1, T)
                                ) + (T-1)*p

        if samples[0][4][T] < minimum_capital or samples[0][4][T] < samples[1][4][T] or samples[0][4][T] < samples[2][4][T] or samples[0][4][T] < samples[3][4][T] or samples[0][4][T] < samples[4][4][T] or samples[0][4][T-1] < samples[5][4][T] :
            loss_count = 1
        
    return samples , loss_count


def RunTests(
             c0, T,
             ln_L_0,
             d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
             d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1, 
             quantizer, prem, weights, gamma, v, mu,  Phi1, Phi2, 
             div_lo, div_up, stock_lo, stock_up, cbond_lo, cbond_up, 
             input_scaler, V_hat_theta,
             numTest
            ):    
    
    start = time.perf_counter() 
    results = {}
    total_loss_coun = 0
    start_i = time.perf_counter() 

    for path in range(1,numTest-1):

        samples, loss_coun = IndividualTest(c0, #gamma, 
                                               path, T,
                                               ln_L_0, 
                                               d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                                               d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1, 
                                               quantizer, prem, weights, gamma, v, mu,  Phi1, Phi2, 
                                               div_lo, div_up, stock_lo, stock_up, cbond_lo, cbond_up, 
                                               input_scaler, V_hat_theta
                                            )
     
        results[path] = samples
        total_loss_coun += loss_coun 
        
        if path == numTest/2: 
            end_i = time.perf_counter() 
            duration_i = round((end_i-start_i)/60,2)
            print('50% done: ' + str(duration_i) + " min.")
            
        elif path == numTest/4:
            end_i = time.perf_counter() 
            duration_i = round((end_i-start_i)/60,2)
            print('25% done: ' + str(duration_i) + " min.")
        elif path == numTest/4*3:
            end_i = time.perf_counter() 
            duration_i = round((end_i-start_i)/60,2)
            print('75% done: ' + str(duration_i) + " min.")
    
    end = time.perf_counter() 
    duration = (end-start)/60
    print("Duration: " + str(duration) + " min.")
        
    return results, total_loss_coun


def certainty_equivalent(utility_values, gamma):
    mean_util = np.mean(utility_values)
    if gamma == 1:
        return np.exp(mean_util)
    return (mean_util * (1 - gamma))**(1 / (1 - gamma))

def expected_shortfall(data, alpha=0.05):
    sorted_data = np.sort(data)
    cutoff_index = int(np.floor(alpha * len(sorted_data)))
    return np.mean(sorted_data[:cutoff_index]) if cutoff_index > 0 else np.min(sorted_data)
