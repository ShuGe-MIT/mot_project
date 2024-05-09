import numpy as np
from functools import reduce
from math import log, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pickle as pkl
import time
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Run MOT Solver')
    parser.add_argument('--target_epsilon', type=float, default=1e-3, help='target epsilon')
    parser.add_argument('--start_epsilon', type=float, default=1, help='start epsilon')
    parser.add_argument('--epsilon_scale_num', type=float, default=0.99, help='epsilon_scale_num')
    parser.add_argument('--epsilon_scale_gap', type=float, default=100, help='epsilon_scale_gap')
    parser.add_argument('--cost_type', type=str, default='square', help='type of cost')
    parser.add_argument('--verbose', type=int, default=2, help='verbose')
    parser.add_argument('--cost_scale', type=float, default=1, help='cost_scale')
    parser.add_argument('--max_iter', type=int, default=5000, help='max iter')
    parser.add_argument('--iter_gap', type=int, default=100, help='iter_gap')
    parser.add_argument('--solver', type=str, default='sinkhorn', help='MOT solver')
    parser.add_argument('--data_file', type=str, default='weight_loss', help='data file')
    # Add more arguments as needed
    return parser.parse_args()

def rotate(l, n):
    return l[n:] + l[:n]

def tensor_sum(list_of_lists):
    # return a tensor of size len(list_of_lists[0]) x ... x len(list_of_lists[-1]) where the value at index (i_1, ..., i_n) is the sum of the values at the same index in the lists
    shape = tuple(len(lst) for lst in list_of_lists)
    tensor = np.zeros(shape)
    M = len(shape)
    idxes = list(range(len(shape)))
    for idx, l in enumerate(list_of_lists):
        tensor += torch.from_numpy(l).reshape(*([1]*idx + [-1] + [1]*(M - idx - 1))).expand(*shape).numpy()
    return tensor

def binary_search(a, b, f, delta = 1e-2):
    while True:
        c = (a+b)/2
        if f(c+delta) < f(c):
            a = c
        else:
            b = c
        if abs(a-b) <= delta:
            break
    return (a+b)/2

def create_normed_cost_tensor(list_list_of_lists):
    shape = tuple(len(lst) for lst in list_list_of_lists[0])
    res = np.zeros(shape)
    for list_of_lists in list_list_of_lists:
        tensor = tensor_sum(list_of_lists)
        res += (tensor / len(shape)) ** 2
    return res.astype(np.float64), np.array([1/n for n in shape for _ in range(n)]).astype(np.float64)

def create_cost_tensor(list_list_of_lists):
    shape = tuple(len(lst) for lst in list_list_of_lists[0])
    res = np.zeros(shape)
    for list_of_lists in list_list_of_lists:
        res += tensor_sum(list_of_lists) ** 2
    return res.astype(np.float64), np.array([1/n for n in shape for _ in range(n)]).astype(np.float64)

def create_cov_cost_tensor(list_list_of_lists):
    shape = tuple(len(lst) for lst in list_list_of_lists[0])
    res = np.ones(shape)
    for list_of_lists in list_list_of_lists:
        res *= tensor_sum(list_of_lists)
    return res.astype(np.float64), np.array([1/n for n in shape for _ in range(n)]).astype(np.float64)

def ravel_index(dim, index, shape):
    return sum([shape[i] for i in range(dim)]) + index

def convert_to_list(U, shape):
    U_list = []
    i = 0
    for s in shape:
        U_list.append(U[i:i+s])
        i += s
    return U_list
def marginal_k(X, k):
    # marginalize tensor X over dimension k
    return np.sum(X, axis=tuple(axis for axis in range(X.ndim) if axis != k))
def get_marginal_k(p, k, shape):
    return p[ravel_index(k, 0, shape): ravel_index(k, shape[k], shape)]

def projection(X, p):
    V = X.copy()
    m, shape = X.ndim, X.shape
    idxes = list(range(len(shape)))
    for r in range(m):
        X_r = np.minimum(get_marginal_k(p, r, shape) / marginal_k(V, r), 1)
        V *= np.transpose(np.tensordot(X_r, np.ones(shape[r+1:] + shape[:r]), axes=0), axes = rotate(idxes, -r))

    err_list = [get_marginal_k(p, r, shape) - marginal_k(V, r) for r in range(m)]
    V += reduce(np.multiply, np.ix_(*err_list)) / (np.abs(err_list[-1]).sum() ** (m-1))
    return V

def rho(a, b):
    return np.exp(a) @ (a - b)


def solve_lp(costs, target_mu):
    shape = costs.shape
    A = np.zeros((np.sum(shape), np.prod(shape)))
    for j, s in enumerate(shape):
        for i in range(s):
            for idx in product(*map(range, shape)):
                if idx[j] == i:
                    A[ravel_index(j, i, shape), np.ravel_multi_index(idx, shape)] = 1
    c = costs.flatten()
    res = linprog(c, A_eq=A, b_eq=target_mu, bounds=[0, 1])
    return res.fun, res.x.reshape(shape)

def solve_sinkhorn(costs, target_mu, epsilon=1e-2, target_epsilon=1e-4, verbose = 0, epsilon_scale_num = 0.99, epsilon_scale_gap = 100, cost_scale = 1, iter_gap = 100, max_iter = 5000, out_dir = 'test'):
    """solve using Greenkhorn's algorithm

    Args:
        costs (array)
        target_mu (list): target probability
        epsilon (float): precision. Defaults to 1e-2.
        verbose (int, optional): print intermediate results if verbose >= 2. Defaults to 0.
        cost_scale (int, optional): divide the cost matrix by cost_scale before calculating, scale back in the end. Defaults to 1.
        iter_gap (int, optional): print results every `iter_gap` iterations. Defaults to 100.

    Returns:
        _type_: _description_
    """
    ########## initialization ###########
    
    costs /= cost_scale
    shape = costs.shape
    M = len(shape)
    # print("shape: ", shape)
    eta = 4 * sum([log (n) for n in shape]) / epsilon
    epsilon_prime = epsilon / 8 / costs.max()
    min_cost = costs.min()
    A_stable = np.exp(-eta * (costs - min_cost))
    A = np.exp(-eta * min_cost) * A_stable
    m = [np.zeros(s) for s in shape]
    B = A_stable / np.sum(np.abs(A_stable))
    iter = 0
    
    obj_list = []
    lb_list = []
    eps_list = []
    dis_list = []
    epsp_list = []
    
    ########## helper function ###########
    
    def dist(B):
        return sum(np.sum(np.abs(marginal_k(B, i) - get_marginal_k(target_mu, i, shape))) for i in range(M))
    def stable_update():
        target_marginal_list = []
        gamma_marginal_list = []
        rho_list = []
        for k in range(M):
            exponents = tensor_sum(m) - eta * costs
            idxes = list(range(M))
            idxes.remove(k)
            tmp_max = exponents.max(axis = tuple(idxes))
            tmp_tensor = torch.from_numpy(tmp_max).reshape(*([1]*k + [-1] + [1]*(M - k - 1))).expand(*shape).numpy()
            target_marginal_list.append(np.log(get_marginal_k(target_mu, k, shape)))
            gamma_marginal_list.append(np.log(np.exp(exponents - tmp_tensor).sum(axis = tuple(idxes))) + tmp_max)
            rho_list.append(rho(target_marginal_list[k], gamma_marginal_list[k]))
        k_star = np.argmax(rho_list)
        m[k_star] += target_marginal_list[k_star] - gamma_marginal_list[k_star]
    
    
    ########## training starts ###########
    
    while True:
    # while iter < 500:
        if iter == 0: # update first step
            exponents = tensor_sum(m) - eta * costs
            idxes = list(range(M))
            idxes.remove(1)
            tmp_max = exponents.max(axis = tuple(idxes))
            tmp_tensor = torch.from_numpy(tmp_max).reshape(*([1]*1 + [-1] + [1]*(M - 1 - 1))).expand(*shape).numpy()
            m[1] += np.log(get_marginal_k(target_mu, 1, shape)) - np.log(np.exp(exponents - tmp_tensor).sum(axis = tuple(idxes))) - tmp_max
        else:
            stable_update()
            
        B = np.exp(tensor_sum(m) - eta * costs)
        distance = dist(B)
        # primal objective
        obj = np.sum(projection(B, target_mu) * costs) * cost_scale
        # lower bound
        lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta     
        
    ########## logging results ###########
    
        eps_list.append(epsilon)
        dis_list.append(distance)
        epsp_list.append(epsilon_prime)
        obj_list.append(obj)
        lb_list.append(lb)
        
        if verbose >= 100: 
            print("tensor_sum max", tensor_sum(m).max(), tensor_sum(m).min())
        if ((tensor_sum(m) - eta * costs) > 0).any().any():
            raise Exception("tensor_sum(m) can't be greater than eta * costs")
            # print("tensor_sum(m) - eta * costs", tensor_sum(m) - eta * costs)

        if iter % iter_gap == 0:
            memo = {"m": m, "obj_list":obj_list, "lb_list": lb_list, 
                    "eps_list": eps_list, "dis_list": dis_list, "epsp_list": epsp_list,
                    "iter": iter, "obj": obj, "lb": lb, "dist": distance, "epsp": epsilon_prime,
                    "eps": epsilon, "eta": eta, 
                    "iter_gap": iter_gap, "epsilon_scale_num": epsilon_scale_num, 
                    "epsilon_scale_gap": epsilon_scale_gap, "cost_scale": cost_scale,
                   "data_file": args.data_file, "start_epsilon": args.start_epsilon,
                   "target_epsilon": target_epsilon, "error": None}
            pkl.dump(memo, open( 'log/' + out_dir + '.pkl', 'wb'))
            if verbose >= 2:
                print("iter: ", iter, 
                      "obj: ", round(obj, 6), 
                      "lb: ", round(lb, 6), 
                      "dist: ", round(dist(B), 6),
                      "eps: ", round(epsilon, 10),
                      "eps_prime: ", round(epsilon_prime, 10),
                     )
        
    ########## update parameters ###########
        iter += 1
        if distance < epsilon_prime:
            if abs(obj) > 4 * epsilon:
                break # stop when epsilon is small enough and converges
            else:
                if abs(obj) < 2 * target_epsilon:
                    raise Exception("obj too small. smaller than target 2 * eps: ", 2 * target_epsilon)
                elif  iter >= max_iter:
                    raise Exception("exceed %d iterations" % max_iter)
                else:
                    scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                    epsilon *= scale_factor
                    epsilon_prime *= scale_factor
                    eta /= scale_factor
                    m = [l / scale_factor for l in m]
        else:
            if iter >= max_iter:
                raise Exception("exceed %d iterations" % max_iter)
            elif iter % epsilon_scale_gap == 0:
                scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                epsilon *= scale_factor
                epsilon_prime *= scale_factor
                eta /= scale_factor
                m = [l / scale_factor for l in m]
    
    ########## return final results ###########
    
    B = np.exp(tensor_sum(m) - eta * costs)
    weights = projection(B, target_mu)
    lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta
    return np.sum(weights * costs) * cost_scale, lb, weights


def solve_rrsinkhorn(costs, target_mu, epsilon=1e-2, target_epsilon=1e-4, verbose = 0, epsilon_scale_num = 0.99, epsilon_scale_gap = 100, cost_scale = 1, iter_gap = 100, max_iter = 5000, out_dir = 'test'):
    """solve using Greenkhorn's algorithm

    Args:
        costs (array)
        target_mu (list): target probability
        epsilon (float): precision. Defaults to 1e-2.
        verbose (int, optional): print intermediate results if verbose >= 2. Defaults to 0.
        cost_scale (int, optional): divide the cost matrix by cost_scale before calculating, scale back in the end. Defaults to 1.
        iter_gap (int, optional): print results every `iter_gap` iterations. Defaults to 100.

    Returns:
        _type_: _description_
    """
    ########## initialization ###########
    
    costs /= cost_scale
    shape = costs.shape
    M = len(shape)
    # print("shape: ", shape)
    eta = 4 * sum([log (n) for n in shape]) / epsilon
    epsilon_prime = epsilon / 8 / costs.max()
    min_cost = costs.min()
    A_stable = np.exp(-eta * (costs - min_cost))
    A = np.exp(-eta * min_cost) * A_stable
    m = [np.zeros(s) for s in shape]
    B = A_stable / np.sum(np.abs(A_stable))
    iter = 0
    
    obj_list = []
    lb_list = []
    eps_list = []
    dis_list = []
    epsp_list = []
    
    ########## helper function ###########
    
    def dist(B):
        return sum(np.sum(np.abs(marginal_k(B, i) - get_marginal_k(target_mu, i, shape))) for i in range(M))
    def stable_update():
        for k in range(M):
            exponents = tensor_sum(m) - eta * costs
            idxes = list(range(M))
            idxes.remove(k)
            tmp_max = exponents.max(axis = tuple(idxes))
            tmp_tensor = torch.from_numpy(tmp_max).reshape(*([1]*k + [-1] + [1]*(M - k - 1))).expand(*shape).numpy()
            m[k] += np.log(get_marginal_k(target_mu, k, shape)) - np.log(np.exp(exponents - tmp_tensor).sum(axis = tuple(idxes))) - tmp_max
    
    ########## training starts ###########
    
    while True:
    # while iter < 500:
        stable_update()
        B = np.exp(tensor_sum(m) - eta * costs)
        distance = dist(B)
        # primal objective
        obj = np.sum(projection(B, target_mu) * costs) * cost_scale
        # lower bound
        lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta     
        
    ########## logging results ###########
    
        eps_list.append(epsilon)
        dis_list.append(distance)
        epsp_list.append(epsilon_prime)
        obj_list.append(obj)
        lb_list.append(lb)
        
        if verbose >= 100: 
            print("tensor_sum max", tensor_sum(m).max(), tensor_sum(m).min())
        if ((tensor_sum(m) - eta * costs) > 0).any().any():
            raise Exception("tensor_sum(m) can't be greater than eta * costs")
            # print("tensor_sum(m) - eta * costs", tensor_sum(m) - eta * costs)

        if iter % iter_gap == 0:
            memo = {"m": m, "obj_list":obj_list, "lb_list": lb_list, 
                    "eps_list": eps_list, "dis_list": dis_list, "epsp_list": epsp_list,
                    "iter": iter, "obj": obj, "lb": lb, "dist": distance, "epsp": epsilon_prime,
                    "eps": epsilon, "eta": eta, 
                    "iter_gap": iter_gap, "epsilon_scale_num": epsilon_scale_num, 
                    "epsilon_scale_gap": epsilon_scale_gap, "cost_scale": cost_scale,
                   "data_file": args.data_file, "start_epsilon": args.start_epsilon,
                   "target_epsilon": target_epsilon, "error": None}
            pkl.dump(memo, open( 'log/' + out_dir + '.pkl', 'wb'))
            if verbose >= 2:
                print("iter: ", iter, 
                      "obj: ", round(obj, 6), 
                      "lb: ", round(lb, 6), 
                      "dist: ", round(dist(B), 6),
                      "eps: ", round(epsilon, 10),
                      "eps_prime: ", round(epsilon_prime, 10),
                     )
        
    ########## update parameters ###########
        iter += 1
        if distance < epsilon_prime:
            if abs(obj) > 4 * epsilon:
                break # stop when epsilon is small enough and converges
            else:
                if abs(obj) < 2 * target_epsilon:
                    raise Exception("obj too small. smaller than target 2 * eps: ", 2 * target_epsilon)
                elif  iter >= max_iter:
                    raise Exception("exceed %d iterations" % max_iter)
                else:
                    scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                    epsilon *= scale_factor
                    epsilon_prime *= scale_factor
                    eta /= scale_factor
                    m = [l / scale_factor for l in m]
        else:
            if iter >= max_iter:
                raise Exception("exceed %d iterations" % max_iter)
            elif iter % epsilon_scale_gap == 0:
                scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                epsilon *= scale_factor
                epsilon_prime *= scale_factor
                eta /= scale_factor
                m = [l / scale_factor for l in m]
    
    ########## return final results ###########
    
    B = np.exp(tensor_sum(m) - eta * costs)
    weights = projection(B, target_mu)
    lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta
    return np.sum(weights * costs) * cost_scale, lb, weights

def Rho(a, b):
    return b - a + a * np.log(a / b)

def solve_greenkhorn(costs, target_mu, epsilon=1e-2, target_epsilon=1e-4, verbose = 0, epsilon_scale_num = 0.99, epsilon_scale_gap = 100, cost_scale = 1, iter_gap = 100, max_iter = 5000, out_dir = 'test'):
    costs /= cost_scale
    shape = costs.shape
    epsilon = start_epsilon
    eta = 4 * sum([log(n) for n in shape]) / epsilon
    epsilon_prime = epsilon / 8 / costs.max()
    min_cost = costs.min()
    A_stable = np.exp(-eta * (costs - min_cost))
    A = np.exp(-eta * min_cost) * A_stable
    B = A_stable / np.sum(np.abs(A_stable))
    m = [np.zeros(s) for s in shape]
    iter = 0
    
    obj_list = []
    lb_list = []
    eps_list = []
    dis_list = []
    epsp_list = []
    
    def dist(B):
        return sum(np.sum(np.abs(marginal_k(B, i) - get_marginal_k(target_mu, i, shape))) for i in range(B.ndim))
        
    while True:
        max_v = []
        buffer = []
        for k in range(B.ndim):
            marginal = marginal_k(B, k)
            tmp = Rho(marginal, get_marginal_k(target_mu, k, shape))
            im = np.argmax(tmp)
            max_v.append(tmp[im])
            buffer.append((marginal[im], im))
        kmax = np.argmax(max_v)
        rki, imax = buffer[kmax]
        m[kmax][imax] += np.log(target_mu[ravel_index(kmax, imax, shape)] / rki)
        B = np.exp(tensor_sum(m) - eta * costs)
        
        distance = dist(B)
        # primal objective
        obj = np.sum(projection(B, target_mu) * costs) * cost_scale
        # lower bound
        lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta     
        
    ########## logging results ###########
    
        eps_list.append(epsilon)
        dis_list.append(distance)
        epsp_list.append(epsilon_prime)
        obj_list.append(obj)
        lb_list.append(lb)
        
        if iter % iter_gap == 0:
            memo = {"m": m, "obj_list":obj_list, "lb_list": lb_list, 
                    "eps_list": eps_list, "dis_list": dis_list, "epsp_list": epsp_list,
                    "iter": iter, "obj": obj, "lb": lb, "dist": distance, "epsp": epsilon_prime,
                    "eps": epsilon, "eta": eta, 
                    "iter_gap": iter_gap, "epsilon_scale_num": epsilon_scale_num, 
                    "epsilon_scale_gap": epsilon_scale_gap, "cost_scale": cost_scale,
                   "data_file": args.data_file, "start_epsilon": args.start_epsilon,
                   "target_epsilon": target_epsilon, "error": None}
            pkl.dump(memo, open( 'log/' + out_dir + '.pkl', 'wb'))
            if verbose >= 2:
                print("iter: ", iter, 
                      "obj: ", round(obj, 6), 
                      "lb: ", round(lb, 6), 
                      "dist: ", round(dist(B), 6),
                      "eps: ", round(epsilon, 10),
                      "eps_prime: ", round(epsilon_prime, 10),
                     )
        
        
    ########## update parameters ###########
        iter += 1
        if distance < epsilon_prime:
            if abs(obj) > 4 * epsilon:
                break # stop when epsilon is small enough and converges
            else:
                if abs(obj) < 2 * target_epsilon:
                    raise Exception("obj too small. smaller than target 2 * eps: ", 2 * target_epsilon)
                elif  iter >= max_iter:
                    raise Exception("exceed %d iterations" % max_iter)
                else:
                    scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                    epsilon *= scale_factor
                    epsilon_prime *= scale_factor
                    eta /= scale_factor
                    m = [l / scale_factor for l in m]
        else:
            if iter >= max_iter:
                raise Exception("exceed %d iterations" % max_iter)
            elif iter % epsilon_scale_gap == 0:
                scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                epsilon *= scale_factor
                epsilon_prime *= scale_factor
                eta /= scale_factor
                m = [l / scale_factor for l in m]
    
    ########## return final results ###########
    
    B = np.exp(tensor_sum(m) - eta * costs)
    weights = projection(B, target_mu)
    lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta
    return np.sum(weights * costs) * cost_scale, lb, weights


def solve_aam(costs, target_mu, epsilon = 1e-2, verbose = 0, print_itr = 0, max_iterate = 10, method = "binary_search"):
    
    print("start solving AAM", "Current Time:", datetime.datetime.now())
    
    shape = costs.shape
    m, n = len(shape), max(shape) # (n = (5, 5, 8, 12))
    MAX_DIGIT = 50
    C_norm = costs.max().max()
    gamma = epsilon / (2 * m * np.log(n))
    epsilon_p = epsilon / (8 * C_norm)
    p = np.array(target_mu)
    uniform = np.array([1/n for n in shape for _ in range(n)])
    p_tilde = (1 - epsilon_p/(4 * m)) * np.array(target_mu) + epsilon_p / (4 * m) * uniform
    # if verbose >= 2 and itr > print_itr: print("p", p, "p_tilde", p_tilde)
    def marginal_k(X, k):
        # marginalize tensor X over dimension k
        return np.sum(X, axis=tuple(axis for axis in range(X.ndim) if axis != k))
    def get_marginal_k(p, k):
        return p[ravel_index(k, 0, shape): ravel_index(k, shape[k], shape)]
    def l1_error(X, p):
        # compute l1 error of X from the target marginal
        return np.sum(np.abs(marginal_k(X, k) - get_marginal_k(p, k)).sum() for k in range(X.ndim)) 
    def F(X):
        # if verbose >= 2 and itr > print_itr: print("F:", (X * np.log(X)).sum().sum())
        return (costs * X).sum().sum() - gamma * (X * np.log(X)).sum().sum()
    def B(U):
        U_tmp = np.zeros(costs.shape)
        for idx in product(*map(range, costs.shape)):
            U_tmp[idx] = np.exp(sum(U[ravel_index(k, i, shape)] for k, i in enumerate(idx)) - costs[idx] / gamma)
        return U_tmp
    def log_B(U):
        U_tmp = np.zeros(costs.shape)
        for idx in product(*map(range, costs.shape)):
            U_tmp[idx] = sum(U[ravel_index(k, i, shape)] for k, i in enumerate(idx)) - costs[idx] / gamma
        max_U_tmp = U_tmp.max().max()
        U_tmp = U_tmp * (U_tmp >= max_U_tmp - MAX_DIGIT)
        min_U_tmp = U_tmp.min().min()
        return min_U_tmp + np.log(np.exp(U_tmp - min_U_tmp).sum().sum())
    def B_1(U):
        U_tmp = np.zeros(costs.shape)
        for idx in product(*map(range, costs.shape)):
            U_tmp[idx] = np.exp(sum(U[ravel_index(k, i, shape)] for k, i in enumerate(idx)) - costs[idx] / gamma + 1)
            if U_tmp[idx] == 0:
                if verbose >= 2 and itr > print_itr: print("U_tmp[idx] == 0", U_tmp[idx], sum(U[ravel_index(k, i, shape)] for k, i in enumerate(idx)), - costs[idx] / gamma + 1)
        return U_tmp
    def psi(U, p):
        return gamma * (log_B(U) - U @ p)
    def primal(U):
        L_tmp = B_1(U)
        L_norm = L_tmp.sum().sum()
        return L_tmp / L_norm
    def g(theta, p):
        U_tmp = B(theta)
        return np.concatenate([marginal_k(U_tmp, k) / U_tmp.sum().sum() - get_marginal_k(p, k) for k in range(m)])
    def g_l2_norm(theta, p):
        U_tmp = B(theta)
        return np.sum([np.square(marginal_k(U_tmp, k) / U_tmp.sum().sum() - get_marginal_k(p, k)).sum() for k in range(m)])
    def g_l2_norm_k(k, theta, p):
        U_tmp = B(theta)
        return np.square(marginal_k(U_tmp, k) / U_tmp.sum().sum() - get_marginal_k(p, k)).sum()
    def update_eta(theta, i): # lemma 2 iteration step
        U_tmp = B(theta)
        eta = theta.copy()
        eta_update = get_marginal_k(theta, i) + np.log(get_marginal_k(p, i)) - np.log(marginal_k(U_tmp, i))
        eta[ravel_index(i, 0, shape): ravel_index(i, shape[i], shape)] = eta_update
        return eta
    def larger_root_qr(a, b, c):
        # calculating  the discriminant
        delta = (b**2) - (4 * a*c)
        # find two results
        assert delta >= 0
        ans1 = (-b - sqrt(delta))/(2 * a)
        ans2 = (-b + sqrt(delta))/(2 * a)
        return max(ans1, ans2)
    def projection(X, p):
        V = X.copy()
        for r in range(m-1):
            X_r = np.minimum(get_marginal_k(p, r) / marginal_k(V, r), 1)
            for idx in product(*map(range, shape)):
                V[idx] = V[idx] * X_r[idx[r]] # TODO: optimize this
        err_list = []
        for r in range(m):
            err_list.append(get_marginal_k(p, r) - marginal_k(V, r))
        # outer product of err_list
        V += reduce(np.multiply, np.ix_(*err_list)) / (np.abs(err_list[-1]).sum() ** (m-1))
        return V
    
    A = 0
    a = 0
    eta = np.zeros(len(p))
    zeta = np.zeros(len(p))
    theta = np.zeros(len(p))
    x0 = 0.1

    X = np.ones(costs.shape)

    obj_list = []
    condition_list = []
    beta_list = []
    itr = 0
    
    while 2 * l1_error(X, p_tilde) + F(X) + psi(eta, p) > epsilon / 2:
        itr += 1
        if itr > max_iterate:
            if verbose >= 2 and itr > print_itr: print("exit %d iterations " % (max_iterate) + "!" * 80, 2 * l1_error(X, p_tilde) + F(X) + psi(eta, p), epsilon / 2)
            break
    
        # apply PD-AAM to the dual problem
        def f(beta):
            return psi(eta + beta * (zeta - eta), p_tilde)
        # find the minimum of f(beta)
        # add constraints that beta is in [0, 1]
        if method == "minimize":
            # method 1
            const = ({'type': 'ineq', 'fun': lambda beta: beta}, {'type': 'ineq', 'fun': lambda beta: 1 - beta})
            beta = max(min(optimize.minimize(f, x0 = x0, constraints=const).x[0], 1), 0)
        elif method == "binary_search":
            # method 2
            beta = binary_search(0, 1, f)
        else:
            # method 3
            beta = optimize.minimize_scalar(f, bounds=(0, 1), method = 'bounded').x
        
        
        beta_list.append(beta)
        theta = eta + beta * (zeta - eta)
        if verbose >= 2 and itr > print_itr: print("beta, eta, zeta", beta, eta, zeta)

        # pick margin
        i = np.argmax([g_l2_norm_k(k, theta, p_tilde) for k in range(m)])
        
        eta = update_eta(theta, i)
        
        tmp = (psi(theta, p_tilde) - psi(eta, p_tilde)) / g_l2_norm(theta, p_tilde)
        if tmp < 0: 
            if verbose >= 2 and itr > print_itr: print("early stopping " + "!" * 80, tmp)
            a = 0
            # break
        else: a = larger_root_qr(1, - 2 * tmp, - 2 * tmp * A)
        A_t = A
        A += a
        zeta = zeta - a * g(theta, p_tilde)
        
        X = (a * primal(theta) + A_t * X) / A
        
        obj = (costs * X).sum().sum()
        
        obj_list.append(obj)
        condition_list.append(2 * l1_error(X, p_tilde) + F(X) + psi(eta, p))
    weight = projection(X, p)
    obj = (costs * weight).sum().sum()
    return obj, -float('inf'), weight

def get_res_single(tmp_list, solver = solve_sinkhorn, return_res = False, print_res = True, cost_type='square'):
    if cost_type == 'cov':
        cost_fn = create_cov_cost_tensor
    elif cost_type == 'normed_square':
        cost_fn = create_normed_cost_tensor
    else:
        cost_fn = create_cost_tensor
    costs, target_mu = cost_fn(tmp_list)
    lb = np.sum([np.mean([np.mean(t) for t in tmp])**2 for tmp in tmp_list])
    try:
        dis, dlb, weight = solver(costs, target_mu)
    except Exception as error:
        print("An exception occurred:", error)
    if print_res: print("shape:", weight.shape)
    if print_res: print("mean sq:", lb, "primal obj value: ", dis, "dual lb: ", dlb)
    if return_res: return dis, dlb, weight

if __name__ == "__main__":
    
    args = parse_args()
    out_dir = "solver-{:}data_file-{:}cost-{:}start_eps-{:}tgt_eps-{:}eps_scale-{:}scale_gap-{:}max_iter-{:}".format(args.solver, args.data_file, args.cost_type, args.start_epsilon, args.target_epsilon, args.epsilon_scale_num, args.epsilon_scale_gap, args.max_iter)
    print(args)
    tmp_list = pkl.load(open('data/' + args.data_file + ".pkl", "rb"))
    tmp_list = [[l.astype(np.float64) for l in tmp_l] for tmp_l in tmp_list]
    tmp_list = [[l - l.mean() for l in tmp_l] for tmp_l in tmp_list]
    tic = time.perf_counter()
    
    if args.solver == 'aam':
        lb, dis, weight = get_res_single(tmp_list, 
                                 solver = solve_aam(a, b, epsilon = 0.01), return_res = True, cost_type = args.cost_type)
    else:
        if args.solver == "sinkhorn":
            solver_fn = solve_sinkhorn
        elif args.solver == "rrsinkhorn":
            solver_fn = solve_rrsinkhorn
        elif args.solver == "greenkhorn":
            solver_fn = solve_greenkhorn
        else:
            raise ValueError("Solver not found")
        lb, dis, weight = get_res_single(tmp_list, 
                                 solver = lambda a, b: solver_fn(a, b,
                                                                      epsilon=args.start_epsilon, 
                                                                      target_epsilon = args.target_epsilon, 
                                                                      verbose = args.verbose, 
                                                                      epsilon_scale_num = args.epsilon_scale_num, 
                                                                      epsilon_scale_gap = args.epsilon_scale_gap, 
                                                                      cost_scale = args.cost_scale, 
                                                                      iter_gap = args.iter_gap, 
                                                                      max_iter = args.max_iter,
                                                                      out_dir = out_dir), 
                                 return_res = True, cost_type = args.cost_type)
    
    toc = time.perf_counter()
    print(f"single run: {toc - tic} seconds")