import numpy as np
from functools import reduce
from math import log, sqrt
import matplotlib.pyplot as plt
import pickle as pkl
import time
import argparse
import torch
from mpmath import mp
mp.dps = 10000


def parse_args():
    parser = argparse.ArgumentParser(description='Run MOT Solver')
    # parser.add_argument('--log_name', type=str, default='convcode_initialization_freeze_embed_freeze_revert', help='Log name')
    parser.add_argument('--target_epsilon', type=float, default=1e-3, help='target epsilon')
    parser.add_argument('--start_epsilon', type=float, default=1, help='start epsilon')
    # parser.add_argument('--epsilon_scale', type=str, default='fixed', help='epsilon scaling method')
    parser.add_argument('--epsilon_scale_num', type=float, default=0.99, help='epsilon_scale_num')
    parser.add_argument('--epsilon_scale_gap', type=float, default=100, help='epsilon_scale_gap')
    # parser.add_argument('--epsilon_decay_factor', type=float, default=2, help='epsilon_decay_factor')
    parser.add_argument('--cost_type', type=str, default='square', help='type of cost')
    parser.add_argument('--verbose', type=int, default=2, help='verbose')
    parser.add_argument('--cost_scale', type=float, default=1, help='cost_scale')
    parser.add_argument('--max_iter', type=int, default=5000, help='max iter')
    parser.add_argument('--iter_gap', type=int, default=100, help='iter_gap')
    parser.add_argument('--out_dir', type=str, default='test_out', help='output directory')
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
        # tensor += np.transpose(np.tensordot(l, np.ones(shape[idx+1:] + shape[:idx]), axes=0), axes = rotate(idxes, -idx))
    return tensor

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
    
    if args.solver == "sinkhorn":
        lb, dis, weight = get_res_single(tmp_list, 
                                     solver = lambda a, b: solve_sinkhorn(a, b,
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
    elif args.solver == "rrsinkhorn":
        lb, dis, weight = get_res_single(tmp_list, 
                                     solver = lambda a, b: solve_rrsinkhorn(a, b,
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
    # elif args.solver == "greenkhorn":
    #     lb, dis, weight = get_res_single(tmp_list, solver = lambda a, b: solve_greenkhorn(a, b, epsilon=args.target_epsilon, verbose = args.verbose, cost_scale = args.cost_scale, iter_gap = args.iter_gap), return_res = True)
    # elif args.solver == "greenkhorn_decay":
    #     lb, dis, weight = get_res_single(tmp_list, solver = lambda a, b: solve_greenkhorn_decay(a, b, verbose = args.verbose, cost_scale = args.cost_scale, iter_gap = args.iter_gap, start_epsilon = args.start_epsilon, epsilon_decay_factor = args.epsilon_decay_factor, epsilon_target = args.target_epsilon), return_res = True)
    
    toc = time.perf_counter()
    print(f"single run: {toc - tic} seconds")