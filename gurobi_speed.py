import random as rd
import gurobipy as gp
from gurobipy import GRB, GurobiError
import numpy as np
import time
from ufp_optuna import problem_grouping, data_preprocess
from ufp_optuna import KnapsackPredictor
import os
import torch
import math
from inference import selection_algorithm_count_util_all


def problem_constructor(filename, num_of_clients, arrival_time='uniform', end_time='uniform',
                        util_demand_relation='linear', dim_size=3, time_length=96):
    # arrival_time--uniform,poisson
    # end_time--uniform,gaussian
    # relation--random,quadratic,linear
    size = dim_size + time_length + 4  # demands_per_dim,time_s,time_e,time_interval,utility,index
    lin_comb = np.random.rand(dim_size) * 3 + 1
    clients = np.zeros((num_of_clients, size))

    # Create client instances
    for index in range(num_of_clients):
        clients[index][-1] = index
        for dim in range(dim_size):
            # determine demand of the client for each dimension
            clients[index][dim] = rd.uniform(10, 100)
        # start time
        if arrival_time == 'uniform':
            s_time = rd.randint(1, int(0.75 * time_length))  # uniformly random
        else:
            lam = 10
            s_time = np.random.poisson(lam)
            s_time = int(max(1, min(s_time, time_length)))  # starting time poisson
        # end time
        if end_time == 'uniform':
            median = (time_length + s_time) // 2
            e_time = rd.randint(median, time_length + 1)  # uniform endtime
        else:
            mean = s_time + (time_length + 1 - s_time) // 2
            deviation = (time_length + 1 - mean) // 3
            e_time = np.random.normal(mean, deviation)
            e_time = int(max(s_time + 1, min(time_length + 1, e_time)))  # gaussian endtime
        clients[index][dim_size] = s_time
        clients[index][dim_size + 1] = e_time  # end_time excluded
        clients[index][dim_size + 1 + s_time:dim_size + 1 + e_time] = np.ones(e_time - s_time)
        if util_demand_relation == 'linear':
            clients[index][-2] = np.dot(lin_comb, clients[index][:dim_size])  # utility
        elif util_demand_relation == 'quadratic':
            clients[index][-2] = np.dot(lin_comb, clients[index][:dim_size] ** 2)
        else:
            clients[index][-2] = rd.uniform(1, 100)
    total_demand_per_dim = clients[:, :dim_size].sum(axis=0)
    max_cap_per_dim = (0.7 * total_demand_per_dim)
    min_cap_per_dim = (0.3 * max_cap_per_dim)
    time_slot_capacity = np.zeros((dim_size, time_length))
    for dim in range(dim_size):
        time_slot_capacity[dim][0] = rd.uniform(min_cap_per_dim[dim], max_cap_per_dim[dim])
        for step in range(1, time_length):
            change = rd.randint(0, 1)
            if change:
                time_slot_capacity[dim][step] = rd.uniform(min_cap_per_dim[dim], max_cap_per_dim[dim])
            else:
                time_slot_capacity[dim][step] = time_slot_capacity[dim][step - 1]
    B_constraint = 0.9 * time_slot_capacity.mean()

    ufp_model = gp.Model()
    opt_choice = ufp_model.addVars(num_of_clients, vtype=GRB.BINARY, name="opt_choice")
    constr_1 = ufp_model.addConstrs((gp.quicksum([opt_choice[index] * clients[index][dim] *
                                                  clients[index][dim_size + 1 + time]
                                                  for index in range(num_of_clients)]) <= time_slot_capacity[dim][
                                         time - 1]
                                     for dim in range(dim_size) for time in range(1, time_length + 1)), name="c")
    constr_2 = ufp_model.addConstrs((gp.quicksum([gp.quicksum([opt_choice[index] * clients[index][dim] *
                                                               clients[index][dim_size + 1 + time]
                                                               for index in range(num_of_clients)]) ** 2 for dim in
                                                  range(dim_size)]) <= B_constraint ** 2 for time in
                                     range(1, time_length + 1)), name='b')

    ufp_model.setObjective(gp.quicksum([opt_choice[i] * clients[i][-2] for i in range(num_of_clients)]),
                           GRB.MAXIMIZE)
    np.savez(f'{filename}', cl=clients, tslot=time_slot_capacity, quad_constr=B_constraint)
    ufp_model.write(f"model_{filename}.lp")


def problem_reader(filename):
    data = np.load(f'{filename}.npz')
    clients = data['cl']
    time_slot_capacity = data['tslot']
    B_constraint = data['quad_constr']
    data.close()
    ufp_model = gp.read(f"model_{filename}.lp")

    return clients, time_slot_capacity, B_constraint, ufp_model


def problem_solver(ufp_model, initial_solution=None):
    size = initial_solution.shape[0]
    ufp_model.setParam('MIPGap', 1e-3)
    ufp_model.setParam('IntFeasTol', 1e-6)
    if initial_solution is not None:
        variables = ufp_model.getVars()
        for index in range(len(variables)):
            variables[index].start = initial_solution[index]
    try:
        c = time.time()
        ufp_model.optimize()
        print(time.time() - c)
    except GurobiError as e:
        if 'Model too large for size-limited license' in str(e):
            print(f"Skipping problem due to license size limitation.")
            return np.zeros(size), 0
        print(f"Error encountered for problem: {e}")
        return np.zeros(size), 0
    answer = np.array([item.x for item in ufp_model.getVars()])

    return answer, ufp_model.ObjVal


def selection_algorithm_count_util_all_infeasible(groups, tot_util, y_pred, num_of_clients=1500, ):
    if type(y_pred) != np.ndarray:
        prediction = y_pred.cpu().detach().numpy()
    else:
        prediction = y_pred

    sum_utils = prediction[:, 2] * tot_util
    final_total_utility = 0
    final_selected_items = []
    for _ in range(10):
        total_utility = 0
        selected_items = []
        for i, group in enumerate(groups):
            if group.size > 0:
                max_res = 0
                best_vec = np.array([])
                prob_vec_uniform = np.ones(group.shape[0]) / group.shape[0]
                for id in range(10):
                    sel_size = min(math.ceil(prediction[i][0] * num_of_clients), group.shape[0])
                    selection = np.random.choice(range(group.shape[0]), sel_size, replace=False, p=prob_vec_uniform)
                    sel_vec = group[selection]
                    util_sum = np.sum(sel_vec[:, -2])
                    if util_sum > max_res and np.abs(util_sum - sum_utils[i]) <= 0.1 * sum_utils[i]:  # 0.05
                        max_res = util_sum
                        best_vec = sel_vec
                if best_vec.size > 0:
                    total_utility += max_res
                    selected_items.extend(best_vec)

            if final_total_utility < total_utility:
                final_selected_items = selected_items.copy()
                final_total_utility = total_utility
    if len(final_selected_items) > 0:
        return np.stack(final_selected_items), final_total_utility
    else:
        return [], 0


if __name__ == '__main__':
    problem_size = 1500
    # Need to be executed only once
    problem_constructor('test_example', problem_size, arrival_time='uniform', end_time='uniform',
                        util_demand_relation='quadratic')
    ###################################################################################
    CHECKPOINT_PATH = "Checkpoint Path"
    root_dir = os.path.join(CHECKPOINT_PATH, "UfpCheckPoint")
    pretrained_filename = os.path.join(root_dir, "UfpCheckPoint.ckpt")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    knapsack_model = KnapsackPredictor.load_from_checkpoint(pretrained_filename)
    knapsack_model = knapsack_model.to(device)
    ###################################################################################
    clients, time_slot_capacity, quad_constr, ufp_model = problem_reader('test_example')
    f_groups, tot_dem, mi_av_max_caps, av_cap_time, q_c, mi_util, ma_util, tot_util, max_cap, clients = problem_grouping(
        clients,
        time_slot_capacity,
        quad_constr)
    inp, _ = data_preprocess(f_groups, tot_dem, q_c, mi_av_max_caps, av_cap_time,
                             mi_util, ma_util, tot_util)
    inp = torch.from_numpy(np.float32(inp[None, :])).to(device)
    y_pred = knapsack_model(inp)
    y_pred = y_pred.cpu().detach().numpy().squeeze()

    selected_items_feasible, _ = selection_algorithm_count_util_all(time_slot_capacity, quad_constr,
                                                                    f_groups,
                                                                    tot_util,
                                                                    y_pred, num_of_clients=problem_size, attempts=100)
    selected_items_infeasible, _ = selection_algorithm_count_util_all_infeasible(f_groups, tot_util, y_pred,
                                                                                 num_of_clients=problem_size)

    if len(selected_items_feasible) != 0:
        selected_indices = selected_items_feasible[:, -1].astype(np.int32)

        initial_solution = np.zeros(problem_size, dtype=np.float32)
        initial_solution[selected_indices] = 1.0

        # With initial feasible solution
        s_time = time.time()
        ans_1, obj_1 = problem_solver(ufp_model, initial_solution)
        run_time = time.time() - s_time
        print('With initial feasible solution:')
        print(f'Time : {run_time},Objective : {obj_1}')
        # Without initial solution
        s_time = time.time()
        ans_2, obj_2 = problem_solver(ufp_model)
        run_time = time.time() - s_time
        print('Without initial solution:')
        print(f'Time : {run_time},Objective : {obj_2}')
        # With initial infeasible solution
        selected_indices = selected_items_infeasible[:, -1].astype(np.int32)
        initial_solution = np.zeros(problem_size, dtype=np.float32)
        initial_solution[selected_indices] = 1.0
        s_time = time.time()
        ans_3, obj_3 = problem_solver(ufp_model, initial_solution)
        run_time = time.time() - s_time
        print('With initial infeasible solution:')
        print(f'Time : {run_time},Objective : {obj_3}')
