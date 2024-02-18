import random as rd
import gc
from scipy.stats import beta
import gurobipy as gp
from gurobipy import GRB, GurobiError
import numpy as np
import os
import time
from multiprocessing import Pool


def problem_constructor_time(filename, num_of_clients, arrival_time='uniform', end_time='uniform',
                             util_demand_relation='linear', dim_size=3, time_length=96):
    # arrival_time--uniform,poisson
    # end_time--uniform,gaussian
    # relation--random,quadratic,linear
    size = dim_size + time_length + 4  # demands_per_dim,time_s,time_e,time_interval,utility,chosen or not
    lin_comb = np.random.rand(dim_size) * 3 + 1
    clients = np.zeros((num_of_clients, size))
    # Create client instances
    for index in range(num_of_clients):
        for dim in range(dim_size):
            # determine demand of the client for each dimension
            clients[index][dim] = rd.uniform(10, 100)
        # start time
        large_interval = rd.random() > 0.95
        if arrival_time == 'uniform':
            if large_interval:
                s_time = rd.randint(1, 16)
            else:
                s_time = rd.randint(1, time_length - 6)
        else:
            lam = 10
            if large_interval:
                s_time = np.random.poisson(lam) % 16 + 1
            else:
                s_time = np.random.poisson(lam) % (time_length - 6) + 1
        # end time
        if end_time == 'uniform':
            if large_interval:
                e_time = rd.randint(s_time + 60, time_length + 1)
            else:
                e_time = rd.randint(s_time + 1, s_time + 7)
        else:
            if large_interval:
                mean = ((time_length + 1) + (s_time + 60)) // 2
                deviation = (time_length + 1 - mean) // 3
            else:
                mean = s_time + 3
                deviation = 2
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
    if not os.path.exists(f'states_time'):
        os.makedirs(f'states_time')
    np.savez(f'states_time//{filename}_{util_demand_relation}', cl=clients, tslot=time_slot_capacity,
             quad_constr=B_constraint)
    ufp_model.write(f"states_time//model_{filename}_{util_demand_relation}.lp")
    del clients
    del time_slot_capacity
    del constr_1
    del constr_2
    del ufp_model
    gc.collect()


def problem_constructor_demand(filename, num_of_clients, arrival_time='uniform', end_time='uniform',
                               util_demand_relation='linear', dim_size=3, time_length=96):
    # arrival_time--uniform,poisson
    # end_time--uniform,gaussian
    # relation--random,quadratic,linear
    size = dim_size + time_length + 4  # demands_per_dim,time_s,time_e,time_interval,utility,chosen or not
    lin_comb = np.random.rand(dim_size) * 3 + 1
    clients = np.zeros((num_of_clients, size))

    # Create client instances
    for index in range(num_of_clients):
        big_demand = rd.random() > 0.95
        if big_demand:
            for dim in range(dim_size):
                clients[index][dim] = rd.uniform(95, 100)
        else:
            for dim in range(dim_size):
                clients[index][dim] = rd.uniform(10, 15)
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
    if not os.path.exists(f'states_demand'):
        os.makedirs(f'states_demand')
    np.savez(f'states_demand//{filename}_{util_demand_relation}', cl=clients, tslot=time_slot_capacity,
             quad_constr=B_constraint)
    ufp_model.write(f"states_demand//model_{filename}_{util_demand_relation}.lp")
    del clients
    del time_slot_capacity
    del constr_1
    del constr_2
    del ufp_model
    gc.collect()


def problem_constructor_constraint(filename, num_of_clients, arrival_time='uniform', end_time='uniform',
                                   util_demand_relation='linear', dim_size=3, time_length=96):
    # arrival_time--uniform,poisson
    # end_time--uniform,gaussian
    # relation--random,quadratic,linear
    size = dim_size + time_length + 4  # demands_per_dim,time_s,time_e,time_interval,utility,chosen or not
    lin_comb = np.random.rand(dim_size) * 3 + 1
    clients = np.zeros((num_of_clients, size))

    # Create client instances
    for index in range(num_of_clients):
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
    B_constraint = 0.5 * time_slot_capacity.mean()

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
    if not os.path.exists(f'states_constraint'):
        os.makedirs(f'states_constraint')
    np.savez(f'states_constraint//{filename}_{util_demand_relation}', cl=clients, tslot=time_slot_capacity,
             quad_constr=B_constraint)
    ufp_model.write(f"states_constraint//model_{filename}_{util_demand_relation}.lp")
    del clients
    del time_slot_capacity
    del constr_1
    del constr_2
    del ufp_model
    gc.collect()


def __bimodal_distribution__(mean_1=20, var_1=15, mean_2=70, var_2=15, probability=0.5):
    if rd.random() > probability:
        return np.random.normal(mean_1, var_1)

    return np.random.normal(mean_2, var_2)


def problem_constructor_arrival(filename, num_of_clients, arrival_time='bimodal', end_time='uniform',
                                util_demand_relation='linear', dim_size=3, time_length=96):
    # arrival_time--bimodal,beta
    # end_time--uniform,gaussian
    # relation--random,quadratic,linear
    size = dim_size + time_length + 4  # demands_per_dim,time_s,time_e,time_interval,utility,chosen or not
    lin_comb = np.random.rand(dim_size) * 3 + 1
    clients = np.zeros((num_of_clients, size))

    # Create client instances
    for index in range(num_of_clients):
        for dim in range(dim_size):
            # determine demand of the client for each dimension
            clients[index][dim] = rd.uniform(10, 100)
        # start time
        if arrival_time == 'bimodal':
            s_time = __bimodal_distribution__()
            s_time = int(max(1, min(s_time, time_length)))
        elif arrival_time == 'beta':
            s_time = 50 + beta.rvs(1 / 2, 1 / 2) * 30
            s_time = int(max(1, min(s_time, time_length)))
        else:
            raise ValueError("Arrival time must be bimodal or beta")
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
    if not os.path.exists(f'states_arrival'):
        os.makedirs(f'states_arrival')
    np.savez(f'states_arrival//{filename}_{util_demand_relation}', cl=clients, tslot=time_slot_capacity,
             quad_constr=B_constraint)
    ufp_model.write(f"states_arrival//model_{filename}_{util_demand_relation}.lp")
    del clients
    del time_slot_capacity
    del constr_1
    del constr_2
    del ufp_model
    gc.collect()


def problem_solver(filename, type, utility):
    """
    :param filename: name of the file
    :param type: type of the problem,can be 'time','demand','constraint','arrival'
    :param utility: how utility is depends on demand,can be 'linear','random','quadratic'
    """

    data = np.load(f'states_{type}//{filename}_{utility}.npz')
    clients = data['cl']
    time_slot_capacity = data['tslot']
    B_constraint = data['quad_constr']
    data.close()
    ufp_model = gp.read(f"states_{type}//model_{filename}_{utility}.lp")
    try:
        os.remove(f"states_{type}//model_{filename}_{utility}.lp")
        os.remove(f'states_{type}//{filename}_{utility}.npz')
    except Exception as e:
        print(f"Error deleting file {filename}: {e}")
    ufp_model.setParam('TimeLimit', 200)
    ufp_model.setParam('MIPGap', 1e-3)

    ufp_model.setParam('IntFeasTol', 1e-6)
    try:
        c = time.time()
        ufp_model.optimize()
        print(time.time() - c)
    except GurobiError as e:
        if 'Model too large for size-limited license' in str(e):
            print(f"Skipping problem due to license size limitation.")
            return
        print(f"Error encountered for problem: {e}")
        return
    if ufp_model.Status == GRB.Status.TIME_LIMIT:
        print(f"Problem  terminated due to reaching the time limit.")
        return
    answer = np.array([item.x for item in ufp_model.getVars()])
    clients[:, -1] = answer
    if not os.path.exists(f'final_states_{type}'):
        os.makedirs(f'final_states_{type}')
    if not os.path.exists(f'answers_{type}'):
        os.makedirs(f'answers_{type}')
    np.savez(f'final_states_{type}//{filename}_{utility}', cl=clients, tslot=time_slot_capacity,
             quad_constr=B_constraint)
    np.save(f'answers_{type}//{filename}_{utility}', answer)


def construct(problem_number, problem_type, utility_type):
    if problem_type == 'arrival':
        index1 = rd.randint(2, 3)
    else:
        index1 = rd.randint(0, 1)
    index2 = rd.randint(0, 1)
    ar_t = parameters['arrival_time'][index1]
    en_t = parameters['end_time'][index2]
    print(f'__________{problem_number}__________{ar_t}_{en_t}_{utility_type}')
    function = globals()[f'problem_constructor_{problem_type}']
    function(f'{problem_number}', 1500, ar_t, en_t, utility_type)


def problem_grouping(clients_array, time_slot, quad_constr, util_rate=1.5,
                     time_len=16, num_of_util_groups=15,  # 24
                     num_of_demand_groups=16,  # 7-8
                     max_quadratic_demand=174,
                     max_utility=400,
                     utility_type='linear'):
    """
    Group by utility,then demand,then by time_slot_length,and normalize
    """
    clients = clients_array.copy()
    num_of_dims = time_slot.shape[0]
    # Group by utilities
    if utility_type == 'linear':
        clients[:, -2] /= 10
    elif utility_type == 'quadratic':
        clients[:, -2] /= 300
    util_groups = []
    steps = np.zeros(num_of_util_groups)
    steps[0] = util_rate
    for i in range(1, steps.size):
        steps[i] = steps[i - 1] * util_rate
    bool_idx = (clients[:, -2] <= steps[0])
    util_groups.append(clients[np.where(bool_idx)[0]])
    for i in range(1, num_of_util_groups):
        bool_idx = (clients[:, -2] <= steps[i]) & (
                clients[:, -2] > steps[i - 1])
        util_groups.append(clients[np.where(bool_idx)[0]])
    # Normalizing by max_utility
    clients[:, -2] /= max_utility
    max_util = clients[:, -2].max()
    min_util = clients[:, -2].min()
    total_utility = clients[:, -2].sum()
    for group in util_groups:
        if group.size > 0:
            group[:, -2] /= max_utility
    # Group by demand of 4th - quadratic dimension
    demand_groups = []
    step = max_quadratic_demand / num_of_demand_groups
    demand_steps = np.arange(0, max_quadratic_demand + 1, step)
    for i in range(num_of_util_groups):
        if util_groups[i].size > 0:
            group_demands = [np.sqrt((client[:num_of_dims] ** 2).sum()) for client in util_groups[i]]
            for j in range(num_of_demand_groups):
                bool_idx = (group_demands > demand_steps[j]) & (group_demands <= demand_steps[j + 1])
                demand_groups.append(util_groups[i][np.where(bool_idx)[0]])
        else:
            for _ in range(num_of_demand_groups):
                demand_groups.append(util_groups[i])

    # Normalizing demands in each dimension by max capacity
    max_cap = time_slot.max()
    for group in demand_groups:
        group[:, :num_of_dims] /= max_cap
    # sum of demands of the first dimension normalized by max capacity
    total_demand = 0
    for group in demand_groups:
        total_demand += group[:, 0].sum()
    # Normalizing capacities by max capacity
    time_slot_normalized = time_slot / max_cap
    time_slot_min = time_slot_normalized.min(axis=1)
    time_slot_av = time_slot_normalized.mean(axis=1)
    time_slot_max = time_slot_normalized.max(axis=1)
    min_av_max_caps = np.concatenate((time_slot_min, time_slot_av, time_slot_max))
    quad_constr_normalized = quad_constr / max_cap
    av_cap_bytime = time_slot.mean(axis=0) / max_cap
    # Group by time_length
    final_groups = []
    time_horizon = time_slot.shape[1]
    num_of_time_groups = int(time_horizon / time_len)
    time_steps = np.arange(0, time_horizon + 1, time_len)
    for i in range(num_of_demand_groups * num_of_util_groups):
        if demand_groups[i].size > 0:
            group_time_lengths = demand_groups[i][:, num_of_dims + 1] - demand_groups[i][:, num_of_dims]
            for j in range(num_of_time_groups):
                bool_idx = (group_time_lengths > time_steps[j]) & (group_time_lengths <= time_steps[j + 1])
                final_groups.append(demand_groups[i][np.where(bool_idx)[0]])
        else:
            for _ in range(num_of_time_groups):
                final_groups.append(demand_groups[i])
    return final_groups, total_demand, min_av_max_caps, av_cap_bytime, quad_constr_normalized, min_util, max_util, total_utility, max_cap, clients


parameters = {'arrival_time': ['uniform', 'poisson', 'bimodal', 'beta'],
              'end_time': ['uniform', 'gaussian'],
              'relation': ['random', 'quadratic', 'linear']}

if __name__ == '__main__':
    # Problem making part
    batch_size = 10  # Change the batch size according to your cpu features
    count = 200 // batch_size

    problem_types = ['time', 'demand', 'constraint', 'arrival']

    for type in problem_types:
        for relation in parameters['relation']:
            for batch in range(count):
                t1 = time.time()
                with Pool(processes=batch_size) as p_construct:
                    args_list = [(i, type, relation) for i in range(batch * batch_size, (batch + 1) * batch_size)]
                    p_construct.starmap(construct, args_list)
                for problem in range(batch * batch_size, (batch + 1) * batch_size):
                    problem_solver(problem, type, relation)
                print(time.time() - t1)
    # For every type of problem there must be 200 linear,200 random,200 quadratic ones
    # After generation for next steps it will be better
    # to use "problem_grouping" function defined in this file,
    # since utility types are already included in the problem's file name
