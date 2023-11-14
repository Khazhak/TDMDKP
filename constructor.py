import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random as rd
from memory_profiler import profile
import gc


@profile
def problem_constructor(filename, num_of_clients, arrival_time='uniform', end_time='uniform',
                        util_demand_relation='linear', dim_size=3, time_length=96):
    # arrival_time--uniform,poisson
    # end_time--uniform,gaussian
    # relation--random,quadratic,linear
    size = dim_size + time_length + 4  # demands_per_dim,time_s,time_e,time_interval,utility,chosen or not
    lin_comb = np.random.rand(dim_size) * 3 + 1
    clients = np.zeros((num_of_clients, size))

    for index in range(num_of_clients):
        for dim in range(dim_size):
            clients[index][dim] = rd.uniform(10, 100)
        # starting time
        if arrival_time == 'uniform':
            s_time = rd.randint(1, int(0.75 * time_length))  # uniformly random
        else:
            lam = 10
            s_time = np.random.poisson(lam)
            s_time = int(max(1, min(s_time, time_length)))  # starting time poisson
        ##end time
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
    # max_total_demand_per_dim_time = np.matmul(np.transpose(clients[:, :dim_size]),
    #                                          clients[:, dim_size + 2:dim_size + time_length + 2])
    # for dim in range(dim_size):
    #     time_slot_capacity[dim][0] = rd.uniform(0.21 * max_total_demand_per_dim_time[dim][0],
    #                                             0.7 * max_total_demand_per_dim_time[dim][0])
    #     for step in range(1, time_length):
    #         change = rd.randint(0, 1)
    #         if change:
    #             time_slot_capacity[dim][step] = rd.uniform(0.21 * max_total_demand_per_dim_time[dim][step],
    #                                                        0.7 * max_total_demand_per_dim_time[dim][step])
    #         else:
    #             time_slot_capacity[dim][step] = time_slot_capacity[dim][step - 1]
    # B_constraint=0.9*time_slot_capacity.mean()

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
    np.savez(f'problem_states_new//{filename}', cl=clients, tslot=time_slot_capacity, quad_constr=B_constraint)
    ufp_model.write(f"problem_states_new//model_{filename}.lp")
    del clients
    del time_slot_capacity
    del constr_1
    del constr_2
    del ufp_model
    gc.collect()


parameters = {'arrival_time': ['uniform', 'poisson'],
              'end_time': ['uniform', 'gaussian'],
              'relation': ['random', 'quadratic', 'linear']}


def construct(problem):
    index1 = rd.randint(0, 1)
    index2 = rd.randint(0, 1)
    index3 = rd.randint(0, 2)
    ar_t = parameters['arrival_time'][index1]
    en_t = parameters['end_time'][index2]
    rel = parameters['relation'][index3]
    print(f'__________{problem}__________{ar_t}_{en_t}_{rel}')
    problem_constructor(f'{problem}', 1500, ar_t, en_t, rel)
