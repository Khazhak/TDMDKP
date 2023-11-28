import numpy as np
import os

state_paths = os.listdir('states/')
answer_paths = os.listdir('answers/')

pr_states = np.load(f'states/{state_paths[1]}')
pr_answer = np.load(f'answers/{answer_paths[1]}')

clients_list = pr_states['cl']
time_slot_capacity = pr_states['tslot']
quad_constr = pr_states['quad_constr']

"""
random
u_min=1
u_max=100
=> (1,100)
linear
u_min=30
u_max=1200
=> (3,120)
quadratic
u_min=300
u_max=120000
=> (1,400)


max demand in the 4-th dimension can be sqrt(10000+10000+10000)=100*sqrt(3)<=174,min=sqrt(3)>1

Total demand per dimension can be max 96*100=9600,so 0.7 part of it is 6720 = max capacity

max_utility=1200

overall (1->1200)
"""


def problem_grouping(clients_array, time_slot, quad_constr, util_rate=1.5,
                     time_len=16, num_of_util_groups=15,
                     num_of_demand_groups=16,
                     max_quadratic_demand=174,
                     max_cap=105000,
                     max_utility=400):  ##grouping by utility,then demand,then by time_slot_length
    clients = clients_array.copy()
    num_of_dims = time_slot.shape[0]
    ##Grouping by utilities
    max_util = clients[:, -2].max()
    if 100 < max_util <= 1200:
        clients[:, -2] /= 10
    elif max_util > 1200:
        clients[:, -2] /= 300
    util_groups = []
    time_steps = np.zeros(num_of_util_groups)
    time_steps[0] = util_rate
    for i in range(1, time_steps.size):
        time_steps[i] = time_steps[i - 1] * util_rate
    bool_idx = (clients[:, -2] <= time_steps[0])
    util_groups.append(clients[np.where(bool_idx)[0]])
    for i in range(1, num_of_util_groups):
        bool_idx = (clients[:, -2] <= time_steps[i]) & (
                clients[:, -2] > time_steps[i - 1])
        util_groups.append(clients[np.where(bool_idx)[0]])
    # Normalizing by max_utility
    clients /= max_utility
    max_util = clients[:, -2].max()
    min_util = clients[:, -2].min()
    total_utility = clients[:, -2].sum()
    for group in util_groups:
        if group.size > 0:
            group[:, -2] /= max_utility
    ##Grouping by demand of 4th - quadratic dimension
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

    # Normalizing demands in each dimension by min capacity
    time_slot_min = time_slot.min(axis=1)
    for group in demand_groups:
        for dim in range(num_of_dims):
            group[:, dim] /= time_slot_min[dim]
    # sum of demands of the first dimension normalized by min capacity of the first dimension
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
    ##Grouping by time_length
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

    return final_groups, total_demand, min_av_max_caps, av_cap_bytime, quad_constr_normalized, min_util, max_util, total_utility


def data_preprocess(groups, total_demand, quad_cap, min_av_max_caps, av_cap_bytime,
                    min_util, max_util, total_utility, num_of_dims=3, num_of_clients=1500, time_length=96):
    length = len(groups)
    inputs = np.zeros((length + 1, 3 * num_of_dims + 1 + time_length + 2))
    labels = np.zeros((length + 1, 3))
    for i, group in enumerate(groups):
        if group.size > 0:
            inputs[i][:num_of_dims] = np.min(group[:, :num_of_dims], axis=0)  # minimum demand per dimension
            inputs[i][num_of_dims:2 * num_of_dims] = np.mean(group[:, :num_of_dims],
                                                             axis=0)  # average demand per dimension
            inputs[i][2 * num_of_dims:3 * num_of_dims] = np.max(group[:, :num_of_dims],
                                                                axis=0)  # maximum demand per dimension
            quadr_demands = ((group[:, :num_of_dims] ** 2).sum(axis=1)).mean()  # mean demand in the quadratic dimension
            inputs[i][3 * num_of_dims] = quadr_demands
            inputs[i][3 * num_of_dims + 1:3 * num_of_dims + time_length + 1] = np.sum(
                group[:, num_of_dims + 2:num_of_dims + 2 + time_length],
                axis=0) / num_of_clients  # average time_occupancy

            inputs[i][-2] = np.mean(group[:, -2])  # average utility
            inputs[i][-1] = group.shape[0] / num_of_clients  # what part of clients is in this group
            labels[i][0] = np.sum(group[:, -1]) / num_of_clients  # what part is selected in the final answer
            if group[np.where(group[:, -1] == 1)].size > 0:
                # total demand of selected clients/total demand of the first dimension
                selected = group[np.where(group[:, -1] == 1)]
                labels[i][1] = selected[:, 0].sum() / total_demand
                labels[i][2] = selected[:, -2].sum() / total_utility
    inputs[-1][:3 * num_of_dims] = min_av_max_caps
    inputs[-1][3 * num_of_dims] = quad_cap
    inputs[-1][3 * num_of_dims + 1:3 * num_of_dims + time_length + 1] = av_cap_bytime
    inputs[-1][-2] = min_util
    inputs[-1][-1] = max_util
    labels[-1][0] = 1 - labels[:, 0].sum()
    labels[-1][1] = 1 - labels[:, 1].sum()
    labels[-1][2] = 1 - labels[:, 2].sum()
    return inputs, labels


if __name__ == '__main__':
    f_groups, tot_dem, min_av_max_caps, av_cap_bytime, q_c, min_util, max_util, tot_util = problem_grouping(
        clients_list,
        time_slot_capacity,
        quad_constr)
    inp, lab = data_preprocess(f_groups, tot_dem, q_c, min_av_max_caps, av_cap_bytime,
                               min_util, max_util, tot_util)