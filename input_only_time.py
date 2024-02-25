import numpy as np
from torch.utils.data import Dataset
import glob

'''
While using train_knapsack function for training,input dim must be changed to 96 
'''


def problem_grouping(clients_array, time_slot, util_rate=1.5,
                     time_len=16, num_of_util_groups=15,
                     num_of_demand_groups=16,
                     max_quadratic_demand=174,
                     max_utility=400):
    """Group by utility,then demand,then by time_slot_length"""
    clients = clients_array.copy()
    num_of_dims = time_slot.shape[0]
    # Group by utilities
    max_util = clients[:, -2].max()
    if 100 < max_util <= 1200:
        clients[:, -2] /= 10
    elif max_util > 1200:
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
    clients /= max_utility
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
            for j in range(num_of_demand_groups)[::-1]:
                bool_idx = (group_demands > demand_steps[j - 1]) & (group_demands <= demand_steps[j])
                demand_groups.append(util_groups[i][np.where(bool_idx)[0]])
        else:
            for _ in range(num_of_demand_groups):
                demand_groups.append(util_groups[i])

    # Normalizing demands in each dimension by max capacity
    max_cap = time_slot.max()
    for group in demand_groups:
        group[:, num_of_dims] /= max_cap
    # sum of demands of the first dimension normalized by max capacity
    total_demand = 0
    for group in demand_groups:
        total_demand += group[:, 0].sum()
    # Normalizing capacities by max capacity
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

    return final_groups, total_demand, av_cap_bytime, total_utility, max_cap


def data_preprocess(groups, total_demand, av_cap_bytime, total_utility, num_of_dims=3, num_of_clients=1500,
                    time_length=96):
    length = len(groups)
    inputs = np.zeros((length + 1, time_length))
    labels = np.zeros((length + 1, 3))
    for i, group in enumerate(groups):
        if group.size > 0:
            # average time_occupancy
            inputs[i] = np.sum(group[:, num_of_dims + 2:num_of_dims + 2 + time_length], axis=0) / num_of_clients
            labels[i][0] = np.sum(group[:, -1]) / num_of_clients  # what part is selected in the final answer
            if group[np.where(group[:, -1] == 1)].size > 0:
                # total demand of selected clients/total demand of the first dimension
                selected = group[np.where(group[:, -1] == 1)]
                labels[i][1] = selected[:, 0].sum() / total_demand
                labels[i][2] = selected[:, -2].sum() / total_utility
    inputs[-1] = av_cap_bytime
    labels[-1][0] = 1 - labels[:, 0].sum()
    labels[-1][1] = 1 - labels[:, 1].sum()
    labels[-1][2] = 1 - labels[:, 2].sum()
    return inputs, labels


class DataSetMaker(Dataset):
    def __init__(self):
        self.states = list(glob.glob('states/*.npz'))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = np.load(self.states[idx])
        clients_list = state['cl']
        time_slot_capacity = state['tslot']
        f_groups, tot_dem, av_cap_time, tot_util, max_cap = problem_grouping(
            clients_list,
            time_slot_capacity)
        inp, lab = data_preprocess(f_groups, tot_dem, av_cap_time, tot_util)
        return np.float32(inp), np.float32(lab), tot_util, tot_dem, max_cap
