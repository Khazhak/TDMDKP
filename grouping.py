import numpy as np

pr_states = np.load(f'1500.npz')
pr_answer = np.load(f'1500.npy')

clients_list = pr_states['cl']
time_slot_capacity = pr_states['tslot']
quad_constr = pr_states['quad_constr']

"""
u_min=1
u_max=120000

"""


def problem_grouping(clients, time_slot, util_rate=1.5,
                     time_len=6, num_of_util_groups=29):  ##grouping by utility,then by time_slot_length
    num_of_dims = time_slot.shape[0]
    time_slot_min = time_slot.min(axis=1)
    time_slot_av = time_slot.mean(axis=1)
    time_slot_max = time_slot.max(axis=1)
    max_cap = time_slot.max()
    min_av_max_caps = np.concatenate((time_slot_min, time_slot_av, time_slot_max))
    av_cap_bytime = time_slot.mean(axis=0) / max_cap
    # Normalizing demands in each dimension by min capacity
    for dim in range(num_of_dims):
        clients[:, dim] /= time_slot_min[dim]
    # sum of demands of the first dimension normalized by min capacity of the first dimension
    total_demand = clients[:, 0].sum()
    # normalizing time capacities
    min_av_max_caps /= max_cap
    ##Grouping by utilities
    max_util = clients[:, -2].max()
    min_util = clients[:, -2].min()
    av_util = clients[:, -2].mean()
    util_groups = []
    steps = np.zeros(num_of_util_groups)
    steps[0] = util_rate
    for i in range(1, steps.size):
        steps[i] = steps[i-1]*util_rate
    bool_idx = (clients[:, -2] <= steps[0])
    util_groups.append(clients[np.where(bool_idx)[0]])
    for i in range(1, num_of_util_groups):
        bool_idx = (clients[:, -2] <= steps[i]) & (
                clients[:, -2] > steps[i - 1])
        util_groups.append(clients[np.where(bool_idx)[0]])
    print(util_groups[0][:,-2])
    # Normalizing by max_util
    for group in util_groups:
        group[:, -2] /= max_util

    ##Grouping by time_length
    final_groups = []
    gr_time = int(time_slot.shape[1] / time_len)
    for i in range(num_of_util_groups):
        if (util_groups[i].size > 0):
            group_time_lengths = util_groups[i][:, num_of_dims + 1] - util_groups[i][:, num_of_dims]
            min_time_len = np.min(group_time_lengths)
            max_time_len = np.max(group_time_lengths)
            steps = np.linspace(min_time_len, max_time_len, gr_time + 1)
            for j in range(gr_time - 1):
                bool_idx = (group_time_lengths >= steps[j]) & (group_time_lengths < steps[j + 1])
                final_groups.append(util_groups[i][np.where(bool_idx)[0]])
            bool_idx = (group_time_lengths >= steps[-2]) & (group_time_lengths <= steps[-1])
            final_groups.append(util_groups[i][np.where(bool_idx)[0]])
        else:
            for _ in range(gr_time):
                final_groups.append(util_groups[i])
    return final_groups, total_demand, min_av_max_caps, av_cap_bytime, min_util, av_util


if __name__ == '__main__':
    f_groups, tot_dem, min_av_max_caps, av_cap_bytime, min_util, av_util = problem_grouping(clients_list,
                                                                                            time_slot_capacity)
    sum = 0
    for idx, group in enumerate(f_groups):
        print(f'group_{idx}--', group.shape)
