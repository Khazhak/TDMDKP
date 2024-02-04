import os
import glob
from ufp_optuna import KnapsackPredictor, problem_grouping, data_preprocess
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from time import time
from tqdm import tqdm
import gc
import gurobipy as gp
from gurobipy import GRB
from memory_profiler import profile
import random

CHECKPOINT_PATH = "C:\\Users\\zhira\\CSIE_PYTHON_PROJECTS\\UFP_FINAL"
root_dir = os.path.join(CHECKPOINT_PATH, "UfpCheckPointNew2")
pretrained_filename = os.path.join(root_dir, "UfpCheckPointNew2.ckpt")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

knapsack_model = KnapsackPredictor.load_from_checkpoint(pretrained_filename)
knapsack_model = knapsack_model.to(device)


def instance_generator(num_of_instances=1000):
    states = list(glob.glob('states/*.npz'))
    answers = list(glob.glob('answers/*.npy'))
    random_indices = np.random.randint(0, len(states) - 1, num_of_instances)
    index = 0
    while index < len(random_indices):
        idx = random_indices[index]
        state = np.load(states[idx])
        answer = np.load(answers[idx])
        clients_list = state['cl']
        time_slot_capacity = state['tslot']
        quad_constr = state['quad_constr']
        state.close()
        f_groups, tot_dem, mi_av_max_caps, av_cap_time, q_c, mi_util, ma_util, tot_util, max_cap, clients = problem_grouping(
            clients_list,
            time_slot_capacity,
            quad_constr)
        inp, lab = data_preprocess(f_groups, tot_dem, q_c, mi_av_max_caps, av_cap_time,
                                   mi_util, ma_util, tot_util)
        yield inp, lab, clients, time_slot_capacity, quad_constr, f_groups, tot_dem, tot_util, answer
        gc.collect()
        index += 1


def validation_check(selected_items, time_slot, quad_constr, max_cap, num_of_dims, time_length):
    is_correct_1 = np.stack([sum([selected_items[index][dim] * max_cap * selected_items[index][
        num_of_dims + 1 + time] for index in range(len(selected_items))]) <= time_slot[dim][time - 1]
                             for dim in range(num_of_dims) for time in range(1, time_length + 1)]).reshape(time_length,
                                                                                                           num_of_dims)
    is_correct_2 = np.stack([np.sum([np.sum([selected_items[index][dim] * max_cap *
                                             selected_items[index][num_of_dims + 1 + time] for index in
                                             range(len(selected_items))]) ** 2 for dim in
                                     range(num_of_dims)]) <= quad_constr ** 2 for time in
                             range(1, time_length + 1)]).reshape(time_length,
                                                                 1)
    return np.hstack([is_correct_1, is_correct_2])


def valid_maker(selected_items, time_slot, quad_constr, max_cap, num_of_dims, time_length, total_utility,
                final_total_utility=0, attempts=1500):
    is_correct = validation_check(selected_items, time_slot, quad_constr, max_cap, num_of_dims, time_length)
    violated = not np.all(is_correct)
    attempt = 1
    while not np.all(is_correct) and total_utility > final_total_utility and len(
            selected_items) > 0 and attempt <= attempts:
        min_utility = 100000
        min_item_index = 0
        rows, cols = np.where(~is_correct)
        times_f = rows + 1  # In which time step the violation occurs
        time_indices = num_of_dims + 1 + times_f
        for item in range(len(selected_items)):
            x = selected_items[item]
            if any(x[time_indices] == 1):
                if x[-2] < min_utility:
                    min_utility = x[-2]
                    min_item_index = item
        selected_items.pop(min_item_index)
        total_utility = sum([item[-2] for item in selected_items])
        is_correct = validation_check(selected_items, time_slot, quad_constr, max_cap, num_of_dims, time_length)
        if np.all(is_correct):
            violated = False
            del is_correct
            break
        attempt += 1
    return selected_items, total_utility, violated


def selection_algorithm_count_util_group(time_slot, quad_constr, groups, tot_util, y_pred, num_of_clients=1500):
    if type(y_pred) != np.ndarray:
        prediction = y_pred.cpu().detach().numpy()
    else:
        prediction = y_pred

    num_of_dims = time_slot.shape[0]
    time_length = time_slot.shape[1]
    max_cap = time_slot.max()

    sum_utils = prediction[:, 2] * tot_util

    final_total_utility = 0
    final_selected_items = []

    for _ in tqdm(range(10), desc='Progress bar'):
        total_utility = 0
        selected_items = []
        rand_perm = np.random.permutation(len(groups))

        for gr_num in rand_perm:
            group = groups[gr_num]
            if group.size > 0:
                max_res = 0
                best_vec = np.array([])

                sel_size = min(math.ceil(prediction[gr_num][0] * num_of_clients), group.shape[0])
                indexes = set(range(sel_size))
                chosen = set()
                invalid_indices = set()

                for _ in range(10):

                    sel_vec = []

                    for index in range(sel_size):
                        for _ in range(10):
                            available_indexes = list(indexes - chosen - invalid_indices)
                            if not available_indexes:
                                break
                            choice = np.random.choice(available_indexes)
                            curr_item = group[choice]
                            temp_selected_items = selected_items + [curr_item]
                            is_correct = validation_check(temp_selected_items, time_slot, quad_constr, max_cap,
                                                          num_of_dims, time_length)
                            if np.all(is_correct):
                                del is_correct
                                sel_vec.append(curr_item)
                                chosen.add(choice)
                                break
                            else:
                                invalid_indices.add(choice)

                    if sel_vec:
                        sel_vec = np.array(sel_vec)
                        util_sum = sel_vec[:, -2].sum()

                        if util_sum > max_res and np.abs(util_sum - sum_utils[gr_num]) <= 0.1 * sum_utils[gr_num]:
                            max_res = util_sum
                            best_vec = sel_vec

                if best_vec.size > 0:
                    total_utility += max_res
                    selected_items.extend(best_vec)

        if final_total_utility < total_utility:
            final_total_utility = total_utility
            final_selected_items = selected_items
        gc.collect()
    return np.stack(final_selected_items), final_total_utility


def selection_algorithm_count_util_group_new(time_slot, quad_constr, groups, tot_util, y_pred, num_of_clients=1500):
    if type(y_pred) != np.ndarray:
        prediction = y_pred.cpu().detach().numpy()
    else:
        prediction = y_pred

    num_of_dims = time_slot.shape[0]
    time_length = time_slot.shape[1]
    max_cap = time_slot.max()

    sum_utils = prediction[:, 2] * tot_util

    final_total_utility = 0
    final_selected_items = []

    for _ in tqdm(range(100), desc='Progress bar'):  # 100
        total_utility = 0
        selected_items = []
        rand_perm = np.random.permutation(len(groups))
        for gr_num in rand_perm:
            group = groups[gr_num]
            if group.size > 0:
                sel_size = min(math.ceil(prediction[gr_num][0] * num_of_clients), group.shape[0])
                prob_vec_uniform = np.ones(group.shape[0]) / group.shape[0]
                for _ in range(5):
                    selected_indices = np.random.choice(range(group.shape[0]), sel_size, replace=False,
                                                        p=prob_vec_uniform)
                    selection = group[selected_indices]
                    selected_utility_sum = selection[:, -2].sum()
                    temp_selected_items = selected_items + list(selection)
                    is_correct = validation_check(temp_selected_items, time_slot, quad_constr, max_cap, num_of_dims,
                                                  time_length)
                    if np.abs(selected_utility_sum - sum_utils[gr_num]) <= 1 / group.shape[0] * sum_utils[gr_num]:
                        if np.all(is_correct):
                            selected_items = temp_selected_items
                            total_utility += selected_utility_sum
                            break
        if total_utility > final_total_utility:
            final_total_utility = total_utility
            final_selected_items = selected_items
    gc.collect()
    if len(final_selected_items) > 0:
        return np.stack(final_selected_items), final_total_utility
    else:
        return [], 0


def selection_algorithm_count_util_all(time_slot, quad_constr, groups, tot_util, y_pred, num_of_clients=1500,
                                       attempts=1500):
    if type(y_pred) != np.ndarray:
        prediction = y_pred.cpu().detach().numpy()
    else:
        prediction = y_pred
    num_of_dims = time_slot.shape[0]
    time_length = time_slot.shape[1]
    max_cap = time_slot.max()

    sum_utils = prediction[:, 2] * tot_util
    final_total_utility = 0
    final_selected_items = []
    for _ in tqdm(range(10), desc='Progress bar'):
        total_utility = 0
        selected_items = []
        for i, group in enumerate(groups):
            if group.size > 0:
                max_res = 0
                best_vec = np.array([])
                # prob_vec=np.array([item[-2]/group_util_sum for item in group])
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

        selected_items, total_utility, violated = valid_maker(selected_items, time_slot, quad_constr, max_cap,
                                                              num_of_dims,
                                                              time_length, total_utility, final_total_utility,
                                                              attempts=attempts)
        if not violated:
            if final_total_utility < total_utility:
                final_selected_items = selected_items.copy()
                final_total_utility = total_utility
    gc.collect()
    if len(final_selected_items) > 0:
        return np.stack(final_selected_items), final_total_utility
    else:
        return [], 0


def innerfunction(items: np.ndarray, W):
    if items.size == 0 or items.min() > W:
        return []

    # Convert a list of items (weight) to a list of the form ((weight, index))
    # for consistent tie-breaking in comparison operations
    augmented_items = [(weight, index) for index, weight in enumerate(items)]

    stack = [(augmented_items, W)]  # Initialize stack with the initial state
    solution = []  # This will store the solution items

    while stack:
        remaining, current_W = stack.pop()
        if not remaining or min(weight[0] for weight in remaining) > current_W:
            continue
        random_item = np.median([weight[0] for weight in remaining])
        # random_item = random.choice(remaining)
        smaller = [item for item in remaining if item[0] <= random_item]
        not_smaller = [item for item in remaining if item[0] > random_item]
        smaller_cost = sum(weight for weight, _ in smaller)

        if smaller_cost <= current_W:
            solution.extend(smaller)  # Include all smaller items in the solution
            # Proceed to process not_smaller items with the updated capacity
            stack.append((not_smaller, current_W - smaller_cost))
        else:
            # If we cannot include all smaller items, only process smaller items
            stack.append((smaller, current_W))

    # Return the un-augmented items from the solution
    return [index for _, index in solution]


def main_algorithm(time_slot, quad_constr, groups, total_demand, y_pred, attempts=1500):
    if type(y_pred) != np.ndarray:
        prediction = y_pred.cpu().detach().numpy()
    else:
        prediction = y_pred
    num_of_dims = time_slot.shape[0]
    time_length = time_slot.shape[1]
    max_cap = time_slot.max()
    demands = prediction[:, 1]
    sum_demands = demands * total_demand

    total_utility = 0
    selected_items = []
    for i, group in enumerate(groups):
        if group.size > 0:
            demands = group[:, 0]
            W = sum_demands[i]
            choice = innerfunction(demands, W)
            total_utility += group[choice][:, -2].sum()
            selected_items.extend(group[choice])
    selected_items, total_utility, violated = valid_maker(selected_items, time_slot, quad_constr, max_cap,
                                                          num_of_dims,
                                                          time_length, total_utility, attempts=attempts)

    if not violated and len(selected_items) > 0:
        return np.stack(selected_items), total_utility
    else:
        return [], 0


def gurobi_maximization(time_slot, quad_constr, groups, total_demand, tot_util, y_pred, attempts=1500):
    if type(y_pred) != np.ndarray:
        prediction = y_pred.cpu().detach().numpy()
    else:
        prediction = y_pred
    num_of_dims = time_slot.shape[0]
    time_length = time_slot.shape[1]
    max_cap = time_slot.max()
    sum_demands = prediction[:, 1] * total_demand
    # sum_utils = prediction[:, 2] * tot_util

    total_utility = 0
    selected_items = []
    for i, group in enumerate(groups):
        if group.size > 0:
            num_of_group_clients = group.shape[0]
            utils = group[:, -2]
            demands = group[:, 0]
            ufp_model = gp.Model()
            opt_choice = ufp_model.addVars(num_of_group_clients, vtype=GRB.BINARY, name="opt_choice")
            W = sum_demands[i]
            # U = sum_utils[i]
            # ufp_model.addConstr(
            #     (gp.quicksum([opt_choice[index] * utils[index] for index in range(num_of_group_clients)]) <= U), name='utility')
            ufp_model.addConstr(
                (gp.quicksum([opt_choice[index] * demands[index] for index in range(num_of_group_clients)]) <= W),
                name='demand')
            ufp_model.setObjective(gp.quicksum([opt_choice[i] * utils[i] for i in range(num_of_group_clients)]),
                                   GRB.MAXIMIZE)
            ufp_model.optimize()
            answer = np.array([item.x for item in ufp_model.getVars()])
            chosen = group[np.where([answer == 1])[0]]
            selected_items.extend(chosen)
            total_utility += chosen[:, -2].sum()
    selected_items, total_utility, violated = valid_maker(selected_items, time_slot, quad_constr, max_cap, num_of_dims,
                                                          time_length, total_utility, attempts=attempts)
    if not violated and len(selected_items) > 0:
        return np.stack(selected_items), total_utility
    else:
        return [], 0


if __name__ == '__main__':
    number_of_iterations = 50
    generator = instance_generator(number_of_iterations)
    algorithm_1_results = np.zeros(number_of_iterations)
    algorithm_2_results = np.zeros(number_of_iterations)
    algorithm_3_results = np.zeros(number_of_iterations)
    algorithm_4_results = np.zeros(number_of_iterations)
    label_results = np.zeros(number_of_iterations)
    for idx, instance in enumerate(generator):
        print(f'Sample No : {idx}')
        inp, lab, clients_list, time_slot_capacity, quad_constr, f_groups, tot_dem, tot_util, ans = instance
        inp = torch.from_numpy(np.float32(inp[None, :])).to(device)
        y_pred = knapsack_model(inp)
        y_pred = y_pred.cpu().detach().numpy().squeeze()
        selected_items, final_total_utility = selection_algorithm_count_util_group(time_slot_capacity,
                                                                                   quad_constr,
                                                                                   f_groups,
                                                                                   tot_util,
                                                                                   y_pred)

        selected_items_2, final_total_utility_2 = selection_algorithm_count_util_all(time_slot_capacity, quad_constr,
                                                                                     f_groups,
                                                                                     tot_util,
                                                                                     y_pred)

        selected_items_3, final_total_utility_3 = main_algorithm(time_slot_capacity, quad_constr, f_groups, tot_dem,
                                                                 y_pred)
        selected_items_4, final_total_utility_4 = gurobi_maximization(time_slot_capacity, quad_constr, f_groups,
                                                                      tot_dem, tot_util,
                                                                      y_pred)
        algorithm_1_results[idx] = final_total_utility * 400
        algorithm_2_results[idx] = final_total_utility_2 * 400
        algorithm_3_results[idx] = final_total_utility_3 * 400
        algorithm_4_results[idx] = final_total_utility_4 * 400
        label_results[idx] = clients_list[clients_list[:, -1] == 1.0][:, -2].sum() * 400
        print('selected_items_shape: ', selected_items.shape)
        print('selected_utility', final_total_utility * 400)
        print('selected_items_shape_2: ', selected_items_2.shape)
        print('selected_utility_2', final_total_utility_2 * 400)
        print('selected_items_shape_3: ', selected_items_3.shape)
        print('selected_utility_3', final_total_utility_3 * 400)
        print('selected_items_shape_4: ', len(selected_items_4))
        print('selected_utility_4', final_total_utility_4 * 400)
        print('answer_shape : ', clients_list[clients_list[:, -1] == 1].shape)
        print('answer_utility : ', clients_list[clients_list[:, -1] == 1.0][:, -2].sum() * 400)
    np.savez('results/results.npz', algorithm=algorithm_1_results, labels=label_results)
    np.savez('results/results2.npz', algorithm=algorithm_2_results, labels=label_results)
    np.savez('results/results3.npz', algorithm=algorithm_3_results, labels=label_results)
    np.savez('results/results4.npz', algorithm=algorithm_4_results, labels=label_results)
    # file = np.load('results/results.npz')
    # algorithm_1_results = file['algorithm1']
    # label_results = file['labels']
    percentage1 = algorithm_1_results / label_results * 100
    percentage2 = algorithm_2_results / label_results * 100
    percentage3 = algorithm_3_results / label_results * 100
    percentage4 = algorithm_4_results / label_results * 100
    indices = np.arange(number_of_iterations)
    plt.scatter(indices, percentage1, color='red', label='algorithm1', s=5)
    plt.scatter(indices, percentage2, color='blue', label='algorithm1', s=5)
    plt.scatter(indices, percentage3, color='green', label='algorithm3', s=5)
    plt.scatter(indices, percentage4, color='orange', label='algorithm4', s=5)
    plt.show()
