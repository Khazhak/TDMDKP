import gurobipy as gp
from gurobipy import GRB, GurobiError
import numpy as np
import os
import time
from multiprocessing import Pool
from constructor import construct


def problem_solver(filename):
    data = np.load(f'problem_states_new//{filename}.npz')
    clients = data['cl']
    time_slot_capacity = data['tslot']
    B_constraint = data['quad_constr']
    data.close()
    ufp_model = gp.read(f"problem_states_new//model_{filename}.lp")
    try:
        os.remove(f"problem_states_new//model_{filename}.lp")
        os.remove(f'problem_states_new//{filename}.npz')
    except Exception as e:
        print(f"Error deleting file {filename}: {e}")
    ufp_model.setParam('TimeLimit', 150)
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
    np.savez(f'states_new//{filename}', cl=clients, tslot=time_slot_capacity, quad_constr=B_constraint)
    np.save(f'answers_new//{filename}', answer)


if __name__ == '__main__':
    batch_size = 10
    count = 30000 // batch_size

    for batch in range(count):
        t1=time.time()
        with Pool(processes=batch_size) as p_construct:
            p_construct.map(construct, range(batch * batch_size,(batch + 1) * batch_size))
        for problem in range(batch * batch_size, (batch + 1) * batch_size):
            problem_solver(problem)
        print(time.time()-t1)
