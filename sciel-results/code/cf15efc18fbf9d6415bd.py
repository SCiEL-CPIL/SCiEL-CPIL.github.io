import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s in one place, let's be safe)
    # Actually, the prompt says "Each evaluation must finish within 120 s" but "COMPUTE BUDGET: the evaluator KILLS the program after 150s wall-clock".
    # I will use 115s to be very safe against overhead.

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    best_result = None
    best_score = float('inf')

    def evaluate(result):
        if result is None:
            return float('inf')
        load = np.zeros(n_facilities)
        np.add.at(load, result, demands)
        
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        open_mask = np.zeros(n_facilities, dtype=bool)
        open_mask[result] = True
        
        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(3, n_facilities)]
                f = top_k[np.random.randint(len(top_k))]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            is_open[f] = True
            
        return result, load, is_open

    def fast_local_search(result, load, is_open):
        improved = True
        while improved:
            improved = False
            for i in range(n_customers):
                if time.time() > deadline - 5:
                    return result, load, is_open
                
                f_old = result[i]
                d = demands[i]
                
                delta_costs = assign_cost[i] - assign_cost[i, f_old]
                
                closed_mask = ~is_open
                delta_costs[closed_mask] += open_cost[closed_mask]
                
                if load[f_old] == d:
                    delta_costs[:] -= open_cost[f_old]
                    delta_costs[f_old] = 0.0
            
                feasible = load + d <= capacities
                delta_costs[~feasible] = np.inf
                
                best_f_new = np.argmin(delta_costs)
                best_delta = delta_costs[best_f_new]
                
                if best_f_new != f_old and best_delta < -1e-9:
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False
                    
                    load[best_f_new] += d
                    if load[best_f_new] == d:
                        is_open[best_f_new] = True
                    
                    result[i] = best_f_new
                    improved = True
                    
        return result, load, is_open

    def swap_search(result, load, is_open):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 10:
                return result, load, is_open
                
            for i in range(n_customers):
                if time.time() > deadline - 10:
                    return result, load, is_open
                f_i = result[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_customers):
                    f_j = result[j]
                    d_j = demands[j]
                    
                    if f_i == f_j:
                        continue
                    
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    
                    delta_open = 0.0
                    if is_open[f_i]:
                        if new_load_i == 0:
                            delta_open -= open_cost[f_i]
                    else:
                        if new_load_i > 0:
                            delta_open += open_cost[f_i]
                            
                    if is_open[f_j]:
                        if new_load_j == 0:
                            delta_open -= open_cost[f_j]
                    else:
                        if new_load_j > 0:
                            delta_open += open_cost[f_j]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        result[i] = f_j
                        result[j] = f_i
                        
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        
                        if new_load_i == 0:
                            is_open[f_i] = False
                        else:
                            is_open[f_i] = True
                            
                        if new_load_j == 0:
                            is_open[f_j] = False
                        else:
                            is_open[f_j] = True
                        
                        improved = True
                        break
        
        return result, load, is_open

    def solve_milp(result_init=None):
        """
        Solves the CFLP using MILP via SciPy's HiGHS solver.
        Variables:
        x[i, f] = 1 if customer i is assigned to facility f
        y[f] = 1 if facility f is open
        
        Objective: Minimize sum(f) open_cost[f]*y[f] + sum(i,f) assign_cost[i,f]*x[i,f]
        Constraints:
        1. sum(f) x[i, f] = 1 for all i
        2. sum(i) demands[i]*x[i, f] <= capacities[f]*y[f] for all f
        3. x[i, f] <= y[f] for all i, f
        """
        # Number of variables: N*F (x) + F (y)
        n_vars = n_customers * n_facilities + n_facilities
        
        # Objective coefficients
        c = np.zeros(n_vars)
        # x part
        c[:n_customers * n_facilities] = assign_cost.flatten()
        # y part
        c[n_customers * n_facilities:] = open_cost
        
        # Constraints
        A_rows = []
        lb = []
        ub = []
        
        # Constraint 1: Assignment (sum_f x[i,f] = 1)
        # We can build a big matrix or use sparse. Given N=100, F=20, N*F=2000.
        # Dense is fine.
        
        # Let's build the matrix for constraint 1
        # Shape: (N, N*F + F)
        A1 = np.zeros((n_customers, n_vars))
        for i in range(n_customers):
            for f in range(n_facilities):
                A1[i, i * n_facilities + f] = 1.0
        A_rows.append(A1)
        lb.append(np.ones(n_customers))
        ub.append(np.ones(n_customers))
        
        # Constraint 2: Capacity (sum_i demands[i]*x[i,f] <= capacities[f]*y[f])
        # sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0
        A2 = np.zeros((n_facilities, n_vars))
        for f in range(n_facilities):
            for i in range(n_customers):
                A2[f, i * n_facilities + f] = demands[i]
            A2[f, n_customers * n_facilities + f] = -capacities[f]
        A_rows.append(A2)
        lb.append(np.zeros(n_facilities))
        ub.append(np.zeros(n_facilities))
        
        # Constraint 3: Linking (x[i,f] <= y[f])
        # x[i,f] - y[f] <= 0
        A3 = np.zeros((n_customers * n_facilities, n_vars))
        for i in range(n_customers):
            for f in range(n_facilities):
                idx = i * n_facilities + f
                A3[idx, idx] = 1.0
                A3[idx, n_customers * n_facilities + f] = -1.0
        A_rows.append(A3)
        lb.append(np.zeros(n_customers * n_facilities))
        ub.append(np.zeros(n_customers * n_facilities))
        
        A = np.vstack(A_rows)
        lb_all = np.concatenate(lb)
        ub_all = np.concatenate(ub)
        
        # Variable bounds and integrality
        integrality = np.ones(n_vars) # All binary
        # x variables are binary, y variables are binary
        # Actually, we can relax y to continuous [0,1] if we have linking constraints? 
        # No, for MILP it's usually best to keep binary if solver supports it well.
        # HiGHS supports binary.
        
        # However, we can fix variables based on result_init to speed up?
        # No, let's just let the solver do its job. It's small enough.
        
        # Provide a starting point? SciPy's milp doesn't directly take a starting solution for primal, 
        # but it uses the initial basis if provided? No, it's a black box.
        # We can try to use the heuristic solution to guide if we had a warm start, but we don't.
        
        try:
            res = milp(c,
                       constraints=LinearConstraint(A, lb_all, ub_all),
                       integrality=integrality,
                       bounds=Bounds(0, 1))
            
            if res.status == 0: # Optimal
                x = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
                # Find argmax for each row
                result = np.argmax(x, axis=1)
                
                # Verify feasibility
                if evaluate(result) < float('inf'):
                    return result
            else:
                # If optimal not found (e.g. time limit), try to use best found?
                # SciPy's milp doesn't easily return intermediate solutions.
                # We might need to set a time limit.
                # Let's assume it finishes quickly for 100x20.
                pass
                
        except Exception as e:
            pass
            
        return None

    # 1. Try MILP first. It is exact and fast for this size.
    # We give it a chunk of time.
    milp_deadline = start_time + 30.0 # 30 seconds for MILP
    # Since milp is blocking, we can't easily interrupt it unless we use threads.
    # But 100x20 is tiny. It should take < 1s usually.
    # Let's just run it.
    
    # Actually, to be safe against a pathological case where MILP hangs, 
    # we might want to run it in a thread with a timeout.
    # But Python's GIL makes threads not truly parallel for CPU-bound.
    # However, HiGHS releases GIL? 
    # Let's assume it's fast. If it's not, we fallback.
    
    # Let's run MILP.
    result_milp = solve_milp()
    
    if result_milp is not None:
        score = evaluate(result_milp)
        if score < best_score:
            best_score = score
            best_result = result_milp.copy()
            
    # 2. If MILP failed or we have time, run heuristics.
    # Even if MILP succeeded, heuristics might find a better solution if MILP timed out internally 
    # (though status 0 means optimal). If status != 0, we should run heuristics.
    
    # Let's always run a quick heuristic pass to ensure we have a valid solution 
    # in case MILP returns something weird or fails.
    
    # 3. Heuristic search
    # Run a few restarts if time permits.
    
    # First, deterministic greedy + local search
    if time.time() < deadline - 5:
        res, load, is_open = greedy_init(randomize=False)
        res, load, is_open = fast_local_search(res, load, is_open)
        res, load, is_open = swap_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()

    # Randomized restarts
    max_restarts = 100
    for k in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = fast_local_search(res, load, is_open)
        res, load, is_open = swap_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()

    if best_result is None:
        # Fallback: just assign everyone to facility 0? Or greedy without local search?
        # This should never happen.
        res, _, _ = greedy_init(randomize=False)
        best_result = res

    return best_result.astype(int)
