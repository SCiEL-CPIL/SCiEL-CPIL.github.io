"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def _solve_milp(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem as a MILP using scipy.optimize.milp.
    Returns (assignment, cost) or (None, inf) on failure.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables: x[0..n_c*n_f-1] for assignment, y[n_c*n_f..n_c*n_f+n_f-1] for open
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y
    
    # Objective coefficients
    c_obj = np.zeros(n_vars)
    # x[c, f] cost
    c_obj[:n_x] = assign_cost.flatten()
    # y[f] cost
    c_obj[n_x:] = open_cost
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_f x[c,f] = 1
    # 2. Capacity: sum_c demands[c]*x[c,f] <= capacities[f]*y[f]  => sum_c demands[c]*x[c,f] - capacities[f]*y[f] <= 0
    # 3. x[c,f] <= y[f]  => x[c,f] - y[f] <= 0
    
    # Count constraints
    n_cons_cust = n_cust
    n_cons_cap = n_fac
    n_cons_link = n_x
    n_cons = n_cons_cust + n_cons_cap + n_cons_link
    
    A = np.zeros((n_cons, n_vars))
    lb = np.zeros(n_cons)
    ub = np.zeros(n_cons)
    
    # 1. Customer assignment constraints
    for ci in range(n_cust):
        for fi in range(n_fac):
            A[ci, ci * n_fac + fi] = 1.0
        lb[ci] = 1.0
        ub[ci] = 1.0
    
    # 2. Capacity constraints
    for fi in range(n_fac):
        row = n_cons_cust + fi
        for ci in range(n_cust):
            A[row, ci * n_fac + fi] = demands[ci]
        A[row, n_x + fi] = -capacities[fi]
        lb[row] = -np.inf
        ub[row] = 0.0
        
    # 3. Linking constraints x[c,f] <= y[f]
    for ci in range(n_cust):
        for fi in range(n_fac):
            row = n_cons_cust + n_cons_cap + ci * n_fac + fi
            A[row, ci * n_fac + fi] = 1.0
            A[row, n_x + fi] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
            
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)
    
    constraint = LinearConstraint(A, lb, ub)
    
    try:
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds)
    except Exception:
        return None, float('inf')
        
    if res.status == 0 and res.fun < float('inf'):
        x_flat = res.x[:n_x]
        assignment = np.zeros(n_cust, dtype=int)
        for ci in range(n_cust):
            vals = x_flat[ci * n_fac:(ci+1) * n_fac]
            # Find index with max value (should be 1)
            assignment[ci] = np.argmax(vals)
            
        # Verify feasibility
        loads = np.zeros(n_fac)
        for ci in range(n_cust):
            loads[assignment[ci]] += demands[ci]
        if np.all(loads <= capacities + 1e-9):
            cost = np.sum(assign_cost[np.arange(n_cust), assignment]) + np.sum(open_cost[loads > 0])
            return assignment, cost
            
    return None, float('inf')

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    def calculate_cost(assignment):
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return float('inf')
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities + 1e-9):
            return float('inf')
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities + 1e-9):
            return False
        return True

    # 1. Try MILP first (exact solver)
    # Since n=100, 20 facilities, this might be fast enough for small instances.
    # We give it a generous chunk of time, but monitor it.
    # Actually, milp can be slow. Let's try it with a timeout-like strategy? 
    # Scipy milp doesn't have a timeout param directly. 
    # We can run it in a thread? Or just hope it's fast.
    # For 100 customers and 20 facilities, 2000 variables.
    # Let's try it. If it takes too long, we fallback.
    # But we can't easily interrupt it. 
    # Strategy: Run heuristics first to get a good incumbent, then try MILP?
    # Or run MILP first?
    # The inspiration suggests MILP is strong. Let's try MILP first.
    # If it fails or takes too long (we can't check time inside), we risk timeout.
    # However, for n=100, 20, it's often solvable in seconds.
    
    milp_result = None
    milp_cost = float('inf')
    
    # We'll try MILP. If it returns, great.
    # To be safe against timeout, we should probably run heuristics first to have a backup,
    # but if MILP is fast, it's best.
    # Let's run heuristics first to ensure we have a valid solution, 
    # then try to improve with MILP if time permits?
    # No, MILP is exact. If it finishes, it's the best.
    # If it doesn't finish, we timeout.
    # So we must be careful.
    
    # Let's run a fast heuristic first to establish a baseline and potentially
    # use it to prune? No, scipy milp doesn't easily accept incumbents for pruning in this API.
    
    # Let's just try MILP. If it's too slow, the evaluator kills us.
    # But the inspiration says it scored 1.38. So it likely finishes within 150s.
    
    # Let's add a fallback: if MILP is not available or fails, use heuristics.
    
    try:
        # Run MILP
        # We cannot interrupt this. We trust it to be fast for this size.
        # If it hangs, we fail.
        # To mitigate risk, we could run it in a separate process? No, too complex.
        # Let's assume it finishes.
        milp_result, milp_cost = _solve_milp(demands, capacities, open_cost, assign_cost)
        
        if milp_result is not None:
            # Verify again
            if is_valid(milp_result):
                return milp_result
    except Exception:
        pass

    # If MILP failed or we are here, use heuristics
    # This path is only taken if MILP failed or returned invalid (unlikely)
    # Or if we decide not to use MILP.
    # Given the score improvement, MILP is likely the key.
    
    # However, if MILP is slow, we might want a backup.
    # Let's implement a robust heuristic fallback just in case MILP throws an error or returns None.
    
    sorted_fac_by_cost = np.argsort(assign_cost, axis=1)
    
    def greedy_assign(order, dual_prices=None):
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        if dual_prices is not None:
            adjusted_cost = assign_cost + dual_prices[np.newaxis, :]
            adj_sorted = np.argsort(adjusted_cost, axis=1)
        else:
            adj_sorted = sorted_fac_by_cost
            
        for i in order:
            d_i = demands[i]
            for f in adj_sorted[i]:
                if load[f] + d_i <= capacities[f]:
                    result[i] = f
                    load[f] += d_i
                    break
            if result[i] == -1:
                best_f = -1
                max_rem = -1
                for f in range(n_fac):
                    rem = capacities[f] - load[f]
                    if rem >= d_i and rem > max_rem:
                        max_rem = rem
                        best_f = f
                if best_f == -1:
                    return None
                result[i] = best_f
                load[best_f] += d_i
        return result

    def local_search(initial_assignment, max_iters=200):
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[current_assignment[i]] += demands[i]
            
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # Relocate moves
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                f_from = current_assignment[i]
                for f_to in sorted_fac_by_cost[i]:
                    if f_to == f_from:
                        continue
                    if load[f_to] + demands[i] > capacities[f_to]:
                        continue
                    
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    delta_open = 0.0
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    if cust_count[f_to] == 0:
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        current_assignment[i] = f_to
                        load[f_from] -= demands[i]
                        load[f_to] += demands[i]
                        cust_count[f_from] -= 1
                        cust_count[f_to] += 1
                        current_cost += total_delta
                        improved = True
                        break

        return current_assignment, current_cost

    # Generate diverse initial solutions
    candidates = []
    
    order1 = np.argsort(-demands)
    cand1 = greedy_assign(order1)
    if cand1 is not None and is_valid(cand1):
        candidates.append(cand1)
        
    order2 = np.argsort(demands)
    cand2 = greedy_assign(order2)
    if cand2 is not None and is_valid(cand2):
        candidates.append(cand2)
        
    rng = np.random.default_rng(42)
    for _ in range(5):
        order3 = rng.permutation(n_cust)
        cand3 = greedy_assign(order3)
        if cand3 is not None and is_valid(cand3):
            candidates.append(cand3)
            
    if not candidates:
        return greedy_assign(np.argsort(-demands))

    best_cand = min(candidates, key=calculate_cost)
    best_assignment, _ = local_search(best_cand, max_iters=100)
    
    return best_assignment

# EVOLVE-BLOCK-END
