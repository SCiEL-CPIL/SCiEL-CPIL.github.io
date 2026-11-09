"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _solve_milp(demands, capacities, open_cost, assign_cost, time_limit=10):
    """
    Solve the capacitated facility location problem using MILP.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables:
    # x[c, f] for c in 0..n_cust-1, f in 0..n_fac-1 (n_cust * n_fac vars)
    # y[f] for f in 0..n_fac-1 (n_fac vars)
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y
    
    # Objective coefficients
    c = np.zeros(n_vars)
    c[:n_x] = assign_cost.flatten()
    c[n_x:] = open_cost
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_f x[c,f] = 1
    #    Matrix A_eq: n_cust x n_vars
    A_eq = np.zeros((n_cust, n_vars))
    for c_idx in range(n_cust):
        for f_idx in range(n_fac):
            A_eq[c_idx, c_idx * n_fac + f_idx] = 1.0
    b_eq = np.ones(n_cust)
    
    # 2. Capacity constraints: sum_c demands[c] * x[c,f] <= capacities[f] * y[f]
    #    sum_c demands[c] * x[c,f] - capacities[f] * y[f] <= 0
    #    Matrix A_ub: n_fac x n_vars
    A_ub = np.zeros((n_fac, n_vars))
    for f_idx in range(n_fac):
        for c_idx in range(n_cust):
            A_ub[f_idx, c_idx * n_fac + f_idx] = demands[c_idx]
        A_ub[f_idx, n_x + f_idx] = -capacities[f_idx]
    b_ub = np.zeros(n_fac)
    
    # 3. Linking constraints: x[c,f] <= y[f]
    #    x[c,f] - y[f] <= 0
    #    This is often redundant if capacities are positive and demands > 0, 
    #    but good to include for robustness or if capacities are 0.
    #    Actually, if y[f]=0, then sum_c d_c x_cf <= 0 implies x_cf=0 for all c.
    #    So x_cf <= y_f is implied by capacity constraint if d_c > 0.
    #    We can skip this to reduce constraint count, or include it.
    #    Let's skip it to keep the model smaller, relying on capacity constraint.
    
    # Integrality: x and y are binary
    integrality = np.ones(n_vars)
    
    # Bounds: all variables between 0 and 1
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    # Combine constraints
    # LinearConstraint(A, lb, ub)
    # We have equality and inequality.
    # scipy.optimize.milp takes a list of LinearConstraint or a single one.
    # Let's combine into one if possible or pass as list.
    # Actually, milp accepts a single LinearConstraint or a sequence.
    # Let's create separate constraints.
    
    constraint_eq = LinearConstraint(A_eq, lb=b_eq, ub=b_eq)
    constraint_ub = LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    
    # Solve
    try:
        res = milp(c=c, constraints=[constraint_eq, constraint_ub], integrality=integrality, bounds=bounds, options={"time_limit": time_limit})
        
        if res.status == 0:
            x_flat = res.x[:n_x]
            # Rounding: x should be close to 0 or 1
            x = np.round(x_flat).reshape(n_cust, n_fac)
            
            # Verify assignment
            assignment = np.argmax(x, axis=1)
            
            # Check validity
            loads = np.zeros(n_fac)
            for i in range(n_cust):
                loads[assignment[i]] += demands[i]
            if np.any(loads > capacities + 1e-6):
                # If invalid due to numerical issues, try to repair or return None
                return None
            return assignment
        else:
            return None
    except Exception:
        return None

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Try MILP first with reduced time to free budget for local search
    assignment = _solve_milp(demands, capacities, open_cost, assign_cost, time_limit=5)
    
    if assignment is not None:
        # Still run local search to improve the MILP solution
        pass
    else:
        assignment = None
    
    # Helper functions
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

    def greedy_assign(order, dual_prices=None):
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        if dual_prices is not None:
            adjusted_cost = assign_cost + dual_prices[np.newaxis, :]
            adj_sorted = np.argsort(adjusted_cost, axis=1)
        else:
            adj_sorted = np.argsort(assign_cost, axis=1)
            
        for i in order:
            d_i = demands[i]
            for f in adj_sorted[i]:
                if load[f] + d_i <= capacities[f] + 1e-9:
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

        sorted_fac_by_cost = np.argsort(assign_cost, axis=1)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                f_from = current_assignment[i]
                for f_to in sorted_fac_by_cost[i]:
                    if f_to == f_from:
                        continue
                    if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
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

    def two_opt_swap(assignment, max_iters=50):
        """Try swapping pairs of customers between facilities."""
        current_assignment = assignment.copy()
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
            
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                for j in range(i+1, n_cust):
                    if time.time() > deadline:
                        break
                    f_i = current_assignment[i]
                    f_j = current_assignment[j]
                    if f_i == f_j:
                        continue
                    
                    # Check if swap is feasible
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                        continue
                    
                    # Calculate delta
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Open cost change
                    delta_open = 0.0
                    if cust_count[f_i] == 1:
                        # f_i becomes empty
                        delta_open -= open_cost[f_i]
                    if cust_count[f_j] == 1:
                        # f_j becomes empty
                        delta_open -= open_cost[f_j]
                    # After swap, if either was empty before, it's still empty (count stays 1)
                    # Actually need to be more careful:
                    # Before: f_i has cust_count[f_i], f_j has cust_count[f_j]
                    # After swap: f_i has cust_count[f_i] - 1 + 1 = same, f_j has same
                    # So open cost only changes if a facility becomes empty or was empty
                    # Let me recalculate:
                    # Before swap: f_i count = c_i, f_j count = c_j
                    # After swap: f_i count = c_i - 1 + 1 = c_i, f_j count = c_j - 1 + 1 = c_j
                    # So open cost doesn't change for 2-opt swaps!
                    # Wait, that's only if both facilities stay non-empty.
                    # If c_i == 1, after removing customer i, f_i has 0, then adding j makes it 1.
                    # So it was open, becomes temporarily closed, then open again. No net change.
                    # Actually the open cost is based on final state, not intermediate.
                    # Final state: f_i has c_i customers (same as before), f_j has c_j customers (same as before)
                    # So delta_open = 0 for 2-opt swaps.
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        current_assignment[i] = f_j
                        current_assignment[j] = f_i
                        load[f_i] -= demands[i]
                        load[f_j] -= demands[j]
                        load[f_i] += demands[j]
                        load[f_j] += demands[i]
                        # cust_count doesn't change for 2-opt swaps
                        current_cost += total_delta
                        improved = True
                        break
                if improved:
                    continue  # Restart from i=0 after a swap

        return current_assignment, current_cost

    # Get initial assignment
    if assignment is None:
        order1 = np.argsort(-demands)
        cand1 = greedy_assign(order1)
        
        if cand1 is None or not is_valid(cand1):
            # Last resort
            return np.zeros(n_cust, dtype=int)
        best_assignment = cand1
    else:
        best_assignment = assignment

    # Run local search
    best_assignment, _ = local_search(best_assignment, max_iters=150)
    
    # Run 2-opt swap
    if time.time() < deadline - 5:
        best_assignment, _ = two_opt_swap(best_assignment, max_iters=30)
    
    # Final high-precision polish with more iterations
    if time.time() < deadline - 3:
        best_assignment, _ = local_search(best_assignment, max_iters=100)
    
    if is_valid(best_assignment):
        return best_assignment
    return best_assignment

# EVOLVE-BLOCK-END
