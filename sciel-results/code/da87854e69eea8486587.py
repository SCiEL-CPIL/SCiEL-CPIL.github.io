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

    def lagrangian_relaxation(num_iters=300, step_init=1.0):
        """Lagrangian relaxation with subgradient optimization on capacity constraints."""
        best_primal = None
        best_primal_cost = float('inf')
        
        # Dual variables for capacity constraints
        lambda_ = np.zeros(n_fac)
        step_size = step_init
        
        # Precompute sorted facility indices for each customer
        for it in range(num_iters):
            if time.time() > deadline - 10:
                break
            
            # Adjusted costs
            adjusted_cost = assign_cost + lambda_[np.newaxis, :]
            
            # Greedy assignment: process customers in order of largest demand first
            # This helps with capacity feasibility
            order = np.argsort(-demands)
            result = greedy_assign(order, dual_prices=lambda_)
            
            if result is None:
                # Try without dual prices
                result = greedy_assign(order, dual_prices=None)
                if result is None:
                    break
            
            # Check if feasible
            loads = np.zeros(n_fac)
            for i in range(n_cust):
                loads[result[i]] += demands[i]
            
            if np.all(loads <= capacities + 1e-9):
                # Feasible solution
                cost = calculate_cost(result)
                if cost < best_primal_cost:
                    best_primal_cost = cost
                    best_primal = result.copy()
            
            # Subgradient update: g_f = sum_c demands[c]*x[c,f] - capacities[f]*y_f
            # Since we don't have y_f explicitly, use: g_f = loads[f] - capacities[f]
            # (when facility is used, y_f=1; when not used, y_f=0 but loads[f]=0)
            # Actually for Lagrangian: g_f = sum_c d_c x_cf - cap_f * y_f
            # We approximate y_f = 1 if loads[f] > 0 else 0
            y_f = (loads > 0).astype(float)
            g = loads - capacities * y_f
            
            # Step size update (Polyak-style)
            # LB is a lower bound; we use the best feasible cost as UB
            # Simple step: step = alpha * (UB - LB) / ||g||^2
            # We don't have a good LB, so use diminishing step
            step_size = step_init / (1.0 + 0.01 * it)
            
            # Update multipliers (project to non-negative)
            lambda_ = np.maximum(0, lambda_ + step_size * g)
        
        return best_primal, best_primal_cost

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
            
            swapped = False
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
                    
                    # For 2-opt swaps, cust_count doesn't change, so delta_open = 0
                    total_delta = delta_assign
                    
                    if total_delta < -1e-9:
                        current_assignment[i] = f_j
                        current_assignment[j] = f_i
                        load[f_i] -= demands[i]
                        load[f_j] -= demands[j]
                        load[f_i] += demands[j]
                        load[f_j] += demands[i]
                        current_cost += total_delta
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break  # Restart the outer loop from i=0

        return current_assignment, current_cost

    # Phase 1: Try MILP with short time limit
    assignment_milp = _solve_milp(demands, capacities, open_cost, assign_cost, time_limit=3)
    
    # Phase 2: Lagrangian relaxation to find good feasible solutions
    lag_assignment, lag_cost = lagrangian_relaxation(num_iters=200, step_init=0.5)
    
    # Phase 3: Greedy fallback
    order1 = np.argsort(-demands)
    greedy_cand = greedy_assign(order1)
    
    # Collect all candidate solutions
    candidates = []
    if assignment_milp is not None and is_valid(assignment_milp):
        candidates.append(assignment_milp)
    if lag_assignment is not None and is_valid(lag_assignment):
        candidates.append(lag_assignment)
    if greedy_cand is not None and is_valid(greedy_cand):
        candidates.append(greedy_cand)
    
    if not candidates:
        # Last resort
        return np.zeros(n_cust, dtype=int)
    
    # Phase 4: Local search on each candidate, keep best
    best_assignment = None
    best_cost = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        ls_result, ls_cost = local_search(cand, max_iters=100)
        if ls_cost < best_cost and is_valid(ls_result):
            best_cost = ls_cost
            best_assignment = ls_result
    
    if best_assignment is None:
        best_assignment = candidates[0]
    
    # Phase 5: 2-opt swap on best
    if time.time() < deadline - 10:
        best_assignment, _ = two_opt_swap(best_assignment, max_iters=30)
    
    # Phase 6: Final polish
    if time.time() < deadline - 5:
        best_assignment, _ = local_search(best_assignment, max_iters=150)
    
    if is_valid(best_assignment):
        return best_assignment
    return best_assignment

# EVOLVE-BLOCK-END
