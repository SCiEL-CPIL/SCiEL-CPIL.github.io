# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def solve_mcf(demands, capacities, open_facilities, assign_cost):
    """
    Solve the min-cost flow (transportation) problem for a fixed set of open facilities.
    Returns: assignment array (length n_customers) or None if infeasible.
    """
    n_customers = len(demands)
    n_open = len(open_facilities)
    
    if n_open == 0:
        return None
        
    # Map open facility indices to 0..n_open-1
    # Variables x[i, j] where i is customer, j is index in open_facilities
    # Total variables: n_customers * n_open
    
    # Objective coefficients
    c = np.zeros(n_customers * n_open)
    for i in range(n_customers):
        for j, f in enumerate(open_facilities):
            c[i * n_open + j] = assign_cost[i, f]
            
    # Inequality constraints: capacity constraints for each open facility
    # sum_i demands[i] * x[i, j] <= capacities[f]
    A_ub = np.zeros((n_open, n_customers * n_open))
    for j, f in enumerate(open_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_open + j] = demands[i]
    b_ub = np.array([capacities[f] for f in open_facilities])
    
    # Equality constraints: each customer assigned to exactly one facility
    # sum_j x[i, j] = 1
    A_eq = np.zeros((n_customers, n_customers * n_open))
    for i in range(n_customers):
        for j in range(n_open):
            A_eq[i, i * n_open + j] = 1
    b_eq = np.ones(n_customers)
    
    # Bounds: x >= 0
    bounds = [(0, None)] * (n_customers * n_open)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        return None
        
    x = res.x.reshape(n_customers, n_open)
    
    # Recover integer assignment
    # Due to LP integrality, x should be close to 0 or 1
    # But to be safe, pick the argmax for each customer
    assignment = np.zeros(n_customers, dtype=int)
    for i in range(n_customers):
        # Find the open facility with max flow
        max_val = -1
        best_j = -1
        for j in range(n_open):
            if x[i, j] > max_val:
                max_val = x[i, j]
                best_j = j
        assignment[i] = open_facilities[best_j]
        
    return assignment

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Initial set of open facilities: all of them
    current_open = set(range(n_facilities))
    
    # Initial assignment using MCF on all facilities
    current_assignment = solve_mcf(demands, capacities, list(current_open), assign_cost)
    
    if current_assignment is None:
        # Fallback to greedy if MCF fails (should not happen if total cap > total demand)
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_cost = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    cost = assign_cost[i, f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f >= 0:
                load[best_f] += demands[i]
                result[i] = best_f
            else:
                result[i] = 0
        return result

    # Calculate current cost
    def calc_cost(assignment, open_set):
        cost = 0.0
        used = set(assignment.tolist())
        for f in used:
            if f in open_set:
                cost += open_cost[f]
        for i in range(n_customers):
            cost += assign_cost[i, assignment[i]]
        return cost

    current_cost = calc_cost(current_assignment, current_open)
    
    improved = True
    max_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 1. Try closing facilities
        # Iterate over a copy of the list to allow modification
        facilities_to_check = sorted(list(current_open))
        for f in facilities_to_check:
            # Only try to close if it's currently used (if not used, it's already effectively closed in cost, but keeping it in set doesn't hurt MCF, 
            # but removing it simplifies the problem. However, if it's not used, closing it saves no cost but might simplify. 
            # Actually, if it's not used, it contributes 0 to assignment cost and we don't pay opening cost if we define cost based on *used* facilities.
            # My calc_cost pays opening cost only if f is in used AND f is in open_set? 
            # Wait, the problem says "open_cost over facilities with at least one customer".
            # So if a facility is in `current_open` but has 0 customers, we don't pay for it.
            # So we should remove unused facilities from `current_open` to keep the set clean.
            
            # Check if f is used in current_assignment
            if np.sum(current_assignment == f) == 0:
                # Remove it from open set, no cost change
                current_open.remove(f)
                improved = True
                # Re-solve? No, removing an unused facility doesn't change the optimal assignment for the remaining customers.
                # But it changes the MCF problem size. We can just continue.
                # However, the current_assignment is still valid for the reduced set.
                continue
            
            # Try closing a used facility
            trial_open = current_open - {f}
            if not trial_open:
                continue
                
            trial_assignment = solve_mcf(demands, capacities, list(trial_open), assign_cost)
            if trial_assignment is None:
                continue
                
            trial_cost = calc_cost(trial_assignment, trial_open)
            if trial_cost < current_cost - 1e-9:
                current_open = trial_open
                current_assignment = trial_assignment
                current_cost = trial_cost
                improved = True
                
        # 2. Try opening facilities
        # Iterate over facilities NOT in current_open
        facilities_to_check = sorted(set(range(n_facilities)) - current_open)
        for f in facilities_to_check:
            trial_open = current_open | {f}
            trial_assignment = solve_mcf(demands, capacities, list(trial_open), assign_cost)
            if trial_assignment is None:
                continue
                
            trial_cost = calc_cost(trial_assignment, trial_open)
            if trial_cost < current_cost - 1e-9:
                current_open = trial_open
                current_assignment = trial_assignment
                current_cost = trial_cost
                improved = True
                
    return current_assignment
# EVOLVE-BLOCK-END
