"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Solves the Capacitated Facility Location Problem using MILP.
    
    Parameters:
    demands: array of shape (100,)
    capacities: array of shape (20,)
    open_cost: array of shape (20,)
    assign_cost: array of shape (100, 20)
    
    Returns:
    assignment: array of shape (100,) with facility indices
    """
    I = len(demands)
    J = len(capacities)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Fallback heuristic in case MILP fails
    def greedy_fallback():
        result = np.zeros(I, dtype=int)
        load = np.zeros(J)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Sort facilities by cost for this customer
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                # Should not happen if problem is feasible, but handle just in case
                # Assign to facility with most remaining capacity
                rem = capacities - load
                f = np.argmax(rem)
                result[i] = f
                load[f] += demands[i]
        return result

    try:
        # Variable ordering:
        # x_ij for i in 0..I-1, j in 0..J-1 -> index i*J + j
        # y_j for j in 0..J-1 -> index I*J + j
        n_x = I * J
        n_y = J
        n_vars = n_x + n_y
        
        # Objective coefficients
        c = np.zeros(n_vars)
        c[:n_x] = assign_cost.flatten()
        c[n_x:] = open_cost
        
        # Equality constraints: sum_j x_ij = 1 for each i
        A_eq = np.zeros((I, n_vars))
        for i in range(I):
            for j in range(J):
                A_eq[i, i * J + j] = 1.0
        b_eq = np.ones(I)
        
        # Inequality constraints:
        # 1. Capacity: sum_i d_i x_ij <= cap_j
        # 2. Linking: x_ij - y_j <= 0
        
        n_cap = J
        n_link = I * J
        n_ineq = n_cap + n_link
        
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)
        
        # Capacity constraints
        for j in range(J):
            for i in range(I):
                A_ub[j, i * J + j] = demands[i]
            b_ub[j] = capacities[j]
            
        # Linking constraints
        # x_ij <= y_j  =>  x_ij - y_j <= 0
        idx = n_cap
        for i in range(I):
            for j in range(J):
                A_ub[idx, i * J + j] = 1.0
                A_ub[idx, n_x + j] = -1.0
                # b_ub[idx] is already 0
                idx += 1
        
        # Bounds
        bounds = [(0, 1)] * n_vars
        
        # Integrality: all binary
        integrality = np.ones(n_vars)
        
        # Solve
        res = so.milp(c=c, constraints=[so.LinearConstraint(A_ub, ub=b_ub), 
                                         so.LinearConstraint(A_eq, lb=b_eq, ub=b_eq)],
                      integrality=integrality, bounds=bounds, 
                      options={'time_limit': 100})
        
        if time.time() > deadline:
            return greedy_fallback()
            
        if res.status == 0: # Optimal
            x_flat = res.x[:n_x]
            # Round to nearest integer (should be close to 0 or 1)
            x = np.round(x_flat).reshape(I, J).astype(int)
            
            # Verify and extract assignment
            assignment = np.zeros(I, dtype=int)
            for i in range(I):
                # Find the facility with x=1
                # Due to numerical issues, it might not be exactly 1, so take argmax
                assignment[i] = np.argmax(x[i])
            
            # Final validation
            load = np.zeros(J)
            for i in range(I):
                load[assignment[i]] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return assignment
            else:
                # If invalid due to rounding error, try to fix or fallback
                # Usually rounding 0.999 -> 1 is fine. If it's 0.5, it's bad.
                # Given binary integrality, this should be robust.
                return greedy_fallback()
        else:
            # Solver failed or timed out
            return greedy_fallback()
            
    except Exception:
        return greedy_fallback()

# EVOLVE-BLOCK-END
