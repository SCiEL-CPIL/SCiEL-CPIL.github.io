# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Uses Lagrangian relaxation with subgradient method for capacity constraints,
    followed by MILP polish and local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 118.0  # safety margin for 150s limit
    
    def _compute_obj(result):
        """Compute objective for a valid assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities):
            return float('inf')
        open_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            open_mask[f] = True
        return np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_cust), result])
    
    def _greedy(order):
        """Greedy assignment in given order."""
        fac_order = np.argsort(assign_cost, axis=1)
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Try to find any facility with remaining capacity
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
            if not assigned:
                return None
        return result
    
    best_result = None
    best_cost = float('inf')
    
    # Try greedy with largest demand first
    order1 = np.argsort(-demands)
    r = _greedy(order1)
    if r is not None:
        c = _compute_obj(r)
        if c < best_cost:
            best_cost = c
            best_result = r.copy()
    
    # Try greedy with smallest demand first
    order2 = np.argsort(demands)
    r = _greedy(order2)
    if r is not None:
        c = _compute_obj(r)
        if c < best_cost:
            best_cost = c
            best_result = r.copy()
    
    # Lagrangian relaxation with subgradient method
    def _lagrangian_subgradient(max_iters=200):
        """
        Solve Lagrangian relaxation of capacity constraints.
        Multipliers lambda[j] for capacity constraints.
        Subproblem: for each customer i, min over j of (assign_cost[i,j] + lambda[j]*demands[i]) 
                   plus open_cost if facility j is first used.
        """
        if time.time() > deadline - 10:
            return None, None
        
        # Initialize multipliers
        lam = np.zeros(n_fac, dtype=np.float64)
        best_lagrangian_result = None
        best_lagrangian_cost = float('inf')
        best_primal_result = None
        best_primal_cost = float('inf')
        
        # Precompute for efficiency
        # For each iteration, we need to solve the relaxed problem
        
        step_size_factor = 1.0
        prev_dual = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Solve the Lagrangian subproblem
            # For each customer i, choose facility j minimizing:
            #   assign_cost[i,j] + lam[j] * demands[i] + (open_cost[j] if j is new)
            # This is decoupled per customer, but open_cost coupling exists.
            # Approximation: ignore open_cost coupling in subproblem, handle in rounding.
            
            # Vectorized: cost_matrix[i,j] = assign_cost[i,j] + lam[j] * demands[i]
            cost_matrix = assign_cost + np.outer(demands, lam)
            
            # For each customer, pick cheapest facility
            fac_choice = np.argmin(cost_matrix, axis=1)
            
            # Compute loads
            load = np.zeros(n_fac, dtype=np.int64)
            for i, f in enumerate(fac_choice):
                load[f] += demands[i]
            
            # Compute dual objective
            # Dual obj = sum_i min_j (assign_cost[i,j] + lam[j]*demands[i]) + sum_j lam[j]*capacities[j]
            min_costs = np.min(cost_matrix, axis=1)
            dual_obj = np.sum(min_costs) + np.dot(lam, capacities)
            
            # Try to round/repair the solution to get a feasible primal
            # Start with fac_choice and fix capacity violations
            result_primal = fac_choice.copy()
            load_primal = load.copy()
            
            # Fix over-capacity facilities by reassigning customers
            for _ in range(50):  # max repair iterations
                if time.time() > deadline - 10:
                    break
                
                # Find facilities over capacity
                over_fac = np.where(load_primal > capacities)[0]
                if len(over_fac) == 0:
                    break
                
                # For each over-capacity facility, move some customers
                for f_over in over_fac:
                    if load_primal[f_over] <= capacities[f_over]:
                        continue
                    
                    # Find customers assigned to f_over, sort by how much they "prefer" it
                    custs_at_f = np.where(result_primal == f_over)[0]
                    if len(custs_at_f) == 0:
                        continue
                    
                    # For each customer, compute cost of moving to their 2nd best facility
                    move_costs = []
                    for i in custs_at_f:
                        # Find best alternative facility
                        sorted_facs = np.argsort(cost_matrix[i])
                        best_alt = None
                        for f_alt in sorted_facs:
                            if f_alt != f_over and load_primal[f_alt] + demands[i] <= capacities[f_alt]:
                                best_alt = f_alt
                                break
                        if best_alt is not None:
                            delta = cost_matrix[i, best_alt] - cost_matrix[i, f_over]
                            move_costs.append((delta, i, best_alt))
                        else:
                            # No feasible alternative, must move anyway (will cause infeasibility)
                            move_costs.append((float('inf'), i, -1))
                    
                    if len(move_costs) == 0:
                        continue
                    
                    # Move customers with lowest cost delta first
                    move_costs.sort(key=lambda x: x[0])
                    
                    excess = load_primal[f_over] - capacities[f_over]
                    for delta, i, f_alt in move_costs:
                        if excess <= 0:
                            break
                        if f_alt >= 0:
                            load_primal[f_over] -= demands[i]
                            load_primal[f_alt] += demands[i]
                            result_primal[i] = f_alt
                            excess -= demands[i]
                    
                    # If still over capacity after moving all, we have infeasibility
                    if load_primal[f_over] > capacities[f_over]:
                        # Force move the largest demand customer to any facility with space
                        custs_at_f = np.where(result_primal == f_over)[0]
                        if len(custs_at_f) > 0:
                            # Find customer with largest demand
                            max_d_idx = custs_at_f[np.argmax(demands[custs_at_f])]
                            # Find any facility with space
                            for f_alt in range(n_fac):
                                if f_alt != f_over and load_primal[f_alt] + demands[max_d_idx] <= capacities[f_alt]:
                                    load_primal[f_over] -= demands[max_d_idx]
                                    load_primal[f_alt] += demands[max_d_idx]
                                    result_primal[max_d_idx] = f_alt
                                    break
        
        # Check if primal solution is feasible
        if np.all(load_primal <= capacities):
            c_primal = _compute_obj(result_primal)
            if c_primal < best_primal_cost:
                best_primal_cost = c_primal
                best_primal_result = result_primal.copy()
        
        # Update multipliers using subgradient
        # Subgradient of dual w.r.t. lam[j] is: capacities[j] - load[j]
        subgrad = capacities - load
        subgrad_norm = np.linalg.norm(subgrad)
        
        if subgrad_norm < 1e-10:
            break
        
        # Step size: use Polyak step size if we have an upper bound
        upper_bound = best_primal_cost if best_primal_cost < float('inf') else dual_obj + 1000
        gap = upper_bound - dual_obj
        if gap > 0:
            step_size = gap / (subgrad_norm ** 2)
        else:
            step_size = 1.0 / (it + 1)
        
        # Damp the step size
        step_size = min(step_size, 10.0)
        
        lam_new = lam + step_size * subgrad
        lam_new = np.maximum(lam_new, 0.0)  # Multipliers should be non-negative
        
        # Check if we should stop
        if abs(dual_obj - prev_dual) < 1e-6:
            break
        prev_dual = dual_obj
        lam = lam_new
    
    # Run Lagrangian relaxation
    if time.time() < deadline - 15:
        lagr_result, lagr_cost = _lagrangian_subgradient(max_iters=150)
        if lagr_result is not None:
            c = _compute_obj(lagr_result)
            if c < best_cost:
                best_cost = c
                best_result = lagr_result.copy()
    
    # MILP polish if time allows
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        if time.time() < deadline - 25:
            # Variables: x[i*n_fac + j] = 1 if customer i assigned to facility j
            #             x[n_cust*n_fac + j] = 1 if facility j is open
            n_vars = n_cust * n_fac + n_fac
            
            # Objective
            c_obj = np.zeros(n_vars, dtype=np.float64)
            for i in range(n_cust):
                for j in range(n_fac):
                    c_obj[i * n_fac + j] = assign_cost[i, j]
            for j in range(n_fac):
                c_obj[n_cust * n_fac + j] = open_cost[j]
            
            # Constraints
            n_constraints = n_cust + n_fac + n_cust * n_fac
            A = np.zeros((n_constraints, n_vars), dtype=np.float64)
            lb = np.zeros(n_constraints, dtype=np.float64)
            ub = np.zeros(n_constraints, dtype=np.float64)
            
            # Constraint 1: sum_j x[i,j] = 1
            for i in range(n_cust):
                for j in range(n_fac):
                    A[i, i * n_fac + j] = 1.0
                lb[i] = 1.0
                ub[i] = 1.0
            
            # Constraint 2: sum_i demands[i]*x[i,j] - capacities[j]*y[j] <= 0
            for j in range(n_fac):
                row = n_cust + j
                for i in range(n_cust):
                    A[row, i * n_fac + j] = demands[i]
                A[row, n_cust * n_fac + j] = -capacities[j]
                lb[row] = -np.inf
                ub[row] = 0.0
            
            # Constraint 3: x[i,j] - y[j] <= 0
            for i in range(n_cust):
                for j in range(n_fac):
                    row = n_cust + n_fac + i * n_fac + j
                    A[row, i * n_fac + j] = 1.0
                    A[row, n_cust * n_fac + j] = -1.0
                    lb[row] = -np.inf
                    ub[row] = 0.0
            
            # Variable bounds
            var_lb = np.zeros(n_vars, dtype=np.float64)
            var_ub = np.ones(n_vars, dtype=np.float64)
            
            # Integrality
            integrality = np.ones(n_vars, dtype=np.int8)
            
            constraint = LinearConstraint(A, lb, ub)
            bounds = Bounds(var_lb, var_ub)
            
            remaining_time = deadline - time.time() - 5
            if remaining_time > 5:
                res = milp(
                    c=c_obj,
                    constraints=constraint,
                    integrality=integrality,
                    bounds=bounds,
                    options={"time_limit": remaining_time, "mip_rel_gap": 0.01}
                )
                
                if res.status in (0, 2) and res.x is not None:
                    x_sol = res.x
                    result_milp = np.zeros(n_cust, dtype=np.int64)
                    valid = True
                    for i in range(n_cust):
                        assigned = False
                        for j in range(n_fac):
                            if x_sol[i * n_fac + j] > 0.5:
                                result_milp[i] = j
                                assigned = True
                                break
                        if not assigned:
                            valid = False
                            break
                    
                    if valid:
                        c_milp = _compute_obj(result_milp)
                        if c_milp < best_cost:
                            best_cost = c_milp
                            best_result = result_milp.copy()
    
    except Exception:
        pass
    
    # Final local search polish
    if best_result is not None and time.time() < deadline - 3:
        fac_order = np.argsort(assign_cost, axis=1)
        
        def _local_search(result, max_iters=200):
            result = result.copy()
            load = np.zeros(n_fac, dtype=np.int64)
            for i, f in enumerate(result):
                load[f] += demands[i]
            
            current_obj = _compute_obj(result)
            
            improved = True
            iters = 0
            while improved and iters < max_iters and time.time() < deadline:
                improved = False
                iters += 1
                for i in range(n_cust):
                    if time.time() > deadline:
                        break
                    cur_f = result[i]
                    for f in fac_order[i]:
                        if f == cur_f:
                            continue
                        if load[f] + demands[i] > capacities[f]:
                            continue
                        delta_assign = assign_cost[i, f] - assign_cost[i, cur_f]
                        delta_open = 0.0
                        if load[cur_f] - demands[i] == 0:
                            delta_open -= open_cost[cur_f]
                        if load[f] == 0:
                            delta_open += open_cost[f]
                        total_delta = delta_assign + delta_open
                        if total_delta < -1e-9:
                            load[cur_f] -= demands[i]
                            load[f] += demands[i]
                            result[i] = f
                            current_obj += total_delta
                            improved = True
                            break
            
            return result
        
        best_result = _local_search(best_result, max_iters=200)
        best_cost = _compute_obj(best_result)
    
    if best_result is None:
        # Ultimate fallback
        result = np.array([i % n_fac for i in range(n_cust)], dtype=np.int64)
        return result
    
    return best_result.astype(int)

# EVOLVE-BLOCK-END
