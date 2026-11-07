import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Flatten assign_cost for LP objective
    # x variables: n_customers * n_facilities
    # y variables: n_facilities
    # Total variables: n_customers * n_facilities + n_facilities
    
    c_vec = assign_cost.flatten()
    f_vec = open_cost
    
    # Objective: sum(c_ij * x_ij) + sum(f_j * y_j)
    c_obj = np.concatenate([c_vec, f_vec])
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility:
    # sum_j x_ij = 1 for all i
    # This gives n_customers equality constraints.
    A_eq = np.zeros((n_customers, n_customers * n_facilities + n_facilities))
    for i in range(n_customers):
        # x_ij are at indices i*n_facilities to (i+1)*n_facilities - 1
        A_eq[i, i*n_facilities : (i+1)*n_facilities] = 1.0
    b_eq = np.ones(n_customers)
    
    # 2. Capacity constraints: sum_i d_i * x_ij <= C_j * y_j
    # sum_i d_i * x_ij - C_j * y_j <= 0 for all j
    # This gives n_facilities inequality constraints.
    A_ub = np.zeros((n_facilities, n_customers * n_facilities + n_facilities))
    b_ub = np.zeros(n_facilities)
    
    for j in range(n_facilities):
        # x_ij terms for facility j
        # x_ij is at index i*n_facilities + j
        for i in range(n_customers):
            A_ub[j, i*n_facilities + j] = demands[i]
        # y_j term is at index n_customers * n_facilities + j
        A_ub[j, n_customers * n_facilities + j] = -capacities[j]
        b_ub[j] = 0.0
        
    # 3. Linking constraints: x_ij <= y_j  =>  x_ij - y_j <= 0
    # This gives n_customers * n_facilities inequality constraints.
    A_link = np.zeros((n_customers * n_facilities, n_customers * n_facilities + n_facilities))
    b_link = np.zeros(n_customers * n_facilities)
    
    idx = 0
    for i in range(n_customers):
        for j in range(n_facilities):
            A_link[idx, i*n_facilities + j] = 1.0
            A_link[idx, n_customers * n_facilities + j] = -1.0
            b_link[idx] = 0.0
            idx += 1
            
    # Combine inequality constraints
    A_ub_total = np.vstack([A_ub, A_link])
    b_ub_total = np.concatenate([b_ub, b_link])
    
    # Bounds: 0 <= x_ij <= 1, 0 <= y_j <= 1
    bounds = [(0, 1)] * (n_customers * n_facilities + n_facilities)
    
    # Solve LP
    x_solution = None
    try:
        res = linprog(c_obj, A_ub=A_ub_total, b_ub=b_ub_total, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            x_solution = res.x
    except Exception:
        pass
        
    if x_solution is not None:
        x_vals = x_solution[:n_customers * n_facilities].reshape(n_customers, n_facilities)
        y_vals = x_solution[n_customers * n_facilities:]
        
        # Initial Assignment from LP
        # Round x_vals to get integer assignment.
        # To handle capacity violations, we use a repair strategy.
        # First, get a tentative assignment by taking argmax of x_vals for each customer.
        # However, this might violate capacities.
        
        # Strategy: 
        # 1. Get LP-based priority.
        # 2. Sort customers by demand descending (greedy repair order).
        # 3. Assign to best feasible facility based on LP value or cost.
        
        # Let's try a local search starting from a greedy solution guided by LP.
        
        # Generate a valid initial solution using Greedy (largest demand first)
        # but using LP values as tie-breakers or primary costs?
        # Actually, just use the standard greedy to ensure a valid start, then improve.
        # Or, better: use the LP solution to identify "likely" facilities and try to fit them.
        
        # Let's stick to a robust Local Search on a valid initial solution.
        # Initial solution: Greedy (Largest Demand First, Cheapest Cost)
        demands_arr = np.array(demands)
        capacities_arr = np.array(capacities)
        open_cost_arr = np.array(open_cost)
        assign_cost_arr = np.array(assign_cost)
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        
        # Greedy Initialization
        order = np.argsort(-demands_arr)
        for i in order:
            # Find cheapest feasible facility
            feasible = [f for f in range(n_facilities) if load[f] + demands_arr[i] <= capacities_arr[f]]
            if not feasible:
                # Fallback: assign to least loaded or cheapest, might be invalid? 
                # Task says invalid result is bad. We must ensure validity.
                # If no feasible facility, we might have to break. 
                # But total capacity should be sufficient. 
                # If not, we can't solve. Assume solvable.
                pass
            else:
                # Choose facility with min assign_cost[i, f]
                costs = [assign_cost_arr[i, f] for f in feasible]
                best_f = feasible[np.argmin(costs)]
                result[i] = best_f
                load[best_f] += demands_arr[i]
                
        # Local Search Improvement
        # Objective: Sum of assign_cost[i, result[i]] + Sum of open_cost[j] for used facilities
        
        def get_cost(res):
            used = set(res)
            open_c = sum(open_cost_arr[j] for j in used)
            assign_c = sum(assign_cost_arr[i, res[i]] for i in range(n_customers))
            return open_c + assign_c
            
        current_cost = get_cost(result)
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[result[i]] += demands_arr[i]
            
        improved = True
        max_iterations = 2000 # Safety cap
        iter_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            iter_count += 1
            if iter_count > max_iterations:
                break
                
            # Try moving each customer to a better facility
            for i in range(n_customers):
                current_f = result[i]
                # Try all other facilities
                for new_f in range(n_facilities):
                    if new_f == current_f:
                        continue
                    # Check capacity
                    if load[new_f] + demands_arr[i] <= capacities_arr[new_f]:
                        # Calculate delta cost
                        # Assign cost change
                        delta_assign = assign_cost_arr[i, new_f] - assign_cost_arr[i, current_f]
                        
                        # Open cost change
                        # Current facility load after removal
                        new_load_current = load[current_f] - demands_arr[i]
                        new_load_new = load[new_f] + demands_arr[i]
                        
                        # If current facility becomes empty, we save its open cost
                        # If new facility was empty, we pay its open cost
                        delta_open = 0
                        if new_load_current == 0:
                            delta_open -= open_cost_arr[current_f]
                        if new_load_new == demands_arr[i] and load[new_f] == 0:
                            # New facility was empty
                            delta_open += open_cost_arr[new_f]
                            
                        total_delta = delta_assign + delta_open
                        
                        if total_delta < -1e-9:
                            # Accept move
                            load[current_f] -= demands_arr[i]
                            load[new_f] += demands_arr[i]
                            result[i] = new_f
                            current_cost += total_delta
                            improved = True
                            # Break inner loop to restart evaluation for this customer?
                            # Or continue? Restarting is safer for convergence.
                            break
                            
    else:
        # Fallback if LP failed or wasn't used: Pure Greedy
        demands_arr = np.array(demands)
        capacities_arr = np.array(capacities)
        open_cost_arr = np.array(open_cost)
        assign_cost_arr = np.array(assign_cost)
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        
        order = np.argsort(-demands_arr)
        for i in order:
            feasible = [f for f in range(n_facilities) if load[f] + demands_arr[i] <= capacities_arr[f]]
            if feasible:
                costs = [assign_cost_arr[i, f] for f in feasible]
                best_f = feasible[np.argmin(costs)]
                result[i] = best_f
                load[best_f] += demands_arr[i]
            else:
                # Should not happen if instance is solvable
                result[i] = 0 
                load[0] += demands_arr[i]

    return result
