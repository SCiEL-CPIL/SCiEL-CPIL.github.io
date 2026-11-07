"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Pre-compute heuristic info for ACO
    # eta[i, k] = 1 / assign_cost[i, k]
    # Avoid division by zero
    assign_cost_safe = np.where(assign_cost == 0, 1e-6, assign_cost)
    eta = 1.0 / assign_cost_safe
    
    def compute_cost(assignment, load):
        assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        open_sum = np.sum(open_cost[load > 0])
        return assign_sum + open_sum

    def local_search(assignment, load, max_passes=100):
        """
        Fast local search using 1-moves and 2-swaps.
        """
        current_cost = compute_cost(assignment, load)
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # 1-Moves
            for i in range(n_customers):
                f_old = assignment[i]
                d_i = demands[i]
                # Find best move for i
                best_delta = 0
                best_f = -1
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    if load[f_new] + d_i > capacities[f_new]:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    if load[f_old] == d_i:
                        delta_open -= open_cost[f_old]
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f = f_new
                
                if best_f != -1:
                    load[f_old] -= d_i
                    load[best_f] += d_i
                    assignment[i] = best_f
                    current_cost += best_delta
                    improved = True
            
            # 2-Swaps
            for i in range(n_customers):
                f_i = assignment[i]
                d_i = demands[i]
                for j in range(i + 1, n_customers):
                    f_j = assignment[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    new_load_fi = load[f_i] - d_i + d_j
                    new_load_fj = load[f_j] - d_j + d_i
                    
                    if new_load_fi > capacities[f_i] or new_load_fj > capacities[f_j]:
                        continue
                    
                    old_assign_cost = assign_cost[i, f_i] + assign_cost[j, f_j]
                    new_assign_cost = assign_cost[i, f_j] + assign_cost[j, f_i]
                    delta_assign = new_assign_cost - old_assign_cost
                    delta_open = 0.0
                    # Swaps between distinct facilities don't change open status
                    # unless one becomes empty, but d_j, d_i > 0 ensures non-empty
                    # if they were non-empty before.
                    # Wait, if load[f_i] == d_i, then new_load_fi = d_j > 0.
                    # If load[f_j] == d_j, then new_load_fj = d_i > 0.
                    # So open cost doesn't change.
                    
                    total_delta = delta_assign
                    if total_delta < -1e-9:
                        load[f_i] = new_load_fi
                        load[f_j] = new_load_fj
                        assignment[i] = f_j
                        assignment[j] = f_i
                        current_cost += total_delta
                        improved = True
                        break
                if improved:
                    break
        
        return assignment, load, current_cost

    def aco_solve(num_ants=20, max_iters=50, alpha=1.0, beta=2.0, rho=0.5, Q=100.0):
        # Pheromone matrix
        tau = np.ones((n_customers, n_facilities)) * 1e-3
        # Initialize with heuristic info?
        tau = tau * eta
        
        best_assignment = None
        best_load = None
        best_cost = float('inf')
        
        for it in range(max_iters):
            for ant in range(num_ants):
                assignment = np.full(n_customers, -1, dtype=int)
                load = np.zeros(n_facilities)
                unassigned = list(range(n_customers))
                
                # Construct solution
                while unassigned:
                    # Pick random customer
                    idx = np.random.randint(len(unassigned))
                    i = unassigned.pop(idx)
                    d_i = demands[i]
                    
                    # Find feasible facilities
                    feasible = []
                    probs = []
                    for k in range(n_facilities):
                        if load[k] + d_i <= capacities[k]:
                            # Probability proportional to tau^alpha * eta^beta
                            p = (tau[i, k] ** alpha) * (eta[i, k] ** beta)
                            feasible.append(k)
                            probs.append(p)
                    
                    if not feasible:
                        # Should not happen if total capacity is sufficient
                        # Fallback: assign to any facility with space (relaxing capacity temporarily? No, must be feasible)
                        # Find facility with most remaining capacity
                        remaining = capacities - load
                        k = np.argmax(remaining)
                        assignment[i] = k
                        load[k] += d_i
                        continue
                    
                    probs = np.array(probs)
                    total_prob = np.sum(probs)
                    if total_prob == 0:
                        k = feasible[np.random.randint(len(feasible))]
                    else:
                        probs = probs / total_prob
                        k = np.random.choice(feasible, p=probs)
                    
                    assignment[i] = k
                    load[k] += d_i
                
                # Evaluate
                cost = compute_cost(assignment, load)
                
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = assignment.copy()
                    best_load = load.copy()
                
                # Update pheromones
                tau = (1 - rho) * tau
                for i in range(n_customers):
                    k = assignment[i]
                    tau[i, k] += Q / cost
            
            # Optional: Run local search on best solution every few iterations
            if it % 10 == 0 and best_assignment is not None:
                best_assignment, best_load, best_cost = local_search(best_assignment, best_load)
        
        # Final local search
        if best_assignment is not None:
            best_assignment, best_load, best_cost = local_search(best_assignment, best_load)
        
        return best_assignment

    # Run ACO
    np.random.seed(42)
    result = aco_solve()
    
    return result
# EVOLVE-BLOCK-END
