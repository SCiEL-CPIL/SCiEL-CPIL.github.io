"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Pre-compute sorted facility indices for each customer based on assignment cost
    # This helps in greedy initialization and potentially in local search candidates
    sorted_facilities = np.argsort(assign_cost, axis=1)
    
    def compute_cost(assignment, load):
        assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        # Open cost: sum of open_cost for facilities where load > 0
        open_sum = np.sum(open_cost[load > 0])
        return assign_sum + open_sum
    
    def local_search(assignment, load, max_passes=200):
        """
        Fast local search using 1-moves and 2-swaps with O(1) delta cost calculation.
        """
        current_cost = compute_cost(assignment, load)
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # 1-Moves: Try moving each customer to a better facility
            # We iterate through customers. To be faster, we can shuffle order, but fixed is fine for now.
            for i in range(n_customers):
                f_old = assignment[i]
                if f_old < 0:
                    continue
                    
                d_i = demands[i]
                # Try moving to all other facilities
                # We prioritize facilities with lower assign_cost[i, f]
                # sorted_facilities[i] gives indices sorted by cost
                for f_new in sorted_facilities[i]:
                    if f_new == f_old:
                        continue
                    
                    # Capacity check
                    if load[f_new] + d_i > capacities[f_new]:
                        continue
                    
                    # Calculate delta
                    # Assignment cost change
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    # Open cost change
                    delta_open = 0.0
                    # Check if f_old becomes empty
                    if load[f_old] == d_i:
                        delta_open -= open_cost[f_old]
                    # Check if f_new becomes open
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        # Apply move
                        load[f_old] -= d_i
                        load[f_new] += d_i
                        assignment[i] = f_new
                        current_cost += total_delta
                        improved = True
                        break # Restart customer loop after improvement
        
            # 2-Swaps: Try swapping customers i and j between facilities f_i and f_j
            # This is O(N^2 * F) if done naively, but we can optimize.
            # N=100, so 100*100 = 10k pairs. For each pair, check feasibility and cost.
            # This is fast enough.
            for i in range(n_customers):
                f_i = assignment[i]
                d_i = demands[i]
                for j in range(i + 1, n_customers):
                    f_j = assignment[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    # Check capacity for swap
                    # New load f_i: load[f_i] - d_i + d_j
                    # New load f_j: load[f_j] - d_j + d_i
                    new_load_fi = load[f_i] - d_i + d_j
                    new_load_fj = load[f_j] - d_j + d_i
                    
                    if new_load_fi > capacities[f_i] or new_load_fj > capacities[f_j]:
                        continue
                    
                    # Calculate delta
                    # Assignment cost change
                    old_assign_cost = assign_cost[i, f_i] + assign_cost[j, f_j]
                    new_assign_cost = assign_cost[i, f_j] + assign_cost[j, f_i]
                    delta_assign = new_assign_cost - old_assign_cost
                    
                    # Open cost change
                    # f_i becomes empty if new_load_fi == 0?
                    # new_load_fi = load[f_i] - d_i + d_j. Since d_j >= 5, new_load_fi >= 5 if load[f_i]-d_i >= 0.
                    # So f_i cannot become empty in a swap if d_j > 0.
                    # Same for f_j.
                    # Thus delta_open = 0 for swaps between distinct facilities.
                    delta_open = 0.0
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        # Apply swap
                        load[f_i] = new_load_fi
                        load[f_j] = new_load_fj
                        assignment[i] = f_j
                        assignment[j] = f_i
                        current_cost += total_delta
                        improved = True
                        break # Restart inner loop
                if improved:
                    break
        
        return assignment, load, current_cost

    def greedy_init(seed=None):
        """
        Generate an initial feasible solution using greedy assignment.
        """
        if seed is not None:
            np.random.seed(seed)
        
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending to place hardest first
        # Add some randomness to break ties or vary order for ILS
        order = np.argsort(-demands)
        
        for i in order:
            d_i = demands[i]
            assigned = False
            # Try facilities in order of increasing assignment cost
            for f in sorted_facilities[i]:
                if load[f] + d_i <= capacities[f]:
                    assignment[i] = f
                    load[f] += d_i
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: assign to facility with most remaining capacity
                # This should rarely happen if total capacity is sufficient
                remaining = capacities - load
                f = np.argmax(remaining)
                assignment[i] = f
                load[f] += d_i
        
        return assignment, load

    # Iterated Local Search
    best_assignment = None
    best_load = None
    best_cost = float('inf')
    
    # Run multiple iterations with different random seeds for perturbation
    num_iterations = 20
    
    for it in range(num_iterations):
        # Generate initial solution
        # Use a different seed for each iteration to explore different starting points
        # Or perturb the previous best solution
        if it == 0:
            assignment, load = greedy_init(seed=42 + it)
        else:
            # Perturb the best solution found so far
            assignment = best_assignment.copy()
            load = best_load.copy()
            
            # Randomly reassign 10% of customers
            num_perturb = max(1, n_customers // 10)
            perturb_indices = np.random.choice(n_customers, num_perturb, replace=False)
            
            for i in perturb_indices:
                # Remove from current facility
                f_old = assignment[i]
                load[f_old] -= demands[i]
                
                # Assign to a random feasible facility
                # Try random facilities until one fits
                attempts = 0
                while True:
                    f_new = np.random.randint(n_facilities)
                    if load[f_new] + demands[i] <= capacities[f_new]:
                        assignment[i] = f_new
                        load[f_new] += demands[i]
                        break
                    attempts += 1
                    if attempts > 50:
                        # Fallback: find any feasible facility
                        feasible = np.where(capacities - load >= demands[i])[0]
                        if len(feasible) > 0:
                            f_new = np.random.choice(feasible)
                            assignment[i] = f_new
                            load[f_new] += demands[i]
                            break
                        else:
                            # Infeasible perturbation, revert
                            load[f_old] += demands[i]
                            assignment[i] = f_old
                            break
        
        # Improve with local search
        assignment, load, cost = local_search(assignment, load)
        
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
            best_load = load.copy()
            
            # Early stopping if we found a very good solution?
            # No, keep going.
        
        # Optional: If no improvement in last 5 iterations, increase perturbation?
        # Keep it simple for now.

    return best_assignment
# EVOLVE-BLOCK-END
