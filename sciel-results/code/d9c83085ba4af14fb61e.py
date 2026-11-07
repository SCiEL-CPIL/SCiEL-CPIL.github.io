"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Pre-compute sorted facility indices for each customer based on assignment cost
    # This helps in greedy initialization
    sorted_facilities = np.argsort(assign_cost, axis=1)
    
    def compute_cost(assignment, load):
        assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        # Open cost: sum of open_cost for facilities where load > 0
        open_sum = np.sum(open_cost[load > 0])
        return assign_sum + open_sum

    def greedy_init():
        """
        Generate an initial feasible solution using greedy assignment.
        """
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending to place hardest first
        order = np.argsort(-demands)
        
        for i in order:
            d_i = demands[i]
            # Try facilities in order of increasing assignment cost
            for f in sorted_facilities[i]:
                if load[f] + d_i <= capacities[f]:
                    assignment[i] = f
                    load[f] += d_i
                    break
            
            if assignment[i] == -1:
                # Fallback: assign to facility with most remaining capacity
                remaining = capacities - load
                f = np.argmax(remaining)
                if remaining[f] >= d_i:
                    assignment[i] = f
                    load[f] += d_i
                else:
                    # This instance might be infeasible or my greedy failed
                    # For robustness, just pick the first feasible one found in sorted order
                    for f in range(n_facilities):
                        if load[f] + d_i <= capacities[f]:
                            assignment[i] = f
                            load[f] += d_i
                            break
        
        return assignment, load

    def lns_improve(assignment, load, max_iters=500):
        """
        Large Neighborhood Search (LNS) improvement.
        """
        current_cost = compute_cost(assignment, load)
        best_assignment = assignment.copy()
        best_load = load.copy()
        best_cost = current_cost
        
        # Temperature or acceptance strategy: Simple Hill Climbing (only accept improvements)
        # To escape local minima, we can add some randomization in destruction/reconstruction.
        
        for it in range(max_iters):
            # 1. Destroy: Select a subset of customers to re-optimize
            # Strategy: Random subset of size k
            # k can vary. Let's try k=10 to 20.
            k = np.random.randint(5, 25)
            destroy_indices = np.random.choice(n_customers, k, replace=False)
            
            # Store original assignment and loads for these customers
            old_assignment = assignment[destroy_indices].copy()
            old_load = load.copy()
            
            # Remove them from current assignment
            for i in destroy_indices:
                f = assignment[i]
                load[f] -= demands[i]
                assignment[i] = -1 # Mark as unassigned
            
            # 2. Repair: Re-assign these customers
            # Sort destroy_indices by demand descending to place larger ones first (greedy heuristic)
            destroy_order = destroy_indices[np.argsort(-demands[destroy_indices])]
            
            success = True
            for i in destroy_order:
                d_i = demands[i]
                best_f = -1
                best_delta = float('inf')
                
                # Evaluate all facilities for this customer
                # We want to minimize: assign_cost[i, f] + open_cost[f] if load[f]==0
                # Note: load[f] is the current load AFTER removing the destroy set.
                # If we assign i to f, and load[f] was 0, we pay open_cost[f].
                
                for f in range(n_facilities):
                    if load[f] + d_i > capacities[f]:
                        continue
                    
                    # Calculate marginal cost
                    marginal_assign = assign_cost[i, f]
                    marginal_open = 0.0
                    if load[f] == 0:
                        marginal_open = open_cost[f]
                    
                    total_marginal = marginal_assign + marginal_open
                    
                    if total_marginal < best_delta:
                        best_delta = total_marginal
                        best_f = f
                
                if best_f == -1:
                    # No feasible facility found for this customer given current loads
                    # This can happen if capacity is tight.
                    # Try to find any feasible facility, even if cost is high
                    for f in range(n_facilities):
                        if load[f] + d_i <= capacities[f]:
                            best_f = f
                            break
                
                if best_f == -1:
                    success = False
                    break
                    
                assignment[i] = best_f
                load[best_f] += d_i
            
            if not success:
                # Revert changes if repair failed
                assignment[destroy_indices] = old_assignment
                load = old_load
                continue
            
            # 3. Calculate new cost
            new_cost = compute_cost(assignment, load)
            
            # 4. Acceptance
            if new_cost < best_cost:
                best_cost = new_cost
                best_assignment = assignment.copy()
                best_load = load.copy()
            else:
                # Revert to best known solution
                assignment = best_assignment.copy()
                load = best_load.copy()
                
        return best_assignment, best_load, best_cost

    # Main Logic
    # 1. Greedy Initialization
    assignment, load = greedy_init()
    
    # 2. LNS Improvement
    assignment, load, cost = lns_improve(assignment, load)
    
    return assignment
# EVOLVE-BLOCK-END
