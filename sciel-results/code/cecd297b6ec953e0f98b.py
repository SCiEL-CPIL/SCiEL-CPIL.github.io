import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities to minimize total cost.
    
    Args:
        demands: Array of customer demands.
        capacities: Array of facility capacities.
        open_cost: Array of facility opening costs.
        assign_cost: 2D array of assignment costs (customers x facilities).
        
    Returns:
        Array of facility indices for each customer.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Ensure inputs are numpy arrays for speed
    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)
    
    best_solution = None
    best_cost = float('inf')
    
    # Multiple restarts with different random seeds for greedy initialization
    # We will run until we hit the deadline or find a very good solution
    # Let's do a fixed number of restarts that fit in time, or a time-based loop.
    # Given N=100, local search is fast. We can do many restarts.
    
    def compute_cost(assignment):
        """Calculate total cost for a given assignment."""
        total_assign = 0.0
        total_open = 0.0
        used_facilities = set()
        for i in range(n_cust):
            f = assignment[i]
            total_assign += assign_cost[i, f]
            used_facilities.add(f)
        for f in used_facilities:
            total_open += open_cost[f]
        return total_assign + total_open

    def is_valid(assignment):
        """Check if assignment respects capacities."""
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities):
            return False
        return True

    def local_search(assignment, time_limit):
        """
        Perform local search starting from a valid assignment.
        Uses Move and Swap operations.
        """
        current_assignment = assignment.copy()
        current_cost = compute_cost(current_assignment)
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[current_assignment[i]] += demands[i]
        
        # Precompute which facilities are open
        open_facs = set(current_assignment)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Randomize the order of customers to escape local minima
            order = np.random.permutation(n_cust)
            
            # --- Pass 1: Move Operations ---
            for i in order:
                if time.time() >= time_limit:
                    break
                f_old = current_assignment[i]
                # Try moving to every other facility
                # Sort potential new facilities by cost to find improvements faster?
                # Or just iterate. N=100, F=20 is small.
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    # Check capacity
                    if loads[f_new] + demands[i] > capacities[f_new]:
                        continue
                    
                    # Calculate cost change
                    # Delta Assign: cost[i, f_new] - cost[i, f_old]
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    # Delta Open:
                    # If f_old becomes empty, we save open_cost[f_old]
                    # If f_new was empty, we pay open_cost[f_new]
                    delta_open = 0.0
                    if loads[f_old] == demands[i]: # f_old becomes empty
                        delta_open -= open_cost[f_old]
                    if loads[f_new] == 0: # f_new becomes open
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9: # Strictly better
                        # Perform move
                        current_assignment[i] = f_new
                        loads[f_old] -= demands[i]
                        loads[f_new] += demands[i]
                        current_cost += total_delta
                        improved = True
                        # We don't break immediately; we continue to see if other moves help
                        # But typically, after a move, we restart the loop or continue.
                        # Continuing is fine.
            
            # --- Pass 2: Swap Operations ---
            # Swapping is crucial for capacity constraints.
            # We can limit swaps to pairs of customers to keep it fast.
            # Or try swapping customer i with any other customer j.
            # N=100, so 100*100 = 10000 pairs. F=20. 
            # Cost: 10000 * 20 = 200,000 ops per pass. Very fast.
            
            for i in order:
                if time.time() >= time_limit:
                    break
                f_i = current_assignment[i]
                for j in range(i+1, n_cust):
                    f_j = current_assignment[j]
                    if f_i == f_j:
                        continue
                    
                    # Check if swap is capacity feasible
                    # i moves f_i -> f_j, j moves f_j -> f_i
                    if loads[f_j] - demands[j] + demands[i] > capacities[f_j]:
                        continue
                    if loads[f_i] - demands[i] + demands[j] > capacities[f_i]:
                        continue
                    
                    # Calculate cost change
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Delta Open
                    # Facilities involved: f_i, f_j
                    # Old state: both open (since they have customers)
                    # New state:
                    # f_i loses i, gains j. Load changes from L_i to L_i - d_i + d_j.
                    # f_j loses j, gains i. Load changes from L_j to L_j - d_j + d_i.
                    
                    # Check if f_i becomes empty
                    new_load_i = loads[f_i] - demands[i] + demands[j]
                    # Check if f_j becomes empty
                    new_load_j = loads[f_j] - demands[j] + demands[i]
                    
                    delta_open = 0.0
                    # If f_i was open and becomes empty, save cost
                    # It was open because it had i (and maybe others).
                    # It becomes empty if new_load_i == 0.
                    # Note: loads[f_i] >= demands[i] is always true.
                    if new_load_i == 0:
                        delta_open -= open_cost[f_i]
                    # If f_j was open and becomes empty, save cost
                    if new_load_j == 0:
                        delta_open -= open_cost[f_j]
                    
                    # What if a facility was closed and becomes open?
                    # In a swap, both f_i and f_j are currently open (they have i and j respectively).
                    # So they cannot become "newly open". They can only stay open or become empty.
                    # Wait, is it possible that f_i == f_j? We skipped that.
                    # So both are currently open.
                    # The only change in open cost is if one or both become empty.
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        # Perform swap
                        current_assignment[i] = f_j
                        current_assignment[j] = f_i
                        loads[f_i] = new_load_i
                        loads[f_j] = new_load_j
                        current_cost += total_delta
                        improved = True
            
        return current_assignment, current_cost

    # Greedy Initialization Function
    def greedy_init(random_seed):
        rng = np.random.default_rng(random_seed)
        assignment = np.zeros(n_cust, dtype=int)
        loads = np.zeros(n_fac)
        
        # Sort customers by demand descending to pack large ones first
        # This helps with capacity feasibility
        order = np.argsort(-demands)
        
        for i in order:
            # Get costs for this customer
            costs = assign_cost[i].copy()
            # Add a penalty for facilities that are full or nearly full?
            # Or just check feasibility.
            feasible_facs = []
            for f in range(n_fac):
                if loads[f] + demands[i] <= capacities[f]:
                    # Cost to consider: assignment cost + potential open cost
                    # If facility is not yet open, we should consider open cost
                    # But open cost is fixed per facility, not per customer.
                    # For greedy, we can just look at assign_cost.
                    # However, opening a new facility is expensive.
                    # Heuristic: Effective Cost = assign_cost[i, f] + (open_cost[f] if loads[f]==0 else 0) * (1/n_cust_remaining?)
                    # Simpler: Just use assign_cost, but prefer already open facilities?
                    # Let's use a soft penalty: if load is 0, add open_cost / expected_customers_per_facility?
                    # Let's stick to simple assign_cost first, then local search fixes the rest.
                    # To improve, let's add a small bias towards already open facilities.
                    # If loads[f] > 0, cost is assign_cost[i, f]
                    # If loads[f] == 0, cost is assign_cost[i, f] + open_cost[f] * 0.1 (arbitrary weight)
                    # Actually, let's just pick min assign_cost among feasible.
                    feasible_facs.append(f)
            
            if not feasible_facs:
                # Infeasible? Should not happen if total capacity > total demand.
                # If it does, we have a problem.
                # For now, pick any facility (will be invalid).
                return None
            
            # To add diversity, we can pick randomly from the top k cheapest
            # or just the cheapest.
            # Let's pick the cheapest.
            # To allow for multiple restarts to be different, we can shuffle the order of facilities if costs are close?
            # Or just use the random seed to shuffle the facility list before sorting by cost?
            # Let's just pick the min cost feasible.
            min_cost = np.inf
            best_f = -1
            for f in feasible_facs:
                c = assign_cost[i, f]
                # Add a random jitter to break ties and encourage diversity
                c += rng.uniform(0, 0.1) 
                if c < min_cost:
                    min_cost = c
                    best_f = f
            
            assignment[i] = best_f
            loads[best_f] += demands[i]
        
        return assignment

    # Main Loop: Multiple Restarts
    seed = 0
    while time.time() < deadline:
        seed += 1
        init_assignment = greedy_init(seed)
        
        if init_assignment is None:
            continue
        
        if not is_valid(init_assignment):
            continue
            
        # Run local search with a time slice
        # Give local search a portion of the remaining time
        # Or just run it until no improvement, which is fast.
        # Local search on 100x20 is very fast (ms).
        # So we can do many restarts.
        
        local_time_limit = min(deadline, time.time() + 2.0) # Cap each local search at 2s
        # Actually, local search converges quickly. 0.5s is plenty.
        local_time_limit = min(deadline, time.time() + 0.5)
        
        sol, cost = local_search(init_assignment, local_time_limit)
        
        if cost < best_cost:
            best_cost = cost
            best_solution = sol.copy()
            
        # If we found a very good solution, we might want to stop early?
        # No, keep trying.
        
        # Check if we are running out of time
        if time.time() > deadline - 1.0:
            break

    if best_solution is None:
        # Fallback: Return a valid assignment if possible, else zeros
        # This should not happen
        return np.zeros(n_cust, dtype=int)
        
    return best_solution
