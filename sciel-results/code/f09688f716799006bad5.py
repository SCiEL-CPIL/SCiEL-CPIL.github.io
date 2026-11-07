"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    N = n_customers
    F = n_facilities
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    demands = np.array(demands, dtype=np.float64)
    capacities = np.array(capacities, dtype=np.float64)
    open_cost = np.array(open_cost, dtype=np.float64)
    assign_cost = np.array(assign_cost, dtype=np.float64)
    
    def is_valid(assignment):
        for f in range(F):
            mask = assignment == f
            if mask.any():
                if demands[mask].sum() > capacities[f] + 1e-6:
                    return False
        return True
    
    def calc_cost(assignment):
        total_assign = np.sum(assign_cost[np.arange(N), assignment])
        used = set(assignment.tolist())
        total_open = np.sum(open_cost[list(used)])
        return total_assign + total_open
    
    def greedy_init():
        # Sort customers by demand descending to handle large demands first
        order = np.argsort(-demands)
        assignment = np.zeros(N, dtype=int)
        load = np.zeros(F)
        used_facilities = set()
        
        for i in order:
            # Find feasible facility with lowest cost
            # Cost = assign_cost[i, f] + (open_cost[f] if f not used else 0)
            # But we must check capacity.
            best_f = -1
            best_cost = float('inf')
            
            # Pre-compute costs for this customer? No, just loop F=20 is fast.
            for f in range(F):
                if load[f] + demands[i] <= capacities[f]:
                    # Potential cost
                    c = assign_cost[i, f]
                    if f not in used_facilities:
                        c += open_cost[f]
                    
                    if c < best_cost:
                        best_cost = c
                        best_f = f
            
            if best_f == -1:
                # Fallback: find any facility with space, lowest assign cost
                for f in range(F):
                    if load[f] + demands[i] <= capacities[f]:
                        best_f = f
                        break
                if best_f == -1:
                    # Should not happen if total demand <= total capacity, but safe guard
                    best_f = 0
            
            assignment[i] = best_f
            load[best_f] += demands[i]
            used_facilities.add(best_f)
            
        return assignment

    # 1. Initialize with Greedy
    current_assignment = greedy_init()
    
    if not is_valid(current_assignment):
        # If greedy failed (unlikely if capacities are sufficient), try random assignment
        current_assignment = np.random.randint(0, F, N)
        # Force validity if possible (simple repair)
        load = np.zeros(F)
        for i in range(N):
            load[current_assignment[i]] += demands[i]
        for f in range(F):
            if load[f] > capacities[f]:
                excess = load[f] - capacities[f]
                customers_at_f = np.where(current_assignment == f)[0]
                # Move customers with smallest demand to other facilities
                customers_at_f = customers_at_f[np.argsort(demands[customers_at_f])]
                for i in customers_at_f:
                    if excess <= 0:
                        break
                    # Find alternative
                    for f2 in range(F):
                        if f2 != f and load[f2] + demands[i] <= capacities[f2]:
                            load[f] -= demands[i]
                            load[f2] += demands[i]
                            current_assignment[i] = f2
                            excess -= demands[i]
                            break
                    if excess > 0:
                        # If still excess, we have a problem. But problem statement implies feasible instances.
                        pass
    
    # 2. Simulated Annealing / Local Search
    current_load = np.zeros(F)
    for i in range(N):
        current_load[current_assignment[i]] += demands[i]
    
    facility_count = np.zeros(F, dtype=int)
    for i in range(N):
        facility_count[current_assignment[i]] += 1
    
    current_cost = calc_cost(current_assignment)
    best_assignment = current_assignment.copy()
    best_cost = current_cost
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Precompute indices for speed
    indices = np.arange(N)
    
    while time.time() < deadline - 2.0:
        T = max(T_min, T * cooling_rate)
        
        # Random move: reassign one customer
        i = random.randint(0, N - 1)
        current_f = current_assignment[i]
        f = random.randint(0, F - 1)
        if f == current_f:
            continue
        
        # Check capacity
        if current_load[f] + demands[i] > capacities[f]:
            continue
        
        # Calculate delta cost
        delta = assign_cost[i, f] - assign_cost[i, current_f]
        
        # Open cost changes
        if facility_count[current_f] == 1:
            delta -= open_cost[current_f]
        if facility_count[f] == 0:
            delta += open_cost[f]
        
        # Acceptance probability
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Apply move
            current_load[current_f] -= demands[i]
            current_load[f] += demands[i]
            facility_count[current_f] -= 1
            facility_count[f] += 1
            current_assignment[i] = f
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignment = current_assignment.copy()
        
        # Occasionally try a swap (2-opt style) to escape local optima
        if random.random() < 0.1:
            i1 = random.randint(0, N - 1)
            i2 = random.randint(0, N - 1)
            if i1 == i2:
                continue
            f1 = current_assignment[i1]
            f2 = current_assignment[i2]
            
            # Swap: i1 -> f2, i2 -> f1
            # Check capacities
            # New load at f2: old_load[f2] - demand[i2] + demand[i1]
            # New load at f1: old_load[f1] - demand[i1] + demand[i2]
            # But we need to consider if they are the same facility.
            if f1 == f2:
                continue
            
            # Check if swap is feasible
            # Case 1: i1 moves to f2, i2 moves to f1
            # Load f1 becomes: current_load[f1] - demands[i1] + demands[i2]
            # Load f2 becomes: current_load[f2] - demands[i2] + demands[i1]
            
            new_load_f1 = current_load[f1] - demands[i1] + demands[i2]
            new_load_f2 = current_load[f2] - demands[i2] + demands[i1]
            
            if new_load_f1 <= capacities[f1] and new_load_f2 <= capacities[f2]:
                # Calculate cost delta
                # Assign cost: (cost[i1, f2] + cost[i2, f1]) - (cost[i1, f1] + cost[i2, f2])
                delta_assign = (assign_cost[i1, f2] + assign_cost[i2, f1]) - (assign_cost[i1, f1] + assign_cost[i2, f2])
                
                # Open cost delta
                # Facility f1 count: old_count[f1] - 1 (i1 leaves) + 1 (i2 enters) = same count?
                # Wait, if i1 was the only one at f1, it closes. If i2 was the only one at f2, it closes.
                # Let's track counts carefully.
                count_f1 = facility_count[f1]
                count_f2 = facility_count[f2]
                
                new_count_f1 = count_f1
                new_count_f2 = count_f2
                # i1 leaves f1, enters f2
                # i2 leaves f2, enters f1
                
                # Open cost change for f1:
                # If count_f1 == 1, it was open, now 0? No, i2 enters. So count stays 1?
                # If count_f1 == 1, i1 is the only one. i1 leaves -> count 0. i2 enters -> count 1.
                # So open status doesn't change for f1 unless count was 1 and i2 doesn't enter? But i2 DOES enter.
                # So open status of f1 remains OPEN (count 1 -> 1).
                # Wait, if count_f1 > 1, it stays open.
                # So open cost for f1 does NOT change in a swap between two distinct facilities?
                # Let's verify.
                # Before: f1 has k customers, f2 has m customers.
                # After: f1 has k-1+1 = k customers. f2 has m-1+1 = m customers.
                # So the number of customers per facility does NOT change in a swap between two distinct facilities.
                # Therefore, the set of open facilities does NOT change.
                # So delta_open = 0.
                
                delta = delta_assign
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Apply swap
                    current_assignment[i1] = f2
                    current_assignment[i2] = f1
                    current_load[f1] = new_load_f1
                    current_load[f2] = new_load_f2
                    # facility_count unchanged
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = current_assignment.copy()

    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
