# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Iterated Local Search (ILS).
    
    Strategy:
    1. Greedy initialization: Assign largest demands first to cheapest feasible facility,
       penalizing the opening of new facilities.
    2. Local Search:
       - 1-Move: Move a single customer to a different feasible facility if it reduces cost.
       - 2-Swap: Swap facilities of two customers if it reduces cost.
       Vectorized cost delta calculations for speed.
    3. Perturbation: Randomly reassign a subset of customers to escape local optima.
    4. Repeat until time limit.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Convert to numpy arrays for speed
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Pre-compute customer indices for vectorization
    cust_idx = np.arange(n_customers)
    
    def compute_cost(assignment):
        """Compute total cost for a given assignment."""
        assign_total = np.sum(assign_cost[cust_idx, assignment])
        used = np.unique(assignment)
        open_total = np.sum(open_cost[used])
        return assign_total + open_total

    def get_loads(assignment):
        """Compute load array for a given assignment."""
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        return load

    def greedy_init(rng):
        """
        Generate an initial feasible assignment.
        Sort by demand descending. Assign to facility minimizing 
        assign_cost + (open_cost if facility is empty).
        """
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Find feasible facilities
            feasible_mask = (load + d) <= capacities
            feasible_indices = np.where(feasible_mask)[0]
            
            if len(feasible_indices) == 0:
                # Fallback: assign to least loaded (should rarely happen if capacities are sufficient)
                f = int(np.argmin(load))
                result[i] = f
                load[f] += d
                continue
            
            # Calculate effective cost: assignment cost + opening cost if new
            # If load[f] == 0, we pay open_cost[f]
            # We want to minimize: assign_cost[i, f] + (open_cost[f] if load[f]==0 else 0)
            
            costs = assign_cost[i, feasible_indices]
            # Add open cost penalty for empty facilities
            empty_mask = (load[feasible_indices] == 0)
            costs += np.where(empty_mask, open_cost[feasible_indices], 0.0)
            
            # Pick best
            best_local_idx = np.argmin(costs)
            f = feasible_indices[best_local_idx]
            
            result[i] = f
            load[f] += d
            
        return result, load

    def local_search(assignment, load):
        """
        Improve assignment via 1-moves and 2-swaps.
        Returns improved assignment and load.
        """
        assignment = assignment.copy()
        load = load.copy()
        current_cost = compute_cost(assignment)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # --- 1-Move Phase ---
            # Iterate through customers. For each, try all other feasible facilities.
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                
                cur_f = assignment[i]
                d = demands[i]
                
                # Feasible facilities: load[f] + d <= capacities[f]
                # Exclude current facility
                feasible_mask = (load + d) <= capacities
                feasible_mask[cur_f] = False
                
                if not np.any(feasible_mask):
                    continue
                
                feasible_indices = np.where(feasible_mask)[0]
                
                # Calculate delta for moving i to each feasible f
                # Delta Assign: assign_cost[i, f] - assign_cost[i, cur_f]
                delta_assign = assign_cost[i, feasible_indices] - assign_cost[i, cur_f]
                
                # Delta Open:
                # Source cur_f: if load[cur_f] - d == 0, we save open_cost[cur_f]
                delta_open_source = -open_cost[cur_f] if (load[cur_f] - d) == 0 else 0.0
                
                # Target f: if load[f] == 0, we pay open_cost[f]
                delta_open_target = np.where(load[feasible_indices] == 0, open_cost[feasible_indices], 0.0)
                
                total_delta = delta_assign + delta_open_target + delta_open_source
                
                # Find best move
                min_idx = np.argmin(total_delta)
                min_delta = total_delta[min_idx]
                
                if min_delta < -1e-9:
                    best_f = feasible_indices[min_idx]
                    
                    # Apply move
                    load[cur_f] -= d
                    load[best_f] += d
                    assignment[i] = best_f
                    current_cost += min_delta
                    improved = True
            
            # --- 2-Swap Phase ---
            # Try swapping facilities of pairs of customers
            # To keep it fast, we do a few passes. 
            # O(N^2) is 10,000 pairs. With M=20, checking feasibility is fast.
            
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                fi = assignment[i]
                di = demands[i]
                
                for j in range(i + 1, n_customers):
                    if time.time() > deadline:
                        break
                    fj = assignment[j]
                    dj = demands[j]
                    
                    if fi == fj:
                        continue
                    
                    # Check feasibility of swap
                    # New load fi: load[fi] - di + dj
                    # New load fj: load[fj] - dj + di
                    if load[fi] - di + dj > capacities[fi]:
                        continue
                    if load[fj] - dj + di > capacities[fj]:
                        continue
                    
                    # Calculate delta
                    # Assign cost change
                    delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                   (assign_cost[i, fi] + assign_cost[j, fj])
                    
                    # Open cost change
                    # This is tricky. 
                    # Old open set: includes fi, fj (and others)
                    # New open set: includes fi, fj (and others)
                    # Unless the swap causes a facility to become empty or a new one to open.
                    # But since we are swapping between two existing facilities, 
                    # and both have other customers (or at least one of them has >1 customer? No, they could be singletons).
                    
                    # Let's check if fi becomes empty or fj becomes empty.
                    # If load[fi] == di, then after swap, load[fi] = dj. If dj > 0, fi is still open.
                    # If load[fj] == dj, then after swap, load[fj] = di. If di > 0, fj is still open.
                    # So open costs only change if:
                    # 1. fi was open, becomes empty? No, dj > 0.
                    # 2. fj was open, becomes empty? No, di > 0.
                    # 3. Did we open a new facility? No, we are swapping between existing ones.
                    # Wait, what if fi and fj are the only customers?
                    # If load[fi] == di and load[fj] == dj.
                    # After swap: load[fi] = dj, load[fj] = di. Both > 0.
                    # So the set of open facilities remains {fi, fj} (plus others).
                    # Thus, delta_open = 0 for 2-swaps between two non-empty facilities.
                    
                    # Exception: If one of the facilities becomes empty?
                    # Since d > 0, load[fi] - di + dj. If di was the only load, load[fi] becomes dj > 0.
                    # So facilities never close in a 2-swap between two customers with positive demand.
                    
                    total_delta = delta_assign
                    
                    if total_delta < -1e-9:
                        # Apply swap
                        load[fi] = load[fi] - di + dj
                        load[fj] = load[fj] - dj + di
                        assignment[i] = fj
                        assignment[j] = fi
                        current_cost += total_delta
                        improved = True
                        # Continue checking other j for this i? Or break?
                        # Breaking j loop might miss better swaps for i. 
                        # But updating assignment[i] changes fi. 
                        # To be safe and simple, we can just continue to next i 
                        # or restart the j loop. 
                        # Given N=100, a full pass is cheap.
                        # Let's break inner loop to avoid using stale fi for subsequent j in same i-pass?
                        # Actually, if we swap i with j, i is now at fj. 
                        # The next j' might swap with i (now at fj).
                        # This is valid.
                        # However, `fi` variable is stale.
                        # Let's just break the inner loop to be safe and efficient (first improvement).
                        # Or better: restart the local search loop?
                        # For now, break inner loop.
                        break
                        # If we don't break, we need to update fi = fj.
                        # Let's try to just break for simplicity and speed.
            
        return assignment, load, current_cost

    # --- Main ILS Loop ---
    
    rng = np.random.default_rng(42)
    
    # Initial solution
    best_assignment, best_load = greedy_init(rng)
    best_cost = compute_cost(best_assignment)
    current_assignment, current_load = best_assignment.copy(), best_load.copy()
    
    # Initial local search
    current_assignment, current_load, current_cost = local_search(current_assignment, current_load)
    if current_cost < best_cost:
        best_cost = current_cost
        best_assignment = current_assignment.copy()
    
    # ILS Loop
    perturbation_size = 5  # Number of customers to perturb
    
    while time.time() < deadline:
        # Perturb
        perturbed_assignment = current_assignment.copy()
        perturbed_load = current_load.copy()
        
        # Select random customers to reassign
        # We need to ensure feasibility.
        # Strategy: Pick k customers, clear their assignment (subtract from load),
        # then greedily reassign them to cheapest feasible facility.
        
        indices_to_perturb = rng.choice(n_customers, size=perturbation_size, replace=False)
        
        # Remove from current facilities
        for i in indices_to_perturb:
            f = perturbed_assignment[i]
            perturbed_load[f] -= demands[i]
        
        # Reassign
        for i in indices_to_perturb:
            d = demands[i]
            feasible_mask = (perturbed_load + d) <= capacities
            feasible_indices = np.where(feasible_mask)[0]
            
            if len(feasible_indices) == 0:
                # Should not happen if total capacity is sufficient
                f = int(np.argmin(perturbed_load))
            else:
                # Greedy reassignment
                costs = assign_cost[i, feasible_indices]
                empty_mask = (perturbed_load[feasible_indices] == 0)
                costs += np.where(empty_mask, open_cost[feasible_indices], 0.0)
                best_local_idx = np.argmin(costs)
                f = feasible_indices[best_local_idx]
                
            perturbed_assignment[i] = f
            perturbed_load[f] += d
            
        # Local Search on perturbed solution
        new_assignment, new_load, new_cost = local_search(perturbed_assignment, perturbed_load)
        
        # Accept if better or with some probability (Simulated Annealing style? 
        # For simplicity and robustness, accept if better. 
        # Or accept if better than best, and sometimes accept worse to explore?
        # Given the time limit, accepting only better or equal is safer for stability.
        # But ILS usually accepts worse to escape basins. 
        # Let's accept if new_cost < best_cost.
        # Also update current to new if new_cost < current_cost?
        # Standard ILS: accept new solution if better than current, or with prob exp(-delta/T).
        # Let's keep it simple: if new_cost < best_cost, update best.
        # If new_cost < current_cost, update current.
        
        if new_cost < current_cost:
            current_assignment = new_assignment
            current_load = new_load
            current_cost = new_cost
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_assignment = new_assignment.copy()
            
        # Optional: If no improvement for a while, increase perturbation size?
        # Keep it simple for now.
        
    return best_assignment.astype(int)

# EVOLVE-BLOCK-END
