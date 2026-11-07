import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Uses Simulated Annealing with Greedy Initialization.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin
    rng = np.random.default_rng(42)
    
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

    def _greedy_init():
        """Greedy initialization: assign largest demands to cheapest feasible facility."""
        order = np.argsort(-demands)
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
                # If no facility has space, this greedy fails. 
                # In this problem, total capacity should be sufficient.
                # If it fails, we need a more robust initialization.
                # Let's try random assignment as fallback.
                return _random_init()
        return result

    def _random_init():
        """Random valid initialization."""
        result = np.full(n_cust, -1, dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        # Shuffle customers
        order = np.arange(n_cust)
        rng.shuffle(order)
        for i in order:
            # Find feasible facilities
            feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
            if not feasible:
                return None
            # Pick one (weighted by cost or random)
            f = feasible[rng.integers(len(feasible))]
            load[f] += demands[i]
            result[i] = f
        return result

    # Initial Solution
    best_result = _greedy_init()
    if best_result is None:
        best_result = _random_init()
    
    if best_result is None:
        # Last resort: distribute round robin (might violate capacity, but let's check)
        # Actually, if greedy fails, it means capacities are tight.
        # We must ensure validity.
        # Let's try a more careful greedy: sort facilities by capacity desc, assign largest to largest cap.
        result = np.full(n_cust, -1, dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        cust_order = np.argsort(-demands)
        fac_order = np.argsort(-capacities)
        for i in cust_order:
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # This instance is infeasible or my greedy is too naive.
                # But the problem statement implies valid instances.
                pass
        if np.any(result == -1):
             # Fallback to random
             best_result = _random_init()
             if best_result is None:
                 # Absolute fallback: return something valid if possible
                 # If even random fails, the problem is infeasible.
                 return np.zeros(n_cust, dtype=int)
        else:
            best_result = result

    best_cost = _compute_obj(best_result)
    
    # Simulated Annealing
    current_result = best_result.copy()
    current_load = np.zeros(n_fac, dtype=np.int64)
    for i, f in enumerate(current_result):
        current_load[f] += demands[i]
    current_cost = best_cost
    
    T_start = 100.0
    T_end = 1.0
    cooling_rate = 0.995
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and T > T_end:
        # Generate a random move
        # 50% chance: Reassign 1 customer
        # 50% chance: Swap 2 customers
        
        if rng.random() < 0.5:
            # Reassign
            i = rng.integers(n_cust)
            cur_f = current_result[i]
            # Pick a new facility
            # Prefer facilities with available capacity
            feasible = [f for f in range(n_fac) if f != cur_f and current_load[f] + demands[i] <= capacities[f]]
            if not feasible:
                continue
            new_f = feasible[rng.integers(len(feasible))]
            
            # Calculate Delta
            delta_assign = assign_cost[i, new_f] - assign_cost[i, cur_f]
            delta_open = 0.0
            if current_load[cur_f] - demands[i] == 0:
                delta_open -= open_cost[cur_f]
            if current_load[new_f] == 0:
                delta_open += open_cost[new_f]
            
            delta = delta_assign + delta_open
            
            # Accept
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Apply move
                current_load[cur_f] -= demands[i]
                current_load[new_f] += demands[i]
                current_result[i] = new_f
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = current_result.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Swap
            i, j = rng.choice(n_cust, size=2, replace=False)
            fi = current_result[i]
            fj = current_result[j]
            if fi == fj:
                continue
            
            # Check capacity constraints for swap
            # Load at fi after swap: load[fi] - d[i] + d[j]
            # Load at fj after swap: load[fj] - d[j] + d[i]
            new_load_fi = current_load[fi] - demands[i] + demands[j]
            new_load_fj = current_load[fj] - demands[j] + demands[i]
            
            if new_load_fi > capacities[fi] or new_load_fj > capacities[fj]:
                continue
            
            # Calculate Delta
            delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
            
            delta_open = 0.0
            # Check if fi becomes empty
            if current_load[fi] - demands[i] == 0:
                # fi loses i, gains j. If it was only i, it becomes empty? No, it gains j.
                # Wait.
                # Before: load[fi] includes i.
                # After: load[fi] = load[fi] - d[i] + d[j].
                # If load[fi] - d[i] == 0, then before swap, fi had ONLY i.
                # After swap, fi has j. So it stays open (if j>0, which it is).
                # So no open cost change for fi in this case?
                # Let's re-evaluate open cost logic.
                
                # Open cost depends on whether load > 0.
                # Before: load[fi] > 0 (since i is there).
                # After: new_load_fi > 0 (since j is there, demands > 0).
                # So open status of fi doesn't change if it had only i and gets j.
                
                # What if fi had i and others?
                # Before: load[fi] > 0.
                # After: new_load_fi > 0.
                # No change.
                
                # So open cost only changes if a facility goes from 0 to >0 or >0 to 0.
                
                # Case 1: fi becomes empty.
                # This happens if current_load[fi] == demands[i] (only i) AND we move i away.
                # But in a swap, i moves to fj. j moves to fi.
                # So fi receives j.
                # So fi cannot become empty if it was open, because it receives j.
                # Unless j is 0? No, demands are >= 5.
                
                # So in a swap between two open facilities, open costs don't change?
                # Let's check:
                # If fi had only i, and fj had only j.
                # Before: fi open, fj open.
                # After: fi has j (open), fj has i (open).
                # No change.
                
                # If fi had i and others, fj had j and others.
                # No change in open status.
                
                # What if one facility was empty?
                # We only swap between assigned customers. So both fi and fj are open (load > 0).
                # And after swap, both receive a customer, so both remain open.
                # So delta_open is ALWAYS 0 for a swap between two assigned customers?
                # Yes, because loads are strictly positive before and after.
                
                delta_open = 0.0
            
            delta = delta_assign + delta_open
            
            # Accept
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Apply move
                current_load[fi] = new_load_fi
                current_load[fj] = new_load_fj
                current_result[i] = fj
                current_result[j] = fi
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = current_result.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        T *= cooling_rate
        if no_improve_count > max_no_improve:
            # Restart with a new random perturbation? Or just break?
            # Let's try a restart with a new greedy/random init if stuck
            # But we must keep the best found so far.
            # Let's just cool down faster.
            T *= 0.5
            no_improve_count = 0

    return best_result.astype(int)
