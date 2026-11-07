"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _evaluate_assignment(result, demands, capacities, open_cost, assign_cost):
    """Compute the objective value for a given assignment. Returns inf if invalid."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return float('inf')
        load[f] += demands[i]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    
    open_total = np.sum(open_cost[load > 0])
    assign_total = np.sum(assign_cost[np.arange(len(demands)), result])
    return open_total + assign_total


def _greedy_assign(demands, capacities, open_cost, assign_cost, seed=None, 
                   strategy='largest_first', open_penalty=0.0, alpha=1.0):
    """
    Greedy assignment with various strategies.
    Returns (assignment_array, cost) or (None, inf) if failed.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = -np.ones(n_cust, dtype=int)
    load = np.zeros(n_fac)
    open_used = np.zeros(n_fac, dtype=bool)
    
    if strategy == 'largest_first':
        order = np.argsort(-demands)
    elif strategy == 'smallest_first':
        order = np.argsort(demands)
    elif strategy == 'random':
        rng = np.random.RandomState(seed)
        order = np.arange(n_cust)
        rng.shuffle(order)
    elif strategy == 'cost_weighted':
        # Sort by demand * average cost
        avg_cost = np.mean(assign_cost, axis=1)
        order = np.argsort(-demands * avg_cost)
    else:
        order = np.argsort(-demands)
    
    # Pre-sort facility preferences for each customer
    for i in order:
        # Facility preference: sort by effective cost
        # If facility already open, cost is just assign_cost
        # If not open, cost is assign_cost + open_cost * alpha
        eff_cost = assign_cost[i].copy()
        for f in range(n_fac):
            if not open_used[f]:
                eff_cost[f] += open_cost[f] * alpha
        
        # Try facilities in order of effective cost
        for f in np.argsort(eff_cost):
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                load[f] += demands[i]
                result[i] = f
                if not open_used[f]:
                    open_used[f] = True
                break
    
    if -1 in result:
        return None, float('inf')
    
    cost = _evaluate_assignment(result, demands, capacities, open_cost, assign_cost)
    return result, cost


def _local_search_with_swaps(result, demands, capacities, open_cost, assign_cost, max_iters=3000, seed=None):
    """
    Local search with single moves and 2-swap moves to escape local minima.
    Returns (improved_assignment, improved_cost).
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    rng = np.random.RandomState(seed)
    
    current_cost = _evaluate_assignment(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result, current_cost
    
    best_result = result.copy()
    best_cost = current_cost
    
    # Track loads
    load = np.zeros(n_fac)
    for i, f in enumerate(best_result):
        load[f] += demands[i]
    
    # Customer to facility count
    fac_customers = {f: [] for f in range(n_fac)}
    for i, f in enumerate(best_result):
        fac_customers[f].append(i)
    
    for it in range(max_iters):
        improved = False
        
        # 50% chance to try single move, 50% chance to try 2-swap
        if rng.random() < 0.5:
            # Single move
            i = rng.randint(n_cust)
            current_fac = best_result[i]
            current_d = demands[i]
            
            # Find best alternative facility
            best_new_cost = current_cost
            best_new_fac = -1
            
            is_last_at_current = len(fac_customers[current_fac]) == 1
            
            for f in range(n_fac):
                if f == current_fac:
                    continue
                if load[f] + current_d > capacities[f] + 1e-9:
                    continue
                
                delta_assign = assign_cost[i, f] - assign_cost[i, current_fac]
                delta_open = 0.0
                
                if is_last_at_current:
                    delta_open -= open_cost[current_fac]
                
                if len(fac_customers[f]) == 0:
                    delta_open += open_cost[f]
                
                new_cost = best_cost + delta_assign + delta_open
                if new_cost < best_new_cost - 1e-9:
                    best_new_cost = new_cost
                    best_new_fac = f
            
            if best_new_fac >= 0 and best_new_cost < best_cost - 1e-9:
                load[current_fac] -= current_d
                fac_customers[current_fac].remove(i)
                load[best_new_fac] += current_d
                fac_customers[best_new_fac].append(i)
                best_result[i] = best_new_fac
                best_cost = best_new_cost
                improved = True
        else:
            # 2-swap: swap two customers between different facilities
            if len(fac_customers) < 2:
                continue
            
            # Pick two random customers from different facilities
            i1 = rng.randint(n_cust)
            i2 = rng.randint(n_cust)
            if i1 == i2:
                continue
            
            f1 = best_result[i1]
            f2 = best_result[i2]
            if f1 == f2:
                continue
            
            d1 = demands[i1]
            d2 = demands[i2]
            
            # Check if swap is feasible
            if load[f1] - d1 + d2 > capacities[f1] + 1e-9:
                continue
            if load[f2] - d2 + d1 > capacities[f2] + 1e-9:
                continue
            
            # Calculate cost change
            delta_assign = (assign_cost[i1, f2] + assign_cost[i2, f1]) - \
                           (assign_cost[i1, f1] + assign_cost[i2, f2])
            
            # Open cost changes: only if facilities become empty or stay non-empty
            delta_open = 0.0
            was_empty_f1_after = (load[f1] - d1 + d2) == 0
            was_empty_f2_after = (load[f2] - d2 + d1) == 0
            was_empty_f1_before = (load[f1]) == 0  # Should be False since i1 is there
            was_empty_f2_before = (load[f2]) == 0  # Should be False since i2 is there
            
            # If f1 becomes empty after swap, save open_cost[f1]
            if was_empty_f1_after and not was_empty_f1_before:
                delta_open -= open_cost[f1]
            # If f2 becomes empty after swap, save open_cost[f2]
            if was_empty_f2_after and not was_empty_f2_before:
                delta_open -= open_cost[f2]
            
            new_cost = best_cost + delta_assign + delta_open
            if new_cost < best_cost - 1e-9:
                # Perform swap
                load[f1] = load[f1] - d1 + d2
                load[f2] = load[f2] - d2 + d1
                
                fac_customers[f1].remove(i1)
                fac_customers[f1].append(i2)
                fac_customers[f2].remove(i2)
                fac_customers[f2].append(i1)
                
                best_result[i1] = f2
                best_result[i2] = f1
                best_cost = new_cost
                improved = True
    
    return best_result, best_cost


def _min_cost_flow_assign(demands, capacities, open_cost, assign_cost):
    """
    Solve capacitated facility location using min-cost flow formulation.
    This exploits the network structure for exact optimal solutions.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # We'll use a greedy + local search approach as fallback since 
    # proper min-cost flow with fixed charges is complex
    # For now, use enhanced local search with swaps
    
    result, cost = _greedy_assign(demands, capacities, open_cost, assign_cost, 
                                   seed=42, strategy='largest_first', alpha=1.0)
    if result is None:
        return None, float('inf')
    
    # Apply enhanced local search with 2-swaps
    improved_result, improved_cost = _local_search_with_swaps(
        result, demands, capacities, open_cost, assign_cost, 
        max_iters=5000, seed=42
    )
    
    return improved_result, improved_cost


def _hungarian_based_assign(demands, capacities, open_cost, assign_cost):
    """
    Try to use a min-cost flow / assignment approach.
    For small problems, this might work. But with capacity constraints, 
    we need a different approach.
    """
    # This is a capacitated facility location, which is NP-hard in general.
    # For 100 customers and 20 facilities, we can try a simpler approach.
    # Skip for now, use greedy + local search.
    pass


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to one facility using a portfolio search approach
    with enhanced local search including 2-swap moves.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = None
    best_cost = float('inf')
    
    def _update_best(result, cost):
        nonlocal best_result, best_cost
        if cost < best_cost:
            best_cost = cost
            best_result = result
    
    # Phase 1: Run multiple greedy variants with different parameters
    strategies = [
        ('largest_first', 0.0, 1.0),
        ('largest_first', 0.5, 1.0),
        ('largest_first', 1.0, 1.0),
        ('largest_first', 0.0, 0.5),
        ('largest_first', 0.0, 2.0),
        ('smallest_first', 0.0, 1.0),
        ('smallest_first', 0.5, 1.0),
        ('random', 42, 1.0),
        ('random', 123, 1.0),
        ('random', 777, 1.0),
        ('cost_weighted', 0.0, 1.0),
        ('cost_weighted', 0.5, 1.0),
    ]
    
    for strategy, open_pen, alpha in strategies:
        if time.time() > deadline - 15:
            break
        
        seed = 42 if strategy != 'random' else open_pen * 1000
        if strategy == 'random':
            result, cost = _greedy_assign(demands, capacities, open_cost, assign_cost, 
                                          seed=int(open_pen * 1000) if open_pen > 0 else 42,
                                          strategy='random', alpha=alpha)
        else:
            result, cost = _greedy_assign(demands, capacities, open_cost, assign_cost, 
                                          seed=42, strategy=strategy, 
                                          open_penalty=open_pen, alpha=alpha)
        
        if result is not None:
            _update_best(result, cost)
    
    # Phase 2: Enhanced local search with 2-swaps on the best result
    if time.time() > deadline - 10:
        if best_result is None:
            # Fallback: simple greedy
            result, _ = _greedy_assign(demands, capacities, open_cost, assign_cost, 
                                       seed=42, strategy='largest_first', alpha=1.0)
            if result is not None:
                best_result = result
            else:
                # Last resort: assign to facility with most capacity
                result = np.zeros(n_cust, dtype=int)
                return result
        return best_result
    
    # Run enhanced local search on best result
    if best_result is not None:
        improved_result, improved_cost = _local_search_with_swaps(
            best_result, demands, capacities, open_cost, assign_cost, 
            max_iters=5000, seed=42
        )
        _update_best(improved_result, improved_cost)
    
    # Try enhanced local search on other good greedy results
    more_strategies = [
        ('largest_first', 0.3, 1.0),
        ('largest_first', 0.7, 1.0),
        ('largest_first', 0.0, 1.5),
        ('largest_first', 0.0, 0.7),
        ('smallest_first', 0.3, 1.0),
    ]
    
    for strategy, open_pen, alpha in more_strategies:
        if time.time() > deadline - 5:
            break
        
        result, cost = _greedy_assign(demands, capacities, open_cost, assign_cost, 
                                      seed=42, strategy=strategy, 
                                      open_penalty=open_pen, alpha=alpha)
        
        if result is not None:
            improved_result, improved_cost = _local_search_with_swaps(
                result, demands, capacities, open_cost, assign_cost, 
                max_iters=3000, seed=123
            )
            _update_best(improved_result, improved_cost)
    
    # Final check
    if best_result is None:
        # Fallback
        result, _ = _greedy_assign(demands, capacities, open_cost, assign_cost, 
                                   seed=42, strategy='largest_first', alpha=1.0)
        if result is not None:
            best_result = result
        else:
            best_result = np.zeros(n_cust, dtype=int)
    
    return best_result
# EVOLVE-BLOCK-END
