# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: Greedy initialization + aggressive vectorized local search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 118.0  # safety margin for 120s limit

    # Precompute sorted facility indices by assignment cost for each customer
    fac_order = np.argsort(assign_cost, axis=1)
    
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

    def _greedy_assign():
        """Greedy assignment: sort customers by demand descending, assign to cheapest feasible facility."""
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        order = np.argsort(-demands)
        
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Fallback: find any facility with space
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
            if not assigned:
                # Should not happen if total capacity >= total demand
                result[i] = 0
                load[0] += demands[i]
        
        return result

    def _force_valid(result):
        """Emergency repair: ensure validity by re-greedy if needed."""
        n_cust = len(demands)
        n_fac = len(capacities)
        result = result.copy()
        load = np.zeros(n_fac, dtype=np.int64)
        
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                result[i] = 0
            load[result[i]] += demands[i]
            
        if np.any(load > capacities):
            load = np.zeros(n_fac, dtype=np.int64)
            result = np.full(n_cust, -1, dtype=np.int64)
            order = np.argsort(-demands)
            
            for i in order:
                assigned = False
                for f in fac_order[i]:
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
                if not assigned:
                    for f in range(n_fac):
                        if load[f] + demands[i] <= capacities[f]:
                            load[f] += demands[i]
                            result[i] = f
                            assigned = True
                            break
                if not assigned:
                    result[i] = 0
                    load[0] += demands[i]
                    
        return result

    def _local_search(result, deadline):
        """
        Vectorized local search with single-move and swap improvements.
        """
        n_cust = len(result)
        n_fac = len(capacities)
        result = result.copy()
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            load[f] += demands[i]
        
        # Ensure valid start
        if np.any(load > capacities):
            result = _force_valid(result)
            load = np.zeros(n_fac, dtype=np.int64)
            for i, f in enumerate(result):
                load[f] += demands[i]
        
        current_obj = _compute_obj(result)
        
        max_rounds = 500
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds and time.time() < deadline - 3:
            improved = False
            rounds += 1
            
            # Vectorized single-move evaluation
            # delta[i, j] = assign_cost[i,j] - assign_cost[i, result[i]] + open_cost changes
            cur_assign_cost = np.array([assign_cost[i, result[i]] for i in range(n_cust)])
            delta_base = assign_cost - cur_assign_cost[:, np.newaxis]
            
            cur_fac = result
            is_last_at_cur = (load[cur_fac] == demands)
            zero_load_fac = (load == 0)
            
            not_cur = np.ones((n_cust, n_fac), dtype=bool)
            not_cur[np.arange(n_cust), cur_fac] = False
            
            # Open cost savings if moving away from a facility that becomes empty
            term1 = np.where(is_last_at_cur[:, np.newaxis], -open_cost[cur_fac][:, np.newaxis], 0.0)
            term1 = np.where(not_cur, term1, 0.0)
            
            # Open cost penalty if moving to a facility that was empty
            term2 = np.where(zero_load_fac[np.newaxis, :], open_cost[np.newaxis, :], 0.0)
            term2 = np.where(not_cur, term2, 0.0)
            
            delta_open = term1 + term2
            total_delta = delta_base + delta_open
            
            # Mask current assignments
            total_delta[np.arange(n_cust), cur_fac] = np.inf
            
            # Mask infeasible moves
            infeasible = (load[np.newaxis, :] + demands[:, np.newaxis]) > capacities[np.newaxis, :]
            total_delta[infeasible] = np.inf
            
            best_f = np.argmin(total_delta, axis=1)
            best_delta = np.min(total_delta, axis=1)
            
            improving = best_delta < -1e-9
            if not np.any(improving):
                # Try swaps if no single moves improve
                if time.time() < deadline - 3:
                    swap_improved = _swap_search(result, load, deadline)
                    if swap_improved:
                        improved = True
                        continue
                break
            
            improving_indices = np.where(improving)[0]
            # Sort by most improving first
            improving_indices = improving_indices[np.argsort(best_delta[improving_indices])]
            
            modified_fac = np.zeros(n_fac, dtype=bool)
            modified_cust = np.zeros(n_cust, dtype=bool)
            
            for i in improving_indices:
                if time.time() > deadline - 3:
                    break
                if modified_cust[i]:
                    continue
                f_new = best_f[i]
                f_old = result[i]
                if modified_fac[f_new] or modified_fac[f_old]:
                    continue
                
                if load[f_new] + demands[i] > capacities[f_new]:
                    continue
                
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                modified_fac[f_old] = True
                modified_fac[f_new] = True
                modified_cust[i] = True
                improved = True
        
        return result

    def _swap_search(result, load, deadline):
        """Try 2-swap moves: swap customers i and j if it improves cost."""
        n_cust = len(result)
        n_fac = len(capacities)
        
        # Only try swaps for a subset to keep it fast
        # Random sample of customer pairs
        np.random.seed(42)
        max_swap_attempts = 2000
        attempts = 0
        improved_any = False
        
        while attempts < max_swap_attempts and time.time() < deadline - 3:
            i = np.random.randint(0, n_cust)
            j = np.random.randint(0, n_cust)
            if i == j:
                continue
            attempts += 1
            
            f_i = result[i]
            f_j = result[j]
            if f_i == f_j:
                continue
            
            # Check if swap is feasible
            # After swap: i goes to f_j, j goes to f_i
            # Load change at f_i: -demands[i] + demands[j]
            # Load change at f_j: -demands[j] + demands[i]
            new_load_i = load[f_i] - demands[i] + demands[j]
            new_load_j = load[f_j] - demands[j] + demands[i]
            
            if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                continue
            
            # Cost change
            old_cost = assign_cost[i, f_i] + assign_cost[j, f_j]
            new_cost = assign_cost[i, f_j] + assign_cost[j, f_i]
            
            # Open cost changes
            old_open = 0.0
            new_open = 0.0
            if load[f_i] == demands[i]:  # f_i becomes empty after i leaves
                old_open += open_cost[f_i]
            if load[f_j] == demands[j]:  # f_j becomes empty after j leaves
                old_open += open_cost[f_j]
            
            if new_load_i == 0:  # f_i becomes empty after swap
                new_open += open_cost[f_i]
            if new_load_j == 0:  # f_j becomes empty after swap
                new_open += open_cost[f_j]
            
            delta = (new_cost - old_cost) + (new_open - old_open)
            
            if delta < -1e-9:
                # Apply swap
                load[f_i] -= demands[i]
                load[f_i] += demands[j]
                load[f_j] -= demands[j]
                load[f_j] += demands[i]
                result[i] = f_j
                result[j] = f_i
                improved_any = True
        
        return improved_any

    # Main logic
    # Start with greedy assignment
    result = _greedy_assign()
    
    # Run local search
    if time.time() < deadline - 5:
        result = _local_search(result, deadline)
    
    # Final validity check
    result = _force_valid(result)
    
    return result.astype(int)

# EVOLVE-BLOCK-END
