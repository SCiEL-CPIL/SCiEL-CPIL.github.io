"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities.
    Strategy:
    1. Generate diverse initial solutions (Greedy largest-first, Greedy smallest-first, LP-rounding if fast).
    2. Polish the best initial solution using local search with:
       - 1-move (reassign customer)
       - 2-swap (swap customers between facilities)
       - Close-facility moves (move all customers from a facility to others, if beneficial)
    3. Return the best valid solution found within the time limit.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # --- Helper Functions ---
    
    def compute_cost(result):
        """Compute total cost for a given assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

    def greedy_init(order_type, randomize=False, seed=None):
        """
        Greedy initialization.
        order_type: 'desc' for largest demand first, 'asc' for smallest.
        """
        if seed is not None:
            np.random.seed(seed)
            
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        if order_type == 'desc':
            order = np.argsort(-demands)
        else:
            order = np.argsort(demands)
            
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            if randomize:
                # Add small random noise to break ties / explore
                noise = np.random.uniform(0, 10) * np.ones(n_fac)
                fac_order = np.argsort(assign_cost[i] + noise)
                
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: find facility with most remaining capacity
                remaining = capacities - load
                feasible_facs = np.where(remaining >= demands[i])[0]
                if len(feasible_facs) > 0:
                    # Pick the one with min cost among feasible
                    best_f = min(feasible_facs, key=lambda f: assign_cost[i, f])
                    load[best_f] += demands[i]
                    result[i] = best_f
                else:
                    # Infeasible input or bug, return None
                    return None
                    
        return result

    def lp_rounding_init():
        """
        Try to solve LP relaxation and round.
        Returns None if too slow or fails.
        """
        try:
            from scipy.optimize import linprog
            # Check time budget
            if time.time() > start_time + 5.0:
                return None
                
            # Variables: x[i, j] (n_cust * n_fac) + y[j] (n_fac)
            n_vars = n_cust * n_fac + n_fac
            c_obj = np.concatenate([assign_cost.flatten(), open_cost])
            
            bounds = [(0, 1)] * n_vars
            
            # Equality constraints: sum_j x[i, j] = 1
            A_eq = np.zeros((n_cust, n_vars))
            for i in range(n_cust):
                for j in range(n_fac):
                    A_eq[i, i * n_fac + j] = 1.0
            b_eq = np.ones(n_cust)
            
            # Inequality constraints:
            # 1. Capacity: sum_i demand[i] * x[i, j] <= capacity[j] * y[j]
            #    sum_i demand[i] * x[i, j] - capacity[j] * y[j] <= 0
            # 2. Linking: x[i, j] <= y[j]  =>  x[i, j] - y[j] <= 0
            
            n_ineq = n_fac + (n_cust * n_fac)
            A_ub = np.zeros((n_ineq, n_vars))
            b_ub = np.zeros(n_ineq)
            
            # Capacity constraints
            for j in range(n_fac):
                row_idx = j
                for i in range(n_cust):
                    A_ub[row_idx, i * n_fac + j] = demands[i]
                A_ub[row_idx, n_cust * n_fac + j] = -capacities[j]
                
            # Linking constraints
            for i in range(n_cust):
                for j in range(n_fac):
                    row_idx = n_fac + i * n_fac + j
                    A_ub[row_idx, i * n_fac + j] = 1.0
                    A_ub[row_idx, n_cust * n_fac + j] = -1.0
                    
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.status != 0 or res.x is None:
                return None
                
            x_lp = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
            
            # Round: Sort customers by demand desc. For each, pick facility with highest LP value that fits.
            result = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac)
            order = np.argsort(-demands)
            
            for i in order:
                # Sort facilities by LP value descending
                fac_order = np.argsort(-x_lp[i])
                assigned = False
                for f in fac_order:
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
                if not assigned:
                    # Fallback to min cost feasible
                    feasible = np.where(capacities - load >= demands[i])[0]
                    if len(feasible) > 0:
                        best_f = min(feasible, key=lambda f: assign_cost[i, f])
                        load[best_f] += demands[i]
                        result[i] = best_f
                    else:
                        return None
            return result
        except Exception:
            return None

    # --- Local Search ---
    
    def local_search(result, time_limit):
        """
        Local search with 1-moves, 2-swaps, and facility closing moves.
        """
        current = result.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        def get_cost():
            return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), current])
            
        current_cost = get_cost()
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        improved = True
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment (1-move)
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_fac):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f]:
                        continue
                    
                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                    
                    # Open cost change
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != -1:
                    # Apply move
                    old_f = cur_f
                    new_f = best_new_f
                    current[i] = new_f
                    load[old_f] -= d_i
                    load[new_f] += d_i
                    fac_count[old_f] -= 1
                    fac_count[new_f] += 1
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
            
            # 2. Swap (2-move)
            # Try swapping two customers i, j between facilities f_i, f_j
            # Only consider if f_i != f_j
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                f_i = current[i]
                d_i = demands[i]
                for j in range(i + 1, n_cust):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    # Check capacity feasibility
                    # After swap:
                    # load[f_i] - d_i + d_j <= capacities[f_i]
                    # load[f_j] - d_j + d_i <= capacities[f_j]
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Open cost change
                    # If fac_count[f_i] becomes 0, save open_cost[f_i]
                    # If fac_count[f_j] becomes 0, save open_cost[f_j]
                    # If fac_count[f_i] was 1 and becomes >1 (impossible since d_j > 0), etc.
                    # Actually, if fac_count[f_i] == 1, then after swap, if d_j > 0, fac_count becomes 1 (j moves in). 
                    # Wait, i moves out, j moves in. So count stays same unless one was 1 and the other was 0? 
                    # No, i is in f_i, j is in f_j. 
                    # If fac_count[f_i] == 1, then only i is there. After swap, j is there. Count is still 1.
                    # If fac_count[f_i] > 1, count decreases by 1, increases by 1. Net 0.
                    # So open cost doesn't change in a simple swap between two occupied facilities.
                    # Unless one facility becomes empty? 
                    # If fac_count[f_i] == 1, then i is the only one. i moves to f_j. f_i becomes empty.
                    # j moves to f_i. f_i becomes occupied.
                    # So open cost is unchanged.
                    
                    if delta < 0:
                        # Apply swap
                        current[i] = f_j
                        current[j] = f_i
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        # fac_count unchanged
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        break # Restart i loop? Or just continue. Break j loop to restart i?
                        # Actually, restarting the whole loop is safer to avoid missing other moves.
                        # But for speed, just break j and let i increment.
            
            # 3. Close Facility Move
            # Try to close a facility f by moving all its customers to other facilities.
            # Only consider if fac_count[f] > 0.
            # This is expensive, so only try if time allows.
            if time.time() - start < time_limit - 1.0:
                for f in range(n_fac):
                    if fac_count[f] == 0:
                        continue
                    # Try to move all customers in f to other facilities
                    # This is a complex sub-problem.
                    # Heuristic: Try to move each customer in f to the best alternative.
                    # If all can be moved feasibly, and cost decreases, accept.
                    
                    # Backup state
                    old_current = current.copy()
                    old_load = load.copy()
                    old_fac_count = fac_count.copy()
                    old_cost = current_cost
                    
                    # List of customers in f
                    custs_in_f = [i for i in range(n_cust) if current[i] == f]
                    
                    # Sort by demand desc to place large ones first?
                    custs_in_f.sort(key=lambda i: -demands[i])
                    
                    success = True
                    new_cost = old_cost - open_cost[f] # Save open cost
                    
                    for i in custs_in_f:
                        d_i = demands[i]
                        # Find best alternative
                        best_alt_f = -1
                        best_alt_cost = float('inf')
                        
                        for alt_f in range(n_fac):
                            if alt_f == f:
                                continue
                            if load[alt_f] + d_i <= capacities[alt_f]:
                                # Cost of moving i to alt_f
                                # Note: load[alt_f] is current load, which might include other moved customers
                                c_move = assign_cost[i, alt_f]
                                if c_move < best_alt_cost:
                                    best_alt_cost = c_move
                                    best_alt_f = alt_f
                                    
                        if best_alt_f == -1:
                            success = False
                            break
                            
                        # Apply move temporarily
                        current[i] = best_alt_f
                        load[best_alt_f] += d_i
                        new_cost += best_alt_cost - assign_cost[i, f]
                        
                    if success and new_cost < old_cost:
                        # Accept move
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        # Update fac_count
                        fac_count = np.zeros(n_fac, dtype=int)
                        for i in range(n_cust):
                            fac_count[current[i]] += 1
                    else:
                        # Revert
                        current = old_current
                        load = old_load
                        fac_count = old_fac_count
                        current_cost = old_cost

        return best_result

    # --- Main Logic ---
    
    candidates = []
    
    # 1. Greedy Desc
    init_1 = greedy_init('desc')
    if init_1 is not None:
        candidates.append(init_1)
        
    # 2. Greedy Asc
    init_2 = greedy_init('asc')
    if init_2 is not None:
        candidates.append(init_2)
        
    # 3. Randomized Greedy (a few seeds)
    for s in [42, 123, 456]:
        if time.time() > deadline - 10:
            break
        init_r = greedy_init('desc', randomize=True, seed=s)
        if init_r is not None:
            candidates.append(init_r)
            
    # 4. LP Rounding (if fast enough)
    if time.time() < start_time + 5.0:
        init_lp = lp_rounding_init()
        if init_lp is not None:
            candidates.append(init_lp)
            
    if not candidates:
        # Last resort
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        candidates.append(result)
        
    # Pick best initial candidate
    best_init = min(candidates, key=compute_cost)
    
    # Polish
    # Allocate remaining time to local search
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    if time_remaining > 1.0:
        polished = local_search(best_init, time_remaining - 2.0)
        if polished is not None:
            best_init = polished
    
    return best_init

# EVOLVE-BLOCK-END
