"""Capacitated facility location: multi-strategy solver with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a combination of:
    1. Lagrangian relaxation for initial solution
    2. Intensive local search with relocate and swap moves
    3. Multiple restarts with perturbation
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    def greedy_init():
        """Greedy initialization: assign customers in order of demand (largest first)."""
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    break
            else:
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    res[i] = fac_order[0]
        return res

    def lagrangian_init():
        """
        Lagrangian relaxation: solve relaxed problem where capacity constraints
        are penalized with multipliers, then repair to feasibility.
        """
        # Initialize multipliers
        mu = np.zeros(n_fac)
        # Solve relaxed problem iteratively
        best_res = None
        best_score = np.inf
        
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # For each customer, find best facility considering multiplier
            # cost_i_f = assign_cost[i,f] + mu[f] * demands[i]
            adjusted_cost = assign_cost + np.outer(demands, mu)
            
            # Assign each customer to cheapest facility
            res = np.argmin(adjusted_cost, axis=1)
            
            # Check feasibility
            load = np.zeros(n_fac)
            np.add.at(load, res, demands)
            violations = load - capacities
            
            # Update multipliers using subgradient
            step = 0.1 / (iteration + 1)
            mu = np.maximum(0, mu + step * violations)
            
            # If feasible, evaluate
            if np.all(violations <= 1e-9):
                score = compute_score(res)
                if score < best_score:
                    best_score = score
                    best_res = res.copy()
        
        # If no feasible solution found, use greedy
        if best_res is None:
            best_res = greedy_init()
        
        return best_res

    def local_search(res, load, max_passes=300):
        """
        Intensive local search with relocate and swap moves.
        """
        improved = True
        it = 0
        while improved and it < max_passes:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            
            # Try relocate moves
            for i in np.random.permutation(n_cust):
                if time.time() > deadline:
                    return res, load
                
                current_f = res[i]
                d_i = demands[i]
                
                # Find best alternative facility
                base_delta = assign_cost[i, :] - assign_cost[i, current_f]
                
                feasible = (load + d_i <= capacities + 1e-9)
                feasible[current_f] = False
                
                if not np.any(feasible):
                    continue
                
                deltas = base_delta.copy()
                newly_opened = feasible & (load == 0)
                deltas[newly_opened] += open_cost[newly_opened]
                
                if load[current_f] == d_i:
                    deltas[feasible] -= open_cost[current_f]
                
                feasible_indices = np.where(feasible)[0]
                if len(feasible_indices) == 0:
                    continue
                
                min_idx = np.argmin(deltas[feasible_indices])
                best_f = feasible_indices[min_idx]
                best_delta = deltas[best_f]
                
                if best_delta < -1e-9:
                    load[current_f] -= d_i
                    load[best_f] += d_i
                    res[i] = best_f
                    improved = True
            
            # Try swap moves (exchange customers between two facilities)
            if time.time() > deadline:
                return res, load
            
            # Group customers by facility
            fac_customers = {}
            for f in range(n_fac):
                custs = np.where(res == f)[0]
                if len(custs) > 0:
                    fac_customers[f] = custs
            
            # Try swaps between pairs of facilities
            fac_list = list(fac_customers.keys())
            for a in range(len(fac_list)):
                if time.time() > deadline:
                    return res, load
                for b in range(a + 1, len(fac_list)):
                    fa = fac_list[a]
                    fb = fac_list[b]
                    
                    custs_a = fac_customers[fa]
                    custs_b = fac_customers[fb]
                    
                    # Try swapping one customer from each
                    for i in custs_a[:5]:  # Limit for speed
                        if time.time() > deadline:
                            return res, load
                        for j in custs_b[:5]:
                            d_i = demands[i]
                            d_j = demands[j]
                            
                            # Check if swap is feasible
                            new_load_a = load[fa] - d_i + d_j
                            new_load_b = load[fb] - d_j + d_i
                            
                            if new_load_a > capacities[fa] + 1e-9 or new_load_b > capacities[fb] + 1e-9:
                                continue
                            
                            # Compute cost change
                            old_cost = assign_cost[i, fa] + assign_cost[j, fb]
                            new_cost = assign_cost[i, fb] + assign_cost[j, fa]
                            
                            # Open cost changes
                            open_delta = 0
                            if new_load_a == 0 and load[fa] > 0:
                                open_delta -= open_cost[fa]
                            if new_load_b == 0 and load[fb] > 0:
                                open_delta -= open_cost[fb]
                            if new_load_a > 0 and load[fa] == 0:
                                open_delta += open_cost[fa]
                            if new_load_b > 0 and load[fb] == 0:
                                open_delta += open_cost[fb]
                            
                            total_delta = (new_cost - old_cost) + open_delta
                            
                            if total_delta < -1e-9:
                                # Apply swap
                                load[fa] = new_load_a
                                load[fb] = new_load_b
                                res[i] = fb
                                res[j] = fa
                                improved = True
                                # Update customer lists
                                fac_customers[fa] = np.where(res == fa)[0]
                                fac_customers[fb] = np.where(res == fb)[0]
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return res, load

    def perturbation(res, load, num_moves=10):
        """Perturb solution by moving random customers."""
        res_p = res.copy()
        load_p = load.copy()
        
        for _ in range(num_moves):
            if time.time() > deadline:
                break
            i = np.random.randint(n_cust)
            current_f = res_p[i]
            d_i = demands[i]
            
            feasible = (load_p + d_i <= capacities + 1e-9)
            feasible[current_f] = False
            
            if np.any(feasible):
                feasible_indices = np.where(feasible)[0]
                new_f = feasible_indices[np.random.randint(len(feasible_indices))]
                load_p[current_f] -= d_i
                load_p[new_f] += d_i
                res_p[i] = new_f
        
        return res_p, load_p

    # Generate initial solutions using different strategies
    best_score = np.inf
    best_assign = None
    
    # Strategy 1: Greedy
    res1 = greedy_init()
    load1 = np.zeros(n_fac)
    np.add.at(load1, res1, demands)
    res1, load1 = local_search(res1, load1)
    score1 = compute_score(res1)
    if score1 < best_score:
        best_score = score1
        best_assign = res1.copy()
    
    # Strategy 2: Lagrangian
    if time.time() < deadline:
        res2 = lagrangian_init()
        load2 = np.zeros(n_fac)
        np.add.at(load2, res2, demands)
        res2, load2 = local_search(res2, load2)
        score2 = compute_score(res2)
        if score2 < best_score:
            best_score = score2
            best_assign = res2.copy()
    
    # Strategy 3: Multiple restarts with perturbation
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Perturb best solution
        res_p, load_p = perturbation(best_assign, np.zeros(n_fac), num_moves=5 + r)
        np.add.at(load_p, res_p, demands)
        
        # Run local search
        res_p, load_p = local_search(res_p, load_p, max_passes=100)
        score_p = compute_score(res_p)
        
        if score_p < best_score:
            best_score = score_p
            best_assign = res_p.copy()
    
    # Final intensive local search on best solution
    if time.time() < deadline - 10:
        final_load = np.zeros(n_fac)
        np.add.at(final_load, best_assign, demands)
        best_assign, final_load = local_search(best_assign, final_load, max_passes=200)
    
    return best_assign.astype(int)
# EVOLVE-BLOCK-END
