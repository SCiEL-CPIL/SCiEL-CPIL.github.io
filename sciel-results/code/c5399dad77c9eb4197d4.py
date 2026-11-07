"""Combined program: Lagrangian decomposition + vectorized local search + SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _greedy_assign(demands, capacities, assign_cost, order, lambdas):
    """Greedy assignment with given order and Lagrangian multipliers."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in order:
        d = demands[i]
        # Effective cost: assignment cost + lambda * demand
        costs = assign_cost[i] + lambdas * d
        # Mask infeasible facilities
        feasible = (load + d) <= capacities
        if not np.any(feasible):
            # Fallback: pick facility with most remaining capacity
            remaining = capacities - load
            f = np.argmin(remaining)
        else:
            costs_masked = np.where(feasible, costs, np.inf)
            f = np.argmin(costs_masked)
        load[f] += d
        result[i] = f
    return result

def _compute_cost(assignment, demands, capacities, open_cost, assign_cost):
    """Compute total cost: assignment + opening costs."""
    n_customers = len(assignment)
    n_facilities = len(capacities)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[assignment[i]] += demands[i]
    
    open_sum = 0.0
    for f in range(n_facilities):
        if load[f] > 0:
            open_sum += open_cost[f]
    
    assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
    return open_sum + assign_sum, load

def _local_search(demands, capacities, open_cost, assign_cost, assignment, time_limit=10.0):
    """Vectorized local search: try moving individual customers."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = assignment.copy().astype(int)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    current_cost, _ = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_result = result.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try moving each customer
        for i in np.random.permutation(n_customers):
            if time.time() - start_time > time_limit:
                break
            
            d = demands[i]
            current_f = result[i]
            
            # Vectorized cost computation for all facilities
            # delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
            delta_assign = assign_cost[i] - assign_cost[i, current_f]
            
            # delta_open: depends on whether source/dest become empty/non-empty
            # If load[current_f] - d == 0, facility closes: -open_cost[current_f]
            # If load[f] == 0, facility opens: +open_cost[f]
            
            # Precompute for each facility
            load_after_source = load[current_f] - d
            closes_source = (load_after_source == 0)
            
            # For each facility f, check if it would be opened
            opens_dest = (load == 0)
            
            # delta_open for each facility
            delta_open = np.zeros(n_facilities)
            if closes_source:
                delta_open[:] -= open_cost[current_f]
            # Add open cost for facilities that are currently empty
            delta_open[opens_dest] += open_cost[opens_dest]
            
            # Total delta
            total_delta = delta_assign + delta_open
            
            # Mask infeasible: capacity violation or same facility
            feasible = (load + d) <= capacities
            feasible[current_f] = False  # Don't move to same facility
            
            if not np.any(feasible):
                continue
            
            # Find best improvement
            total_delta_masked = np.where(feasible, total_delta, np.inf)
            best_f = np.argmin(total_delta_masked)
            best_delta = total_delta_masked[best_f]
            
            if best_delta < -1e-9:
                # Perform move
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if not improved:
            if no_improve_count > max_no_improve:
                break
            # Random perturbation: swap two customers
            if np.random.random() < 0.5:
                i1, i2 = np.random.choice(n_customers, 2, replace=False)
                f1, f2 = result[i1], result[i2]
                if f1 != f2:
                    # Check feasibility
                    if load[f1] - demands[i1] + demands[i2] <= capacities[f1] and \
                       load[f2] - demands[i2] + demands[i1] <= capacities[f2]:
                        load[f1] = load[f1] - demands[i1] + demands[i2]
                        load[f2] = load[f2] - demands[i2] + demands[i1]
                        result[i1], result[i2] = f2, f1
                        
    return best_result, best_cost

def _simulated_annealing(demands, capacities, open_cost, assign_cost, initial_assignment, time_limit=5.0):
    """Simulated annealing for facility location."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = initial_assignment.copy().astype(int)
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    current_cost, _ = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_result = result.copy()
    
    start_time = time.time()
    temp = 100.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        # Pick a random customer
        i = np.random.randint(n_customers)
        d = demands[i]
        current_f = result[i]
        
        # Pick a random facility
        f = np.random.randint(n_facilities)
        if f == current_f:
            continue
        if load[f] + d > capacities[f]:
            continue
        
        # Compute cost delta
        delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
        delta_open = 0.0
        if load[current_f] - d == 0:
            delta_open -= open_cost[current_f]
        if load[f] == 0:
            delta_open += open_cost[f]
        
        total_delta = delta_assign + delta_open
        
        if total_delta < 0 or np.random.random() < np.exp(-total_delta / max(temp, 1e-6)):
            # Accept
            load[current_f] -= d
            load[f] += d
            result[i] = f
            current_cost += total_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
        
        temp *= cooling_rate
    
    return best_result, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_result = None
    best_cost = np.inf
    
    # Initialize Lagrangian multipliers
    lambdas = np.zeros(n_facilities)
    
    # Phase 1: Lagrangian relaxation with greedy repair
    # Run multiple iterations of Lagrangian subgradient method
    max_lag_iters = 50
    lag_iter = 0
    U = np.inf  # Upper bound (best feasible cost)
    L_best = -np.inf  # Best Lagrangian lower bound
    
    while time.time() < deadline - 30 and lag_iter < max_lag_iters:
        lag_iter += 1
        
        # Solve relaxed problem: assign each customer to min cost facility
        # Cost: assign_cost[i, f] + lambdas[f] * demands[i]
        term = demands[:, None] * lambdas[None, :]
        C_relax = assign_cost + term
        f_relax = np.argmin(C_relax, axis=1)
        
        # Calculate loads from relaxed solution
        load_relax = np.zeros(n_facilities)
        for i in range(n_customers):
            load_relax[f_relax[i]] += demands[i]
        
        # Calculate Lagrangian value
        assign_sum_relax = np.sum(assign_cost[np.arange(n_customers), f_relax])
        penalty_sum = np.sum(lambdas * (load_relax - capacities))
        L_val = assign_sum_relax + penalty_sum
        
        if L_val > L_best:
            L_best = L_val
        
        # Subgradient: g = load_relax - capacities
        g = load_relax - capacities
        g_norm_sq = np.sum(g**2)
        
        if g_norm_sq < 1e-10:
            break
        
        # Polyak step size
        if U <= L_val:
            alpha = 0.1
        else:
            alpha = (U - L_val) / g_norm_sq
            alpha = min(alpha, 1.0)
        
        # Update lambdas
        lambdas = np.maximum(0, lambdas + alpha * g)
        
        # Repair to feasible solution using greedy with current lambdas
        # Try multiple orderings
        orders = [
            np.argsort(-demands),  # Largest demand first
            np.argsort(assign_cost.min(axis=1)),  # Cheapest first
            np.random.permutation(n_customers)  # Random
        ]
        
        for order in orders:
            if time.time() > deadline - 25:
                break
            
            sol = _greedy_assign(demands, capacities, assign_cost, order, lambdas)
            cost, _ = _compute_cost(sol, demands, capacities, open_cost, assign_cost)
            
            if cost < best_cost:
                best_cost = cost
                best_result = sol.copy()
                U = cost
    
    # Phase 2: Local search on best solution
    if best_result is not None and time.time() < deadline - 15:
        ls_result, ls_cost = _local_search(demands, capacities, open_cost, assign_cost, 
                                            best_result, time_limit=12.0)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_result = ls_result.copy()
    
    # Phase 3: Simulated annealing on best
    if best_result is not None and time.time() < deadline - 5:
        sa_result, sa_cost = _simulated_annealing(demands, capacities, open_cost, assign_cost,
                                                   best_result, time_limit=8.0)
        if sa_cost < best_cost:
            best_cost = sa_cost
            best_result = sa_result.copy()
    
    # Final safety check
    if best_result is None:
        # Fallback to simple greedy
        order = np.argsort(-demands)
        best_result = _greedy_assign(demands, capacities, assign_cost, order, np.zeros(n_facilities))
    
    # Verify and repair if needed
    result = best_result.astype(int)
    loads = np.zeros(n_facilities)
    for i in range(n_customers):
        loads[result[i]] += demands[i]
    
    if np.any(loads > capacities + 1e-6):
        # Repair capacity violations
        for f in range(n_facilities):
            if loads[f] > capacities[f]:
                excess = loads[f] - capacities[f]
                customers_at_f = np.where(result == f)[0]
                customers_at_f = customers_at_f[np.argsort(demands[customers_at_f])]
                for i in customers_at_f:
                    if excess <= 0:
                        break
                    d = demands[i]
                    best_alt = -1
                    best_alt_cost = np.inf
                    for alt_f in range(n_facilities):
                        if alt_f == f:
                            continue
                        if loads[alt_f] + d <= capacities[alt_f]:
                            alt_cost = assign_cost[i, alt_f]
                            if alt_cost < best_alt_cost:
                                best_alt_cost = alt_cost
                                best_alt = alt_f
                    if best_alt >= 0:
                        loads[f] -= d
                        loads[best_alt] += d
                        result[i] = best_alt
                        excess -= d
    
    return result
# EVOLVE-BLOCK-END
