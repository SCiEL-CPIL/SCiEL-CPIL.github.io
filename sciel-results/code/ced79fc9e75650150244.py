"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def compute_cost(result):
        """Compute total cost for an assignment."""
        cost = 0.0
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            f = int(result[i])
            cost += assign_cost[i, f]
            load[f] += demands[i]
        for f in range(n_facilities):
            if load[f] > 1e-9:
                cost += open_cost[f]
        return cost, load

    def solve_lp_relaxation():
        """
        Solve LP relaxation of CFLP.
        """
        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y
        
        c = np.zeros(n_vars)
        for i in range(n_customers):
            for f in range(n_facilities):
                c[i * n_facilities + f] = assign_cost[i, f]
        for f in range(n_facilities):
            c[n_x + f] = open_cost[f]
        
        A_eq = np.zeros((n_customers, n_vars))
        for i in range(n_customers):
            for f in range(n_facilities):
                A_eq[i, i * n_facilities + f] = 1.0
        b_eq = np.ones(n_customers)
        
        A_ub = np.zeros((n_facilities, n_vars))
        for f in range(n_facilities):
            for i in range(n_customers):
                A_ub[f, i * n_facilities + f] = demands[i]
            A_ub[f, n_x + f] = -capacities[f]
        b_ub = np.zeros(n_facilities)
        
        bounds = []
        for _ in range(n_vars):
            bounds.append((0, 1))
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if not result.success:
                return None
            x = result.x[:n_x].reshape(n_customers, n_facilities)
            return x
        except Exception:
            return None

    def round_lp_solution(x):
        """
        Round LP solution using greedy assignment guided by fractional values.
        """
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort by max fractional value descending (most confident first)
        max_frac = np.max(x, axis=1)
        order = np.argsort(-max_frac)
        
        for i in order:
            # Find best feasible facility
            best_f = -1
            best_score = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f] + 1e-6:
                    # Prefer high LP value, penalize high load slightly
                    score = -x[i, f] + 0.001 * load[f]
                    if score < best_score:
                        best_score = score
                        best_f = f
            if best_f == -1:
                # Fallback: most remaining capacity
                remaining = capacities - load
                best_f = int(np.argmax(remaining))
            result[i] = best_f
            load[best_f] += demands[i]
        
        return result, load

    def greedy_assignment(strategy='demand'):
        """
        Greedy assignment.
        strategy: 'demand' (sort by demand desc), 'cost' (sort by min assign cost asc)
        """
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        
        if strategy == 'demand':
            order = np.argsort(-demands)
        elif strategy == 'cost':
            min_costs = np.min(assign_cost, axis=1)
            order = np.argsort(min_costs)
        else:
            order = np.arange(n_customers)
            
        for i in order:
            best_f = -1
            best_cost = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f] + 1e-6:
                    # Cost includes opening cost if facility is currently empty
                    cost = assign_cost[i, f]
                    if load[f] == 0:
                        cost += open_cost[f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f == -1:
                remaining = capacities - load
                best_f = int(np.argmax(remaining))
            result[i] = best_f
            load[best_f] += demands[i]
        
        return result, load

    def local_search(result, load, current_cost, time_limit):
        """Deterministic local search with 1-move and 2-swap."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-move pass
            for i in range(n_customers):
                if time.time() > time_limit:
                    return result, current_cost, load
                old_f = result[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f] + 1e-6:
                        continue
                    
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                    delta_open = 0.0
                    if load[old_f] == demands[i]:
                        delta_open -= open_cost[old_f]
                    if load[new_f] == 0:
                        delta_open += open_cost[new_f]
                    
                    delta_cost = delta_assign + delta_open
                    if delta_cost < best_delta - 1e-9:
                        best_delta = delta_cost
                        best_new_f = new_f
                
                if best_new_f != -1:
                    load[old_f] -= demands[i]
                    load[best_new_f] += demands[i]
                    result[i] = best_new_f
                    current_cost += best_delta
                    improved = True
            
            # 2-swap pass
            for i in range(n_customers):
                if time.time() > time_limit:
                    return result, current_cost, load
                old_f_i = result[i]
                for j in range(i + 1, n_customers):
                    if time.time() > time_limit:
                        return result, current_cost, load
                    old_f_j = result[j]
                    if old_f_i == old_f_j:
                        continue
                    
                    if load[old_f_j] - demands[j] + demands[i] <= capacities[old_f_j] + 1e-6 and \
                       load[old_f_i] - demands[i] + demands[j] <= capacities[old_f_i] + 1e-6:
                        
                        delta_assign = (assign_cost[i, old_f_j] - assign_cost[i, old_f_i]) + \
                                       (assign_cost[j, old_f_i] - assign_cost[j, old_f_j])
                        
                        delta_open = 0.0
                        new_load_i = load[old_f_i] - demands[i] + demands[j]
                        new_load_j = load[old_f_j] - demands[j] + demands[i]
                        
                        if new_load_i == 0 and load[old_f_i] != 0:
                            delta_open -= open_cost[old_f_i]
                        if new_load_j == 0 and load[old_f_j] != 0:
                            delta_open -= open_cost[old_f_j]
                        if new_load_i > 0 and load[old_f_i] == 0:
                            delta_open += open_cost[old_f_i]
                        if new_load_j > 0 and load[old_f_j] == 0:
                            delta_open += open_cost[old_f_j]
                        
                        delta_cost = delta_assign + delta_open
                        if delta_cost < -1e-9:
                            load[old_f_i] -= demands[i]
                            load[old_f_j] -= demands[j]
                            load[old_f_j] += demands[i]
                            load[old_f_i] += demands[j]
                            result[i] = old_f_j
                            result[j] = old_f_i
                            current_cost += delta_cost
                            improved = True
                            break
        
        return result, current_cost, load

    def simulated_annealing(result, load, current_cost, time_limit):
        """
        Simulated Annealing with 1-move and random 2-swap moves.
        """
        temp = 1000.0
        cooling_rate = 0.9995
        min_temp = 0.1
        
        best_result = result.copy()
        best_cost = current_cost
        best_load = load.copy()
        
        while temp > min_temp and time.time() < time_limit:
            # Random move
            if random.random() < 0.7:
                # 1-move
                i = np.random.randint(n_customers)
                old_f = result[i]
                feasible = [f for f in range(n_facilities) if f != old_f and load[f] + demands[i] <= capacities[f] + 1e-6]
                if not feasible:
                    continue
                new_f = random.choice(feasible)
                
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                delta_open = 0.0
                if load[old_f] == demands[i]:
                    delta_open -= open_cost[old_f]
                if load[new_f] == 0:
                    delta_open += open_cost[new_f]
                
                delta_cost = delta_assign + delta_open
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / temp):
                    load[old_f] -= demands[i]
                    load[new_f] += demands[i]
                    result[i] = new_f
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = result.copy()
                        best_load = load.copy()
            else:
                # 2-swap
                i = np.random.randint(n_customers)
                j = np.random.randint(n_customers)
                if i == j:
                    continue
                old_f_i = result[i]
                old_f_j = result[j]
                if old_f_i == old_f_j:
                    continue
                
                if load[old_f_j] - demands[j] + demands[i] <= capacities[old_f_j] + 1e-6 and \
                   load[old_f_i] - demands[i] + demands[j] <= capacities[old_f_i] + 1e-6:
                    
                    delta_assign = (assign_cost[i, old_f_j] - assign_cost[i, old_f_i]) + \
                                   (assign_cost[j, old_f_i] - assign_cost[j, old_f_j])
                    
                    delta_open = 0.0
                    new_load_i = load[old_f_i] - demands[i] + demands[j]
                    new_load_j = load[old_f_j] - demands[j] + demands[i]
                    
                    if new_load_i == 0 and load[old_f_i] != 0:
                        delta_open -= open_cost[old_f_i]
                    if new_load_j == 0 and load[old_f_j] != 0:
                        delta_open -= open_cost[old_f_j]
                    if new_load_i > 0 and load[old_f_i] == 0:
                        delta_open += open_cost[old_f_i]
                    if new_load_j > 0 and load[old_f_j] == 0:
                        delta_open += open_cost[old_f_j]
                    
                    delta_cost = delta_assign + delta_open
                    
                    if delta_cost < 0 or random.random() < np.exp(-delta_cost / temp):
                        load[old_f_i] -= demands[i]
                        load[old_f_j] -= demands[j]
                        load[old_f_j] += demands[i]
                        load[old_f_i] += demands[j]
                        result[i] = old_f_j
                        result[j] = old_f_i
                        current_cost += delta_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = result.copy()
                            best_load = load.copy()
            
            temp *= cooling_rate
        
        return best_result, best_cost, best_load

    # Main logic
    best_overall_result = None
    best_overall_cost = np.inf
    best_overall_load = None
    
    # Strategy 1: LP Relaxation
    if time.time() < deadline - 60:
        x_lp = solve_lp_relaxation()
        if x_lp is not None:
            result, load = round_lp_solution(x_lp)
            cost, load = compute_cost(result)
            
            # Polish with Local Search
            time_limit_ls = min(deadline - 40, time.time() + 10)
            result, cost, load = local_search(result, load, cost, time_limit_ls)
            
            # SA Polish
            time_limit_sa = min(deadline - 20, time.time() + 20)
            result, cost, load = simulated_annealing(result, load, cost, time_limit_sa)
            
            if cost < best_overall_cost:
                best_overall_result = result.copy()
                best_overall_cost = cost
                best_overall_load = load.copy()
    
    # Strategy 2: Greedy Demand + SA
    if time.time() < deadline - 40:
        result, load = greedy_assignment('demand')
        cost, load = compute_cost(result)
        
        time_limit_ls = min(deadline - 30, time.time() + 5)
        result, cost, load = local_search(result, load, cost, time_limit_ls)
        
        time_limit_sa = min(deadline - 15, time.time() + 20)
        result, cost, load = simulated_annealing(result, load, cost, time_limit_sa)
        
        if cost < best_overall_cost:
            best_overall_result = result.copy()
            best_overall_cost = cost
            best_overall_load = load.copy()

    # Strategy 3: Greedy Cost + SA
    if time.time() < deadline - 20:
        result, load = greedy_assignment('cost')
        cost, load = compute_cost(result)
        
        time_limit_ls = min(deadline - 15, time.time() + 5)
        result, cost, load = local_search(result, load, cost, time_limit_ls)
        
        time_limit_sa = min(deadline - 5, time.time() + 10)
        result, cost, load = simulated_annealing(result, load, cost, time_limit_sa)
        
        if cost < best_overall_cost:
            best_overall_result = result.copy()
            best_overall_cost = cost
            best_overall_load = load.copy()

    # Final Polish on the best result found
    if best_overall_result is not None and time.time() < deadline - 5:
        time_limit_final = deadline - 2
        best_overall_result, best_overall_cost, best_overall_load = local_search(
            best_overall_result, best_overall_load, best_overall_cost, time_limit_final
        )

    # Fallback if no valid result
    if best_overall_result is None:
        best_overall_result, best_overall_load = greedy_assignment('demand')
    
    # Verify and fix if necessary
    load_check = np.zeros(n_facilities)
    for i in range(n_customers):
        f = int(best_overall_result[i])
        if f < 0 or f >= n_facilities:
            best_overall_result, best_overall_load = greedy_assignment('demand')
            load_check = np.zeros(n_facilities)
            for i in range(n_customers):
                load_check[int(best_overall_result[i])] += demands[i]
            break
        load_check[f] += demands[i]
    
    if np.any(load_check > capacities + 1e-6):
        # Force fix: move customers from overloaded facilities
        for f in range(n_facilities):
            if load_check[f] > capacities[f]:
                # Find customers in f and move them
                customers_in_f = [i for i in range(n_customers) if best_overall_result[i] == f]
                for i in customers_in_f:
                    # Find a facility with space
                    for new_f in range(n_facilities):
                        if new_f != f and load_check[new_f] + demands[i] <= capacities[new_f]:
                            load_check[f] -= demands[i]
                            load_check[new_f] += demands[i]
                            best_overall_result[i] = new_f
                            break
    
    return best_overall_result.astype(int)
# EVOLVE-BLOCK-END
