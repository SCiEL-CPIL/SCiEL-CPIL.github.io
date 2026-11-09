"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    deadline = time.time() + 140.0  # safety margin before 150s
    
    def total_cost(assignment):
        """Compute total cost: open costs for used facilities + assignment costs."""
        used = set(assignment.tolist())
        open_sum = sum(open_cost[f] for f in used)
        assign_sum = sum(assign_cost[i, assignment[i]] for i in range(n_customers))
        return open_sum + assign_sum
    
    def is_feasible(assignment):
        """Check if assignment respects capacities."""
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)
    
    def greedy_cheapest():
        """Assign each customer to cheapest feasible facility, largest demand first."""
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_cost = np.inf
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    best_f = f
                    best_cost = assign_cost[i, f]
                    break
            if best_f == -1:
                # Fallback: find any feasible
                for f in range(n_facilities):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        best_f = f
                        break
            if best_f == -1:
                best_f = 0  # should not happen
            load[best_f] += demands[i]
            result[i] = best_f
        return result
    
    def greedy_with_penalty(penalty_weight):
        """Greedy with open cost penalty to encourage fewer facilities."""
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        used = np.zeros(n_facilities, dtype=bool)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_val = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    # Effective cost: assignment cost + penalty for opening new facility
                    eff = assign_cost[i, f]
                    if not used[f]:
                        eff += penalty_weight * open_cost[f]
                    if eff < best_val:
                        best_val = eff
                        best_f = f
            if best_f == -1:
                for f in range(n_facilities):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        best_f = f
                        break
            if best_f == -1:
                best_f = 0
            load[best_f] += demands[i]
            used[best_f] = True
            result[i] = best_f
        return result
    
    def local_search(assignment, max_iter=5000):
        """Iterative local search with reassign and close moves."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        # Precompute current cost
        used = np.zeros(n_facilities, dtype=bool)
        for i in range(n_customers):
            used[assignment[i]] = True
        cur_cost = sum(open_cost[f] for f in range(n_facilities) if used[f]) + \
                   sum(assign_cost[i, assignment[i]] for i in range(n_customers))
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if time.time() > deadline:
                break
            
            # Try reassign moves: move customer i to facility f
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                cur_f = assignment[i]
                for f in range(n_facilities):
                    if f == cur_f:
                        continue
                    if load[f] + demands[i] > capacities[f] + 1e-9:
                        continue
                    
                    # Compute delta
                    # Remove i from cur_f
                    new_load_cur = load[cur_f] - demands[i]
                    # Add i to f
                    new_load_f = load[f] + demands[i]
                    
                    # Open cost change
                    delta_open = 0.0
                    if new_load_cur <= 1e-9:  # cur_f becomes empty
                        delta_open -= open_cost[cur_f]
                    if new_load_f > 1e-9 and load[f] <= 1e-9:  # f was empty, now used
                        delta_open += open_cost[f]
                    
                    # Assignment cost change
                    delta_assign = assign_cost[i, f] - assign_cost[i, cur_f]
                    
                    delta = delta_open + delta_assign
                    if delta < -1e-9:
                        # Accept
                        load[cur_f] -= demands[i]
                        load[f] += demands[i]
                        assignment[i] = f
                        cur_cost += delta
                        improved = True
                        break  # restart scanning after a move
        
        return assignment, cur_cost
    
    def local_search_with_swaps(assignment, max_iter=3000):
        """Local search with swap moves to handle capacity constraints."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        used = np.zeros(n_facilities, dtype=bool)
        for i in range(n_customers):
            used[assignment[i]] = True
        cur_cost = sum(open_cost[f] for f in range(n_facilities) if used[f]) + \
                   sum(assign_cost[i, assignment[i]] for i in range(n_customers))
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if time.time() > deadline:
                break
            
            # Try swap: customer i in f1, customer j in f2, swap them
            # Only if it improves and is feasible
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                f1 = assignment[i]
                for j in range(i + 1, n_customers):
                    f2 = assignment[j]
                    if f1 == f2:
                        continue
                    
                    # Check feasibility after swap
                    new_load_f1 = load[f1] - demands[i] + demands[j]
                    new_load_f2 = load[f2] - demands[j] + demands[i]
                    
                    if new_load_f1 > capacities[f1] + 1e-9:
                        continue
                    if new_load_f2 > capacities[f2] + 1e-9:
                        continue
                    
                    # Compute delta
                    delta_open = 0.0
                    # f1: was load[f1], now new_load_f1
                    if load[f1] > 1e-9 and new_load_f1 <= 1e-9:
                        delta_open -= open_cost[f1]
                    elif load[f1] <= 1e-9 and new_load_f1 > 1e-9:
                        delta_open += open_cost[f1]
                    # f2: was load[f2], now new_load_f2
                    if load[f2] > 1e-9 and new_load_f2 <= 1e-9:
                        delta_open -= open_cost[f2]
                    elif load[f2] <= 1e-9 and new_load_f2 > 1e-9:
                        delta_open += open_cost[f2]
                    
                    delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - \
                                   (assign_cost[i, f1] + assign_cost[j, f2])
                    
                    delta = delta_open + delta_assign
                    if delta < -1e-9:
                        # Accept swap
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        assignment[i] = f2
                        assignment[j] = f1
                        cur_cost += delta
                        improved = True
                        break  # restart after swap
                if improved:
                    break
        
        return assignment, cur_cost
    
    # Generate diverse initial solutions
    candidates = []
    
    # Strategy 1: greedy cheapest
    a1 = greedy_cheapest()
    if is_feasible(a1):
        a1, c1 = local_search(a1)
        candidates.append((c1, a1))
    
    # Strategy 2: greedy with various penalties
    for pw in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        if time.time() > deadline:
            break
        a = greedy_with_penalty(pw)
        if is_feasible(a):
            a, c = local_search(a)
            candidates.append((c, a))
    
    # Strategy 3: random feasible initialization
    for trial in range(5):
        if time.time() > deadline:
            break
        # Random assignment, then fix infeasibilities
        a = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.random.permutation(n_customers)
        feasible = True
        for i in order:
            # Try random facilities
            f_order = np.random.permutation(n_facilities)
            placed = False
            for f in f_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    a[i] = f
                    load[f] += demands[i]
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        if not feasible:
            continue
        a, c = local_search(a)
        candidates.append((c, a))
    
    # Select best candidate
    if not candidates:
        # Fallback
        a = greedy_cheapest()
        return a
    
    best_cost, best_assign = min(candidates, key=lambda x: x[0])
    
    # Further polish the best with swap-based local search
    if time.time() < deadline - 10:
        best_assign, best_cost = local_search_with_swaps(best_assign)
    
    # Final local search pass
    if time.time() < deadline - 5:
        best_assign, best_cost = local_search(best_assign)
    
    return best_assign.astype(int)
# EVOLVE-BLOCK-END
