import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    N = len(demands)
    M = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # 1. Greedy Initialization
    # Sort customers by demand descending
    order = np.argsort(-demands)
    current_solution = np.zeros(N, dtype=int)
    load = np.zeros(M)
    
    for i in order:
        feasible_mask = load + demands[i] <= capacities
        if not np.any(feasible_mask):
            # Fallback: pick facility with most remaining capacity
            idx = np.argmax(capacities - load)
            current_solution[i] = idx
            load[idx] += demands[i]
            continue
            
        costs = assign_cost[i].copy()
        costs[~feasible_mask] = np.inf
        f = np.argmin(costs)
        current_solution[i] = f
        load[f] += demands[i]
        
    def calc_cost(sol, loads):
        open_mask = loads > 0
        cost = np.sum(open_cost[open_mask])
        assign_sum = np.sum([assign_cost[i, sol[i]] for i in range(N)])
        return cost + assign_sum

    current_cost = calc_cost(current_solution, load)
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    # 2. Try LP Relaxation (Min-Cost Flow style with Big-M)
    # Variables: x[i, j] = flow from customer i to facility j
    # y[j] = 1 if facility j is open
    # Minimize sum(c[i,j]*x[i,j]) + sum(open_cost[j]*y[j])
    # Subject to:
    # sum_j x[i,j] = demands[i] for all i
    # sum_i x[i,j] <= capacities[j] * y[j] for all j
    # x[i,j] <= demands[i] * y[j] for all i, j (if y[j]=0 then x[i,j]=0)
    # x[i,j] >= 0, y[j] in {0,1}
    
    # Since scipy linprog does LP (not MIP), we can't enforce y[j] binary.
    # However, we can solve the LP relaxation and round.
    # A better approach for LP:
    # Let x[i,j] be the amount assigned.
    # If we just minimize sum(c[i,j]*x[i,j]) subject to capacity constraints,
    # we get a fractional assignment which doesn't account for open costs correctly
    # unless we model it properly.
    
    # Alternative: Use the LP to find a good fractional solution, then repair/round.
    # Or, simply use the LP to guide the local search or as an initial solution.
    
    # Let's try a specific LP formulation:
    # Minimize sum_{i,j} c[i,j] x[i,j] + sum_j open_cost[j] y_j
    # s.t.
    # sum_j x[i,j] = demands[i]
    # sum_i x[i,j] <= capacities[j]
    # x[i,j] <= demands[i] * y_j
    # x[i,j] >= 0
    # 0 <= y_j <= 1
    
    # This is a standard LP.
    # Number of variables: N*M + M = 100*20 + 20 = 2020
    # This is small enough for HiGHS.
    
    def solve_lp():
        n_vars = N * M + M
        # c vector
        c = np.zeros(n_vars)
        for i in range(N):
            for j in range(M):
                c[i * M + j] = assign_cost[i, j]
        for j in range(M):
            c[N * M + j] = open_cost[j]
        
        # A_ub, b_ub:
        # Constraints:
        # 1. x[i,j] - demands[i] * y_j <= 0  for all i, j
        # 2. -x[i,j] <= 0 (handled by bounds)
        
        # Actually, simpler bounds:
        # x[i,j] in [0, demands[i]]
        # y_j in [0, 1]
        
        # But we need x[i,j] <= demands[i] * y_j.
        # This is a bilinear constraint. Linearization:
        # x[i,j] <= demands[i] * y_j
        # x[i,j] <= demands[i]
        # x[i,j] >= 0
        # x[i,j] >= demands[i] * (y_j - 1)
        
        # Let's stick to the simpler capacity constraint and see if open cost is handled.
        # If we just do min sum(c*x) s.t. capacities, we ignore open cost.
        # To include open cost in LP, we need the y variables.
        
        # Let's build the full LP.
        # Variables: x[0..N*M-1], y[0..M-1]
        
        A_ub = []
        b_ub = []
        
        # Constraints: x[i,j] - demands[i]*y[j] <= 0
        for i in range(N):
            for j in range(M):
                row = np.zeros(n_vars)
                row[i * M + j] = 1.0
                row[N * M + j] = -demands[i]
                A_ub.append(row)
                b_ub.append(0.0)
                
        # Constraints: x[i,j] - demands[i]*(1-y[j]) <= 0  => x[i,j] + demands[i]*y[j] <= demands[i]
        # This ensures if y[j]=0, x[i,j]<=0. If y[j]=1, x[i,j]<=demands[i].
        for i in range(N):
            for j in range(M):
                row = np.zeros(n_vars)
                row[i * M + j] = 1.0
                row[N * M + j] = demands[i]
                A_ub.append(row)
                b_ub.append(demands[i])
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # A_eq, b_eq:
        # sum_j x[i,j] = demands[i]
        A_eq = []
        b_eq = []
        for i in range(N):
            row = np.zeros(n_vars)
            for j in range(M):
                row[i * M + j] = 1.0
            A_eq.append(row)
            b_eq.append(demands[i])
            
        # sum_i x[i,j] <= capacities[j]
        for j in range(M):
            row = np.zeros(n_vars)
            for i in range(N):
                row[i * M + j] = 1.0
            A_eq.append(row) # Wait, this is <=, so it goes to A_ub
            
        # Let's restructure.
        # A_ub includes:
        # 1. Capacity: sum_i x[i,j] <= capacities[j]
        # 2. Linking: x[i,j] - demands[i]*y[j] <= 0
        # 3. Linking: x[i,j] + demands[i]*y[j] <= demands[i]
        
        A_ub = []
        b_ub = []
        
        # Capacities
        for j in range(M):
            row = np.zeros(n_vars)
            for i in range(N):
                row[i * M + j] = 1.0
            A_ub.append(row)
            b_ub.append(capacities[j])
            
        # Linking 1
        for i in range(N):
            for j in range(M):
                row = np.zeros(n_vars)
                row[i * M + j] = 1.0
                row[N * M + j] = -demands[i]
                A_ub.append(row)
                b_ub.append(0.0)
                
        # Linking 2
        for i in range(N):
            for j in range(M):
                row = np.zeros(n_vars)
                row[i * M + j] = 1.0
                row[N * M + j] = demands[i]
                A_ub.append(row)
                b_ub.append(demands[i])
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # A_eq
        A_eq = []
        b_eq = []
        for i in range(N):
            row = np.zeros(n_vars)
            for j in range(M):
                row[i * M + j] = 1.0
            A_eq.append(row)
            b_eq.append(demands[i])
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        bounds = []
        for i in range(N):
            for j in range(M):
                bounds.append((0, demands[i]))
        for j in range(M):
            bounds.append((0, 1))
            
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.status != 0:
            return None
            
        x = res.x[:N*M].reshape(N, M)
        y = res.x[N*M:]
        
        # Round x to integer solution
        # We need to assign each customer i to exactly one facility j such that sum(x[i,j]) = demands[i]
        # Since x is fractional, we need to repair.
        # Simple repair: For each customer, pick the facility with max x[i,j].
        # This might violate capacities.
        
        sol = np.zeros(N, dtype=int)
        loads = np.zeros(M)
        
        # Sort customers by demand descending for repair
        order = np.argsort(-demands)
        
        for i in order:
            # Find best facility with capacity
            feasible = loads + demands[i] <= capacities
            if not np.any(feasible):
                # Try to find any feasible, or fallback
                feasible = np.ones(M, dtype=bool)
                
            # Score based on LP solution + cost
            # Prefer high LP flow, then low cost
            scores = np.zeros(M)
            for j in range(M):
                if feasible[j]:
                    scores[j] = x[i, j] * 1000 - assign_cost[i, j] # Weight LP flow heavily
                else:
                    scores[j] = -np.inf
                    
            best_j = np.argmax(scores)
            if scores[best_j] == -np.inf:
                # Fallback to cheapest feasible
                costs = assign_cost[i].copy()
                costs[~feasible] = np.inf
                best_j = np.argmin(costs)
                
            sol[i] = best_j
            loads[best_j] += demands[i]
            
        return sol, loads

    # Try LP if time permits
    if time.time() - start_time < deadline - 10:
        lp_res = solve_lp()
        if lp_res is not None:
            sol_lp, loads_lp = lp_res
            cost_lp = calc_cost(sol_lp, loads_lp)
            if cost_lp < best_cost:
                best_cost = cost_lp
                best_solution = sol_lp.copy()
                current_solution = sol_lp.copy()
                load = loads_lp.copy()
                current_cost = cost_lp

    # 3. Local Search
    # Use the best solution found so far (either greedy or LP-repaired)
    
    while time.time() - start_time < deadline:
        improved = False
        
        # --- Single Move Phase (First Improvement to speed up?) ---
        # Best improvement is safer for quality but slower.
        # With N=100, M=20, N*M = 2000 checks per pass. Very fast.
        best_delta = 0
        best_move = None
        
        for i in range(N):
            curr_f = current_solution[i]
            for k in range(M):
                if k == curr_f:
                    continue
                if load[k] + demands[i] > capacities[k]:
                    continue
                
                d_assign = assign_cost[i, k] - assign_cost[i, curr_f]
                
                d_open = 0
                new_load_curr = load[curr_f] - demands[i]
                if new_load_curr == 0:
                    d_open -= open_cost[curr_f]
                
                if load[k] == 0:
                    d_open += open_cost[k]
                    
                total_delta = d_assign + d_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_move = (i, k)
                    
        if best_move is not None:
            i, k = best_move
            curr_f = current_solution[i]
            load[curr_f] -= demands[i]
            load[k] += demands[i]
            current_solution[i] = k
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
                
        # --- Swap Phase ---
        best_delta = 0
        best_swap = None
        for i in range(N):
            f_i = current_solution[i]
            for m in range(i + 1, N):
                f_m = current_solution[m]
                if f_i == f_m:
                    continue
                
                new_load_fi = load[f_i] - demands[i] + demands[m]
                new_load_fm = load[f_m] - demands[m] + demands[i]
                
                if new_load_fi > capacities[f_i] or new_load_fm > capacities[f_m]:
                    continue
                
                d_assign = (assign_cost[i, f_m] + assign_cost[m, f_i]) - (assign_cost[i, f_i] + assign_cost[m, f_m])
                
                d_open = 0
                if new_load_fi == 0:
                    d_open -= open_cost[f_i]
                if new_load_fm == 0:
                    d_open -= open_cost[f_m]
                
                total_delta = d_assign + d_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_swap = (i, m)
                    
        if best_swap is not None:
            i, m = best_swap
            f_i = current_solution[i]
            f_m = current_solution[m]
            
            load[f_i] -= demands[i]
            load[f_i] += demands[m]
            load[f_m] -= demands[m]
            load[f_m] += demands[i]
            
            current_solution[i] = f_m
            current_solution[m] = f_i
            
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
                
        if not improved:
            break
            
    return best_solution.astype(int)
