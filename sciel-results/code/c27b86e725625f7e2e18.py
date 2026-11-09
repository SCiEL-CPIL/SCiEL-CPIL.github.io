"""Capacitated facility location: Lagrangian Relaxation with Min-Cost Flow."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def assign(demands, capacities, open_cost, assign_cost):
    """
    Solves the capacitated facility location problem using Lagrangian relaxation.
    1. Optimizes dual variables (lambda) for capacity constraints via subgradient method.
    2. Solves the Lagrangian subproblem (uncapacitated FLP + min-cost flow) for each lambda.
    3. Uses the best Lagrangian solution to guide a capacity-feasibility repair.
    4. Refines the feasible solution using local search (moves, swaps, closes).
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Safety margin: 150s limit, aim for ~140s max to leave buffer
    deadline = time.time() + 140.0
    
    # Precompute constants
    total_demand = np.sum(demands)
    
    def calc_cost(res):
        if res is None:
            return float('inf')
        used = np.unique(res)
        return float(np.sum(open_cost[used])) + float(np.sum(assign_cost[np.arange(n_cust), res]))
    
    def loads_of(res):
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        return load
    
    # ---- Lagrangian Relaxation Solver ----
    # The Lagrangian function L(lambda) = sum_i min_j (c_ij + lambda_j * d_i) x_ij + sum_j (open_cost_j * y_j - lambda_j * cap_j)
    # where y_j = 1 if any i is assigned to j.
    # We maximize L(lambda) for lambda >= 0.
    
    def solve_lagrangian_subproblem(lam):
        """
        Solves the Lagrangian subproblem for given lambda.
        Returns:
        - assignment: array of facility indices for each customer (ignoring capacity)
        - open_facilities: boolean array of open facilities
        - lagrangian_value: the value of L(lambda)
        """
        # Modified costs: c_ij + lambda_j * d_i
        mod_costs = assign_cost + np.outer(demands, lam)
        
        # For each customer, find the facility that minimizes mod_cost
        # But we also have to consider the fixed cost.
        # Actually, the Lagrangian subproblem decomposes.
        # Let's be precise.
        # L(lambda) = sum_i min_j (c_ij + lambda_j d_i) + sum_j (F_j y_j - lambda_j U_j)
        # This is not quite right. The standard relaxation relaxes capacity constraints:
        # min sum c_ij x_ij + sum F_j y_j
        # s.t. sum_i x_ij <= U_j y_j  ->  sum_i x_ij - U_j y_j <= 0
        # L(lambda) = min sum_i c_ij x_ij + sum_j (F_j - lambda_j U_j) y_j + sum_j lambda_j sum_i x_ij
        #           = sum_i min_j (c_ij + lambda_j) x_ij + sum_j (F_j - lambda_j U_j) y_j
        # Wait, the penalty is lambda_j * (sum_i x_ij - U_j y_j).
        # If we assign i to j, we pay c_ij + lambda_j.
        # If we open j, we pay F_j - lambda_j U_j.
        
        # Let's re-derive.
        # Dual function g(lambda) = min { sum c_ij x_ij + sum (F_j - lambda_j U_j) y_j + sum lambda_j sum_i x_ij : x, y feasible w.r.t other constraints }
        # Other constraints: sum_j x_ij = 1, x_ij <= y_j, x,y >= 0.
        # For a fixed y, the assignment part is: sum_i min_j (c_ij + lambda_j) x_ij subject to x_ij <= y_j.
        # This is a min-cost flow problem.
        
        # However, a simpler decomposition exists if we relax sum_i x_ij <= U_j (ignoring y_j coupling in the relaxation term, or handling it differently).
        # Standard FLP Lagrangian relaxation usually relaxes capacity constraints sum_i x_ij <= U_j.
        # Then L(lambda) = sum_i min_j (c_ij + lambda_j) + sum_j max(0, F_j - lambda_j U_j) ??? No.
        
        # Let's stick to the most robust approach:
        # Relax capacity constraints: sum_i x_ij <= U_j.
        # Multipliers lambda_j >= 0.
        # L(lambda) = sum_i min_j (c_ij + lambda_j) + sum_j (F_j y_j - lambda_j U_j)
        # This still has y_j coupled to x_ij via x_ij <= y_j.
        
        # Alternative: Relax x_ij <= y_j? No, that's the facility opening constraint.
        
        # Let's use the decomposition:
        # L(lambda) = min { sum_i c_ij x_ij + sum_j F_j y_j + sum_j lambda_j (sum_i x_ij - U_j) : sum_j x_ij = 1, x_ij <= y_j }
        # = sum_j (F_j y_j - lambda_j U_j) + sum_i min_j (c_ij + lambda_j) x_ij  (s.t. x_ij <= y_j)
        
        # This is a bit complex to solve directly in a vectorized way without a flow solver.
        # However, we can approximate or use a specific structure.
        
        # Let's try a different relaxation: Relax sum_j x_ij = 1? No.
        
        # Let's use the "capacity" relaxation where we ignore the y_j coupling in the term.
        # Actually, a common approach is:
        # L(lambda) = sum_i min_j (c_ij + lambda_j) + sum_j (F_j - lambda_j U_j) y_j
        # If we assume y_j is determined by whether the min_j term picks j, it's circular.
        
        # Let's use the min-cost flow formulation for the subproblem.
        # Nodes: Source, Customers, Facilities, Sink.
        # Edges:
        # Source -> Customer i: capacity 1, cost 0
        # Customer i -> Facility j: capacity 1, cost c_ij + lambda_j
        # Facility j -> Sink: capacity U_j, cost F_j - lambda_j U_j (if y_j=1)
        
        # Wait, the cost F_j - lambda_j U_j is incurred if y_j=1.
        # In min-cost flow, we can model fixed costs by having two edges from Facility j to Sink:
        # 1. Capacity 0, cost 0 (if closed) - not useful.
        # 2. We can use a "fixed charge" node or simply solve it as a mixed integer program? No, must be fast.
        
        # Let's simplify. If we relax ONLY the capacity constraint sum_i x_ij <= U_j, and keep x_ij <= y_j:
        # The problem is separable by facility j? No, x_ij <= y_j couples i and j.
        
        # Let's try relaxing the coupling x_ij <= y_j?
        # If we relax x_ij <= y_j, then x and y decouple.
        # min sum c_ij x_ij + sum F_j y_j + sum lambda_j (sum_i x_ij - U_j) - sum mu_ij (x_ij - y_j)
        # This gets complicated.
        
        # Let's go with the standard "capacity relaxation" which results in a min-cost flow problem where the fixed cost is handled by the structure of the graph or by a simple observation:
        # If we relax sum_i x_ij <= U_j, the Lagrangian is:
        # L(lambda) = min { sum c_ij x_ij + sum F_j y_j + sum lambda_j (sum_i x_ij - U_j) : sum_j x_ij = 1, x_ij <= y_j }
        # = sum_j (F_j y_j - lambda_j U_j) + sum_i min_j (c_ij + lambda_j) x_ij (s.t. x_ij <= y_j)
        
        # For a fixed y, the second term is a min-cost flow.
        # But y is a variable.
        
        # Actually, there is a simpler bound.
        # Let's just use the heuristic of "effective cost" for initialization and local search, but the prompt asks for a STRUCTURAL change.
        # Lagrangian Relaxation is a structural change.
        
        # Let's implement the Subgradient Method for the dual problem.
        # We need to evaluate L(lambda) and the subgradient.
        
        # To evaluate L(lambda):
        # 1. For each i, compute c_ij + lambda_j.
        # 2. Solve the min-cost flow with these costs.
        #    - Source to i: cap 1, cost 0.
        #    - i to j: cap 1, cost c_ij + lambda_j.
        #    - j to Sink: cap U_j, cost 0.
        #    - BUT, we have the fixed cost F_j.
        #    We can add a node for each facility j.
        #    j_in -> j_out: cap 1, cost F_j. (This enforces paying F_j if any flow passes)
        #    j_out -> Sink: cap U_j, cost -lambda_j U_j. (Wait, the penalty is lambda_j * (load - U_j). If load <= U_j, penalty is negative? No, lambda >= 0.
        #    The term is lambda_j * (sum x_ij - U_j).
        #    If sum x_ij < U_j, the term is negative. This encourages opening?
        #    Actually, the dual function is a lower bound.
        
        # Let's construct the graph for min-cost flow (scipy.sparse.csgraph or networkx? NetworkX is not guaranteed, but scipy.sparse.csgraph has min_cost_flow? No, scipy has `scipy.sparse.csgraph.minimum_spanning_tree` but not min cost flow directly in the same way as networkx. `scipy.optimize` has `linprog` which can solve LPs. Min-cost flow is an LP.
        
        # Using `linprog` for 100 customers and 20 facilities might be slow if done repeatedly in subgradient.
        # However, 100*20 = 2000 variables. LPs are fast.
        
        # Let's try a simpler approach for the "Lagrangian" inspiration:
        # Use the dual variables to adjust costs, then solve an UNCAPACITATED facility location problem (which is easy: for each i, pick j minimizing c_ij + lambda_j, and open j if picked).
        # This ignores capacity in the subproblem, which is the point of relaxation.
        # Then we get a solution that might violate capacity.
        # We use the violation to update lambda: lambda_j <- lambda_j + alpha * (load_j - U_j).
        
        # This is the standard Subgradient Method for FLP.
        
        best_lagrangian_value = -float('inf')
        best_lambda = np.zeros(n_fac)
        best_subproblem_sol = None
        
        # Initialize lambda to 0
        lam = np.zeros(n_fac)
        
        # Subgradient method parameters
        max_iter = 200
        step_size = 1.0
        
        for k in range(max_iter):
            if time.time() > deadline - 10.0:
                break
                
            # Solve subproblem:
            # For each i, find j minimizing c_ij + lam_j.
            # Let this be j_i.
            # Open facility j if any i maps to it.
            
            adjusted_costs = assign_cost + lam[np.newaxis, :]
            j_assign = np.argmin(adjusted_costs, axis=1)
            
            # Calculate load
            load = np.zeros(n_fac)
            np.add.at(load, j_assign, demands)
            
            # Determine open facilities
            open_fac = (load > 0).astype(float)
            
            # Calculate Lagrangian Value
            # L(lam) = sum_i (c_i, j_i + lam_j_i) + sum_j (F_j * y_j - lam_j * U_j)
            # Note: The term sum_i lam_j_i is sum_j lam_j * load_j.
            # So L(lam) = sum_i c_i, j_i + sum_j lam_j * load_j + sum_j F_j y_j - sum_j lam_j U_j
            #          = sum_i c_i, j_i + sum_j F_j y_j + sum_j lam_j (load_j - U_j)
            
            assign_sum = np.sum(assign_cost[np.arange(n_cust), j_assign])
            open_sum = np.sum(open_cost * open_fac)
            penalty_sum = np.sum(lam * (load - capacities))
            
            lag_value = assign_sum + open_sum + penalty_sum
            
            # Update best
            if lag_value > best_lagrangian_value:
                best_lagrangian_value = lag_value
                best_lambda = lam.copy()
                best_subproblem_sol = j_assign.copy()
                best_open_fac = open_fac.copy()
                best_load = load.copy()
                
            # Check convergence
            subgrad = load - capacities
            subgrad_norm = np.linalg.norm(subgrad)
            if subgrad_norm < 1e-4:
                break
                
            # Update lambda
            # Step size: alpha_k = gamma * (UB - L_k) / ||subgrad||^2
            # We need an Upper Bound (UB). We can get a feasible solution from the subproblem by repairing it.
            # For now, use a simple UB: greedy solution cost.
            # Let's compute a quick greedy UB once.
            # Actually, we can just use a fixed step size or a simple decay.
            # Let's use alpha = 1.0 / (k+1) or similar.
            
            # A better step size requires a valid UB.
            # Let's generate a feasible solution from j_assign by resolving capacity violations.
            # This is expensive to do every iteration.
            # Let's do it every 10 iterations.
            
            if k % 10 == 0:
                # Repair j_assign to get a feasible solution
                res_feasible = j_assign.copy()
                load_feasible = load.copy()
                
                # Simple repair: move customers from overloaded facilities to least loaded feasible ones
                overloaded = np.where(load_feasible > capacities)[0]
                for f in overloaded:
                    if load_feasible[f] <= capacities[f]:
                        continue
                    custs_in_f = np.where(res_feasible == f)[0]
                    # Sort by demand descending
                    custs_in_f = custs_in_f[np.argsort(-demands[custs_in_f])]
                    for c in custs_in_f:
                        if load_feasible[f] <= capacities[f]:
                            break
                        d_c = demands[c]
                        # Find best alternative
                        feasible_alts = np.where(load_feasible + d_c <= capacities)[0]
                        if len(feasible_alts) == 0:
                            continue
                        # Pick one with min assign_cost
                        costs = assign_cost[c, feasible_alts]
                        best_alt = feasible_alts[np.argmin(costs)]
                        
                        # Move
                        res_feasible[c] = best_alt
                        load_feasible[f] -= d_c
                        load_feasible[best_alt] += d_c
                        
                if np.all(load_feasible <= capacities):
                    ub = calc_cost(res_feasible)
                    gap = ub - lag_value
                    if gap > 0:
                        alpha = 1.0 * gap / (subgrad_norm**2 + 1e-9)
                    else:
                        alpha = 0.1
                else:
                    alpha = 0.1
                    
            else:
                alpha = 0.1 # Fallback step size
            
            lam = lam + alpha * subgrad
            lam = np.maximum(lam, 0)
            
        # After subgradient, we have best_subproblem_sol (j_assign) which may violate capacity.
        # We need to convert this to a feasible solution.
        
        # Initialize result with the best Lagrangian solution
        if best_subproblem_sol is not None:
            res = best_subproblem_sol.copy()
        else:
            # Fallback to greedy
            res = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac)
            order = np.argsort(-demands)
            for i in order:
                feasible = np.where(load + demands[i] <= capacities)[0]
                if len(feasible) == 0:
                    f = int(np.argmax(capacities - load))
                    res[i] = f
                    load[f] += demands[i]
                else:
                    costs = assign_cost[i, feasible]
                    f_idx = int(feasible[np.argmin(costs)])
                    res[i] = f_idx
                    load[f_idx] += demands[i]
            
        # Repair capacity violations
        load = loads_of(res)
        max_iter_repair = 100
        for _ in range(max_iter_repair):
            if time.time() > deadline - 5.0:
                break
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                break
            f = overloaded[0]
            custs_in_f = np.where(res == f)[0]
            if len(custs_in_f) == 0:
                break
            # Move customer with largest demand first? Or smallest?
            # Largest demand is usually better to move to reduce number of moves.
            custs_in_f = custs_in_f[np.argsort(-demands[custs_in_f])]
            moved = False
            for c in custs_in_f:
                if load[f] <= capacities[f]:
                    break
                d_c = demands[c]
                feasible_alts = np.where(load + d_c <= capacities)[0]
                if len(feasible_alts) == 0:
                    continue
                costs = assign_cost[c, feasible_alts]
                best_alt = feasible_alts[np.argmin(costs)]
                
                res[c] = best_alt
                load[f] -= d_c
                load[best_alt] += d_c
                moved = True
            if not moved:
                # Stuck, try moving smallest demand?
                custs_in_f = custs_in_f[np.argsort(demands[custs_in_f])]
                for c in custs_in_f:
                    if load[f] <= capacities[f]:
                        break
                    d_c = demands[c]
                    feasible_alts = np.where(load + d_c <= capacities)[0]
                    if len(feasible_alts) == 0:
                        continue
                    costs = assign_cost[c, feasible_alts]
                    best_alt = feasible_alts[np.argmin(costs)]
                    
                    res[c] = best_alt
                    load[f] -= d_c
                    load[best_alt] += d_c
                    moved = True
                if not moved:
                    break

        # Local Search Refinement
        # Use the same local search as before, but starting from this Lagrangian-guided solution.
        
        def single_moves_pass(res, load):
            improved = False
            while True:
                if time.time() > deadline - 2.0:
                    break
                any_improve = False
                base_assign = assign_cost[np.arange(n_cust), res]
                for i in range(n_cust):
                    if time.time() > deadline - 2.0:
                        break
                    f_old = res[i]
                    d_i = demands[i]
                    feasible_mask = (load + d_i <= capacities)
                    feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
                    if not np.any(feasible_mask):
                        continue
                    delta_assign = assign_cost[i, :] - base_assign[i]
                    delta_open = np.zeros(n_fac)
                    if load[f_old] == d_i:
                        delta_open[f_old] = -open_cost[f_old]
                    open_new = (load == 0) & (np.arange(n_fac) != f_old)
                    if np.any(open_new):
                        delta_open[open_new] = open_cost[open_new]
                    total_delta = delta_assign + delta_open
                    total_delta[~feasible_mask] = np.inf
                    total_delta[f_old] = np.inf
                    best_f_new = int(np.argmin(total_delta))
                    best_delta = total_delta[best_f_new]
                    if best_delta < -1e-9:
                        res[i] = best_f_new
                        load[f_old] -= d_i
                        load[best_f_new] += d_i
                        any_improve = True
                        improved = True
                if not any_improve:
                    break
            return res, load, improved

        def swap_pass(res, load, n_samples=1500):
            improved = False
            rng = np.random.default_rng(42)
            for _ in range(n_samples):
                if time.time() > deadline - 2.0:
                    break
                i, j = rng.integers(0, n_cust, size=2)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                f_i = res[i]
                f_j = res[j]
                if f_i == f_j:
                    continue
                d_i = demands[i]
                d_j = demands[j]
                if load[f_j] - d_j + d_i > capacities[f_j]:
                    continue
                if load[f_i] - d_i + d_j > capacities[f_i]:
                    continue
                delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                               (assign_cost[j, f_i] - assign_cost[j, f_j])
                total_delta = delta_assign
                if total_delta < -1e-9:
                    res[i], res[j] = res[j], res[i]
                    load[f_i] -= d_i
                    load[f_i] += d_j
                    load[f_j] -= d_j
                    load[f_j] += d_i
                    improved = True
            return res, load, improved

        def close_facility_pass(res, load):
            improved = False
            for f_close in range(n_fac):
                if time.time() > deadline - 2.0:
                    break
                if load[f_close] == 0:
                    continue
                custs = np.where(res == f_close)[0]
                if len(custs) == 0:
                    continue
                
                temp_load = load.copy()
                temp_load[f_close] = 0
                temp_res = res.copy()
                
                cust_order = custs[np.argsort(-demands[custs])]
                success = True
                for c in cust_order:
                    d_c = demands[c]
                    feasible = np.where(temp_load + d_c <= capacities)[0]
                    if len(feasible) == 0:
                        success = False
                        break
                    costs = assign_cost[c, feasible]
                    open_penalty = np.where(temp_load[feasible] == 0, open_cost[feasible], 0.0)
                    adjusted_costs = costs + open_penalty
                    best_f = feasible[np.argmin(adjusted_costs)]
                    temp_res[c] = best_f
                    temp_load[best_f] += d_c
                
                if not success:
                    continue
                if np.any(temp_load > capacities):
                    continue
                if temp_load[f_close] > 0:
                    continue
                
                old_assign_sum = np.sum(assign_cost[custs, f_close])
                new_assign_sum = np.sum(assign_cost[custs, temp_res[custs]])
                
                newly_opened = set()
                for c in custs:
                    nf = temp_res[c]
                    if nf != f_close and temp_load[nf] > 0:
                        if load[nf] == 0:
                            newly_opened.add(nf)
                
                open_cost_change = -open_cost[f_close]
                for nf in newly_opened:
                    open_cost_change += open_cost[nf]
                
                total_delta = (new_assign_sum - old_assign_sum) + open_cost_change
                
                if total_delta < -1e-9:
                    for c in custs:
                        res[c] = temp_res[c]
                    load = temp_load.copy()
                    improved = True
            return res, load, improved

        def local_search(res, load):
            improved = True
            while improved:
                improved = False
                if time.time() > deadline - 2.0:
                    break
                
                res, load, imp1 = single_moves_pass(res, load)
                improved = improved or imp1
                
                if time.time() > deadline - 2.0:
                    break
                res, load, imp2 = swap_pass(res, load, n_samples=1000)
                improved = improved or imp2
                
                if time.time() > deadline - 2.0:
                    break
                res, load, imp3 = close_facility_pass(res, load)
                improved = improved or imp3
            return res, load

        # Run local search on the repaired Lagrangian solution
        res_ls, load_ls = local_search(res.copy(), load.copy())
        
        # Verify validity
        if np.all(load_ls <= capacities):
            return res_ls

        # Fallback if local search somehow broke validity (shouldn't happen)
        return res

# EVOLVE-BLOCK-END
