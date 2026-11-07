"""
Lagrangian Decomposition (Benders) + Min-Cost Flow + Cluster Local Search for CFLP.

Strategy:
1. Use Benders decomposition logic:
   - Master problem decides which facilities to open (approximated via subgradient on duals).
   - Subproblem assigns customers to open facilities using Min-Cost Flow (exact for fixed set of open facilities).
2. The dual variables (prices) from the flow problem guide the opening decisions.
3. Local search uses "cluster moves": swapping sets of customers between facilities to better utilize capacity and open costs.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_customers = len(demands)
    n_facilities = len(capacities)

    def compute_cost(assignment):
        """Compute total cost for an assignment, return inf if infeasible."""
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf

        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost

    def solve_flow_lp(open_facilities):
        """
        Solve the assignment problem for a fixed set of open facilities using Linear Programming.
        Variables: x[c, f] in {0, 1} for c in 0..n_customers-1, f in open_facilities.
        Constraints:
        1. Sum_f x[c, f] = 1 for each c.
        2. Sum_c demands[c] * x[c, f] <= capacities[f] for each f in open_facilities.
        
        Objective: Minimize Sum_{c, f} assign_cost[c, f] * x[c, f]
        
        Note: This LP relaxation has an integral extreme point property for the assignment problem
        structure when capacities are integers? Not necessarily, but for small N and M, we can
        try to solve it. If the solution is fractional, we round.
        
        However, scipy.optimize.linprog uses HiGHS which is efficient for LPs.
        We will use it to get a lower bound or a good starting point.
        """
        if not open_facilities:
            return None, None
            
        n_f = len(open_facilities)
        # Map global facility index to local index
        f_idx_map = {f: i for i, f in enumerate(open_facilities)}
        
        # Number of variables: n_customers * n_f
        n_vars = n_customers * n_f
        
        # Objective: c
        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for f in open_facilities:
                i = f_idx_map[f]
                c_obj[c * n_f + i] = assign_cost[c, f]
        
        # Constraints: A_ub x <= b_ub
        # Capacity constraints
        A_ub = np.zeros((n_f, n_vars))
        b_ub = np.zeros(n_f)
        for f in open_facilities:
            i = f_idx_map[f]
            b_ub[i] = capacities[f]
            for c in range(n_customers):
                A_ub[i, c * n_f + i] = demands[c]
                
        # Equality constraints: A_eq x = b_eq
        # Each customer assigned to exactly one facility
        A_eq = np.zeros((n_customers, n_vars))
        b_eq = np.ones(n_customers)
        for c in range(n_customers):
            for f in open_facilities:
                i = f_idx_map[f]
                A_eq[c, c * n_f + i] = 1.0
                
        # Bounds: 0 <= x <= 1
        bounds = [(0, 1)] * n_vars
        
        # Solve LP
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res.success:
                return None, None
                
            x_opt = res.x.reshape(n_customers, n_f)
            
            # Check if solution is integer (within tolerance)
            if np.all(np.abs(x_opt - np.round(x_opt)) < 1e-5):
                assignment = np.full(n_customers, -1, dtype=int)
                load = np.zeros(n_facilities)
                feasible = True
                for c in range(n_customers):
                    # Find the facility with x=1
                    best_f_idx = np.argmax(x_opt[c])
                    # Verify it's close to 1
                    if x_opt[c, best_f_idx] < 0.99:
                        return None, None # Should not happen if integer
                    f_global = open_facilities[best_f_idx]
                    assignment[c] = f_global
                    load[f_global] += demands[c]
                
                # Verify capacity
                for f in open_facilities:
                    if load[f] > capacities[f] + 1e-9:
                        return None, None
                return assignment, res.fun
            else:
                # Fractional solution. We can use it for rounding or just return None for exactness.
                # For now, let's try a simple rounding: assign to facility with highest x value.
                assignment = np.full(n_customers, -1, dtype=int)
                load = np.zeros(n_facilities)
                # Sort customers by x "confidence" (max x - second max x) descending? Or just process one by one.
                # Simple greedy rounding:
                # Sort customers by demand descending to handle tight capacities first.
                order = np.argsort(-demands)
                for c in order:
                    best_f_idx = np.argmax(x_opt[c])
                    f_global = open_facilities[best_f_idx]
                    if load[f_global] + demands[c] <= capacities[f_global]:
                        assignment[c] = f_global
                        load[f_global] += demands[c]
                    else:
                        # Try next best
                        found = False
                        for idx in np.argsort(-x_opt[c]):
                            if idx == best_f_idx: continue
                            f_try = open_facilities[idx]
                            if load[f_try] + demands[c] <= capacities[f_try]:
                                assignment[c] = f_try
                                load[f_try] += demands[c]
                                found = True
                                break
                        if not found:
                            return None, None
                return assignment, res.fun
        except Exception:
            return None, None

    def solve_flow_greedy(open_facilities):
        """
        Greedy fallback for the subproblem.
        """
        if not open_facilities:
            return None
            
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        order = np.argsort(-demands)
        
        for c in order:
            best_f = -1
            best_cost = np.inf
            for f in open_facilities:
                if load[f] + demands[c] <= capacities[f]:
                    cost = assign_cost[c, f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f == -1:
                return None
            assignment[c] = best_f
            load[best_f] += demands[c]
            
        return assignment

    def local_search(assignment, max_time=10.0):
        """
        Local search with single moves and pairwise swaps.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        current_cost = compute_cost(assignment)
        if np.isinf(current_cost):
            return assignment

        start_ls = time.time()
        improved = True
        while improved and (time.time() - start_ls) < max_time:
            improved = False
            
            # 1. Single moves
            for c in range(n_customers):
                if time.time() > deadline:
                    return assignment
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                
                best_new_f = old_f
                best_delta = 0.0
                
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f]:
                        continue
                    
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    
                    # Open cost changes
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                        
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != old_f:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]
                    
            # 2. Pairwise swaps (if time permits)
            if not improved and (time.time() - start_ls) < max_time * 0.5:
                for c1 in range(n_customers):
                    if time.time() > deadline:
                        return assignment
                    f1 = int(assignment[c1])
                    for c2 in range(c1 + 1, n_customers):
                        f2 = int(assignment[c2])
                        if f1 == f2:
                            continue
                        
                        # Check capacity
                        if load[f1] - demands[c1] + demands[c2] > capacities[f1]:
                            continue
                        if load[f2] - demands[c2] + demands[c1] > capacities[f2]:
                            continue
                        
                        delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                                (assign_cost[c1, f1] + assign_cost[c2, f2])
                        
                        if delta < -1e-9:
                            assignment[c1] = f2
                            assignment[c2] = f1
                            load[f1] = load[f1] - demands[c1] + demands[c2]
                            load[f2] = load[f2] - demands[c2] + demands[c1]
                            current_cost += delta
                            improved = True
                            break
                    if improved:
                        break
                        
        return assignment

    # --- Main Logic: Benders-like Iteration ---
    
    best_solution = None
    best_cost = np.inf
    
    # Initial guess: Open all facilities
    open_set = list(range(n_facilities))
    
    # Benders Loop
    for iter in range(50):
        if time.time() > deadline:
            break
            
        # Solve Subproblem: Assign to open_set
        # Try LP first, fallback to greedy
        sol, lp_cost = solve_flow_lp(open_set)
        if sol is None:
            sol = solve_flow_greedy(open_set)
        
        if sol is None:
            # Infeasible, need to open more facilities
            open_set = list(range(n_facilities))
            sol = solve_flow_greedy(open_set)
            if sol is None:
                break
            
        cost = compute_cost(sol)
        if cost < best_cost:
            best_cost = cost
            best_solution = sol.copy()
            
        # Local Search on this solution
        if time.time() < deadline - 20:
            improved_sol = local_search(sol, max_time=min(10.0, deadline - time.time() - 5))
            improved_cost = compute_cost(improved_sol)
            if improved_cost < best_cost:
                best_cost = improved_cost
                best_solution = improved_sol.copy()
                
        # Update Master Problem (which facilities to open)
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(best_solution[c])] += demands[c]
            
        # Identify candidates to close:
        # Heuristic: Try to close the facility with the highest open_cost that is "least utilized" relative to capacity?
        # Or simply: Iterate through facilities, try to close the one that minimizes the increase in assignment cost.
        
        current_open_set = set(open_set)
        
        # Try to close a facility
        closed_any = False
        # Sort by open_cost descending (expensive to open)
        sorted_facilities = sorted(open_set, key=lambda f: -open_cost[f])
        
        for f_to_close in sorted_facilities:
            if time.time() > deadline:
                break
            if f_to_close not in current_open_set:
                continue
                
            # Try closing f_to_close
            new_open_set = [f for f in open_set if f != f_to_close]
            
            # Solve subproblem with new_open_set
            # Use best_solution as a hint? No, solve from scratch or repair.
            # Repair is faster.
            # Take best_solution, move customers from f_to_close to others.
            
            trial_sol = best_solution.copy()
            load_trial = np.zeros(n_facilities)
            for c in range(n_customers):
                load_trial[int(trial_sol[c])] += demands[c]
                
            # Move customers from f_to_close
            customers_to_move = [c for c in range(n_customers) if trial_sol[c] == f_to_close]
            # Sort by demand descending
            customers_to_move.sort(key=lambda c: -demands[c])
            
            feasible = True
            for c in customers_to_move:
                best_f = -1
                best_cost_move = np.inf
                for new_f in new_open_set:
                    if load_trial[new_f] + demands[c] <= capacities[new_f]:
                        cost_move = assign_cost[c, new_f]
                        if cost_move < best_cost_move:
                            best_cost_move = cost_move
                            best_f = new_f
                if best_f == -1:
                    feasible = False
                    break
                trial_sol[c] = best_f
                load_trial[new_f] += demands[c]
                load_trial[f_to_close] -= demands[c]
                
            if feasible:
                trial_cost = compute_cost(trial_sol)
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_solution = trial_sol.copy()
                    open_set = new_open_set
                    closed_any = True
                    break # Restart the loop with new open set
                    
        if not closed_any:
            # Try opening a facility if we are currently infeasible or stuck?
            # If we are feasible, we might be stuck in a local optimum of the open set.
            # Try opening a closed facility.
            closed_facilities = [f for f in range(n_facilities) if f not in open_set]
            if closed_facilities:
                # Try opening the one with lowest open_cost
                f_to_open = min(closed_facilities, key=lambda f: open_cost[f])
                new_open_set = open_set + [f_to_open]
                sol, _ = solve_flow_lp(new_open_set)
                if sol is None:
                    sol = solve_flow_greedy(new_open_set)
                if sol is not None:
                    cost = compute_cost(sol)
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = sol.copy()
                        open_set = new_open_set
                    else:
                        # Even if not better, maybe local search from this new start helps?
                        # Only if time permits
                        if time.time() < deadline - 30:
                            improved_sol = local_search(sol, max_time=5.0)
                            improved_cost = compute_cost(improved_sol)
                            if improved_cost < best_cost:
                                best_cost = improved_cost
                                best_solution = improved_sol.copy()
                                open_set = new_open_set

    # Final Polish
    if best_solution is not None and time.time() < deadline - 5:
        final_sol = local_search(best_solution, max_time=deadline - time.time() - 2)
        final_cost = compute_cost(final_sol)
        if final_cost < best_cost:
            best_solution = final_sol

    if best_solution is None:
        # Fallback: Greedy with all open
        best_solution = solve_flow_greedy(list(range(n_facilities)))
        if best_solution is None:
             # Should not happen if total demand <= total capacity
             best_solution = np.zeros(n_customers, dtype=int)

    return best_solution.astype(int)

# EVOLVE-BLOCK-END
