"""Capacitated facility location: Exact MIP solver via SciPy with heuristic fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities.
    Attempts to solve exactly using scipy.optimize.milp.
    Falls back to heuristic local search if MIP fails or is unavailable.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    n_cust = len(demands)
    n_fac = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    # --- Attempt Exact Solution using MIP ---
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Variables: x_ij (binary) for i in 0..n_cust-1, j in 0..n_fac-1
        #            y_j (binary) for j in 0..n_fac-1
        # Total variables: n_cust * n_fac + n_fac
        
        num_x = n_cust * n_fac
        num_y = n_fac
        num_vars = num_x + num_y
        
        # Objective: Min sum(assign_cost[i,j] * x_ij) + sum(open_cost[j] * y_j)
        c = np.zeros(num_vars)
        c[:num_x] = assign_cost.flatten()
        c[num_x:] = open_cost
        
        # Constraints:
        # 1. Sum_j x_ij = 1 for all i
        #    Matrix A1: n_cust x num_x, sparse
        #    Rows: i, Cols: i*n_fac + j
        A1 = sp.lil_matrix((n_cust, num_x))
        for i in range(n_cust):
            for j in range(n_fac):
                A1[i, i * n_fac + j] = 1
        
        # 2. Sum_i demands[i] * x_ij <= capacities[j] * y_j for all j
        #    Sum_i demands[i] * x_ij - capacities[j] * y_j <= 0
        #    Matrix A2: n_fac x num_vars
        A2 = sp.lil_matrix((n_fac, num_vars))
        for j in range(n_fac):
            for i in range(n_cust):
                A2[j, i * n_fac + j] = demands[i]
            A2[j, num_x + j] = -capacities[j]
        
        # 3. x_ij <= y_j for all i, j
        #    x_ij - y_j <= 0
        #    Matrix A3: n_cust * n_fac x num_vars
        A3 = sp.lil_matrix((num_x, num_vars))
        for i in range(n_cust):
            for j in range(n_fac):
                idx = i * n_fac + j
                A3[idx, idx] = 1
                A3[idx, num_x + j] = -1
        
        # Combine constraints
        # A1 part: equality
        # A2 part: inequality (<= 0)
        # A3 part: inequality (<= 0)
        
        # We can combine A2 and A3 into one inequality matrix
        A_ineq = sp.vstack([A2, A3], format='csr')
        b_ineq = np.zeros(A_ineq.shape[0])
        
        # A1 is equality
        A_eq = A1.tocsr()
        b_eq = np.ones(n_cust)
        
        # Bounds: 0 <= x,y <= 1
        bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
        
        # Integrality: all binary
        integrality = np.ones(num_vars)
        
        # Solve
        res = milp(c=c, constraints=[LinearConstraint(A_ineq, -np.inf, b_ineq), 
                                     LinearConstraint(A_eq, b_eq, b_eq)], 
                   integrality=integrality, bounds=bounds)
        
        if res.success:
            x_flat = res.x[:num_x]
            # Round to nearest integer (should be 0 or 1)
            x = np.round(x_flat).reshape(n_cust, n_fac)
            
            # Extract assignment
            result = np.zeros(n_cust, dtype=int)
            for i in range(n_cust):
                # Find the facility with x=1
                facs = np.where(x[i] > 0.5)[0]
                if len(facs) > 0:
                    result[i] = facs[0]
                else:
                    # Fallback if solver failed to assign (shouldn't happen)
                    result[i] = 0
            
            # Verify validity
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[result[i]] += demands[i]
                
            if np.all(load <= capacities + 1e-6):
                return result
            else:
                # Invalid result from MIP, fall back
                pass
        else:
            # MIP failed
            pass
            
    except Exception:
        # Fallback to heuristic
        pass

    # --- Fallback Heuristic ---
    def _total_cost(result):
        load = np.zeros(n_fac)
        for c in range(n_cust):
            load[result[c]] += demands[c]
        used = load > 0
        open_sum = open_cost[used].sum()
        assign_sum = np.sum(assign_cost[np.arange(n_cust), result])
        return open_sum + assign_sum

    def _greedy_seed(strategy='eff'):
        """Generate initial assignment using effective-cost aware greedy."""
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac, dtype=float)

        # Precompute amortized open cost per unit capacity
        amortized = open_cost / np.maximum(capacities, 1.0)

        if strategy == 'eff':
            # Effective cost: assign_cost + demand * amortized_open_cost
            eff_cost = assign_cost + np.outer(demands, amortized)
            # Sort customers by largest effective total cost first (hardest to place)
            total_eff = eff_cost.sum(axis=1)
            order = np.lexsort((-demands, total_eff))
            cost_matrix = eff_cost
        elif strategy == 'eff_desc':
            amortized = open_cost / np.maximum(capacities, 1.0)
            eff_cost = assign_cost + np.outer(demands, amortized)
            order = np.argsort(-demands)
            cost_matrix = eff_cost
        elif strategy == 'raw_desc':
            order = np.argsort(-demands)
            cost_matrix = assign_cost
        elif strategy == 'raw_asc':
            order = np.argsort(demands)
            cost_matrix = assign_cost
        elif strategy == 'random':
            order = np.random.permutation(n_cust)
            cost_matrix = assign_cost
        elif strategy == 'cheapest':
            min_costs = np.min(assign_cost, axis=1)
            order = np.argsort(min_costs)
            cost_matrix = assign_cost
        else:
            order = np.argsort(-demands)
            cost_matrix = assign_cost

        for i in order:
            fac_order = np.argsort(cost_matrix[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Fallback: any feasible by raw assign_cost
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
            if result[i] < 0:
                # Last resort: most remaining capacity
                remaining = capacities - load
                f = int(np.argmax(remaining))
                result[i] = f
                load[f] += demands[i]

        return result, load

    def _local_search(result, load, deadline):
        """Best-improvement local search with Relocate and Swap moves."""
        result = result.copy()
        load = load.copy()
        count = np.zeros(n_fac, dtype=int)
        for c in range(n_cust):
            count[result[c]] += 1

        current_cost = _total_cost(result)

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # --- Phase 1: Best Relocate ---
            best_delta = 0.0
            best_c = -1
            best_f = -1

            for c in range(n_cust):
                if time.time() > deadline - 10.0:
                    break
                old_f = result[c]
                d_c = demands[c]
                base_cost = assign_cost[c, old_f]

                for f in range(n_fac):
                    if f == old_f:
                        continue
                    if load[f] + d_c > capacities[f]:
                        continue

                    delta = assign_cost[c, f] - base_cost

                    # Open cost changes
                    if count[old_f] == 1:
                        delta -= open_cost[old_f]
                    if count[f] == 0:
                        delta += open_cost[f]

                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                        best_f = f

            if best_delta < -1e-9 and best_c != -1:
                c = best_c
                f = best_f
                old_f = result[c]
                result[c] = f
                load[old_f] -= demands[c]
                load[f] += demands[c]
                count[old_f] -= 1
                count[f] += 1
                current_cost += best_delta
                improved = True
                continue

            # --- Phase 2: Best Swap ---
            best_swap_delta = 0.0
            best_c1 = -1
            best_c2 = -1

            for c1 in range(n_cust):
                if time.time() > deadline - 10.0:
                    break
                f1 = result[c1]
                d1 = demands[c1]
                ac1_f1 = assign_cost[c1, f1]

                for c2 in range(c1 + 1, n_cust):
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]

                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1

                    if new_load_f1 > capacities[f1] + 1e-6 or new_load_f2 > capacities[f2] + 1e-6:
                        continue

                    delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                            (ac1_f1 + assign_cost[c2, f2])

                    if delta < best_swap_delta:
                        best_swap_delta = delta
                        best_c1 = c1
                        best_c2 = c2

            if best_swap_delta < -1e-9 and best_c1 != -1:
                c1, c2 = best_c1, best_c2
                f1, f2 = result[c1], result[c2]
                result[c1], result[c2] = f2, f1
                load[f1] -= demands[c1]
                load[f1] += demands[c2]
                load[f2] -= demands[c2]
                load[f2] += demands[c1]
                current_cost += best_swap_delta
                improved = True
                continue

        return result, current_cost

    # Portfolio of seeds
    strategies = ['eff', 'eff_desc', 'raw_desc', 'raw_asc', 'cheapest', 'random']
    best_cost = float('inf')
    best_result = None

    for strategy in strategies:
        if time.time() > deadline - 25.0:
            break

        result, load = _greedy_seed(strategy)
        current_cost = _total_cost(result)

        result, current_cost = _local_search(result, load, deadline)

        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()

    if best_result is None:
        best_result, _ = _greedy_seed('eff')

    return best_result

# EVOLVE-BLOCK-END
