<<<<<<< SEARCH
"""Capacitated facility location: facility-centric + Lagrangian seed + cluster moves."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
=======
"""Capacitated facility location: facility-centric + Lagrangian seed + cluster moves."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import milp, LinearConstraint, Bounds
>>>>>>> REPLACE
<<<<<<< SEARCH
def _true_obj(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        return float('inf')
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + assign_sum
=======
def _true_obj(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        return float('inf')
    open_sum = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_sum += open_cost[f]
    assign_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + assign_sum


def _milp_solve(demands, capacities, open_cost, assign_cost, time_limit=5.0):
    """
    Solve the capacitated facility location problem using MILP via scipy.
    Returns (result, obj) or (None, inf) if infeasible or timeout.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables: x[c, f] in {0, 1} for each customer-facility pair
    # y[f] in {0, 1} for each facility open
    # Total variables: n_cust * n_fac + n_fac
    
    num_vars = n_cust * n_fac + n_fac
    x = np.zeros(num_vars)
    
    # Objective: minimize sum(c_f * x_cf) + sum(f_f * y_f)
    obj = np.zeros(num_vars)
    for c in range(n_cust):
        for f in range(n_fac):
            obj[c * n_fac + f] = assign_cost[c, f]
    for f in range(n_fac):
        obj[n_cust * n_fac + f] = open_cost[f]
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_f x[c,f] = 1
    # 2. Capacity: sum_c demands[c] * x[c,f] <= capacities[f] * y[f]
    
    # Build A matrix and bounds
    n_constraints = n_cust + n_fac
    
    A = np.zeros((n_constraints, num_vars))
    lb = np.zeros(n_constraints)
    ub = np.zeros(n_constraints)
    
    # Constraint 1: assignment
    for c in range(n_cust):
        for f in range(n_fac):
            A[c, c * n_fac + f] = 1.0
        lb[c] = 1.0
        ub[c] = 1.0
    
    # Constraint 2: capacity (sum_c d_c * x_cf <= cap_f * y_f)
    # => sum_c d_c * x_cf - cap_f * y_f <= 0
    for f in range(n_fac):
        for c in range(n_cust):
            A[n_cust + f, c * n_fac + f] = demands[c]
        A[n_cust + f, n_cust * n_fac + f] = -capacities[f]
        lb[n_cust + f] = -np.inf
        ub[n_cust + f] = 0.0
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Variable bounds: all binary
    integrality = np.ones(num_vars)
    var_bounds = Bounds(0, 1)
    
    try:
        res = milp(c=obj, constraints=constraints, integrality=integrality,
                   bounds=var_bounds, options={'time_limit': time_limit, 'presolve': True})
        
        if res.status == 0 or res.status == 1:  # Optimal or near-optimal
            # Extract solution
            result = np.full(n_cust, -1, dtype=int)
            for c in range(n_cust):
                for f in range(n_fac):
                    if res.x[c * n_fac + f] > 0.5:
                        result[c] = f
                        break
            
            # Verify
            if np.any(result < 0):
                return None, float('inf')
            
            obj_val = _true_obj(result, demands, capacities, open_cost, assign_cost)
            if obj_val == float('inf'):
                return None, float('inf')
            
            return result, obj_val
        else:
            return None, float('inf')
    except Exception:
        return None, float('inf')
>>>>>>> REPLACE
<<<<<<< SEARCH
def _greedy_seed(demands, capacities, assign_cost):
    """Largest-first greedy with capacity awareness."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        # Sort facilities by cost, pick first feasible
        best_f = -1
        best_cost = np.inf
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                if assign_cost[i, f] < best_cost:
                    best_cost = assign_cost[i, f]
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += d
        else:
            # Force into least-loaded feasible facility
            for f in range(n_fac):
                if load[f] + d <= capacities[f]:
                    result[i] = f
                    load[f] += d
                    break
            else:
                result[i] = 0
                load[0] += d
    return result
=======
def _greedy_seed(demands, capacities, assign_cost):
    """Largest-first greedy with capacity awareness."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        # Sort facilities by cost, pick first feasible
        best_f = -1
        best_cost = np.inf
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                if assign_cost[i, f] < best_cost:
                    best_cost = assign_cost[i, f]
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += d
        else:
            # Force into least-loaded feasible facility
            for f in range(n_fac):
                if load[f] + d <= capacities[f]:
                    result[i] = f
                    load[f] += d
                    break
            else:
                result[i] = 0
                load[0] += d
    return result


def _diverse_greedy_seeds(demands, capacities, assign_cost, n_seeds=5):
    """Generate diverse greedy seeds with different orderings."""
    n_cust = len(demands)
    n_fac = len(capacities)
    seeds = []
    
    # Seed 0: largest-first (default)
    seeds.append(_greedy_seed(demands, capacities, assign_cost))
    
    # Additional seeds with different orderings
    for trial in range(1, n_seeds):
        rng = np.random.RandomState(42 + trial)
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        # Different ordering strategies
        if trial == 1:
            order = np.argsort(-demands)  # largest first
        elif trial == 2:
            order = np.argsort(demands)  # smallest first
        elif trial == 3:
            order = rng.permutation(n_cust)  # random
        elif trial == 4:
            # Sort by min cost (easy to place first)
            min_costs = np.min(assign_cost, axis=1)
            order = np.argsort(min_costs)
        else:
            order = rng.permutation(n_cust)
        
        for i in order:
            d = demands[i]
            best_f = -1
            best_cost = np.inf
            for f in np.argsort(assign_cost[i]):
                if load[f] + d <= capacities[f]:
                    if assign_cost[i, f] < best_cost:
                        best_cost = assign_cost[i, f]
                        best_f = f
            if best_f >= 0:
                result[i] = best_f
                load[best_f] += d
            else:
                for f in range(n_fac):
                    if load[f] + d <= capacities[f]:
                        result[i] = f
                        load[f] += d
                        break
                else:
                    result[i] = 0
                    load[0] += d
        seeds.append(result)
    
    return seeds
>>>>>>> REPLACE
<<<<<<< SEARCH
def _lagrangian_seed(demands, capacities, assign_cost, n_iters=300, deadline=None):
    """
    Lagrangian relaxation on capacity constraints.
    Dual variables mu[f] >= 0 penalize capacity usage.
    For given mu, each customer picks argmin_f (assign_cost[c,f] + mu[f]).
    Subgradient: g[f] = load[f] - capacities[f] (for f with load>0).
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    total_demand = float(np.sum(demands))
    # Initial step size
    step = 2.0 * total_demand / (n_fac * (n_iters + 1))
    mu = np.zeros(n_fac)
    best_result = None
    best_obj = float('inf')
    best_feasible = None

    for it in range(n_iters):
        if deadline is not None and time.time() >= deadline:
            break
        # Primal: each customer picks argmin of (assign_cost + mu)
        effective = assign_cost + mu[np.newaxis, :]
        result = np.argmin(effective, axis=1)
        load = np.zeros(n_fac)
        for c, f in enumerate(result):
            load[f] += demands[c]

        # Check feasibility
        if np.all(load <= capacities):
            obj = _true_obj(result, demands, capacities, open_cost_placeholder(assign_cost, open_cost=None), assign_cost)
            # We don't have open_cost here; compute a proxy
            # Actually we need open_cost. Let's pass it in.
            pass

        # Subgradient update
        grad = load - capacities
        grad = np.maximum(grad, 0)  # mu >= 0, only positive violations push up
        # Normalize step
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-9:
            break
        step_t = step * (1.0 - it / n_iters)  # decay
        mu = np.maximum(0, mu + step_t * grad / grad_norm * np.max(capacities) / (np.max(capacities) + 1))
        # Cap mu to avoid overflow
        mu = np.minimum(mu, 1e6)

    # Final: use the mu to get a result, then repair
    effective = assign_cost + mu[np.newaxis, :]
    result = np.argmin(effective, axis=1)
    return result


def open_cost_placeholder(assign_cost, open_cost=None):
    # dummy to satisfy signature; we'll restructure
    if open_cost is None:
        return np.zeros(assign_cost.shape[1])
    return open_cost
>>>>>>> REPLACE
<<<<<<< SEARCH
def _repair_capacity(result, demands, capacities, assign_cost):
    """Move customers from overloaded facilities to feasible alternatives."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]

    for _ in range(n_fac * 5):
        excess_mask = load > capacities
        if not np.any(excess_mask):
            break
        for f in np.where(excess_mask)[0]:
            while load[f] > capacities[f]:
                customers_at_f = np.where(result == f)[0]
                if len(customers_at_f) == 0:
                    break
                # Move customer with highest demand first
                c = customers_at_f[np.argmax(demands[customers_at_f])]
                d = demands[c]
                alt_costs = assign_cost[c].copy()
                alt_costs[f] = np.inf
                alt_order = np.argsort(alt_costs)
                moved = False
                for af in alt_order:
                    if load[af] + d <= capacities[af]:
                        load[f] -= d
                        load[af] += d
                        result[c] = af
                        moved = True
                        break
                if not moved:
                    # Try any feasible facility regardless of cost order
                    for af in range(n_fac):
                        if af != f and load[af] + d <= capacities[af]:
                            load[f] -= d
                            load[af] += d
                            result[c] = af
                            moved = True
                            break
                if not moved:
                    break
    return result
=======
def _repair_capacity(result, demands, capacities, assign_cost):
    """Move customers from overloaded facilities to feasible alternatives."""
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]

    for _ in range(n_fac * 5):
        excess_mask = load > capacities
        if not np.any(excess_mask):
            break
        for f in np.where(excess_mask)[0]:
            while load[f] > capacities[f]:
                customers_at_f = np.where(result == f)[0]
                if len(customers_at_f) == 0:
                    break
                # Move customer with highest demand first
                c = customers_at_f[np.argmax(demands[customers_at_f])]
                d = demands[c]
                alt_costs = assign_cost[c].copy()
                alt_costs[f] = np.inf
                alt_order = np.argsort(alt_costs)
                moved = False
                for af in alt_order:
                    if load[af] + d <= capacities[af]:
                        load[f] -= d
                        load[af] += d
                        result[c] = af
                        moved = True
                        break
                if not moved:
                    # Try any feasible facility regardless of cost order
                    for af in range(n_fac):
                        if af != f and load[af] + d <= capacities[af]:
                            load[f] -= d
                            load[af] += d
                            result[c] = af
                            moved = True
                            break
                if not moved:
                    break
    return result


def _polish_with_milp(result, demands, capacities, open_cost, assign_cost, deadline):
    """
    Use MILP to find a better solution if time permits.
    Returns (result, obj) or original if MILP fails or times out.
    """
    remaining = deadline - time.time()
    if remaining < 2.0:
        return result, _true_obj(result, demands, capacities, open_cost, assign_cost)
    
    # Try MILP with a reasonable time limit
    milp_time = min(remaining - 1.0, 30.0)
    milp_result, milp_obj = _milp_solve(demands, capacities, open_cost, assign_cost, time_limit=milp_time)
    
    if milp_result is not None:
        cur_obj = _true_obj(result, demands, capacities, open_cost, assign_cost)
        if milp_obj < cur_obj:
            return milp_result, milp_obj
    
    return result, _true_obj(result, demands, capacities, open_cost, assign_cost)
>>>>>>> REPLACE
<<<<<<< SEARCH
def _multi_start_seed(demands, capacities, open_cost, assign_cost, deadline):
    """Generate multiple seeds and pick the best."""
    n_cust = len(demands)
    n_fac = len(capacities)
    seeds = []

    # Seed 1: Greedy largest-first
    seeds.append(_greedy_seed(demands, capacities, assign_cost))

    # Seed 2: Argmin + repair
    r2 = np.argmin(assign_cost, axis=1).astype(int)
    r2 = _repair_capacity(r2, demands, capacities, assign_cost)
    seeds.append(r2)

    # Seed 3: Randomized greedy (multiple random orderings)
    for trial in range(3):
        if time.time() >= deadline:
            break
        rng = np.random.RandomState(42 + trial)
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        order = rng.permutation(n_cust)
        for i in order:
            d = demands[i]
            best_f = -1
            best_cost = np.inf
            for f in np.argsort(assign_cost[i]):
                if load[f] + d <= capacities[f]:
                    if assign_cost[i, f] < best_cost:
                        best_cost = assign_cost[i, f]
                        best_f = f
            if best_f >= 0:
                result[i] = best_f
                load[best_f] += d
            else:
                for f in range(n_fac):
                    if load[f] + d <= capacities[f]:
                        result[i] = f
                        load[f] += d
                        break
                else:
                    result[i] = 0
                    load[0] += d
        seeds.append(result)

    # Pick best seed
    best_seed = None
    best_seed_obj = float('inf')
    for s in seeds:
        if np.any(np.array(s) < 0) or np.any(np.array(s) >= n_fac):
            continue
        load = np.zeros(n_fac)
        for c, f in enumerate(s):
            load[f] += demands[c]
        if np.any(load > capacities):
            continue
        obj = _true_obj(s, demands, capacities, open_cost, assign_cost)
        if obj < best_seed_obj:
            best_seed_obj = obj
            best_seed = s

    if best_seed is None:
        return seeds[0]
    return best_seed
=======
def _multi_start_seed(demands, capacities, open_cost, assign_cost, deadline):
    """Generate multiple seeds and pick the best."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Generate diverse greedy seeds
    seeds = _diverse_greedy_seeds(demands, capacities, assign_cost, n_seeds=5)
    
    # Add argmin + repair seed
    r2 = np.argmin(assign_cost, axis=1).astype(int)
    r2 = _repair_capacity(r2, demands, capacities, assign_cost)
    seeds.append(r2)
    
    # Pick best seed
    best_seed = None
    best_seed_obj = float('inf')
    for s in seeds:
        if np.any(np.array(s) < 0) or np.any(np.array(s) >= n_fac):
            continue
        load = np.zeros(n_fac)
        for c, f in enumerate(s):
            load[f] += demands[c]
        if np.any(load > capacities):
            continue
        obj = _true_obj(s, demands, capacities, open_cost, assign_cost)
        if obj < best_seed_obj:
            best_seed_obj = obj
            best_seed = s

    if best_seed is None:
        return seeds[0]
    return best_seed
>>>>>>> REPLACE
<<<<<<< SEARCH
def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using multi-phase optimization:
    1. Multi-start seed
    2. Customer local search (moves + swaps)
    3. Facility-set refinement
    4. Cluster moves
    5. Final customer local search
    """
    n_cust = len(demands)
    n_fac = len(capacities)

    # Safety margin: 150s limit, aim for 140s max
    deadline = time.time() + 140.0

    # Phase 1: Multi-start seed
    result = _multi_start_seed(demands, capacities, open_cost, assign_cost, deadline)

    # Phase 2: Customer local search
    result, obj = _customer_local_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 3: Facility-set refinement
    if time.time() < deadline:
        result, obj = _facility_set_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 4: Cluster moves
    if time.time() < deadline:
        result, obj = _cluster_moves(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 5: Final customer local search
    if time.time() < deadline:
        result, obj = _customer_local_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Verify feasibility
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        # Fallback: return greedy result
        return _greedy_seed(demands, capacities, assign_cost).astype(int)

    return result.astype(int)
# EVOLVE-BLOCK-END
=======
def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using multi-phase optimization:
    1. Multi-start seed
    2. Customer local search (moves + swaps)
    3. Facility-set refinement
    4. Cluster moves
    5. Final customer local search
    6. MILP polish (if time permits)
    """
    n_cust = len(demands)
    n_fac = len(capacities)

    # Safety margin: 150s limit, aim for 145s max
    deadline = time.time() + 145.0

    # Phase 1: Multi-start seed
    result = _multi_start_seed(demands, capacities, open_cost, assign_cost, deadline)

    # Phase 2: Customer local search
    result, obj = _customer_local_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 3: Facility-set refinement
    if time.time() < deadline:
        result, obj = _facility_set_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 4: Cluster moves
    if time.time() < deadline:
        result, obj = _cluster_moves(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 5: Final customer local search
    if time.time() < deadline:
        result, obj = _customer_local_search(result, demands, capacities, open_cost, assign_cost, deadline)

    # Phase 6: MILP polish
    if time.time() < deadline:
        result, obj = _polish_with_milp(result, demands, capacities, open_cost, assign_cost, deadline)

    # Verify feasibility
    load = np.zeros(n_fac)
    for c, f in enumerate(result):
        load[f] += demands[c]
    if np.any(load > capacities):
        # Fallback: return greedy result
        return _greedy_seed(demands, capacities, assign_cost).astype(int)

    return result.astype(int)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
