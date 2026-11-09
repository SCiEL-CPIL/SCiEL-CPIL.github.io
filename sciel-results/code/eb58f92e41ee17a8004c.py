# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_cost_flow

def _solve_flow(demands, capacities, open_cost, assign_cost, open_mask):
    """Solve the capacitated assignment exactly via min-cost flow, given which facilities are open."""
    n = len(demands)
    m = len(capacities)
    open_facs = np.where(open_mask)[0]
    if len(open_facs) == 0:
        return None, np.inf

    # Nodes: 0=source, 1..n=customers, n+1..n+m=facilities, n+m+1=sink
    # Total nodes = n + m + 2
    total = n + m + 2
    sink = total - 1

    rows = []
    cols = []
    caps = []
    costs = []

    # Source -> customers (supply = demand)
    for i in range(n):
        rows.append(0)
        cols.append(1 + i)
        caps.append(int(demands[i]))
        costs.append(0)

    # Customers -> open facilities
    for i in range(n):
        for f in open_facs:
            rows.append(1 + i)
            cols.append(n + 1 + f)
            caps.append(int(demands[i]))
            costs.append(int(round(assign_cost[i, f] * 100)))

    # Facilities -> sink (capacity)
    for f in open_facs:
        rows.append(n + 1 + f)
        cols.append(sink)
        caps.append(int(capacities[f]))
        costs.append(0)

    if not rows:
        return None, np.inf

    cap_arr = np.array(caps, dtype=np.int64)
    cost_arr = np.array(costs, dtype=np.int64)
    row_arr = np.array(rows, dtype=np.int64)
    col_arr = np.array(cols, dtype=np.int64)

    cap_matrix = coo_matrix((cap_arr, (row_arr, col_arr)), shape=(total, total)).tocsr()
    cost_matrix = coo_matrix((cost_arr, (row_arr, col_arr)), shape=(total, total)).tocsr()

    try:
        flow, cost_val = min_cost_flow(cap_matrix, cost_matrix)
    except Exception:
        return None, np.inf

    if cost_val is None or flow is None:
        return None, np.inf

    # Extract assignment: customer i -> facility f if flow on edge (1+i, n+1+f) > 0
    result = np.zeros(n, dtype=int)
    flow_coo = flow.tocoo()
    # Build a dict for customer->facility flows
    cust_fac_flows = {}
    for r, c, v in zip(flow_coo.row, flow_coo.col, flow_coo.data):
        if 1 <= r <= n and n + 1 <= c <= n + m and v > 0:
            cust_fac_flows[(r - 1, c - n - 1)] = v

    valid = True
    for i in range(n):
        found = False
        for f in open_facs:
            if (i, f) in cust_fac_flows:
                result[i] = f
                found = True
                break
        if not found:
            valid = False
            break

    if not valid:
        return None, np.inf

    # Verify capacity
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]
    if np.any(load > capacities + 1e-6):
        return None, np.inf

    # Compute true cost
    open_sum = float(open_cost[open_mask].sum())
    assign_sum = float(assign_cost[np.arange(n), result].sum())
    return result, open_sum + assign_sum

def _greedy_feasible(demands, capacities, assign_cost, open_mask):
    """Greedy feasible assignment restricted to open facilities."""
    n = len(demands)
    m = len(capacities)
    open_facs = np.where(open_mask)[0]
    if len(open_facs) == 0:
        return None
    result = np.zeros(n, dtype=int)
    load = np.zeros(m, dtype=float)
    for i in np.argsort(-demands):
        feasible = [f for f in open_facs if load[f] + demands[i] <= capacities[f]]
        if not feasible:
            return None
        f = min(feasible, key=lambda x: assign_cost[i, x])
        result[i] = f
        load[f] += demands[i]
    return result

def _total_cost(result, demands, open_cost, assign_cost):
    used = np.unique(result)
    open_sum = float(open_cost[used].sum())
    assign_sum = float(assign_cost[np.arange(len(demands)), result].sum())
    return open_sum + assign_sum

def _is_valid(result, demands, capacities):
    load = np.zeros(len(capacities), dtype=float)
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    return np.all(load <= capacities + 1e-9)

def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit):
    """Best-improvement local search on the assignment (facilities fixed = those used)."""
    n = len(demands)
    m = len(capacities)
    result = result.copy()
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]

    current_cost = _total_cost(result, demands, open_cost, assign_cost)
    best_cost = current_cost
    best_result = result.copy()
    start_time = time.time()

    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            cur_f = result[i]
            d = demands[i]
            feasible_mask = (load + d) <= capacities
            feasible_mask[cur_f] = False
            if not np.any(feasible_mask):
                continue
            feasible_indices = np.where(feasible_mask)[0]
            d_assign = assign_cost[i, feasible_indices] - assign_cost[i, cur_f]
            target_loads = load[feasible_indices]
            d_open_target = np.where(target_loads == 0, open_cost[feasible_indices], 0.0)
            d_open_source = -open_cost[cur_f] if (load[cur_f] - d) <= 1e-9 else 0.0
            total_delta = d_assign + d_open_target + d_open_source
            min_idx = np.argmin(total_delta)
            min_delta = total_delta[min_idx]
            if min_delta < -1e-9:
                best_f = feasible_indices[min_idx]
                load[cur_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += min_delta
                improved = True
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best_result = result.copy()
        if not improved:
            break
    return best_result, best_cost

def _simulated_annealing(demands, capacities, open_cost, assign_cost, time_limit, rng):
    """Simulated annealing over facility subsets, solving exact flow for each."""
    n = len(demands)
    m = len(capacities)
    start_time = time.time()

    # Initial: open all facilities, solve flow
    open_mask = np.ones(m, dtype=bool)
    result, cost = _solve_flow(demands, capacities, open_cost, assign_cost, open_mask)
    if result is None:
        # Fallback greedy
        result = _greedy_feasible(demands, capacities, assign_cost, open_mask)
        if result is None:
            return None, np.inf
        cost = _total_cost(result, demands, open_cost, assign_cost)

    best_result = result.copy()
    best_cost = cost
    cur_result = result.copy()
    cur_cost = cost
    cur_mask = open_mask.copy()

    T = 50.0
    T_min = 0.01
    cooling = 0.97

    while time.time() - start_time < time_limit:
        if T < T_min:
            break

        # Propose: toggle a random facility
        f = rng.randint(m)
        if cur_mask[f]:
            # Try closing: check feasibility (capacity of remaining must cover demand)
            new_mask = cur_mask.copy()
            new_mask[f] = False
            # Quick check: can remaining facilities cover total demand?
            if np.sum(capacities[new_mask]) < np.sum(demands) - 1e-6:
                continue
            # Check: if f has assigned customers, they need reassignment
            # Just try to solve flow
            new_result, new_cost = _solve_flow(demands, capacities, open_cost, assign_cost, new_mask)
            if new_result is None:
                continue
        else:
            # Try opening
            new_mask = cur_mask.copy()
            new_mask[f] = True
            new_result, new_cost = _solve_flow(demands, capacities, open_cost, assign_cost, new_mask)
            if new_result is None:
                continue

        delta = new_cost - cur_cost
        if delta < 0 or rng.random() < np.exp(-delta / T):
            cur_mask = new_mask
            cur_result = new_result
            cur_cost = new_cost
            if new_cost < best_cost - 1e-9:
                best_cost = new_cost
                best_result = new_result.copy()

        T *= cooling

    return best_result, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n = len(demands)
    m = len(capacities)
    global_start = time.time()
    deadline = global_start + 120.0

    rng = np.random.RandomState(12345)
    best_result = None
    best_cost = np.inf

    # Phase 1: Exact flow with all facilities open
    t1 = time.time()
    result, cost = _solve_flow(demands, capacities, open_cost, assign_cost, np.ones(m, dtype=bool))
    if result is not None and cost < best_cost:
        best_cost = cost
        best_result = result.copy()
    t_flow = time.time() - t1

    # Phase 2: Simulated annealing over subsets
    remaining = deadline - time.time() - 5.0  # reserve 5s for local search
    if remaining > 1.0:
        sa_result, sa_cost = _simulated_annealing(demands, capacities, open_cost, assign_cost, remaining, rng)
        if sa_result is not None and sa_cost < best_cost:
            best_cost = sa_cost
            best_result = sa_result.copy()

    # Phase 3: Local search refinement on best result
    remaining = deadline - time.time() - 2.0
    if remaining > 0.5 and best_result is not None:
        ls_result, ls_cost = _local_search(best_result, demands, capacities, open_cost, assign_cost, remaining)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_result = ls_result.copy()

    # Fallback if nothing worked
    if best_result is None:
        open_mask = np.ones(m, dtype=bool)
        best_result = _greedy_feasible(demands, capacities, assign_cost, open_mask)
        if best_result is None:
            # Last resort: greedy without open mask restriction
            best_result = np.zeros(n, dtype=int)
            load = np.zeros(m, dtype=float)
            for i in np.argsort(-demands):
                feasible = [f for f in range(m) if load[f] + demands[i] <= capacities[f]]
                if feasible:
                    f = min(feasible, key=lambda x: assign_cost[i, x])
                else:
                    f = int(np.argmin(load))
                best_result[i] = f
                load[f] += demands[i]

    if not _is_valid(best_result, demands, capacities):
        # Emergency repair
        best_result = np.zeros(n, dtype=int)
        load = np.zeros(m, dtype=float)
        for i in np.argsort(-demands):
            feasible = [f for f in range(m) if load[f] + demands[i] <= capacities[f]]
            if feasible:
                f = min(feasible, key=lambda x: assign_cost[i, x])
            else:
                f = int(np.argmin(load))
            best_result[i] = f
            load[f] += demands[i]

    return best_result.astype(int)
# EVOLVE-BLOCK-END
