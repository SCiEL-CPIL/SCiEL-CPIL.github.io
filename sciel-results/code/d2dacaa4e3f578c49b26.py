"""Capacitated facility location: greedy seed + vectorized local search with restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 135.0  # safety margin under the 150s wall-clock kill


def _objective(assignment, demands, capacities, open_cost, assign_cost):
    """Total cost = open cost of used facilities + assignment cost. Returns inf if infeasible."""
    n_fac = len(capacities)
    load = np.bincount(assignment, minlength=n_fac)
    if np.any(load > capacities):
        return np.inf
    used = load > 0
    return float(open_cost[used].sum() + assign_cost[np.arange(len(demands)), assignment].sum())


def _seed(demands, capacities, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_c, n_f = len(demands), len(capacities)
    result = np.zeros(n_c, dtype=int)
    load = np.zeros(n_f)
    order = np.argsort(-demands)
    for i in order:
        cands = np.where(load + demands[i] <= capacities)[0]
        if len(cands) == 0:
            # fallback: facility with most remaining capacity
            f = int(np.argmax(capacities - load))
        else:
            f = int(cands[np.argmin(assign_cost[i, cands])])
        load[f] += demands[i]
        result[i] = f
    return result


def _local_search(assignment, demands, capacities, open_cost, assign_cost, deadline):
    """Vectorized 1-move and 2-swap local search until no improving move or deadline."""
    n_c = len(demands)
    n_f = len(capacities)
    cost = _objective(assignment, demands, capacities, open_cost, assign_cost)
    if not np.isfinite(cost):
        return assignment, cost

    improved = True
    while improved and time.time() < deadline:
        improved = False
        load = np.bincount(assignment, minlength=n_f)
        used = load > 0

        # ---- 1-move pass (vectorized) ----
        # cost_delta[i, f] = assign_cost[i, f] - assign_cost[i, cur] + open_change
        cur = assignment
        base = assign_cost[np.arange(n_c), cur]
        delta = assign_cost - base[None, :]  # (n_c, n_f)
        # opening cost change: if f unused, +open_cost[f]; if cur becomes unused, -open_cost[cur]
        open_add = np.where(used[None, :], 0.0, open_cost[None, :])
        # cur becomes unused only if it's the sole occupant
        sole = load == 1
        open_sub = np.where(sole[cur], open_cost[cur], 0.0)
        delta = delta + open_add - open_sub[:, None]

        # feasibility: load[f] + demand[i] - (1 if f==cur else 0) <= capacities[f]
        # new_load[f] = load[f] + demand[i] - (f == cur_i)
        # For each i, f: feasible if load[f] + demand[i] - (f == cur[i]) <= capacities[f]
        new_load = load[None, :] + demands[:, None]
        # subtract 1 where f == cur[i]
        new_load -= np.eye(n_f)[cur]
        feasible = new_load <= capacities[None, :]
        feasible = feasible & (np.arange(n_f)[None, :] != cur[:, None])  # must be a real move

        best_f = np.full(n_c, -1, dtype=int)
        best_d = np.full(n_c, np.inf)
        mask = feasible & (delta < 0)
        if np.any(mask):
            # for each customer, the best improving feasible facility
            d = np.where(mask, delta, np.inf)
            best_d = d.min(axis=1)
            best_f = d.argmin(axis=1)
            ok = best_d < 0
            if np.any(ok):
                # apply moves one at a time (greedy) to keep loads consistent
                idx = np.where(ok)[0]
                # sort by improvement magnitude to apply biggest first
                idx = idx[np.argsort(best_d[idx])]
                applied = 0
                for i in idx:
                    f = best_f[i]
                    if f < 0 or f == cur[i]:
                        continue
                    if load[f] + demands[i] - (1 if f == cur[i] else 0) <= capacities[f]:
                        load[f] += demands[i]
                        load[cur[i]] -= 1
                        cur[i] = f
                        applied += 1
                if applied > 0:
                    improved = True
                    cost = _objective(cur, demands, capacities, open_cost, assign_cost)

        # ---- 2-swap pass (vectorized over pairs sharing a facility) ----
        if time.time() >= deadline:
            break
        # For each facility, customers assigned to it
        for f in range(n_f):
            members = np.where(cur == f)[0]
            if len(members) < 2:
                continue
            # try swapping each pair of members to a common other facility g
            # cost change = (ac[i,g]+ac[j,g]-ac[i,f]-ac[j,f]) + open_change
            # only consider g that is used or cheap to open
            for a in range(len(members)):
                i = members[a]
                for b in range(a + 1, len(members)):
                    j = members[b]
                    if i == j:
                        continue
                    base_ij = assign_cost[i, f] + assign_cost[j, f]
                    # candidate g != f
                    cand = assign_cost[i, :] + assign_cost[j, :] - base_ij
                    # feasibility: g must have room for demand[i]+demand[j]
                    room = capacities - load + (2 if True else 0)
                    # load[f] decreases by 2, so others unchanged; g must fit both
                    feasible_g = (load + demands[i] + demands[j]) <= capacities
                    feasible_g[f] = False
                    if not np.any(feasible_g):
                        continue
                    cg = np.where(feasible_g, cand, np.inf)
                    g = int(cg.argmin())
                    if g < 0 or not np.isfinite(cg[g]):
                        continue
                    # open cost change
                    open_change = 0.0
                    if load[g] == 0:
                        open_change += open_cost[g]
                    if load[f] - 2 == 0:
                        open_change -= open_cost[f]
                    total = cg[g] + open_change
                    if total < -1e-9:
                        # apply
                        load[g] += demands[i] + demands[j]
                        load[f] -= demands[i] + demands[j]
                        cur[i] = g
                        cur[j] = g
                        improved = True
                        cost = _objective(cur, demands, capacities, open_cost, assign_cost)
                        break
                if improved and time.time() >= deadline:
                    break

    return cur, cost


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    start = time.time()
    deadline = start + _DEADLINE

    best_assign = None
    best_cost = np.inf

    n_restarts = 12
    for r in range(n_restarts):
        if time.time() >= deadline:
            break
        # seed: r==0 greedy; else greedy with slight randomization in tie-break
        if r == 0:
            seed = _seed(demands, capacities, assign_cost)
        else:
            # perturb: start from greedy then randomly reassign a few customers
            seed = _seed(demands, capacities, assign_cost).copy()
            n_c = len(demands)
            n_f = len(capacities)
            k = max(1, n_c // 10)
            perm = np.random.permutation(n_c)
            for i in perm[:k]:
                # move to a random feasible facility
                load = np.bincount(seed, minlength=n_f)
                f = seed[i]
                load[f] -= demands[i]
                cands = np.where(load + demands[i] <= capacities)[0]
                if len(cands) > 0:
                    seed[i] = int(cands[np.random.randint(len(cands))])
                else:
                    load[f] += demands[i]

        a, c = _local_search(seed, demands, capacities, open_cost, assign_cost, deadline)
        if np.isfinite(c) and c < best_cost:
            best_cost = c
            best_assign = a

    if best_assign is None:
        best_assign = _seed(demands, capacities, assign_cost)

    # final feasibility check; if infeasible, fall back to pure greedy
    load = np.bincount(best_assign, minlength=len(capacities))
    if np.any(load > capacities):
        best_assign = _seed(demands, capacities, assign_cost)

    return best_assign.astype(int)
# EVOLVE-BLOCK-END
