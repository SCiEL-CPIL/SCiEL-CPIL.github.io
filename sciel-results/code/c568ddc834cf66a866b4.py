"""Capacitated vehicle routing heuristic with savings-based lookahead."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 110.0  # safety margin before 120s kill
_start_time = None


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.perf_counter() - _start_time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()

    dm = np.asarray(distance_matrix)
    dem = np.asarray(demands)
    cur = int(current_node)
    dep = int(depot)

    # If nothing fits, close the route
    feasible = [int(n) for n in unvisited_nodes if dem[n] <= rest_capacity]
    if not feasible:
        return int(depot)

    # Deadline guard: fall back to fast nearest-feasible
    if _elapsed() > _DEADLINE:
        feas_arr = np.array(feasible)
        return int(feas_arr[np.argmin(dm[cur, feas_arr])])

    d_cur = dm[cur]
    d_cur_dep = float(d_cur[dep])
    d_dep = dm[dep]

    # 1-step lookahead: min distance from each candidate to any other feasible/depot
    extra = list(feasible)
    if dep not in extra:
        extra = extra + [dep]
    extra = np.array(extra)
    sub = dm[np.ix_(extra, extra)]
    idx = np.arange(len(extra))
    sub[idx, idx] = np.inf
    next_min = sub.min(axis=1)
    next_min_map = {int(extra[i]): float(next_min[i]) for i in range(len(extra))}

    # Score each feasible customer
    best_score = -np.inf
    best_node = feasible[0]
    for c in feasible:
        c = int(c)
        d1 = float(d_cur[c])
        sav = float(d_cur_dep + d_dep[c] - d_cur[c])
        nxt = next_min_map.get(c, d_cur_dep)
        score = -d1 + 0.5 * sav - 0.3 * nxt
        if score > best_score:
            best_score = score
            best_node = c

    # Score depot (close route)
    dep_score = -d_cur_dep
    if rest_capacity <= 2:
        dep_score += 0.5

    if dep_score > best_score:
        return int(depot)
    return int(best_node)
# EVOLVE-BLOCK-END
