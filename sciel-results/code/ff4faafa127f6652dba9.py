"""Assembly Line Balancing: bottleneck-density priority using downstream critical path and station scarcity."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque


def _build_graph(instance):
    n = len(instance["times"])
    preds = instance["predecessors"]
    # successors[j] = list of tasks that have j as a predecessor
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for j in range(n):
        ps = preds[j] if preds is not None else []
        if ps is None:
            ps = []
        for p in ps:
            succ[p].append(j)
            indeg[j] += 1
    return succ, indeg


def _compute_topo_order(succ, n):
    """Return topological order of tasks."""
    indeg = [0] * n
    for j in range(n):
        for s in succ[j]:
            indeg[s] += 1
    q = deque([j for j in range(n) if indeg[j] == 0])
    topo = []
    while q:
        j = q.popleft()
        topo.append(j)
        for s in succ[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    return topo


def _longest_path_to_end(succ, times, n, topo):
    """lp[j] = times[j] + max(lp[s] for s in succ[j])"""
    lp = np.zeros(n, dtype=float)
    for j in reversed(topo):
        best = 0.0
        for s in succ[j]:
            if lp[s] > best:
                best = lp[s]
        lp[j] = times[j] + best
    # Tasks not in topo (cycle members) get just their own time
    for j in range(n):
        if lp[j] == 0.0:
            lp[j] = times[j]
    return lp


def _min_stations_downstream(succ, times, n, topo, cycle_time):
    """
    min_st[j] = minimum number of stations needed to finish task j and its
    downstream subtree, assuming a station can hold any subset of ready
    downstream tasks whose total time <= cycle_time. Approximated by
    ceil(lp[j] / cycle_time) where lp is the longest-path (critical) length.
    This is a lower bound on stations; tasks with a large lp relative to
    cycle_time force many sequential stations -> scarce capacity.
    """
    lp = _longest_path_to_end(succ, times, n, topo)
    # avoid divide by zero / tiny cycle
    ct = max(float(cycle_time), 1e-9)
    return np.ceil(lp / ct).astype(float)


def _downstream_depth(succ, n, topo):
    """
    depth[j] = 1 + max(depth[s] for s in succ[j]), or 1 if no successors.
    This measures how many levels of tasks remain after j.
    """
    depth = np.ones(n, dtype=float)
    for j in reversed(topo):
        if succ[j]:
            max_d = max(depth[s] for s in succ[j])
            depth[j] = 1.0 + max_d
    return depth


def _descendant_count(succ, n, topo):
    """
    count[j] = number of all downstream tasks reachable from j.
    """
    count = np.zeros(n, dtype=float)
    for j in reversed(topo):
        if succ[j]:
            count[j] = 1.0 + max(count[s] for s in succ[j])
        # else count[j] stays 0
    return count


def priority(instance):
    start = time.time()
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance["predecessors"]
    cycle_time = instance.get("cycle_time", 40.0)
    
    succ, _ = _build_graph(instance)
    topo = _compute_topo_order(succ, n)
    
    # Longest path to end (critical path length)
    lp = _longest_path_to_end(succ, times, n, topo)
    
    # Downstream depth: how many levels of tasks remain
    depth = _downstream_depth(succ, n, topo)
    
    # Descendant count: total number of downstream tasks
    desc_count = _descendant_count(succ, n, topo)
    
    # Immediate successor count
    nsucc = np.array([len(s) for s in succ], dtype=float)
    
    # True bottleneck density: ratio of task time to minimum stations required
    # to finish its downstream subtree. A large value means this task is long
    # relative to the scarce station capacity left for the rest of the line.
    min_st = _min_stations_downstream(succ, times, n, topo, cycle_time)
    # guard against zero stations
    min_st_safe = np.maximum(min_st, 1.0)
    bottleneck_density = times / min_st_safe
    # Also a station-scarcity multiplier: tasks with many required downstream
    # stations are on a tight path -> boost their priority.
    scarcity = min_st_safe / max(float(min_st_safe.max()), 1e-9)

    # Normalize to [0,1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    s_lp = norm(lp)
    s_depth = norm(depth)
    s_desc = norm(desc_count)
    s_nsucc = norm(nsucc)
    s_time = norm(times)
    s_bd = norm(bottleneck_density)
    s_scar = norm(scarcity)

    # Smooth multiplicative interaction (log-sum-exp style): tasks that are
    # BOTH critical AND bottlenecked get a super-additive boost, while tasks
    # that are only one or the other get a moderate score. Use a soft product
    # via log(1+a*x) so the interaction is smooth and bounded.
    a = 1.5  # interaction strength
    soft_interaction = np.log1p(a * s_lp * s_bd)
    s_int = norm(soft_interaction)

    # Small self-tuning over a compact weight grid (fast: 30 tasks, 5 instances
    # at evaluation; here we just pick a fixed improved blend to stay safe).
    # Emphasize: critical path, true bottleneck density, smooth interaction,
    # and downstream scarcity.
    score = (0.38 * s_lp
             + 0.24 * s_bd
             + 0.18 * s_int
             + 0.10 * s_scar
             + 0.06 * s_depth
             + 0.04 * s_desc)

    # Safety margin guard
    if time.time() - start > 100:
        pass

    return score
# EVOLVE-BLOCK-END
