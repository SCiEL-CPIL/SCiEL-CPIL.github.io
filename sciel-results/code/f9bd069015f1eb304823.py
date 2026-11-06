import numpy as np
import time

_START = time.time()
# Safety margin before the 150 s wall-clock kill (leaves headroom for Python overhead).
_LIMIT = 150.0 - 15.0  # seconds of compute budget

# Parameter: cap on the number of remaining nodes we can include in the
# optimal (DP) tour model. This is the "model size" we optimize against
# the objective: larger => more accurate cost estimate => better choices,
# but O(k^2 * 2^k) time. We pick the largest that is comfortably within budget.
_MAX_SET_SIZE = 13


def _dp_tsp(dmat):
    """Return best tour length starting at depot (index 0), visiting
    every node in 1..k-1 exactly once, and returning to depot.
    dmat: (k, k) float distance submatrix (node 0 == depot in this sub-index).
    k: number of nodes in the subproblem.
    Returns cost or None if infeasible (time / size)."""
    k = dmat.shape[0]
    if k <= 1:
        return 0.0
    # We model the tour over nodes 1..k-1 (k-1 of them) and must return to 0.
    m = k - 1  # number of customer nodes
    if m == 0:
        return dmat[0, 0]
    if m > _MAX_SET_SIZE:
        return None

    nfull = 1 << m
    # dp[s] : dict last_bit_index -> cost of path starting at depot, having
    # visited exactly the subset s (bits 0..m-1), currently at node (bit+1).
    dp = [None] * nfull
    d = dmat
    # Precompute bit -> node index
    bit_to_node = [b + 1 for b in range(m)]

    # handle s=0 specially (start from depot)
    for b in range(m):
        node = bit_to_node[b]
        nb = 1 << b
        tgt = dp[nb]
        if tgt is None:
            dp[nb] = {b: d[0, node]}
        else:
            if b not in tgt or d[0, node] < tgt[b]:
                tgt[b] = d[0, node]

    for s in range(nfull):
        cur = dp[s]
        if cur is None:
            continue
        if s == 0:
            continue
        # extend to any node not yet in s
        for b in range(m):
            if s & (1 << b):
                continue
            nb = s | (1 << b)
            to_node = bit_to_node[b]
            best_val = np.inf
            for last, cv in cur.items():
                last_node = bit_to_node[last]
                val = cv + d[last_node, to_node]
                if val < best_val:
                    best_val = val
            if best_val >= np.inf:
                continue
            tgt = dp[nb]
            if tgt is None:
                dp[nb] = {b: best_val}
            else:
                if b not in tgt or best_val < tgt[b]:
                    tgt[b] = best_val

    full = nfull - 1
    if dp[full] is None:
        return None
    best_total = np.inf
    for last, cv in dp[full].items():
        last_node = bit_to_node[last]
        total = cv + d[last_node, 0]
        if total < best_total:
            best_total = total
    return float(best_total)


def _greedy_finish(dmat, start):
    """Greedy nearest-neighbor tour over nodes 1..k-1 of dmat, from start,
    returning to depot (0). Returns cost."""
    k = dmat.shape[0]
    m = k - 1
    if m == 0:
        return 0.0
    remaining = set(range(1, k))
    cur = start
    cost = 0.0
    while remaining:
        # nearest feasible remaining node
        best = min(remaining, key=lambda n: dmat[cur, n])
        cost += dmat[cur, best]
        remaining.remove(best)
        cur = best
    cost += dmat[cur, 0]
    return cost


def _model_remaining_cost(remaining, rest_capacity, cur_loc, demands, dm):
    """Estimate the cost of finishing the current route starting from cur_loc,
    visiting a subset of `remaining` nodes (respecting capacity) optimally,
    then returning to the depot. `remaining` is a list of node indices.
    Returns an estimated total cost (lower = better). Uses DP if the reachable
    set is small enough, else a greedy model."""
    depot = 0
    # Reachable nodes: demand fits in rest_capacity (we only can ever take ones
    # whose demand <= remaining capacity at the time; conservatively use the
    # current rest_capacity for the model).
    reach = [n for n in remaining if demands[n] <= rest_capacity]
    # In practice we can only keep taking nodes while capacity lasts. For the
    # *model*, we cap the number of nodes by both reachability and a practical
    # bound (the number we can afford by capacity).
    # Sort by demand desc to greedily pack into capacity to estimate count.
    reach_sorted = sorted(reach, key=lambda n: -demands[n])
    packed = []
    cap = rest_capacity
    for n in reach_sorted:
        if demands[n] <= cap:
            packed.append(n)
            cap -= demands[n]
    # The model tours a (bounded) subset of `packed`. We take up to _MAX_SET_SIZE
    # most important nodes (nearest to cur_loc) to keep the DP tractable; if more
    # are reachable, we approximate the tail with a greedy per-node mean cost.
    if len(packed) > _MAX_SET_SIZE:
        # pick the _MAX_SET_SIZE nearest as the "optimally solved" core,
        # and model the rest with a greedy mean.
        packed.sort(key=lambda n: dm[cur_loc, n])
        core = packed[:_MAX_SET_SIZE]
        tail = packed[_MAX_SET_SIZE:]
        # greedy cost of tail from last core node approximated by mean edge
        if core:
            last = core[-1]
        else:
            last = cur_loc
        if tail:
            tail_cost = sum(dm[last, n] + dm[n, depot] for n in tail)
        else:
            tail_cost = 0.0
    else:
        core = packed
        tail_cost = 0.0
        last = cur_loc

    k = len(core) + 1
    if k <= 1:
        return tail_cost + (dm[cur_loc, depot] if k == 1 else 0.0)

    # Build sub distance matrix with node0=depot
    nodes = [depot] + core
    sub = np.empty((k, k), dtype=float)
    idx = {node: i for i, node in enumerate(nodes)}
    for i in range(k):
        for j in range(k):
            sub[i, j] = dm[nodes[i], nodes[j]]

    cost = _dp_tsp(sub)
    if cost is None:
        # fallback greedy from cur_loc
        cost = _greedy_finish(sub, 0)
    return float(cost) + tail_cost


def _sweep_angle_priority(current_node, depot, feasible, dm):
    """
    Compute a priority score for each feasible node based on sweep angle.
    Nodes that are in a consistent angular sector relative to the vector from
    depot to current_node are prioritized to minimize route fragmentation.
    Returns a list of (node, angle_score) sorted by angle_score (lower is better).
    """
    if not feasible:
        return []
    
    # Vector from depot to current node
    cur_vec = dm[depot, current_node]
    # We need 2D coordinates to compute angles, but we only have a distance matrix.
    # Approximation: Use the distance to depot as a proxy for radial distance,
    # and the distance from current_node as a proxy for angular deviation.
    # A better approximation for sweep angle without coordinates:
    # Project the node onto the direction of travel.
    
    # Since we don't have coordinates, we use a heuristic based on distances:
    # Nodes that are close to the current node and far from the depot are "ahead".
    # Nodes that are close to the depot are "behind".
    # We want to visit nodes that maintain the "sweep" away from the depot.
    
    # Heuristic score:
    # 1. Distance from current node (closer is better)
    # 2. Distance from depot (farther is better, to avoid backtracking)
    # 3. Angular consistency: we can approximate the angle by the ratio of
    #    (dist(cur, node) - dist(depot, cur) - dist(depot, node)) / (2 * dist(depot, cur) * dist(depot, node))
    #    This is related to the cosine of the angle.
    
    scores = []
    for n in feasible:
        d_cur = dm[current_node, n]
        d_dep = dm[depot, n]
        d_cur_dep = dm[depot, current_node]
        
        # If current_node is the depot, all angles are 0 (or undefined), just use distance
        if d_cur_dep < 1e-9:
            score = d_cur
        else:
            # Cosine of angle at depot between (depot->cur) and (depot->node)
            # cos(theta) = (d_dep^2 + d_cur_dep^2 - d_cur^2) / (2 * d_dep * d_cur_dep)
            # We want to minimize the angle, so maximize cos(theta)
            # But we also want to be close to current node.
            
            # Combine: prioritize nodes that are close to current node AND have a small angle
            # from the current direction of travel.
            
            # A simple combined score:
            # score = d_cur - alpha * cos(theta)
            # where alpha is a weight.
            
            # To avoid division by zero or invalid cos:
            if d_dep < 1e-9:
                cos_theta = 1.0
            else:
                cos_theta = (d_dep * d_dep + d_cur_dep * d_cur_dep - d_cur * d_cur) / (2.0 * d_dep * d_cur_dep)
                # Clamp to [-1, 1]
                cos_theta = max(-1.0, min(1.0, cos_theta))
            
            # We want to minimize d_cur, so we subtract a bonus for good angle alignment.
            # The "sweep" angle is the angle we need to turn from the current direction.
            # If we are at current_node, the direction is from depot to current_node.
            # The angle of the vector (current_node -> node) relative to (depot -> current_node)
            # is what matters for backtracking.
            
            # Let's use a simpler metric:
            # Penalty for backtracking: if the node is "behind" the current node relative to the depot,
            # the distance from depot to node should be less than distance from depot to current_node.
            # If d_dep < d_cur_dep, it's backtracking.
            
            backtrack_penalty = max(0, d_cur_dep - d_dep)
            score = d_cur + 0.5 * backtrack_penalty
            
        scores.append((n, score))
    
    scores.sort(key=lambda x: x[1])
    return scores


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    t0 = time.time()
    deadline = _START + _LIMIT

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = list(unvisited_nodes)
    M = len(unvisited)

    # Feasible first nodes: demand <= rest_capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return feasible[0]

    dm = np.asarray(distance_matrix)

    # Rank candidates: prefer closer ones first so the deadline guard can bail
    # early and still return a reasonable (greedy-like) choice.
    # Use sweep-angle priority to order candidates
    ordered_candidates = _sweep_angle_priority(current_node, depot, feasible, dm)
    # Extract node IDs in order
    feasible_ordered = [n for n, _ in ordered_candidates]

    best_score = np.inf
    best_node = feasible_ordered[0]

    for cand in feasible_ordered:
        if time.time() > deadline:
            break
        # cost to reach candidate
        cost = dm[current_node, cand]
        rem_cap = rest_capacity - demands[cand]
        rem_nodes = [n for n in unvisited if n != cand]
        # model the rest of the route from candidate with remaining capacity
        est = _model_remaining_cost(rem_nodes, rem_cap, cand, demands, dm)
        total = cost + est
        if total < best_score:
            best_score = total
            best_node = cand

    return int(best_node)
