"""Guided local search penalty for TSP: 2-opt move regret-based penalties."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    if time.perf_counter() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    n = edge_distance.shape[0]
    if n <= 2:
        return np.asarray(edge_distance, dtype=float).copy()

    # True distances (always input is the true matrix)
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Lengths of each tour edge (true)
    tour_len = d[i, j]
    mean_tour_len = np.mean(tour_len)

    # For each edge in the tour, compute the 2-opt move regret:
    # For edge (a, b), removing it means we need to reconnect a to some node x
    # and b to some node y via a 2-opt reversal. The minimal additional cost
    # to remove this edge is the regret.
    
    # Efficient computation: for each edge index k with endpoints (i[k], j[k]),
    # the 2-opt move that removes this edge reverses the segment between i[k] and j[k].
    # The cost change is: d[i[k], j[k]] + d[i[k+1], j[k-1]] - d[i[k], i[k+1]] - d[j[k-1], j[k]]
    # Wait, that's not quite right. Let me think more carefully.
    
    # In a 2-opt move, we reverse a segment. If we remove edge (a, b) where a=tour[k], b=tour[k+1],
    # we can reverse the segment from b to a (going the other way around the cycle).
    # The two edges that are removed are (a, b) and the edge connecting the other endpoints.
    
    # Actually, let's use a simpler approach: for each edge (a, b) in the tour,
    # compute the best alternative: find the minimum cost to replace this edge
    # by connecting a to some neighbor and b to some neighbor differently.
    
    # Simpler regret model: for edge (a, b), the regret is how much longer this edge is
    # compared to the best alternative connection that would maintain a valid tour.
    # A reasonable proxy: the difference between this edge's length and the shortest
    # edge incident to either a or b (excluding the other endpoint).
    
    # Let's use a direct 2-opt regret: for each edge, compute the minimal cost increase
    # if we were to remove it via the cheapest 2-opt reversal that eliminates it.
    
    # For edge at position k (connecting tour[k] and tour[k+1]), a 2-opt reversal of
    # segment [k+1, m] for some m changes edges. To specifically remove edge (tour[k], tour[k+1]),
    # we can reverse segment [k+1, n-1] (wrapping around). This replaces:
    #   d[tour[k], tour[k+1]] + d[tour[n-1], tour[0]]
    # with:
    #   d[tour[k], tour[n-1]] + d[tour[k+1], tour[0]]
    
    # But that's just one specific move. Let's compute the minimal regret over all
    # 2-opt moves that remove this edge.
    
    # For edge (a, b) where a=tour[k], b=tour[k+1]:
    # A 2-opt move removing this edge reverses a segment containing exactly one of {a, b}
    # as an endpoint. The reversal connects a to some node c and b to some node d.
    
    # Let's use a simpler but effective metric: for each edge, compute how much it
    # "sticks out" compared to local alternatives.
    
    # Regret for edge (a, b): min over all valid 2-opt moves that remove (a,b) of
    # (new_cost - old_cost). If negative, the edge is already suboptimal.
    
    # For efficiency with n=100, let's compute for each edge the best single-edge
    # replacement: the shortest edge from a to any node other than b, plus the shortest
    # edge from b to any node other than a, minus the current edge length.
    
    # Actually, let's use a cleaner approach: compute the "regret" as the difference
    # between this edge's cost and the best possible replacement in a 2-opt context.
    
    # For edge (a, b) at position k in tour:
    # The two neighbors of a in the tour are tour[k-1] and tour[k+1]=b
    # The two neighbors of b in the tour are a=tour[k] and tour[k+2]
    
    # A 2-opt reversal that removes (a,b) and keeps the tour valid:
    # Reverse segment [k+1, m] for m in {k+2, ..., n-1, 0, ..., k-2}
    # This removes edges (a, b) and (tour[m], tour[m+1 mod n])
    # And adds edges (a, tour[m]) and (b, tour[m+1 mod n])
    
    # The cost change is:
    #   d[a, tour[m]] + d[b, tour[m+1]] - d[a, b] - d[tour[m], tour[m+1]]
    
    # We want the minimum of this over all valid m.
    
    # Let's compute this for each edge.
    tour_len_arr = d[tour, np.roll(tour, -1)]
    
    # For each edge k, compute min regret over all reversals
    regrets = np.zeros(n)
    for k in range(n):
        a = tour[k]
        b = tour[(k + 1) % n]
        min_regret = np.inf
        # Try all reversal endpoints m
        for m in range(n):
            if m == k or m == (k - 1) % n:
                continue  # Skip invalid reversals
            m_next = (m + 1) % n
            if m_next == k or m_next == (k + 1) % n:
                continue
            # Cost change for reversing segment [k+1, m]
            old_cost = d[a, b] + d[tour[m], tour[m_next]]
            new_cost = d[a, tour[m]] + d[b, tour[m_next]]
            regret = new_cost - old_cost
            if regret < min_regret:
                min_regret = regret
        regrets[k] = min_regret
    
    # Regrets can be negative (edge is already suboptimal) or positive
    # We want to penalize edges with high regret (hard to remove)
    # Scale regrets to be in distance units
    regret_max = np.max(regrets)
    regret_min = np.min(regrets)
    if regret_max > regret_min + 1e-9:
        regret_scaled = (regrets - regret_min) / (regret_max - regret_min)
    else:
        regret_scaled = np.zeros_like(regrets)
    
    # Penalty: higher for edges with higher regret (harder to remove)
    # This encourages 2-opt to target these edges
    penalty_mag = 0.5 * regret_scaled * mean_tour_len
    
    # Apply penalties
    penalty = np.zeros((n, n), dtype=float)
    for idx in range(n):
        a, b = i[idx], j[idx]
        mag = penalty_mag[idx]
        if mag > 1e-8:
            penalty[a, b] += mag
            penalty[b, a] += mag
    
    # Add usage-based penalty for exploration
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    if u_max > 0:
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
    else:
        usage_scaled = np.zeros_like(usage)
    penalty += 0.05 * usage_scaled * mean_tour_len
    
    # Symmetrize
    penalty = 0.5 * (penalty + penalty.T)
    
    # Result: true distance + penalty
    result = d + penalty
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)
    
    return result
