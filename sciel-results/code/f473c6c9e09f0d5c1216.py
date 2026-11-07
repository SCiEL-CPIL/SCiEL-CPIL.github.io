"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    start_time = time.time()
    TIMEOUT = 75  # safety margin before 90s kill

    # Base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)

    # Candidate edges: all non-path edges connecting even-odd pairs (bipartite constraint)
    # Path edges connect i to i+1, which are always even-odd, so we exclude those.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in [(a, b) for a, b in P['edges']] or (v, u) in [(a, b) for a, b in P['edges']]:
                continue
            # Must be bipartite: different parity
            if (u % 2) == (v % 2):
                continue
            candidates.append((u, v))

    m = len(candidates)

    def compute_dist(adj):
        """Compute all-pairs shortest distances. Returns dist matrix or None if disconnected."""
        dist = [[-1] * n for _ in range(n)]
        for s in range(n):
            dist[s][s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[s][v] == -1:
                        dist[s][v] = dist[s][u] + 1
                        q.append(v)
        return dist

    def check_validity(edge_indices):
        """Check if the set of edges (by index into candidates) gives a valid solution.
        Returns (diameter, num_violations) or None if disconnected."""
        adj = [set(base_adj[i]) for i in range(n)]
        for idx in edge_indices:
            u, v = candidates[idx]
            adj[u].add(v)
            adj[v].add(u)
        dist = compute_dist(adj)
        if dist is None:
            return None
        max_d = 0
        viol = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i][j] == -1:
                    return None
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                if dist[i][j] > 4:
                    viol += 1
        return max_d, viol

    def greedy_lower_bound(adj, k_limit):
        """Greedy set-cover lower bound: how many more edges are needed to fix all dist>4 pairs.
        Returns a lower bound on additional edges needed."""
        dist = compute_dist(adj)
        if dist is None:
            return float('inf')
        # Find all pairs with dist > 4
        violated_pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i][j] > 4:
                    violated_pairs.add((i, j))
        if not violated_pairs:
            return 0

        # For each candidate edge not yet used, compute which violated pairs it would fix
        # An edge (u,v) fixes pair (a,b) if the new shortest path a->...->u-v->...->b <= 4
        # Simplification: edge (u,v) reduces dist(a,b) to min(dist(a,b), dist(a,u)+1+dist(v,b), dist(a,v)+1+dist(u,b))
        # We use current dist as approximation.

        remaining = set(violated_pairs)
        count = 0
        while remaining and count < k_limit:
            best_fix = 0
            best_edge = None
            for idx in range(m):
                u, v = candidates[idx]
                fixed = 0
                for (a, b) in remaining:
                    # Check if adding (u,v) would make dist(a,b) <= 4
                    # New dist = min(old, dist(a,u)+1+dist(v,b), dist(a,v)+1+dist(u,b))
                    nd1 = dist[a][u] + 1 + dist[v][b] if dist[a][u] != -1 and dist[v][b] != -1 else 99
                    nd2 = dist[a][v] + 1 + dist[u][b] if dist[a][v] != -1 and dist[u][b] != -1 else 99
                    new_d = min(dist[a][b], nd1, nd2)
                    if new_d <= 4:
                        fixed += 1
                if fixed > best_fix:
                    best_fix = fixed
                    best_edge = idx
            if best_edge is None or best_fix == 0:
                return float('inf')  # Can't fix with remaining candidates
            count += 1
            # Remove fixed pairs from remaining (approximate)
            u, v = candidates[best_edge]
            still_remaining = set()
            for (a, b) in remaining:
                nd1 = dist[a][u] + 1 + dist[v][b] if dist[a][u] != -1 and dist[v][b] != -1 else 99
                nd2 = dist[a][v] + 1 + dist[u][b] if dist[a][v] != -1 and dist[u][b] != -1 else 99
                new_d = min(dist[a][b], nd1, nd2)
                if new_d > 4:
                    still_remaining.add((a, b))
            remaining = still_remaining
        if remaining:
            return float('inf')
        return count

    # Branch and bound
    best_solution = None
    best_count = float('inf')

    def branch_and_bound(selected, start_idx, current_adj):
        nonlocal best_solution, best_count
        if time.time() - start_time > TIMEOUT:
            return

        # Check current state
        if len(selected) >= best_count:
            return

        diam, viol = check_validity(selected)
        if diam is not None and diam <= 4:
            if len(selected) < best_count:
                best_count = len(selected)
                best_solution = list(selected)
            return

        if diam is None:
            return

        # Lower bound
        lb = greedy_lower_bound(current_adj, best_count - len(selected))
        if lb is None or lb >= best_count - len(selected):
            return

        # Branch: try adding candidates from start_idx onwards
        for i in range(start_idx, m):
            if time.time() - start_time > TIMEOUT:
                return
            u, v = candidates[i]
            current_adj[u].add(v)
            current_adj[v].add(u)
            selected.append(i)
            branch_and_bound(selected, i + 1, current_adj)
            selected.pop()
            current_adj[u].remove(v)
            current_adj[v].remove(u)

    # Initialize adjacency for B&B
    init_adj = [set(base_adj[i]) for i in range(n)]
    branch_and_bound([], 0, init_adj)

    if best_solution is not None:
        result = [[int(candidates[i][0]), int(candidates[i][1])] for i in best_solution]
        return result

    # Fallback: greedy
    added = []
    adj = [set(base_adj[i]) for i in range(n)]
    while True:
        if time.time() - start_time > TIMEOUT:
            break
        diam, viol = check_validity([i for i, (u, v) in enumerate(candidates) if u in adj[v]])
        # Recompute properly
        dist = compute_dist(adj)
        if dist is None:
            break
        max_d = max(dist[i][j] for i in range(n) for j in range(n) if dist[i][j] != -1)
        if max_d <= 4:
            break
        best_edge = None
        best_viol = float('inf')
        for idx in range(m):
            u, v = candidates[idx]
            if v in adj[u]:
                continue
            adj[u].add(v)
            adj[v].add(u)
            d = compute_dist(adj)
            if d is None:
                adj[u].remove(v)
                adj[v].remove(u)
                continue
            vd = sum(1 for i in range(n) for j in range(i+1, n) if d[i][j] > 4)
            md = max(d[i][j] for i in range(n) for j in range(n) if d[i][j] != -1)
            score = 10000 * md + vd
            adj[u].remove(v)
            adj[v].remove(u)
            if score < best_viol:
                best_viol = score
                best_edge = idx
        if best_edge is None:
            break
        u, v = candidates[best_edge]
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])

    return added

# EVOLVE-BLOCK-END
