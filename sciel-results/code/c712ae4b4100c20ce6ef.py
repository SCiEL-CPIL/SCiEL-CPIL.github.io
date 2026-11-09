"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    # Hardcoded fallback: n=10 random graph (very likely valid)
    # We'll verify and use as fallback
    rng = random.Random(42)
    fallback = _make_fallback(rng)

    best = fallback
    best_n = fallback['n']

    # Try to find larger graphs via local search
    for target_n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 10:
            break
        result = _local_search(target_n, deadline - 5, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']

    return best


def _make_fallback(rng):
    """Generate a verified valid graph on 10 vertices."""
    n = 10
    # Try random graphs until we find one with no regular induced subgraph of size >= 6
    for _ in range(10000):
        edges = []
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = adj[j][i] = 1
                    edges.append([i, j])
        if _is_valid(adj, n):
            return {'n': n, 'edges': edges}
    # Should not reach here, but provide a trivially valid small graph
    return {'n': 5, 'edges': []}


def _is_valid(adj, n):
    """Check if no subset of size >= 6 induces a regular graph."""
    from itertools import combinations
    # Check all subsets of size 6, 7, ..., n
    # For efficiency, check size 6 first (most common violation)
    for size in range(6, n+1):
        for subset in combinations(range(n), size):
            if _is_regular_induced(adj, subset, n):
                return False
    return True


def _is_regular_induced(adj, subset, n):
    """Check if the induced subgraph on subset is regular."""
    s = len(subset)
    if s < 2:
        return True  # trivially regular
    # Count degrees within the induced subgraph
    degrees = [0] * s
    for i in range(s):
        for j in range(i+1, s):
            if adj[subset[i]][subset[j]]:
                degrees[i] += 1
                degrees[j] += 1
    # Check if all degrees are equal
    d0 = degrees[0]
    for i in range(1, s):
        if degrees[i] != d0:
            return False
    return True


def _find_violation(adj, n):
    """Find a subset of size >= 6 that induces a regular graph. Returns (subset, size) or None."""
    from itertools import combinations
    # Check smaller sizes first as they're more likely to be violations
    for size in range(6, n+1):
        for subset in combinations(range(n), size):
            if _is_regular_induced(adj, subset, n):
                return subset, size
    return None


def _local_search(target_n, deadline, rng):
    """Local search to find a valid graph on target_n vertices."""
    n = target_n
    start_time = time.time()

    # Initialize with random graph
    adj = [[0]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj[i][j] = adj[j][i] = 1
                edges.append((i, j))

    # Check if already valid
    if _is_valid(adj, n):
        return _to_result(adj, n, edges)

    # Local search: find violations and flip edges to break them
    best_adj = [row[:] for row in adj]
    best_edges = edges[:]
    best_score = -1  # score = negative number of violations

    for iteration in range(5000):
        if time.time() > deadline:
            break

        # Find a violation
        viol = _find_violation(adj, n)
        if viol is None:
            # No violation found, graph is valid!
            return _to_result(adj, n, edges)

        subset, size = viol
        # Pick a random edge within the subset to flip
        # This should break the regularity
        i, j = rng.sample(range(size), 2)
        u, v = subset[i], subset[j]

        # Flip the edge
        old_val = adj[u][v]
        adj[u][v] = adj[v][u] = 1 - old_val
        if old_val:
            # Remove edge
            edges = [(a, b) for a, b in edges if not ((a == u and b == v) or (a == v and b == u))]
        else:
            edges.append((u, v))

        # Optionally, also try flipping a random edge outside the subset
        if rng.random() < 0.3:
            a, b = rng.sample(range(n), 2)
            old_val2 = adj[a][b]
            adj[a][b] = adj[b][a] = 1 - old_val2
            if old_val2:
                edges = [(x, y) for x, y in edges if not ((x == a and y == b) or (x == b and y == a))]
            else:
                edges.append((a, b))

    # Return best found (check if valid)
    if _is_valid(adj, n):
        return _to_result(adj, n, edges)
    return None


def _to_result(adj, n, edges):
    """Convert adjacency matrix and edge list to result format."""
    edge_list = [[u, v] for u, v in edges]
    return {'n': n, 'edges': edge_list}
# EVOLVE-BLOCK-END
