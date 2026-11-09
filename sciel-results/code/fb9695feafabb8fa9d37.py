"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    rng = random.Random(42)

    best = None
    best_n = 0

    # First, try to quickly find a valid graph via local search from random init
    # for various target sizes, starting large
    for target_n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 30:
            break
        result = _local_search(target_n, deadline - 25, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']
            if best_n >= 21:
                break

    # If we didn't find anything, use fallback
    if best is None or best_n < 10:
        fallback = _make_fallback(rng)
        if fallback['n'] > best_n:
            best = fallback
            best_n = fallback['n']

    return best


def _make_fallback(rng):
    """Generate a verified valid graph on 10 vertices."""
    n = 10
    for _ in range(5000):
        adj = [[0]*n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = adj[j][i] = 1
                    edges.append([i, j])
        if _is_valid_fast(adj, n):
            return {'n': n, 'edges': edges}
    return {'n': 5, 'edges': []}


def _is_valid_fast(adj, n):
    """Check if no subset of size >= 6 induces a regular graph using bitmasks."""
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)

    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset_fast(adj_mask, subset):
                return False
    return True


def _is_regular_subset_fast(adj_mask, subset):
    """Check if induced subgraph on subset is regular using bitmask operations."""
    s = len(subset)
    if s < 2:
        return True

    subset_mask = 0
    for v in subset:
        subset_mask |= (1 << v)

    d0 = None
    for v in subset:
        d = bin(adj_mask[v] & subset_mask).count('1')
        if d0 is None:
            d0 = d
        elif d != d0:
            return False
    return True


def _find_violation_fast(adj_mask, n):
    """Find a subset of size >= 6 that induces a regular graph. Returns subset or None."""
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset_fast(adj_mask, subset):
                return subset
    return None


def _local_search(target_n, deadline, rng):
    """Local search to find a valid graph on target_n vertices."""
    n = target_n
    start_time = time.time()

    # Try multiple random initializations
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize with random graph
        adj = [[0]*n for _ in range(n)]
        adj_mask = [0] * n
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = adj[j][i] = 1
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    edges.append([i, j])

        # Check if already valid
        if _is_valid_fast(adj, n):
            return {'n': n, 'edges': edges}

        # Local search: find violations and flip edges to break them
        best_viol_count = _count_violations(adj_mask, n)
        
        for iteration in range(500):
            if time.time() > deadline:
                break

            # Find a violation
            viol = _find_violation_fast(adj_mask, n)
            if viol is None:
                return {'n': n, 'edges': edges}

            subset = viol
            size = len(subset)

            # Try flipping edges within the violation subset to break it
            improved = False
            # Collect all edges within the subset
            internal_edges = []
            for i in range(size):
                for j in range(i+1, size):
                    u, v = subset[i], subset[j]
                    internal_edges.append((u, v))
            
            rng.shuffle(internal_edges)
            
            for u, v in internal_edges:
                # Flip the edge
                old_val = adj[u][v]
                adj[u][v] = adj[v][u] = 1 - old_val
                if old_val:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    # Remove from edges list
                    for k in range(len(edges)-1, -1, -1):
                        if (edges[k][0] == u and edges[k][1] == v) or (edges[k][0] == v and edges[k][1] == u):
                            edges.pop(k)
                            break
                else:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.append([min(u, v), max(u, v)])

                # Check if this flip helps (reduces violations)
                new_viol_count = _count_violations(adj_mask, n)
                if new_viol_count < best_viol_count:
                    best_viol_count = new_viol_count
                    improved = True
                    break
                else:
                    # Revert
                    adj[u][v] = adj[v][u] = old_val
                    if old_val:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        edges.append([min(u, v), max(u, v)])
                    else:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        for k in range(len(edges)-1, -1, -1):
                            if (edges[k][0] == u and edges[k][1] == v) or (edges[k][0] == v and edges[k][1] == u):
                                edges.pop(k)
                                break

            if not improved:
                # Try flipping a random edge outside the subset
                a, b = rng.sample(range(n), 2)
                old_val2 = adj[a][b]
                adj[a][b] = adj[b][a] = 1 - old_val2
                if old_val2:
                    adj_mask[a] &= ~(1 << b)
                    adj_mask[b] &= ~(1 << a)
                    for k in range(len(edges)-1, -1, -1):
                        if (edges[k][0] == a and edges[k][1] == b) or (edges[k][0] == b and edges[k][1] == a):
                            edges.pop(k)
                            break
                else:
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                    edges.append([min(a, b), max(a, b)])

                new_viol_count = _count_violations(adj_mask, n)
                if new_viol_count < best_viol_count:
                    best_viol_count = new_viol_count
                else:
                    # Revert
                    adj[a][b] = adj[b][a] = old_val2
                    if old_val2:
                        adj_mask[a] |= (1 << b)
                        adj_mask[b] |= (1 << a)
                        edges.append([min(a, b), max(a, b)])
                    else:
                        adj_mask[a] &= ~(1 << b)
                        adj_mask[b] &= ~(1 << a)
                        for k in range(len(edges)-1, -1, -1):
                            if (edges[k][0] == a and edges[k][1] == b) or (edges[k][0] == b and edges[k][1] == a):
                                edges.pop(k)
                                break

            # Occasionally add random noise
            if rng.random() < 0.1:
                for _ in range(3):
                    a, b = rng.sample(range(n), 2)
                    adj[a][b] = adj[b][a] = 1 - adj[a][b]
                    if adj[a][b]:
                        adj_mask[a] |= (1 << b)
                        adj_mask[b] |= (1 << a)
                        edges.append([min(a, b), max(a, b)])
                    else:
                        adj_mask[a] &= ~(1 << b)
                        adj_mask[b] &= ~(1 << a)
                        for k in range(len(edges)-1, -1, -1):
                            if (edges[k][0] == a and edges[k][1] == b) or (edges[k][0] == b and edges[k][1] == a):
                                edges.pop(k)
                                break

    return None


def _count_violations(adj_mask, n):
    """Count the number of subsets of size >= 6 that induce a regular graph."""
    count = 0
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset_fast(adj_mask, subset):
                count += 1
    return count
# EVOLVE-BLOCK-END
