"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    rng = random.Random(42)

    # Strategy: spectral gap heuristic with NumPy-accelerated cost evaluation
    # Cost = sum over all 6-subsets of (sum of squared degree deviations from mean)
    # Flip edges that most reduce this cost; validate exactly when cost == 0

    best_n = 0
    best_result = None

    # Try target sizes from largest to smallest
    for target_n in [22, 21, 20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 30:
            break

        # Multiple restarts for each target size
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break

            result = _spectral_search(target_n, deadline - 15, rng, restart)
            if result is not None and result['n'] > best_n:
                best_n = result['n']
                best_result = result
                # Found a valid graph, try to improve
                break

        if best_n >= target_n:
            break

    # Fallback: verified small graph
    if best_result is None or best_n < 10:
        fallback = _make_fallback(rng)
        if fallback['n'] > best_n:
            best_n = fallback['n']
            best_result = fallback

    return best_result if best_result is not None else {'n': 1, 'edges': []}


def _spectral_search(n, deadline, rng, seed_offset=0):
    """Spectral gap heuristic: minimize sum of squared degree deviations
    across all 6-vertex induced subgraphs using NumPy matrix operations."""

    start_time = time.time()
    rng_local = random.Random(42 + seed_offset)

    # Initialize random graph
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng_local.random() < 0.5:
                A[i, j] = A[j, i] = 1

    # Precompute all 6-combinations as index arrays for vectorized operations
    combos_6 = np.array(list(itertools.combinations(range(n), 6)), dtype=np.int32)
    n_combos = len(combos_6)

    # Also precompute for larger subsets if n allows
    combos_7 = np.array(list(itertools.combinations(range(n), 7)), dtype=np.int32) if n >= 7 else np.empty((0, 7), dtype=np.int32)
    combos_8 = np.array(list(itertools.combinations(range(n), 8)), dtype=np.int32) if n >= 8 else np.empty((0, 8), dtype=np.int32)

    def compute_cost(A):
        """Compute total irregularity cost using NumPy vectorization."""
        total_cost = 0.0

        # For 6-vertex subsets
        if n_combos > 0:
            # Extract degree matrices for all 6-subsets at once
            # For each subset S, degree of vertex v in S = sum of A[v, S\{v}]
            # Vectorized: for combo c, degrees = A[combo, combo].sum(axis=1) - 1 (diagonal is 0)
            # Actually A[combo, combo] gives the subgraph adjacency, sum of each row = degree
            sub = A[np.ix_(combos_6.ravel(), combos_6.ravel())].reshape(n_combos, 6, 6)
            # Degrees within each induced subgraph
            degs = sub.sum(axis=1)  # shape (n_combos, 6)
            # Mean degree per subgraph
            means = degs.mean(axis=1, keepdims=True)  # shape (n_combos, 1)
            # Squared deviations
            sq_dev = (degs - means) ** 2  # shape (n_combos, 6)
            total_cost += sq_dev.sum()

        # For 7-vertex subsets
        if len(combos_7) > 0:
            sub7 = A[np.ix_(combos_7.ravel(), combos_7.ravel())].reshape(len(combos_7), 7, 7)
            degs7 = sub7.sum(axis=1)
            means7 = degs7.mean(axis=1, keepdims=True)
            sq_dev7 = (degs7 - means7) ** 2
            total_cost += sq_dev7.sum()

        # For 8-vertex subsets
        if len(combos_8) > 0:
            sub8 = A[np.ix_(combos_8.ravel(), combos_8.ravel())].reshape(len(combos_8), 8, 8)
            degs8 = sub8.sum(axis=1)
            means8 = degs8.mean(axis=1, keepdims=True)
            sq_dev8 = (degs8 - means8) ** 2
            total_cost += sq_dev8.sum()

        return total_cost

    def is_valid_exact(A):
        """Exact check: no subset of size >= 6 induces a regular graph."""
        adj_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if A[i, j]:
                    adj_mask[i] |= (1 << j)

        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                if _is_regular_subset_fast(adj_mask, subset):
                    return False
        return True

    # Initial cost
    current_cost = compute_cost(A)
    best_cost = current_cost
    best_A = A.copy()

    # Simulated annealing / hill climbing on the cost landscape
    temperature = 10.0
    cooling = 0.98
    max_iterations = 5000

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break

        # Pick a random edge to flip
        u = rng_local.randrange(n)
        v = rng_local.randrange(n)
        if u == v:
            continue

        old_val = A[u, v]
        A[u, v] = A[v, u] = 1 - old_val

        new_cost = compute_cost(A)
        delta = new_cost - current_cost

        # Accept if improvement or with probability exp(-delta/T)
        if delta < 0 or (temperature > 0.1 and rng_local.random() < np.exp(-delta / temperature)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_A = A.copy()
        else:
            # Revert
            A[u, v] = A[v, u] = old_val

        temperature *= cooling

        # Check if we've achieved zero cost (valid graph)
        if best_cost < 1e-9:
            if is_valid_exact(best_A):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if best_A[i, j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}

        # Periodic exact check if cost is very low
        if iteration % 200 == 0 and current_cost < 5.0:
            if is_valid_exact(A):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if A[i, j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}

    # Final check with best found
    if best_cost < 1e-9 and is_valid_exact(best_A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_A[i, j]:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    return None


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

    degrees = []
    for v in subset:
        d = bin(adj_mask[v] & subset_mask).count('1')
        if adj_mask[v] & (1 << v):
            d -= 1
        degrees.append(d)

    d0 = degrees[0]
    for i in range(1, s):
        if degrees[i] != d0:
            return False
    return True
# EVOLVE-BLOCK-END
