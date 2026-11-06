"""Combined program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency, order):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            neighbors = adjacency[v].astype(bool)
            available[neighbors] = False
    return count


def _score_order(adjacency, priority_vec):
    """Evaluate a priority vector by running the greedy decoder."""
    n = len(priority_vec)
    order = np.argsort(-priority_vec, kind='stable')
    return _greedy_size(adjacency, order)


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    A_float = adjacency.astype(float)

    # Precompute degrees
    deg = A.sum(axis=1).astype(float)

    # Compute neighborhood edge counts for each vertex
    # E_N(v) = number of edges among neighbors of v
    C = np.zeros(n)
    for v in range(n):
        if deg[v] < 2:
            C[v] = 0.0
        else:
            nbrs = np.where(A[v])[0]
            if len(nbrs) > 0:
                sub = A[np.ix_(nbrs, nbrs)]
                C[v] = sub.sum() / 2.0

    # Normalize C to [0, 1] range if possible
    max_C = C.max()
    if max_C > 0:
        C_norm = C / max_C
    else:
        C_norm = C

    # Density: 2*C / (d*(d-1))
    density = np.zeros(n)
    mask = deg >= 2
    density[mask] = 2.0 * C[mask] / (deg[mask] * (deg[mask] - 1.0))

    # Mean neighbor degree
    neighbor_deg_sum = A @ deg
    mean_neighbor_deg = np.zeros(n)
    mask_pos = deg > 0
    mean_neighbor_deg[mask_pos] = neighbor_deg_sum[mask_pos] / deg[mask_pos]

    # Build multiple candidate scoring strategies
    # Base: negative degree
    base_score = -deg.copy()

    # Strategy 1: Base + neighborhood sparsity bonus (from current)
    score1 = base_score - 0.5 * C_norm

    # Strategy 2: Pure degree (from current)
    score2 = base_score.copy()

    # Strategy 3: Degree adjusted by neighborhood density (from current)
    score3 = base_score - density

    # Strategy 4: Inverse of (degree + neighborhood_edges) (from current)
    score4 = -1.0 / (deg + C + 1e-9)

    # Strategy 5: From inspiration 2: -deg + lambda * density (dense neighborhoods preferred)
    lambda_val = 1.5
    score5 = -deg + lambda_val * density

    # Strategy 6: From inspiration 3: -deg + alpha * mean_neighbor_deg
    alpha = 0.1
    score6 = -deg + alpha * mean_neighbor_deg

    # Strategy 7: Combined: -deg + lambda*density + alpha*mean_neighbor_deg
    score7 = -deg + lambda_val * density + alpha * mean_neighbor_deg

    # Strategy 8: -deg + 2.0*density (stronger density weight)
    score8 = -deg + 2.0 * density

    # Strategy 9: -deg + 1.0*density
    score9 = -deg + 1.0 * density

    # Strategy 10: -deg - 0.3*C_norm + 0.1*mean_neighbor_deg
    score10 = -deg - 0.3 * C_norm + 0.1 * mean_neighbor_deg

    candidates = [score1, score2, score3, score4, score5, score6, score7, score8, score9, score10]

    best_score = -1
    best_vec = base_score.copy()

    for s in candidates:
        if time.time() > deadline:
            break
        size = _score_order(adjacency, s)
        if size > best_score:
            best_score = size
            best_vec = s.copy()

    # Try some random perturbations of the best strategy
    rng = np.random.default_rng(42)

    for _ in range(50):
        if time.time() > deadline:
            break
        noise = rng.normal(0, 0.1, size=n)
        blended = best_vec + noise
        size = _score_order(adjacency, blended)
        if size > best_score:
            best_score = size
            best_vec = blended.copy()

    # Local search: swap pairs of vertices in the ordering
    if time.time() < deadline - 10.0:
        order = np.argsort(-best_vec, kind='stable')
        current_size = best_score

        # Try random swaps
        for _ in range(200):
            if time.time() > deadline:
                break

            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue

            order[i], order[j] = order[j], order[i]

            new_size = _greedy_size(adjacency, order)

            if new_size > current_size:
                current_size = new_size
                new_priority = np.zeros(n)
                for rank, v in enumerate(order):
                    new_priority[v] = n - rank
                best_vec = new_priority
            else:
                order[i], order[j] = order[j], order[i]

    return best_vec.astype(float)
# EVOLVE-BLOCK-END
