"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _evaluate_fast(nbrs, scores):
    """Fast evaluation using precomputed neighbor lists."""
    n = len(nbrs)
    if n == 0:
        return 0
    indices = np.argsort(-scores, kind='stable')
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in indices:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
    return count


def _fiedler_fast(L, n, iters=200):
    """Approximate Fiedler vector via power iteration on shifted Laplacian."""
    # Shift so that lambda_1=0 maps to largest eigenvalue
    # Use (cI - L) where c is slightly larger than lambda_max
    # For ER graphs, lambda_max ~ 2*avg_degree + 2*sqrt(avg_degree)
    # Simpler: use (I - L/lambda_max_est) and power iterate
    # Actually, for second-smallest, use (cI - L)^k with c > lambda_max
    # But we want the vector for lambda_1 (second smallest).
    # Alternative: deflation approach
    # Simplest: use scipy if available, else power method on (cI-L) with deflation
    
    # Estimate lambda_max
    # For Laplacian, lambda_max <= 2*max_degree
    max_deg = int(np.max(np.sum(L, axis=1)))
    c = 2.0 * max_deg + 1.0
    
    M = c * np.eye(n) - L  # This has eigenvalues c - lambda_i, so c - lambda_0 = c is largest
    # We want the eigenvector for lambda_1 of L, which corresponds to c - lambda_1 of M
    # That's the second-largest eigenvalue of M.
    # Use power iteration with deflation for the top eigenvector
    
    # First, get the dominant eigenvector (corresponds to lambda_0=0 of L, i.e., all-ones)
    # The top eigenvector of M should be close to all-ones
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    
    for _ in range(iters):
        w = M @ v
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-10:
            break
        v = w / norm_w
    
    # Deflate: remove component along v
    # Now power iterate on M with the deflated space
    # Actually, simpler: just use the fact that for connected graphs,
    # the Fiedler vector is orthogonal to all-ones.
    # Let's just do: start with random vector orthogonal to ones, power iterate on (cI-L)
    # but project out the all-ones component each step.
    
    ones = np.ones(n) / np.sqrt(n)
    v = np.random.randn(n)
    v = v - np.dot(v, ones) * ones  # orthogonalize
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        v = np.random.randn(n)
        v = v - np.dot(v, ones) * ones
        norm_v = np.linalg.norm(v)
    v /= norm_v
    
    for _ in range(iters):
        w = M @ v
        w = w - np.dot(w, ones) * ones  # project out all-ones
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-10:
            break
        v = w / norm_w
    
    return v


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Uses a combination of local degree, global centrality (Fiedler vector),
    and local structural features to better approximate which vertices to pick.
    """
    deadline = time.time() + 145  # Safety margin before 150s kill
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])

    # Precompute neighbor lists for fast evaluation
    nbrs = [np.where(adjacency[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)

    # --- Feature Computation ---
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    # Degree
    D_n = norm(degree)

    # Mean neighbor degree
    mean_nd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            mean_nd[i] = np.mean(degree[nbrs[i]])
        else:
            mean_nd[i] = 0
    MeanND_n = norm(mean_nd)

    # Sum of neighbor degrees
    snd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            snd[i] = np.sum(degree[nbrs[i]])
    SND_n = norm(snd)

    # Triangles
    A = adjacency.astype(float)
    A_sq = A @ A
    tri = np.diag(A_sq) / 2.0
    Tri_n = norm(tri)

    # Clustering coefficient
    clus = np.zeros(n)
    mask = degree > 1
    clus[mask] = 2.0 * tri[mask] / (degree[mask] * (degree[mask] - 1.0))
    Clus_n = norm(clus)

    # Fiedler vector (second-smallest eigenvector of Laplacian)
    try:
        D_mat = np.diag(degree)
        L = D_mat - A
        fiedler = _fiedler_fast(L, n, iters=300)
        abs_fiedler = np.abs(fiedler)
        AbsFiedler_n = norm(abs_fiedler)
    except Exception:
        AbsFiedler_n = np.zeros(n)

    # Core numbers (simplified k-core)
    core = np.zeros(n, dtype=int)
    deg_copy = degree.astype(int).copy()
    remaining = np.ones(n, dtype=bool)
    max_deg = int(deg_copy.max()) if n > 0 else 0
    for k in range(max_deg, -1, -1):
        for i in range(n):
            if remaining[i] and deg_copy[i] >= k:
                core[i] = k
                remaining[i] = False
                for j in nbrs[i]:
                    if remaining[j]:
                        deg_copy[j] -= 1
    Core_n = norm(core.astype(float))

    # --- Initial Score Candidates ---
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 1.0),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 1.0),
        (-0.7, -0.2, 0.0, 0.0, 0.0, -0.2, 0.8),
        (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
        (-1.0, -0.3, -0.2, 0.0, 0.0, -0.3, 0.5),
        (-0.9, -0.4, -0.3, -0.2, -0.2, -0.4, 0.8),
        (-0.8, -0.4, -0.4, -0.3, -0.3, -0.4, 0.7),
        (-1.0, -0.6, 0.0, 0.0, 0.0, -0.6, 0.6),
        (-0.9, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5),
    ]

    best_count = 0
    best_scores = -D_n.copy()
    best_weights = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for weights in candidates:
        if time.time() > deadline:
            break
        w = np.array(weights)
        scores = (w[0] * D_n + w[1] * MeanND_n + w[2] * SND_n +
                  w[3] * Tri_n + w[4] * Clus_n + w[5] * Core_n +
                  w[6] * AbsFiedler_n)
        count = _evaluate_fast(nbrs, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()
            best_weights = w.copy()

    # --- Local Polish: Hill Climbing on Score Weights ---
    if time.time() < deadline - 15:
        base_weights = best_weights.copy()
        for _ in range(100):
            if time.time() > deadline:
                break
            perturbation = np.random.normal(0, 0.1, 7)
            new_weights = base_weights + perturbation
            scores = (new_weights[0] * D_n + new_weights[1] * MeanND_n +
                      new_weights[2] * SND_n + new_weights[3] * Tri_n +
                      new_weights[4] * Clus_n + new_weights[5] * Core_n +
                      new_weights[6] * AbsFiedler_n)
            count = _evaluate_fast(nbrs, scores)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                best_weights = new_weights.copy()
                base_weights = new_weights.copy()
            elif np.random.random() < 0.05:
                base_weights = best_weights.copy()

    # --- Final Polish: Targeted Swap Improvement ---
    # Instead of all-pairs, try swapping each vertex with its top-scoring neighbors
    if time.time() < deadline - 5:
        scores = best_scores.copy()
        order = np.argsort(-scores, kind='stable')
        for sweep in range(30):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                # Try swapping vertex i with a few high-score vertices
                # Pick up to 5 candidates from the top of the ordering
                for j in order[:min(10, n)]:
                    if j == i:
                        continue
                    scores[i], scores[j] = scores[j], scores[i]
                    count = _evaluate_fast(nbrs, scores)
                    if count > best_count:
                        best_count = count
                        best_scores = scores.copy()
                        improved = True
                        # Update order
                        order = np.argsort(-scores, kind='stable')
                    else:
                        scores[i], scores[j] = scores[j], scores[i]
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if not improved:
                break
        scores = best_scores

    return best_scores
# EVOLVE-BLOCK-END
