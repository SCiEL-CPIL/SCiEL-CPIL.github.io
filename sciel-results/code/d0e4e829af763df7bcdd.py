"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    picked = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if forbidden[v]:
            continue
        picked[v] = True
        count += 1
        # Mark neighbours as forbidden
        neighbors = np.where(adjacency[v])[0]
        forbidden[neighbors] = True
    return count


def _evaluate_fast(adjacency, nbrs, scores):
    """Fast evaluation using precomputed neighbor lists."""
    n = len(nbrs)
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


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Uses a combination of local degree, hub-avoidance (inverse neighbor degree),
    and spectral features (Fiedler vector) to better approximate which vertices to pick.
    """
    deadline = time.time() + 135  # Safety margin before 150s kill
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

    # Mean neighbor degree (high means connected to hubs)
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
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
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
    # Strategy: prefer low degree, low core, high |Fiedler| (peripheral nodes)
    # Also penalize high mean neighbor degree (hub avoidance)
    
    candidates = [
        # (weight_D, weight_MeanND, weight_SND, weight_Tri, weight_Clus, weight_Core, weight_AbsFiedler)
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, -0.5, 1.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-0.8, -0.3, 0.0, 0.0, 0.0, -0.3, 1.0),
        (-0.7, -0.2, 0.0, 0.0, 0.0, -0.2, 0.8),
        (-1.0, -0.3, -0.2, 0.0, 0.0, -0.3, 0.5),
        (-1.0, -0.5, -0.3, 0.0, 0.0, -0.5, 0.8),
        (-1.0, -0.7, 0.0, 0.0, 0.0, 0.0, 0.5),
    ]

    best_count = 0
    best_scores = -D_n.copy()

    for weights in candidates:
        if time.time() > deadline:
            break
        w_D, w_MeanND, w_SND, w_Tri, w_Clus, w_Core, w_AbsFiedler = weights
        scores = (w_D * D_n + w_MeanND * MeanND_n + w_SND * SND_n +
                  w_Tri * Tri_n + w_Clus * Clus_n + w_Core * Core_n +
                  w_AbsFiedler * AbsFiedler_n)
        count = _evaluate_fast(adjacency, nbrs, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # --- Local Polish: Hill Climbing on Score Weights ---
    # We can adjust the weights slightly to improve the result
    if time.time() < deadline - 10:
        # Start from the best candidate found
        best_w = None
        for weights in candidates:
            w_D, w_MeanND, w_SND, w_Tri, w_Clus, w_Core, w_AbsFiedler = weights
            scores = (w_D * D_n + w_MeanND * MeanND_n + w_SND * SND_n +
                      w_Tri * Tri_n + w_Clus * Clus_n + w_Core * Core_n +
                      w_AbsFiedler * AbsFiedler_n)
            count = _evaluate_fast(adjacency, nbrs, scores)
            if count == best_count:
                best_w = weights
                break
        if best_w is None:
            best_w = (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        base_weights = np.array(best_w)
        # Try small perturbations around the best candidate
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            # Random perturbation
            perturbation = np.random.normal(0, 0.1, 7)
            new_weights = base_weights + perturbation
            scores = (new_weights[0] * D_n + new_weights[1] * MeanND_n +
                      new_weights[2] * SND_n + new_weights[3] * Tri_n +
                      new_weights[4] * Clus_n + new_weights[5] * Core_n +
                      new_weights[6] * AbsFiedler_n)
            count = _evaluate_fast(adjacency, nbrs, scores)
            if count > best_count:
                best_count = count
                best_scores = scores.copy()
                base_weights = new_weights
                improved = True
            if not improved and np.random.random() < 0.1:
                # Occasionally reset to a good baseline
                base_weights = np.array(best_w)

    # --- Final Polish: Swap-based Improvement ---
    if time.time() < deadline - 5:
        scores = best_scores.copy()
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    scores[i], scores[j] = scores[j], scores[i]
                    count = _evaluate_fast(adjacency, nbrs, scores)
                    if count > best_count:
                        best_count = count
                        best_scores = scores.copy()
                        improved = True
                    else:
                        scores[i], scores[j] = scores[j], scores[i]
                if time.time() > deadline:
                    break
            if not improved:
                break

    return best_scores
# EVOLVE-BLOCK-END
