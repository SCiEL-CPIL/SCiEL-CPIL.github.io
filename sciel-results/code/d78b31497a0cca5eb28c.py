import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s/150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)
    A_f = A.astype(np.float64)

    deg = A.sum(axis=1).astype(np.float64)
    has_edge = deg > 0
    BIG = 1e9

    # --- Structural features (from inspiration 1) ---
    # 1. Degree
    # 2. Mean neighbor degree
    sum_neighbor_deg = A_f @ deg
    mean_neighbor_deg = np.zeros(n, dtype=np.float64)
    mask = deg > 0
    mean_neighbor_deg[mask] = sum_neighbor_deg[mask] / deg[mask]

    # 3. Sum of inverse neighbor degrees (lower = neighbors are hubs)
    inv_deg = 1.0 / (deg + 1.0)
    sum_inv_neighbor_deg = A_f @ inv_deg

    # 4. Exact triangle count via diag(A @ A)
    A2 = A_f @ A_f
    tri_counts = np.diag(A2) / 2.0

    # 5. Second-neighbor degree sum (captures 2-hop structure)
    second_nbr_deg = A2 @ deg

    def zscore(x):
        x = x.astype(np.float64)
        m = x.mean()
        s = x.std()
        if s < 1e-9:
            return np.zeros_like(x)
        return (x - m) / s

    z_deg = zscore(deg)
    z_mean_nbr = zscore(mean_neighbor_deg)
    z_sum_inv = zscore(sum_inv_neighbor_deg)
    z_tri = zscore(tri_counts)
    z_second = zscore(second_nbr_deg)

    # --- Fixed-point safety score (from current program, fewer iterations) ---
    # s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    # Vertices whose neighbors have low total score are "safer" to pick
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG

    alpha = 0.5
    for it in range(80):
        if time.time() > deadline - 15.0:
            break
        nbr_sum = A_f @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        damped_s = (1 - alpha) * s + alpha * new_s
        damped_s[~has_edge] = BIG
        if np.allclose(damped_s, s, rtol=1e-6, atol=1e-8):
            s = damped_s
            break
        s = damped_s

    z_safety = zscore(s)

    # --- Spectral refinement: Fiedler vector (periphery = safer) ---
    fiedler_score = np.zeros(n)
    try:
        if time.time() < deadline - 10.0 and n <= 500:
            D = np.diag(deg)
            L = D - A_f
            eigvals, eigvecs = np.linalg.eigh(L)
            positive_mask = eigvals > 1e-10
            if positive_mask.any():
                idx = np.where(positive_mask)[0][0]
                fiedler = eigvecs[:, idx]
                fiedler_score = np.abs(fiedler)
                fmax = fiedler_score.max()
                if fmax > 1e-12:
                    fiedler_score = fiedler_score / fmax
    except Exception:
        pass

    z_fiedler = zscore(fiedler_score)

    # --- Combine all features with tuned weights ---
    # Strategy: reward low degree, high neighbor degree (block hubs),
    # low triangle count (less local redundancy), high safety, high periphery
    w_deg = 1.0
    w_mean_nbr = 0.6
    w_sum_inv = 0.4
    w_tri = 0.3
    w_safety = 0.8
    w_fiedler = 0.2
    w_second = 0.15

    scores = (-w_deg * z_deg
              + w_mean_nbr * z_mean_nbr
              - w_sum_inv * z_sum_inv
              - w_tri * z_tri
              + w_safety * z_safety
              + w_fiedler * z_fiedler
              - w_second * z_second)

    # Isolated vertices: always safe to pick
    scores[~has_edge] = BIG

    # --- Light greedy simulation for value-based refinement (from inspiration 3) ---
    if time.time() < deadline - 5.0:
        try:
            # Quick greedy run to estimate which vertices are "critical"
            order = np.argsort(-scores)
            selected = np.zeros(n, dtype=bool)
            remaining = np.ones(n, dtype=bool)
            for v in order:
                if not remaining[v]:
                    continue
                selected[v] = True
                remaining[v] = False
                remaining[A[v]] = False

            # Vertices that were selected get a small boost (they're "useful")
            # Vertices that blocked many others get a small penalty
            blocked_count = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if has_edge[i]:
                    blocked_count[i] = A[i].sum()

            z_blocked = zscore(blocked_count)
            # Small penalty for high blocked count (hubs block too much)
            scores = scores - 0.1 * z_blocked
            scores[~has_edge] = BIG
        except Exception:
            pass

    # Final: ensure isolated vertices stay at top
    scores[~has_edge] = BIG

    # Tiny tie-breaker: lower degree slightly preferred
    if deg.max() > 0:
        scores = scores + 1e-6 * (-deg / deg.max())

    if time.time() > deadline:
        return scores

    return scores
