import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s / 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)
    A_float = A.astype(float)

    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    # Isolated vertices: always safe to pick -> top priority
    BIG = 1e9

    # Initialize with inverse degree
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG

    # Fixed-point iteration with adaptive damping for faster convergence
    # s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    # This captures "safety" — vertices whose neighbors have low total score are safer
    # Phase 1: aggressive damping for fast convergence
    # Phase 2: gentle damping for precision polish
    for it in range(200):
        if time.time() > deadline - 15.0:
            break
        if it < 100:
            alpha = 0.7  # aggressive for fast convergence
        else:
            alpha = 0.3  # gentle for precision
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        # Damped update for stability
        damped_s = (1 - alpha) * s + alpha * new_s
        damped_s[~has_edge] = BIG
        if np.allclose(damped_s, s, rtol=1e-6, atol=1e-8):
            s = damped_s
            break
        s = damped_s
    
    # Phase 3: high-precision polish with very tight tolerance
    for it in range(100):
        if time.time() > deadline - 5.0:
            break
        alpha = 0.1  # very gentle for precision
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        damped_s = (1 - alpha) * s + alpha * new_s
        damped_s[~has_edge] = BIG
        if np.allclose(damped_s, s, rtol=1e-9, atol=1e-11):
            s = damped_s
            break
        s = damped_s

    # Spectral refinement: use Fiedler vector magnitude as a secondary signal
    # Vertices at the periphery (high |Fiedler component|) are less central
    # and thus safer to include in an independent set
    try:
        if time.time() < deadline - 5.0:
            D = np.diag(deg)
            L = D - A_float
            
            if n <= 500:
                eigvals, eigvecs = np.linalg.eigh(L)
                positive_mask = eigvals > 1e-10
                if positive_mask.any():
                    idx = np.where(positive_mask)[0][0]
                    fiedler = eigvecs[:, idx]
                else:
                    fiedler = np.zeros(n)
                
                fiedler_score = np.abs(fiedler)
                
                fmax = fiedler_score.max()
                if fmax > 1e-12:
                    fiedler_score = fiedler_score / fmax
                else:
                    fiedler_score = np.zeros(n)
                
                # Adaptive spectral weight: stronger when base scores are close
                s_spread = s.max() - s.min()
                if s_spread > 1e-12:
                    spectral_weight = min(0.3, 0.1 + 0.2 * (1.0 / (1.0 + s_spread)))
                else:
                    spectral_weight = 0.3
                
                scores = s + spectral_weight * fiedler_score
            else:
                scores = s.copy()
        else:
            scores = s.copy()
    except Exception:
        scores = s.copy()

    # Local neighborhood refinement: vectorized contested penalty
    # Vertices adjacent to many high-scoring neighbors are more "contested"
    try:
        if time.time() < deadline - 2.0:
            # Vectorized: max neighbor score for each vertex
            # Use matrix multiplication with scores broadcast
            # For each vertex i, nbr_max[i] = max(scores[j] for j in neighbors(i))
            # Approximate using: A @ scores gives sum, but we want max
            # Use a soft-max approximation: log-sum-exp trick
            # Or simpler: just use the mean as a proxy (faster)
            nbr_sum_scores = A_float @ scores
            deg_safe = np.maximum(deg, 1.0)
            nbr_mean_scores = nbr_sum_scores / deg_safe
            
            # Also compute a "contestation" metric: how many neighbors have high scores
            # Normalize scores to [0,1] for this computation
            s_min = scores[has_edge].min() if has_edge.any() else 0
            s_max = scores[has_edge].max() if has_edge.any() else 1
            s_range = max(s_max - s_min, 1e-12)
            scores_norm = (scores - s_min) / s_range
            scores_norm[~has_edge] = 0
            
            # Count how many neighbors have normalized score > 0.5
            high_score_mask = (scores_norm > 0.5).astype(float)
            nbr_high_count = A_float @ high_score_mask
            
            # Penalty: vertices with many high-scoring neighbors are more contested
            max_high = nbr_high_count.max()
            if max_high > 1e-12:
                contested_penalty = nbr_high_count / max_high
                scores = scores - 0.03 * contested_penalty
            scores[~has_edge] = BIG
        else:
            scores[~has_edge] = BIG
    except Exception:
        scores[~has_edge] = BIG

    # Isolated vertices must remain at top
    scores[~has_edge] = BIG

    # Small tie-breaker: lower degree slightly preferred
    deg_rank = -deg / (deg.max() + 1) if deg.max() > 0 else 0.0
    scores = scores + 1e-6 * deg_rank

    if time.time() > deadline:
        return scores

    return scores
