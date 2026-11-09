# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon with equidistant vertices (in the sense of the scoring metric).
    Uses multi-start L-BFGS-B optimization to maximize the minimum vertex score.
    """
    n = 20
    deadline = time.time() + 580.0  # safety margin before 630s limit

    def compute_scores(polygon):
        """Compute the per-vertex scores for a given polygon (20, 2)."""
        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = polygon.mean(axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return np.zeros(n)
        normalized = centered / mean_dist

        # Check for repeated points
        min_pairwise = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(normalized[i] - normalized[j])
                if d < min_pairwise:
                    min_pairwise = d
        if min_pairwise < 0.05:
            # Some pair of points too close; individual vertex scores may be 0
            pass

        # Compute all pairwise distances
        # dists[i, j] = distance from vertex i to vertex j
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dists[i, j] = np.linalg.norm(normalized[i] - normalized[j])

        scores = np.zeros(n)
        for i in range(n):
            # Sort the 19 distances from vertex i to others
            d_sorted = np.sort(dists[i, :])
            # If any distance < 0.05, score is 0
            if d_sorted[0] < 0.05:
                scores[i] = 0.0
                continue

            # Find k = first index minimizing d_{k+1} - d_k
            gaps = d_sorted[1:] - d_sorted[:-1]
            k = int(np.argmin(gaps))

            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            if D < 0.1:
                scores[i] = 0.0
                continue

            # Select four distances
            if k == 0:
                four_d = d_sorted[0:4]
            elif k == 17:
                four_d = d_sorted[15:19]
            else:
                four_d = d_sorted[k-1:k+3]

            # raw = mean of max(d/D, D/d) over the four distances
            ratios = np.maximum(four_d / D, D / four_d)
            raw = np.mean(ratios)
            scores[i] = min(1.0, 1.0 / raw)

        return scores

    def negative_min_score(polygon):
        """Objective: negative of the minimum vertex score (to minimize)."""
        scores = compute_scores(polygon)
        return -np.min(scores)

    def is_valid_polygon(polygon):
        """Check if polygon is strictly convex with no repeated points."""
        # No repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(polygon[i] - polygon[j]) < 1e-6:
                    return False

        # Check convexity: all cross products of consecutive edges have same sign
        cross_products = []
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i+1) % n]
            p2 = polygon[(i+2) % n]
            e1 = p1 - p0
            e2 = p2 - p1
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            cross_products.append(cross)

        cross_products = np.array(cross_products)
        # All must be nonzero and same sign
        if np.any(np.abs(cross_products) < 1e-8):
            return False
        if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
            return False

        return True

    # Initial guess: regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    best_polygon = init.copy()
    best_score = compute_scores(init).min()

    # Multi-start optimization
    n_restarts = 10
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            x0 = init.copy()
        else:
            # Perturb the regular polygon
            x0 = init.copy()
            for i in range(n):
                # Random angular perturbation
                angle = angles[i] + rng.uniform(-0.3, 0.3)
                radius = 1.0 + rng.uniform(-0.2, 0.2)
                x0[i, 0] = radius * np.cos(angle)
                x0[i, 1] = radius * np.sin(angle)
            # Ensure roughly circular order by sorting by angle
            ang = np.arctan2(x0[:, 1], x0[:, 0])
            order = np.argsort(ang)
            x0 = x0[order]

        try:
            result = minimize(
                negative_min_score,
                x0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() > deadline:
                break

            candidate = result.x.reshape(n, 2)
            if is_valid_polygon(candidate):
                score = compute_scores(candidate).min()
                if score > best_score:
                    best_score = score
                    best_polygon = candidate.copy()
        except Exception:
            continue

    # Final validation and normalization
    if not is_valid_polygon(best_polygon):
        # Fallback to regular polygon
        best_polygon = init

    return best_polygon


# EVOLVE-BLOCK-END
