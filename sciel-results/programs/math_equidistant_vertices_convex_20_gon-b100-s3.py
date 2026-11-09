# EVOLVE-BLOCK-START
import numpy as np
import time

def _score_polygon(polygon):
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    # Normalize
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_d = dists_to_centroid.mean()
    if mean_d <= 0.01:
        return 0.0
    centered = centered / mean_d
    
    # Check convexity: no repeated points, all turns same sign, total turn = ±2π
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(centered[i] - centered[j]) < 1e-10:
                return 0.0
    
    # Compute cross products for turn direction
    cross_products = []
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        cross_products.append(cross)
    
    # All must be nonzero and same sign
    nonzero = [c for c in cross_products if abs(c) > 1e-12]
    if len(nonzero) != n:
        return 0.0
    if any(c > 0 for c in nonzero) and any(c < 0 for c in nonzero):
        return 0.0
    
    # Check total turning angle
    angles = []
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        if norm_e1 < 1e-12 or norm_e2 < 1e-12:
            return 0.0
        cos_angle = np.dot(e1, e2) / (norm_e1 * norm_e2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if cross_products[i] > 0:
            angle = angle  # already positive for left turn
        else:
            angle = -angle
        angles.append(angle)
    
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = centered[i]
        # Distances to all other vertices
        d = np.linalg.norm(centered - v, axis=1)
        d[i] = np.inf  # exclude self
        d_sorted = np.sort(d)[:19]
        
        # Check if any distance < 0.05
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        
        # Select four distances
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        min_score = min(min_score, vertex_score)
    
    return min_score


def _fast_score(polygon):
    """Fast scoring that skips some convexity checks for speed during search."""
    n = len(polygon)
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_d = dists_to_centroid.mean()
    if mean_d <= 0.01:
        return 0.0
    centered = centered / mean_d
    
    # Quick convexity check using cross products
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross_products[i] = e1[0]*e2[1] - e1[1]*e2[0]
    
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = centered[i]
        d = np.linalg.norm(centered - v, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)[:19]
        
        if d_sorted[0] < 0.05:
            return 0.0
        
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            return 0.0
        
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        min_score = min(min_score, vertex_score)
        if min_score < 0.5:  # Early exit if score is already low
            return min_score
    
    return min_score


def _generate_candidate(rng, strategy, base_angles, n=20):
    """Generate a candidate polygon using the given strategy."""
    if strategy == 0:
        # Regular polygon
        angles = base_angles.copy()
        radii = np.ones(n)
        
    elif strategy == 1:
        # Small angular perturbations
        noise = rng.normal(0, 0.03, n)
        angles = base_angles + noise
        angles = np.sort(angles)
        # Ensure angles are in [0, 2π)
        while angles[0] < 0:
            angles[0] += 2*np.pi
        while angles[-1] >= 2*np.pi:
            angles[-1] -= 2*np.pi
        angles = np.sort(angles)
        radii = np.ones(n)
        
    elif strategy == 2:
        # Radial perturbations
        angles = base_angles.copy()
        radii = 1.0 + rng.normal(0, 0.08, n)
        radii = np.clip(radii, 0.6, 1.4)
        
    elif strategy == 3:
        # Fourier perturbations
        angles = base_angles.copy()
        radii = np.ones(n)
        for harmonic in [2, 3, 4, 5, 6]:
            amp = rng.uniform(0.005, 0.04)
            phase = rng.uniform(0, 2*np.pi)
            radii += amp * np.cos(harmonic * angles + phase)
        
    elif strategy == 4:
        # Sine-based radial modulation
        angles = base_angles.copy()
        freq = rng.integers(2, 8)
        amp = rng.uniform(0.02, 0.1)
        phase = rng.uniform(0, 2*np.pi)
        radii = 1.0 + amp * np.sin(freq * angles + phase)
        
    elif strategy == 5:
        # Random convex polygon
        angles = np.sort(rng.uniform(0, 2*np.pi, n))
        radii = rng.uniform(0.7, 1.3, n)
        
    else:
        # Mixed perturbations
        angles = base_angles + rng.normal(0, 0.02, n)
        angles = np.sort(angles)
        while angles[0] < 0:
            angles[0] += 2*np.pi
        while angles[-1] >= 2*np.pi:
            angles[-1] -= 2*np.pi
        angles = np.sort(angles)
        radii = 1.0 + rng.normal(0, 0.05, n)
        radii = np.clip(radii, 0.6, 1.4)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _refine_polygon(polygon, rng, max_iter=500, base_step=0.01):
    """Local refinement using gradient-free optimization."""
    n = len(polygon)
    best = polygon.copy()
    best_score = _fast_score(best)
    
    for it in range(max_iter):
        # Adaptive step size
        step = base_step * (0.95 ** (it // 50))
        
        # Try perturbations
        improved = False
        for _ in range(10):
            noise = rng.normal(0, step, (n, 2))
            candidate = best + noise
            score = _fast_score(candidate)
            if score > best_score:
                best = candidate.copy()
                best_score = score
                improved = True
                break
        
        if not improved and step < 0.001:
            break
    
    return best, best_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Search for a convex 20-gon that maximizes the minimum vertex equidistance score.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Leave 50s safety margin
    
    n = 20
    best_polygon = None
    best_score = -1.0
    
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Phase 1: Broad exploration
    rng = np.random.default_rng(12345)
    
    # Start with regular polygon
    polygon = _generate_candidate(rng, 0, base_angles)
    score = _score_polygon(polygon)
    if score > best_score:
        best_score = score
        best_polygon = polygon.copy()
    
    # Try various strategies
    num_exploration = 2000
    for i in range(num_exploration):
        if time.time() > deadline - 100.0:
            break
        
        strategy = i % 7
        polygon = _generate_candidate(rng, strategy, base_angles)
        score = _fast_score(polygon)
        if score > best_score:
            best_score = score
            best_polygon = polygon.copy()
    
    # Phase 2: Local refinement around best candidates
    if best_polygon is not None and time.time() < deadline - 50.0:
        # Refine the best polygon
        refined, refined_score = _refine_polygon(best_polygon, rng, max_iter=300, base_step=0.015)
        if refined_score > best_score:
            best_score = refined_score
            best_polygon = refined.copy()
    
    # Phase 3: More diverse candidates and refinement
    if time.time() < deadline - 30.0:
        for i in range(500):
            if time.time() > deadline - 30.0:
                break
            
            # Generate new candidate
            strategy = i % 7
            polygon = _generate_candidate(rng, strategy, base_angles)
            score = _fast_score(polygon)
            
            if score > best_score:
                best_score = score
                best_polygon = polygon.copy()
            
            # If score is promising, refine it
            if score > best_score * 0.8 and time.time() < deadline - 35.0:
                refined, refined_score = _refine_polygon(polygon, rng, max_iter=100, base_step=0.01)
                if refined_score > best_score:
                    best_score = refined_score
                    best_polygon = refined.copy()
    
    # Phase 4: Final refinement
    if best_polygon is not None and time.time() < deadline - 10.0:
        refined, refined_score = _refine_polygon(best_polygon, rng, max_iter=200, base_step=0.008)
        if refined_score > best_score:
            best_score = refined_score
            best_polygon = refined.copy()
    
    # Verify final score with full scoring
    if best_polygon is not None:
        final_score = _score_polygon(best_polygon)
        if final_score > 0:
            return best_polygon
    
    # Fallback to regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)


# EVOLVE-BLOCK-END
