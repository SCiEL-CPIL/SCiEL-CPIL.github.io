"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _build_points(radii, phases, ring_counts):
    """
    Build point array from ring radii, phases, and counts.
    """
    points = []
    for i, (r, phase, count) in enumerate(zip(radii, phases, ring_counts)):
        if count == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + phase
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points.append(np.column_stack([xs, ys]))
    
    if not points:
        return np.zeros((0, 2))
    
    return np.vstack(points)


def _compute_radius(points):
    """Compute the largest common radius for the given points."""
    if len(points) == 0:
        return 0.0
    if len(points) == 1:
        return 1.0 - np.hypot(*points[0])
    
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)


def _objective(params, ring_counts):
    """
    Objective function: negative of the packing radius.
    """
    k = len(ring_counts)
    radii = params[:k]
    phases = params[k:]
    
    radii = np.maximum(radii, 0.0)
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def _generate_ring_candidates():
    """
    Generate a list of candidate ring count arrays.
    We focus on hexagonal-like structures but vary the counts slightly.
    """
    candidates = []
    
    # Standard hexagonal: 1, 6, 12, ...
    # Sum of first k hexagonal rings: k^2
    # We need sum = 300. 17^2 = 289, 18^2 = 324.
    # So we have 17 full rings + 11 extra in the 18th?
    # 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96
    # Sum 1..16 rings (16^2=256) + 17th ring (102) = 358. Too many.
    # Let's list them:
    # Ring 1: 1 (Sum 1)
    # Ring 2: 6 (Sum 7)
    # Ring 3: 12 (Sum 19)
    # Ring 4: 18 (Sum 37)
    # Ring 5: 24 (Sum 61)
    # Ring 6: 30 (Sum 91)
    # Ring 7: 36 (Sum 127)
    # Ring 8: 42 (Sum 169)
    # Ring 9: 48 (Sum 217)
    # Ring 10: 54 (Sum 271)
    # Ring 11: 60 (Sum 331) -> Too many.
    # So we need 10 full rings (271) + 29 in the 11th ring.
    # Candidate 1: [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
    
    c1 = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    candidates.append(c1)
    
    # Variation: Move some circles from outer ring to inner rings to balance?
    # Or try 11 rings with different distribution.
    
    # Candidate 2: Slightly different distribution
    # Try to make the last ring more full and reduce previous?
    # If we take 1 from ring 10 (54->53) and put in ring 11 (29->30)
    c2 = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 53, 30])
    candidates.append(c2)
    
    # Candidate 3: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29 is standard.
    # What about 12 rings?
    # Sum 1..11 (331) is too high.
    # So max 11 rings if we stick to hexagonal growth.
    
    # Let's try a structure with fewer, denser rings?
    # E.g. 1, 8, 16, 24, 32, 40, 48, 56, 64, 111? No, too uneven.
    
    # Let's add a random perturbation to the standard counts.
    rng = np.random.default_rng(42)
    for _ in range(5):
        c = c1.copy()
        # Perturb inner rings (indices 1-8)
        for i in range(1, 9):
            delta = rng.integers(-2, 3)
            if c[i] + delta > 0 and c[-1] - delta > 0:
                c[i] += delta
                c[-1] -= delta
        if sum(c) == 300 and np.all(c > 0):
            candidates.append(c)
            
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    return unique_candidates


def _initial_guess(ring_counts, jitter_seed=0):
    """Generate initial guess for radii and phases."""
    rng = np.random.default_rng(jitter_seed)
    k = len(ring_counts)
    
    # Estimate optimal radius r based on area: N * pi * r^2 <= pi * 1^2 => r <= 1/sqrt(N)
    # But packing efficiency is ~0.9, so r ~ 0.95 / sqrt(N)
    r_target = 0.95 / np.sqrt(sum(ring_counts))
    
    radii = np.zeros(k)
    for i, n in enumerate(ring_counts):
        if n > 1:
            # Chord length constraint: 2 * r_i * sin(pi/n) >= 2 * r_target
            r_min_ring = r_target / np.sin(np.pi / n)
        else:
            r_min_ring = 0.0
        
        # Radial spacing constraint: r_i >= r_0 + 2*i*r_target
        # Assuming r_0=0 for center
        r_min_adj = 2.0 * i * r_target
        radii[i] = max(r_min_ring, r_min_adj)
    
    # Scale to fit in unit circle
    if radii[-1] > 0:
        scale = (1.0 - r_target) / radii[-1]
        if scale < 1.0:
            radii *= scale
    
    # Jitter
    radii = radii + rng.uniform(-0.005, 0.005, size=k)
    radii = np.maximum(radii, 0.0)
    
    # Phases
    phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i] > 0 and ring_counts[i-1] > 0:
            phases[i] = np.pi / ring_counts[i-1] + rng.uniform(-0.05, 0.05)
    
    return np.concatenate([radii, phases])


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    
    candidates = _generate_ring_candidates()
    
    # Limit number of candidates to ensure time constraint
    # 120 seconds total. Let's allow ~10 seconds per candidate.
    # 12 candidates max.
    candidates = candidates[:12]
    
    for idx, ring_counts in enumerate(candidates):
        num_rings = len(ring_counts)
        
        # Try a few initial guesses for each ring structure
        for seed in range(3):
            x0 = _initial_guess(ring_counts, jitter_seed=seed)
            
            # Bounds
            bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
            
            # Use Powell for better handling of non-smoothness and no gradient needed
            # Nelder-Mead is also good. Let's use Nelder-Mead with adaptive.
            result = minimize(
                _objective, x0, args=(ring_counts,),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
            )
            
            radii = np.maximum(result.x[:num_rings], 0.0)
            phases = result.x[num_rings:]
            
            points = _build_points(radii, phases, ring_counts)
            
            if len(points) != N:
                continue
            
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_points = points
    
    if best_points is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points

# EVOLVE-BLOCK-END
