"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def _build_points(radii, phases, ring_counts):
    """
    Build point array from ring radii, phases, and counts.
    Each ring i has count_i points at radius radii[i] with angular offset phases[i].
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
    
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances using KDTree for efficiency
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)  # k=2 includes self
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_boundary, min_pair)


def _objective(params, ring_counts):
    """
    Objective function: negative of the packing radius.
    params = [r0, r1, ..., r_{k-1}, phi0, phi1, ..., phi_{k-1}]
    First half are radii, second half are angular phases.
    """
    k = len(ring_counts)
    radii = params[:k]
    phases = params[k:]
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def _local_refine(points, ring_counts, max_iters=200, tol=1e-10):
    """
    Local refinement: adjust individual point positions to maximize minimum distance.
    Uses gradient-free local search on the full point set.
    """
    n = len(points)
    best_points = points.copy()
    best_r = _compute_radius(best_points)
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            # Try small perturbations in x and y
            for dx in [-0.005, 0.005]:
                for dy in [-0.005, 0.005]:
                    trial = best_points.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    # Check boundary
                    dist_origin = np.hypot(trial[i, 0], trial[i, 1])
                    if dist_origin > 1.0 - 0.01:  # rough check
                        continue
                    r_trial = _compute_radius(trial)
                    if r_trial > best_r + tol:
                        best_points = trial
                        best_r = r_trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_points, best_r


def _initial_guess(ring_counts, jitter_seed=0):
    """Generate initial guess for radii and phases."""
    rng = np.random.default_rng(jitter_seed)
    
    k = len(ring_counts)
    
    # Start with a reasonable guess for r
    r_target = 0.053  # close to best known
    
    radii = np.zeros(k)
    for i, n in enumerate(ring_counts):
        if n > 1:
            r_min_ring = r_target / np.sin(np.pi / n)
        else:
            r_min_ring = 0.0
        r_min_adj = 2.0 * i * r_target
        radii[i] = max(r_min_ring, r_min_adj)
    
    # Scale so outermost ring + r_target <= 1
    if radii[-1] > 0:
        scale = (1.0 - r_target) / radii[-1]
        if scale < 1.0:
            radii *= scale
    
    # Add small jitter
    radii = radii + rng.uniform(-0.005, 0.005, size=k)
    radii = np.maximum(radii, 0.0)
    
    # Phases: stagger adjacent rings for better packing
    phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i] > 0 and ring_counts[i-1] > 0:
            # Offset by half the angular spacing of the previous ring
            phases[i] = np.pi / ring_counts[i-1] + rng.uniform(-0.05, 0.05)
    
    return np.concatenate([radii, phases])


def _try_ring_structure(ring_counts, n_starts=3, maxiter_nm=3000):
    """Try to optimize a given ring structure."""
    num_rings = len(ring_counts)
    best_points = None
    best_r = 0.0
    
    for seed in range(n_starts):
        x0 = _initial_guess(ring_counts, jitter_seed=seed)
        
        # Bounds: radii in [0, 1], phases in [0, 2*pi]
        bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
        
        # Use Nelder-Mead
        result = minimize(
            _objective, x0, args=(ring_counts,),
            method='Nelder-Mead',
            options={'maxiter': maxiter_nm, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        
        radii = np.maximum(result.x[:num_rings], 0.0)
        phases = result.x[num_rings:]
        
        points = _build_points(radii, phases, ring_counts)
        
        if len(points) != N:
            continue
        
        # Local refinement
        points_refined, r_refined = _local_refine(points, ring_counts, max_iters=50)
        
        if r_refined > best_r:
            best_r = r_refined
            best_points = points_refined
    
    return best_points, best_r


# Define multiple ring structures to try
# Structure 1: Standard hexagonal-like
ring_counts_1 = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])

# Structure 2: Smoother density increase
ring_counts_2 = np.array([1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 51])
# Sum: 1+6+11+16+21+26+31+36+41+46+51+51 = 337... too many
# Let me recalculate: need sum = 300
# 1 + 6 + 11 + 16 + 21 + 26 + 31 + 36 + 41 + 46 = 235, remaining 65 for 2 rings
# Try: 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 32, 33 = 300
ring_counts_2 = np.array([1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 32, 33])

# Structure 3: Fewer rings, larger counts
ring_counts_3 = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
# Same as 1, let's try different
ring_counts_3 = np.array([1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 20])
# Sum: 1+7+13+19+25+31+37+43+49+55+20 = 300

# Structure 4: More rings, smaller counts
ring_counts_4 = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 44])
# Sum: 1+5+10+15+20+25+30+35+40+45+50+44 = 320... too many
# 1+5+10+15+20+25+30+35+40+45 = 226, remaining 74 for 2 rings: 37+37
ring_counts_4 = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 37, 37])

# Structure 5: Optimal-ish based on area
# Area per circle ~ pi*r^2, total area ~ 300*pi*r^2
# With r~0.053, total area ~ 300*pi*0.0028 ~ 2.64, container area = pi ~ 3.14
# Packing fraction ~ 84%, which is high
# Try: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29 (original)
# and variations

def construct():
    """Construct the packing configuration."""
    
    structures = [ring_counts_1, ring_counts_2, ring_counts_3, ring_counts_4]
    
    best_points = None
    best_r = 0.0
    
    for rc in structures:
        if np.sum(rc) != N:
            continue
        points, r = _try_ring_structure(rc, n_starts=2, maxiter_nm=2000)
        if points is not None and r > best_r:
            best_r = r
            best_points = points
    
    # Fallback if no valid configuration found
    if best_points is None:
        # Hexagonal lattice fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            pts = pts[mask]
            radius = radius[mask]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points

# EVOLVE-BLOCK-END
