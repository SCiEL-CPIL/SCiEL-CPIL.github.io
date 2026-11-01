"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


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
    
    # Pairwise distances
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
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


def _ring_constraint_radius(ring_counts):
    """
    For a given ring structure, compute the maximum achievable radius
    by solving for ring radii such that:
    1. Each ring's circumference can fit its circles: 2*pi*r_i >= n_i * 2*r
       => r_i >= n_i * r / pi
    2. Adjacent rings don't overlap: r_{i+1} - r_i >= 2*r (approx, for tangent rings)
       More precisely, the minimum distance between circles on adjacent rings >= 2*r
    3. Outermost ring: r_{k-1} + r <= 1
    
    This gives a lower bound on what's achievable. We use this to guide
    initial guesses and to validate.
    """
    k = len(ring_counts)
    # For ring i with n_i circles at radius r_i:
    #   chord distance between adjacent circles = 2 * r_i * sin(pi/n_i)
    #   This must be >= 2*r  =>  r_i >= r / sin(pi/n_i)
    # For adjacent rings i, i+1:
    #   minimum distance between a circle on ring i and ring i+1 >= 2*r
    #   The closest approach is approximately |r_{i+1} - r_i| (radial)
    #   but actually depends on angular alignment. Worst case: radial gap >= 2*r
    #   So: r_{i+1} >= r_i + 2*r
    # Boundary: r_{k-1} + r <= 1
    
    # From ring i constraint: r_i >= r / sin(pi/n_i)
    # From adjacency: r_i >= r_0 + 2*i*r  (for i=0,1,...)
    # Combined: r_i >= max(r / sin(pi/n_i), r_0 + 2*i*r)
    # And r_0 >= r / sin(pi/n_0) if n_0 > 1, or r_0 >= 0 if n_0 == 1 (center)
    
    # Simplest: assume r_0 = 0 if first ring is center (n_0=1)
    # Then r_i >= max(r / sin(pi/n_i), 2*i*r)
    # And r_{k-1} + r <= 1
    
    # So: max over i of [max(r/sin(pi/n_i), 2*i*r)] + r <= 1
    # => r * max_i [max(1/sin(pi/n_i), 2*i)] + r <= 1
    # => r <= 1 / (1 + max_i [max(1/sin(pi/n_i), 2*i)])
    
    max_term = 0.0
    for i, n in enumerate(ring_counts):
        if n > 1:
            term1 = 1.0 / np.sin(np.pi / n)
        else:
            term1 = 0.0
        term2 = 2.0 * i
        max_term = max(max_term, term1, term2)
    
    return 1.0 / (1.0 + max_term)


# Define ring structure: counts per ring summing to 300
# Hexagonal-like: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
# Sum: 1+6+12+18+24+30+36+42+48+54+29 = 300
ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
num_rings = len(ring_counts)


def _initial_guess(ring_counts, jitter_seed=0):
    """Generate initial guess for radii and phases."""
    rng = np.random.default_rng(jitter_seed)
    
    k = len(ring_counts)
    
    # Compute constraint-based radii
    # r_i >= max(r / sin(pi/n_i), 2*i*r) where r is the circle radius
    # We don't know r yet, so use a parameterization.
    # Let's assume a target radius r_target and compute required ring radii.
    
    # Start with a reasonable guess for r
    r_target = 0.05  # ballpark
    
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
    radii = radii + rng.uniform(-0.01, 0.01, size=k)
    radii = np.maximum(radii, 0.0)
    
    # Phases: stagger adjacent rings for better packing
    phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i] > 0 and ring_counts[i-1] > 0:
            # Offset by half the angular spacing of the previous ring
            phases[i] = np.pi / ring_counts[i-1] + rng.uniform(-0.1, 0.1)
    
    return np.concatenate([radii, phases])


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    
    # Try multiple initial guesses with different jitters
    for seed in range(5):
        x0 = _initial_guess(ring_counts, jitter_seed=seed)
        
        # Bounds: radii in [0, 1], phases in [0, 2*pi]
        bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
        
        # Use Nelder-Mead for robustness with non-smooth objective
        result = minimize(
            _objective, x0, args=(ring_counts,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
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
            pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points

# EVOLVE-BLOCK-END
