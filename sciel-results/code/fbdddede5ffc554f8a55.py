"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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


def _analytic_radii(ring_counts, r_target):
    """
    Compute analytically optimal ring radii for a given target radius r_target.
    
    For a ring with n circles at radius R, the minimum distance between adjacent
    circles is 2*R*sin(pi/n). For non-overlap: 2*R*sin(pi/n) >= 2*r_target
    => R >= r_target / sin(pi/n)
    
    Between adjacent rings i and i+1 with n_i and n_{i+1} circles at radii R_i and R_{i+1},
    the minimum inter-ring distance depends on angular alignment. In the worst case (aligned),
    the distance is |R_{i+1} - R_i|. For non-overlap: |R_{i+1} - R_i| >= 2*r_target
    (this is a conservative bound; actual constraint is more complex).
    
    We use: R_i = r_target / sin(pi/n_i) for n_i > 1, and R_0 = 0 for center.
    Then ensure R_{i+1} - R_i >= 2*r_target * 0.5 (half the gap, conservative).
    """
    k = len(ring_counts)
    radii = np.zeros(k)
    
    for i, n in enumerate(ring_counts):
        if n <= 1:
            radii[i] = 0.0
        else:
            radii[i] = r_target / np.sin(np.pi / n)
    
    # Ensure monotonicity and minimum separation
    for i in range(1, k):
        min_sep = 1.5 * r_target  # Conservative: at least 1.5*r apart
        if radii[i] < radii[i-1] + min_sep:
            radii[i] = radii[i-1] + min_sep
    
    return radii


def _objective_phases(params, radii, ring_counts):
    """
    Optimize only phases, given fixed radii.
    params = [phi_0, phi_1, ..., phi_{k-1}]
    """
    k = len(ring_counts)
    phases = params
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def _objective_full(params, ring_counts):
    """
    Full optimization: radii offsets + phases.
    params = [dR_0, dR_1, ..., dR_{k-1}, phi_0, phi_1, ..., phi_{k-1}]
    """
    k = len(ring_counts)
    dR = params[:k]
    phases = params[k:]
    
    # Base radii from analytic solution
    r_est = 0.05
    base_radii = _analytic_radii(ring_counts, r_est)
    radii = base_radii + dR
    
    if np.any(radii < 0):
        return 1e6
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def _optimize_ring_structure(ring_counts, n_starts=5, maxiter=3000):
    """
    Optimize a given ring structure using multiple starts.
    Returns (best_points, best_r).
    """
    k = len(ring_counts)
    best_r = 0.0
    best_points = None
    
    # Step 1: Get analytic radii
    r_est = 0.05
    base_radii = _analytic_radii(ring_counts, r_est)
    
    # Step 2: Optimize phases with fixed radii (fast, gets good phase alignment)
    phase_bounds = [(0, 2*np.pi)] * k
    for seed in range(n_starts):
        rng = np.random.default_rng(seed)
        x0_phases = rng.uniform(0, 2*np.pi, size=k)
        
        result_phases = minimize(
            _objective_phases, x0_phases, args=(base_radii, ring_counts),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        
        # Evaluate
        phases_opt = result_phases.x
        points = _build_points(base_radii, phases_opt, ring_counts)
        if len(points) == N:
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_points = points
    
    # Step 3: Full optimization (radii + phases) from best phase solution
    if best_points is not None:
        # Extract phases from best solution for initialization
        # We'll re-derive from the points
        # Actually, let's just do full optimization from multiple starts
        for seed in range(n_starts):
            rng = np.random.default_rng(seed + 100)
            dR0 = rng.uniform(-0.01, 0.01, size=k)
            phi0 = rng.uniform(0, 2*np.pi, size=k)
            x0 = np.concatenate([dR0, phi0])
            
            result_full = minimize(
                _objective_full, x0, args=(ring_counts,),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            dR = result_full.x[:k]
            phases = result_full.x[k:]
            radii = base_radii + dR
            
            if np.any(radii < 0):
                continue
                
            points = _build_points(radii, phases, ring_counts)
            if len(points) == N:
                r = _compute_radius(points)
                if r > best_r:
                    best_r = r
                    best_points = points
    
    return best_points, best_r


def _generate_ring_structures():
    """
    Generate candidate ring structures that sum to N=300.
    Uses hexagonal close-packing inspired structures.
    """
    structures = []
    
    # Classic hexagonal: 1 + 6 + 12 + 18 + ... + 6k
    # Sum of first m hexagonal rings: 1 + 6*(1+2+...+(m-1)) = 1 + 3*m*(m-1)
    # For m=10: 1 + 3*10*9 = 271. Need 29 more.
    # For m=11: 1 + 3*11*10 = 331. Too many.
    
    # Base: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54 (sum=271)
    base = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54]
    remaining = 29
    
    # Structure 1: Add remaining as last ring
    structures.append(np.array(base + [remaining]))
    
    # Structure 2: Distribute remaining across last few rings
    # Try: 1,6,12,18,24,30,36,42,48,50,33 (sum=300)
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 50, 33]))
    
    # Structure 3: 1,6,12,18,24,30,36,42,48,51,32
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 51, 32]))
    
    # Structure 4: 1,6,12,18,24,30,36,42,48,52,31
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 52, 31]))
    
    # Structure 5: 1,6,12,18,24,30,36,42,48,53,30
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 53, 30]))
    
    # Structure 6: 1,6,12,18,24,30,36,42,48,54,29 (original)
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]))
    
    # Structure 7: Adjust middle rings
    # 1,6,12,18,24,30,36,40,48,56,29 -> sum = 300
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 40, 48, 56, 29]))
    
    # Structure 8: 1,6,12,18,24,30,36,44,48,52,29 -> sum = 300
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 44, 48, 52, 29]))
    
    # Structure 9: 1,6,12,18,24,30,36,42,46,54,31 -> sum = 300
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 46, 54, 31]))
    
    # Structure 10: 1,6,12,18,24,30,36,42,48,50,33 (same as 2, skip)
    
    # Structure 11: Different inner: 1,6,12,18,24,32,36,42,48,54,29 -> sum=302, no
    # 1,6,12,18,24,30,36,42,48,54,29 is already there
    
    # Structure 12: 1,6,12,18,24,30,36,42,48,55,28
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 55, 28]))
    
    # Structure 13: 1,6,12,18,24,30,36,42,48,56,27
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 56, 27]))
    
    # Structure 14: 1,6,12,18,24,30,36,42,48,57,26
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 57, 26]))
    
    # Structure 15: 1,6,12,18,24,30,36,42,48,58,25
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 58, 25]))
    
    # Filter valid structures
    valid_structures = []
    seen = set()
    for s in structures:
        if sum(s) != N:
            continue
        if any(c < 1 for c in s):
            continue
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            valid_structures.append(s)
    
    return valid_structures


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    
    structures = _generate_ring_structures()
    
    for ring_counts in structures:
        points, r = _optimize_ring_structure(ring_counts, n_starts=4, maxiter=2000)
        
        if points is not None and r > best_r:
            best_r = r
            best_points = points
    
    # Fallback: simple hexagonal grid
    if best_points is None:
        spacing = 0.1
        while spacing > 0.01:
            # Generate hexagonal lattice points
            pts = []
            for row in range(-20, 21):
                for col in range(-20, 21):
                    x = col * spacing + (row % 2) * spacing * 0.5
                    y = row * spacing * np.sqrt(3) / 2
                    if x*x + y*y <= (1 - spacing/2)**2:
                        pts.append([x, y])
            pts = np.array(pts)
            if len(pts) >= N:
                # Select N points closest to center for better packing
                dists = np.hypot(pts[:, 0], pts[:, 1])
                idx = np.argsort(dists, kind='stable')[:N]
                return pts[idx]
            spacing *= 0.95
    
    return best_points

# EVOLVE-BLOCK-END
