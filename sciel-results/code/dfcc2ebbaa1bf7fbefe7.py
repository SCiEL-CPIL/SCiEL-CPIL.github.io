"""
Strategy: Structural Descriptor-Guided Global Optimization for Sutton-Chen clusters.
Combines physics-informed templates with a radial density / spherical harmonic descriptor
that penalizes non-FCC/HCP local environments, guiding the search toward known structural motifs.
Uses adaptive basin hopping with descriptor-based acceptance and high-precision polish.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize
from scipy.spatial import cKDTree

_DEADLINE_SEC = 295.0
_START_TIME = None

def _time_left():
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * np.sqrt(2)
    L = n ** (1.0/3.0) * spacing * 2.0
    n_side = int(np.ceil(L / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    points.append(p)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            points = np.vstack([points, [p]])
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _hcp_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an HCP lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    points = []
    n_layers = int(np.ceil(np.sqrt(n))) + 2
    
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        if layer % 2 == 0:
            shift = np.array([0, 0])
        else:
            shift = np.array([a/2, a*np.sqrt(3)/6])
        
        n_side = int(np.ceil(np.sqrt(n))) + 2
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift[0]
                y = j * a * np.sqrt(3) / 2.0 + shift[1]
                points.append([x, y, z])
                
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            points = np.vstack([points, [p]])
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _bcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * 2.0 / np.sqrt(3)
    L = n ** (1.0/3.0) * spacing * 1.5
    n_side = int(np.ceil(L / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                points.append(base)
                points.append(base + np.array([0.5, 0.5, 0.5]) * a)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            points = np.vstack([points, [p]])
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Generate an icosahedral cluster with shells."""
    if rng is None:
        rng = np.random.default_rng()
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * spacing
    
    points = [np.zeros(3)]
    points.extend(verts.tolist())
    
    for scale in [1.6, 2.2, 2.8]:
        for v in verts:
            points.append(v * scale)
    
    while len(points) < n:
        scale = 1.0 + 0.3 * len(points)
        for v in verts:
            points.append(v * scale)
            if len(points) >= n:
                break
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
        
    radius = spacing * n ** (1.0/3.0) * 1.0
    points = []
    max_attempts = n * 200
    attempts = 0
    min_dist = 0.5 * spacing
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1
            
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _structural_descriptor(positions):
    """
    Compute a structural descriptor that penalizes non-FCC/HCP local environments.
    Uses radial distribution function peaks and coordination number analysis.
    Lower value = more FCC/HCP-like structure.
    """
    n = len(positions)
    if n < 2:
        return 0.0
    
    tree = cKDTree(positions)
    
    # FCC nearest neighbor distance ~ 1.0 (in reduced units)
    # HCP nearest neighbor distance ~ 1.0
    # Typical second neighbor distance ~ 1.414 (FCC) or ~ 1.633 (HCP c-axis)
    
    # Count coordination numbers at various shell radii
    shell1 = tree.query_ball_point(positions, r=1.2, workers=-1)
    shell2 = tree.query_ball_point(positions, r=1.8, workers=-1)
    shell3 = tree.query_ball_point(positions, r=2.5, workers=-1)
    
    # Average coordination numbers
    coord1 = np.mean([len(s) - 1 for s in shell1])
    coord2 = np.mean([len(s) - 1 for s in shell2])
    coord3 = np.mean([len(s) - 1 for s in shell3])
    
    # FCC: CN1 ~ 12, CN2 ~ 6, CN3 ~ 24
    # HCP: CN1 ~ 12, CN2 ~ 6, CN3 ~ 24
    # BCC: CN1 ~ 8, CN2 ~ 6, CN3 ~ 12
    # Icosahedral: CN1 ~ 12, CN2 ~ 0 (no second shell in ideal ico)
    
    # Penalize deviation from FCC/HCP coordination pattern
    penalty = 0.0
    
    # First shell should be ~12
    penalty += (coord1 - 12.0) ** 2
    
    # Second shell should be ~6 (FCC/HCP)
    penalty += (coord2 - 6.0) ** 2
    
    # Third shell should be ~24 (FCC/HCP)
    penalty += (coord3 - 24.0) ** 2 * 0.5
    
    # Also check bond angle distribution for FCC/HCP
    # FCC has tetrahedral angles ~ 109.47 degrees
    # Sample a few atoms and compute bond angles
    if n >= 10:
        sample_idx = np.random.default_rng(42).choice(n, size=min(5, n), replace=False)
        angle_penalty = 0.0
        for idx in sample_idx:
            neighbors = tree.query_ball_point(positions[idx], r=1.2, workers=-1)
            neighbors = [nb for nb in neighbors if nb != idx]
            if len(neighbors) >= 3:
                # Compute some bond angles
                for i in range(min(3, len(neighbors))):
                    for j in range(i+1, min(6, len(neighbors))):
                        v1 = positions[neighbors[i]] - positions[idx]
                        v2 = positions[neighbors[j]] - positions[idx]
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        # FCC tetrahedral angle ~ 1.91 rad
                        angle_penalty += (angle - 1.91) ** 2
        penalty += angle_penalty * 0.1
    
    return penalty

def _perturb(positions, rng, scale=0.5):
    """Perturb positions with Gaussian noise."""
    noise = rng.normal(0, scale, positions.shape)
    return positions + noise

def _polish(positions, potential, parameter, max_iter=5000, ftol=1e-14):
    """High-precision polish using scipy minimize with tight tolerances."""
    try:
        result = minimize(
            fun=lambda x: sutton_chen_energy_gradient(x.reshape(-1, 3), parameter)[0],
            x0=positions.flatten(),
            jac=lambda x: sutton_chen_energy_gradient(x.reshape(-1, 3), parameter)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10, 'disp': False}
        )
        if result.success:
            return result.fun, result.x.reshape(-1, 3)
    except:
        pass
    
    # Fallback to relax
    try:
        energy, pos = relax(positions, potential, parameter)
        return energy, pos
    except:
        return np.inf, positions

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = 1.0
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    best_descriptor = np.inf
    
    # Generate templates
    templates = []
    for gen_func in [_fcc_fragment, _hcp_fragment, _bcc_fragment, _icosahedral_shell, _random_compact]:
        try:
            t = gen_func(size, spacing, rng)
            if len(t) == size:
                templates.append(t)
        except:
            pass
    
    # Relax each template and compute descriptor
    for t in templates:
        if _time_left() < 5.0:
            break
        try:
            energy, positions = relax(t, potential, parameter)
            desc = _structural_descriptor(positions)
            # Combined score: energy + lambda * descriptor
            combined = energy + 0.1 * desc
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                best_descriptor = desc
        except:
            pass
    
    # Adaptive basin hopping with descriptor-guided acceptance
    max_iterations = 400
    lambda_desc = 0.05  # Weight for structural descriptor in acceptance
    
    for i in range(max_iterations):
        if _time_left() < 10.0:
            break
            
        # Adaptive perturbation scale: larger early, smaller later
        progress = i / max_iterations
        scale = 0.6 * (1.0 - progress) + 0.1
        
        # Choose base: mostly best, sometimes random template
        if rng.random() < 0.7 and best_positions is not None:
            base = best_positions
        else:
            base = templates[rng.integers(len(templates))]
            
        new_positions = _perturb(base, rng, scale)
        
        try:
            energy, positions = relax(new_positions, potential, parameter)
            desc = _structural_descriptor(positions)
            
            # Accept if energy improves OR if descriptor significantly improves
            # (allowing escape from local minima with worse energy but better structure)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                best_descriptor = desc
            elif desc < best_descriptor - 0.5 and energy < best_energy + abs(best_energy) * 0.05:
                # Accept if structure is much better and energy is not much worse
                best_positions = positions
                best_descriptor = desc
        except:
            pass
    
    # Final high-precision polish
    if best_positions is not None and _time_left() > 15.0:
        try:
            polished_energy, polished_pos = _polish(best_positions, potential, parameter)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_positions = polished_pos
        except:
            pass
    
    # One more round of small perturbations + polish
    if best_positions is not None and _time_left() > 10.0:
        for scale in [0.05, 0.02, 0.01]:
            if _time_left() < 5.0:
                break
            try:
                perturbed = _perturb(best_positions, rng, scale)
                polished_energy, polished_pos = _polish(perturbed, potential, parameter)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_positions = polished_pos
            except:
                pass
    
    if best_positions is None:
        best_positions = templates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
