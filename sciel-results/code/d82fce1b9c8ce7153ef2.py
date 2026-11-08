"""
Strategy: Multi-start global optimization with diverse structural templates
and adaptive basin hopping for Sutton-Chen transition-metal clusters.

Key improvements over previous versions:
1. More diverse initial templates: FCC, HCP, BCC, icosahedral, and random
2. Adaptive basin hopping with decreasing perturbation scales
3. Multiple independent basin hopping runs from different starting points
4. High-precision final polish with L-BFGS-B
5. Tight deadline management to avoid timeouts
"""

import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_DEADLINE_SEC = 325.0
_START_TIME = None

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)


def _fcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * np.sqrt(2)
    L = n ** (1.0 / 3.0) * spacing * 2.0
    n_side = int(np.ceil(L / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = base + np.array(offset) * a
                    points.append(p)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0 / 3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        # Expand the search radius
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
    
    if len(points) < n:
        return None

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
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    n_layers = int(np.ceil(np.sqrt(n))) + 2
    
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        if layer % 2 == 0:
            shift = np.array([0, 0])
        else:
            shift = np.array([a / 2, a * np.sqrt(3) / 6])
        
        n_side = int(np.ceil(np.sqrt(n))) + 2
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift[0]
                y = j * a * np.sqrt(3) / 2.0 + shift[1]
                points.append([x, y, z])
                
    points = np.array(points)
    R = 0.5 * n ** (1.0 / 3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
        
    if len(points) < n:
        return None
        
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points


def _bcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    points = []
    n_side = int(np.ceil(n ** (1.0 / 3.0))) + 2
    
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                p1 = base
                p2 = base + np.array([0.5, 0.5, 0.5]) * a
                points.append(p1)
                points.append(p2)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0 / 3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
        
    if len(points) < n:
        return None
        
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points


def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Generate an icosahedral cluster with shells."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Generate shells
    points = []
    # Center atom
    points.append(np.zeros(3))
    
    # First shell: 12 atoms at icosahedral positions
    shell1 = ico_verts * spacing
    for v in shell1:
        points.append(v)
    
    # Second shell: 42 atoms (if needed)
    if n > 13:
        # Generate points on a sphere for the second shell
        n_shell2 = min(42, n - 13)
        # Use a simple spherical distribution
        angles = np.linspace(0, 2 * np.pi, n_shell2, endpoint=False)
        radii = spacing * np.sqrt(2)
        for i, angle in enumerate(angles):
            # Distribute in a pattern
            theta = np.arccos(1 - 2 * (i + 0.5) / n_shell2)
            x = radii * np.sin(theta) * np.cos(angle)
            y = radii * np.sin(theta) * np.sin(angle)
            z = radii * np.cos(theta)
            points.append(np.array([x, y, z]))
    
    # Third shell: 92 atoms (if needed)
    if n > 55:
        n_shell3 = min(92, n - len(points))
        angles = np.linspace(0, 2 * np.pi, n_shell3, endpoint=False)
        radii = spacing * np.sqrt(3)
        for i, angle in enumerate(angles):
            theta = np.arccos(1 - 2 * (i + 0.5) / n_shell3)
            x = radii * np.sin(theta) * np.cos(angle)
            y = radii * np.sin(theta) * np.sin(angle)
            z = radii * np.cos(theta)
            points.append(np.array([x, y, z]))
    
    points = np.array(points)
    
    if len(points) < n:
        return None
    
    # Take the closest n to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster (liquid-like)."""
    if rng is None:
        rng = np.random.default_rng()
        
    radius = spacing * n ** (1.0 / 3.0) * 1.0
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


def _perturb(positions, scale, rng):
    """Perturb positions with Gaussian noise."""
    noise = rng.normal(0, scale, positions.shape)
    return positions + noise


def _polish(positions, potential, kind, max_iter=5000, ftol=1e-14):
    """High-precision polish using L-BFGS-B with tight tolerance."""
    if _time_left() < 5.0:
        return positions
    
    try:
        def fun(x):
            pos = x.reshape(-1, 3)
            e, g = sutton_chen_energy_gradient(pos, kind)
            return e, g.flatten()
        
        res = minimize(
            fun, 
            positions.flatten(), 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if res.success and np.isfinite(res.fun):
            result = res.x.reshape(-1, 3)
            # Verify no atoms are too close
            if len(result) > 1:
                d = cdist(result, result)
                np.fill_diagonal(d, np.inf)
                if np.min(d) < 0.35:
                    return positions
            return result
    except Exception:
        pass
        
    return positions


def _basin_hopping(initial_positions, potential, kind, n_iter=50, rng=None):
    """Adaptive basin hopping with decreasing perturbation scale."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_energy = np.inf
    best_positions = initial_positions.copy()
    
    # Initial relaxation
    try:
        energy, positions = relax(initial_positions, potential, kind)
        if np.isfinite(energy):
            best_energy = energy
            best_positions = positions.copy()
    except:
        pass
    
    for i in range(n_iter):
        if _time_left() < 15.0:
            break
        
        # Adaptive perturbation scale
        scale = 0.3 * (1.0 - i / n_iter) + 0.05
        new_positions = _perturb(best_positions, scale, rng)
        
        # Relax
        try:
            energy, positions = relax(new_positions, potential, kind)
            
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            pass
    
    return best_energy, best_positions


def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures
    templates = []
    
    # FCC
    fcc = _fcc_fragment(size, 1.0, rng)
    if fcc is not None:
        templates.append(fcc)
        
    # HCP
    hcp = _hcp_fragment(size, 1.0, rng)
    if hcp is not None:
        templates.append(hcp)
        
    # BCC
    bcc = _bcc_fragment(size, 1.0, rng)
    if bcc is not None:
        templates.append(bcc)
        
    # Icosahedral
    ico = _icosahedral_shell(size, 1.0, rng)
    if ico is not None:
        templates.append(ico)
        
    # Random starts
    for _ in range(3):
        rand = _random_compact(size, 1.0, rng)
        templates.append(rand)
    
    # Relax each template and keep the best
    for t in templates:
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(t, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            pass
            
    if best_positions is None:
        # Fallback: use first template
        best_positions = templates[0]
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except:
            pass
    
    # Multiple basin hopping runs from different starting points
    if _time_left() > 30.0:
        # Determine number of BH runs based on remaining time
        n_bh_runs = min(5, max(2, int(_time_left() / 20.0)))
        n_iter_per_run = max(10, int(_time_left() / (n_bh_runs * 2.0)))
        
        # Start from best and a few other templates
        start_points = [best_positions]
        for t in templates[:2]:
            if t is not best_positions:
                start_points.append(t)
        
        for sp in start_points[:n_bh_runs]:
            if _time_left() < 20.0:
                break
            try:
                bh_energy, bh_positions = _basin_hopping(sp, potential, kind, n_iter_per_run, rng)
                if np.isfinite(bh_energy) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions
            except:
                pass
    
    # Final high-precision polish
    if _time_left() > 10.0:
        polished = _polish(best_positions, potential, kind)
        try:
            final_energy, _ = sutton_chen_energy_gradient(polished, kind)
            if np.isfinite(final_energy) and final_energy < best_energy:
                best_positions = polished
                best_energy = final_energy
        except:
            pass
    
    # Final safety check
    if best_positions is None or len(best_positions) != size:
        # Last resort: random
        best_positions = _random_compact(size, 1.0, rng)
    
    return best_positions
