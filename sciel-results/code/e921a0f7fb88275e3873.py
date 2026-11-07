"""
Strategy: Structure-guided optimization using local environment descriptors.
We use a hybrid approach:
1. Generate diverse initial structures (FCC, HCP, BCC, random).
2. Use a gradient-free global optimizer (basin hopping) but guided by a penalty
   that discourages non-FCC/HCP local environments.
3. The penalty is computed from the radial distribution function (RDF) and
   local coordination numbers, which are cheap to compute and provide a strong
   signal for structural preference.
4. Final relaxation with L-BFGS-B for high precision.

This approach explicitly biases the search towards the known structural motifs
(FCC/HCP) that are favored by Sutton-Chen potentials, rather than relying solely
on stochastic exploration.
"""

import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_DEADLINE_SEC = 295.0
_START_TIME = None

def _time_left():
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _compute_local_environments(positions):
    """
    Compute local environment descriptors for each atom.
    Returns:
        coord_numbers: (n,) array of coordination numbers (neighbors within 1.5 * nearest neighbor distance)
        rdf: (n, 10) array of RDF bins for each atom
    """
    n = len(positions)
    if n < 2:
        return np.zeros(n), np.zeros((n, 10))
    
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    
    # Find nearest neighbor distance
    nn_dists = np.min(dists, axis=1)
    nn_dist = np.median(nn_dists)
    
    # Coordination number: neighbors within 1.5 * nn_dist
    cutoff = 1.5 * nn_dist
    coord_numbers = np.sum(dists < cutoff, axis=1)
    
    # RDF bins
    rdf_bins = np.linspace(0.8 * nn_dist, 2.5 * nn_dist, 11)
    rdf = np.zeros((n, 10))
    for i in range(n):
        d_i = dists[i]
        valid = d_i < 2.5 * nn_dist
        if np.sum(valid) > 0:
            hist, _ = np.histogram(d_i[valid], bins=rdf_bins)
            rdf[i] = hist / (n * valid.sum())
    
    return coord_numbers, rdf

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
        return _fcc_fragment(n, spacing, rng)

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
        return _hcp_fragment(n, spacing, rng)
        
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
    n_side = int(np.ceil(n ** (1.0/3.0))) + 2
    
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                p1 = base
                p2 = base + np.array([0.5, 0.5, 0.5]) * a
                points.append(p1)
                points.append(p2)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.5
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        return _bcc_fragment(n, spacing, rng)
        
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster (liquid-like)."""
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

def _structure_penalty(positions, target_coord=12.0, penalty_weight=0.1):
    """
    Compute a penalty for non-FCC/HCP local environments.
    FCC and HCP have coordination number 12.
    Penalize atoms with coordination number significantly different from 12.
    """
    coord_numbers, rdf = _compute_local_environments(positions)
    
    # Penalty for coordination number deviation
    coord_penalty = np.mean((coord_numbers - target_coord) ** 2)
    
    # Penalty for RDF structure: FCC/HCP have a specific RDF pattern
    # First peak at ~1.0, second peak at ~1.73 (for FCC) or ~1.63 (for HCP)
    if len(rdf) > 0 and np.sum(rdf) > 0:
        # Check if there's a peak in the expected region
        rdf_flat = rdf.flatten()
        expected_peak_region = rdf[:, 1:4]  # bins 1-3
        rdf_penalty = 1.0 - np.mean(np.max(expected_peak_region, axis=1) > 0.1)
    else:
        rdf_penalty = 1.0
    
    return penalty_weight * (coord_penalty + rdf_penalty)

def _guided_basin_hopping(initial_positions, potential, kind, n_iter=100, rng=None):
    """
    Basin hopping with structure-guided perturbations.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    best_energy = np.inf
    best_positions = initial_positions.copy()
    
    for i in range(n_iter):
        if _time_left() < 20.0:
            break
        
        # Perturb
        scale = 0.1 + 0.5 * (1.0 - i / n_iter)
        noise = rng.normal(0, scale, best_positions.shape)
        new_positions = best_positions + noise
        
        # Relax
        try:
            energy, positions = relax(new_positions, potential, kind)
            
            # Apply structure penalty
            penalty = _structure_penalty(positions)
            total_score = energy + penalty
            
            if total_score < best_energy + penalty:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    return best_energy, best_positions

def _polish(positions, potential, kind, max_iter=2000, ftol=1e-12):
    """High-precision polish using L-BFGS-B with tighter tolerance."""
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
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10, 'maxls': 20}
        )
        
        if res.success and res.fun < np.inf:
            return res.x.reshape(-1, 3)
    except Exception:
        pass
        
    return positions

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    spacing = 1.0
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures
    templates = []
    
    # FCC
    try:
        fcc = _fcc_fragment(size, spacing, rng)
        if len(fcc) == size:
            templates.append(fcc)
    except:
        pass
        
    # HCP
    try:
        hcp = _hcp_fragment(size, spacing, rng)
        if len(hcp) == size:
            templates.append(hcp)
    except:
        pass
        
    # BCC
    try:
        bcc = _bcc_fragment(size, spacing, rng)
        if len(bcc) == size:
            templates.append(bcc)
    except:
        pass
        
    # Random (Multiple diverse starts)
    for _ in range(5):
        rand = _random_compact(size, spacing, rng)
        templates.append(rand)
    
    # Relax each template and keep the best
    for t in templates:
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(t, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
            
    if best_positions is None:
        best_positions = templates[0]
    
    # Guided basin hopping
    if _time_left() > 30.0:
        n_iter = int(min(100, max(20, _time_left() / 3.0)))
        bh_energy, bh_positions = _guided_basin_hopping(best_positions, potential, kind, n_iter, rng)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions
    
    # Final polish
    polished_positions = _polish(best_positions, potential, kind)
    
    # Verify energy
    try:
        final_energy, _ = sutton_chen_energy_gradient(polished_positions, kind)
        if final_energy < best_energy:
            best_positions = polished_positions
            best_energy = final_energy
    except:
        pass
        
    return best_positions
