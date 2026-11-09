"""Sutton-Chen transition-metal cluster global minima: multi-start structured seeds + L-BFGS-B relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    # fcc basis vectors
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    # Generate candidate sites
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    site = np.array([i, j, k]) * spacing + b * spacing
                    sites.append(site)
    sites = np.array(sites)
    # Remove duplicates (within tolerance)
    unique_sites = []
    for s in sites:
        if not any(np.linalg.norm(s - u) < 1e-6 for u in unique_sites):
            unique_sites.append(s)
    unique_sites = np.array(unique_sites)
    # Sort by distance from origin
    dists = np.linalg.norm(unique_sites, axis=1)
    idx = np.argsort(dists)
    # Take the n closest, but ensure they form a compact cluster
    selected = unique_sites[idx[:n]]
    # Center the cluster
    selected -= np.mean(selected, axis=0)
    return selected


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    # HCP structure: two interpenetrating triangular lattices
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    sites = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                # First layer (z=0)
                x1 = a * (i + j / 2.0)
                y1 = a * np.sqrt(3.0) / 2.0 * j
                z1 = 0.0
                sites.append([x1, y1, z1])
                # Second layer (z=c/2)
                x2 = a * (i + j / 2.0 + 1.0 / 3.0)
                y2 = a * np.sqrt(3.0) / 2.0 * (j + 1.0 / 3.0)
                z2 = c / 2.0
                sites.append([x2, y2, z2])
    sites = np.array(sites)
    # Remove duplicates
    unique_sites = []
    for s in sites:
        if not any(np.linalg.norm(s - u) < 1e-6 for u in unique_sites):
            unique_sites.append(s)
    unique_sites = np.array(unique_sites)
    dists = np.linalg.norm(unique_sites, axis=1)
    idx = np.argsort(dists)
    selected = unique_sites[idx[:n]]
    selected -= np.mean(selected, axis=0)
    return selected


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    # Icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    # Scale to desired spacing
    verts *= spacing * 0.8
    
    sites = list(verts)
    # Add more layers for larger n
    if n > 12:
        # Add octahedral vertices
        octa = np.array([
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
        ], dtype=float)
        octa /= np.linalg.norm(octa[0])
        octa *= spacing * 0.5
        sites.extend(octa)
    
    if n > 20:
        # Add more vertices
        for _ in range(n - len(sites)):
            angle1 = np.random.uniform(0, 2 * np.pi)
            angle2 = np.random.uniform(0, np.pi)
            r = spacing * 0.6 * (1 + 0.3 * np.random.uniform())
            x = r * np.sin(angle2) * np.cos(angle1)
            y = r * np.sin(angle2) * np.sin(angle1)
            z = r * np.cos(angle2)
            sites.append([x, y, z])
    
    sites = np.array(sites[:n])
    sites -= np.mean(sites, axis=0)
    return sites


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
            points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def construct(instance):
    """Multi-start structured seeds + L-BFGS-B relaxation for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Get parameters
    if kind == "12-6":
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == "9-6":
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == "10-8":
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        n_exp, m_exp, c = 12, 6, 144.41
    
    spacing = 0.72
    
    # Generate diverse starting configurations
    start_configs = []
    
    # FCC-like
    start_configs.append(_fcc_cluster(size, spacing))
    
    # HCP-like
    start_configs.append(_hcp_cluster(size, spacing))
    
    # Icosahedral-like
    start_configs.append(_icosahedral_cluster(size, spacing))
    
    # Random compact with different seeds
    for seed in [1, 2, 3, 4, 5]:
        start_configs.append(_random_compact(size, spacing, seed))
    
    # Track best result
    best_energy = np.inf
    best_positions = None
    
    # Deadline: 325 seconds (5s margin)
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Try each starting configuration
    for i, config in enumerate(start_configs):
        if time.time() > deadline:
            break
        
        try:
            energy, positions = relax(config, potential, kind)
            
            # Check validity
            if np.all(np.isfinite(positions)):
                # Check minimum distance
                min_dist = np.inf
                for a in range(len(positions)):
                    for b in range(a + 1, len(positions)):
                        d = np.linalg.norm(positions[a] - positions[b])
                        min_dist = min(min_dist, d)
                if min_dist > 0.35:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
        except Exception:
            continue
    
    # If no valid result, use the last relaxed positions
    if best_positions is None:
        # Fallback: use a simple random configuration
        rng = np.random.default_rng(0)
        positions = rng.uniform(-2, 2, (size, 3))
        try:
            energy, positions = relax(positions, potential, kind)
            if np.all(np.isfinite(positions)):
                best_positions = positions
        except Exception:
            best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
