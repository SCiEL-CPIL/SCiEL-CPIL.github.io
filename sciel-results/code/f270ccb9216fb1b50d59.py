"""Dzugutov cluster global minima search using icosahedral seeding + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline guard
_START_TIME = None
_DEADLINE = 330.0  # seconds

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts /= norms
    return verts

def _get_icosahedron_edges():
    """Return edge connectivity for icosahedron (12 vertices, 30 edges)."""
    verts = _get_icosahedron_vertices()
    # Edge length in unit icosahedron
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(np.linalg.norm(verts[i] - verts[j]) - edge_len) < 1e-10:
                edges.append((i, j))
    return edges

def _generate_mackay_icosahedron(n_target, spacing=1.1):
    """
    Generate a Mackay icosahedral cluster with approximately n_target atoms.
    Shell structure:
      - Core: 1 atom at origin
      - Shell 1 (12 atoms): icosahedron vertices at radius r1
      - Shell 2 (42 atoms): icosahedral second shell at radius r2
    
    For n up to 60, we use core + shell1 + part of shell2.
    Returns positions array of shape (n_actual, 3) and actual count.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Core atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 icosahedron vertices
    ico_verts = _get_icosahedron_vertices()
    r1 = spacing
    for v in ico_verts:
        points.append(r1 * v)
    
    # If we already have enough atoms, stop
    if len(points) >= n_target:
        return np.array(points[:n_target]), min(n_target, len(points))
    
    # Shell 2: 42 atoms - these are the edge midpoints (30) and face centers (12) of the icosahedron
    # Actually, the second shell of a Mackay icosahedron has 42 atoms:
    # - 30 atoms at edge midpoints (scaled)
    # - 12 atoms at face centers (scaled)
    
    # Edge midpoints: for each edge, place atom at midpoint direction, at radius r2
    r2 = spacing * np.sqrt(2)  # approximately
    ico_edges = _get_icosahedron_edges()
    
    # For edge midpoints, the direction is the normalized sum of the two vertex directions
    for i, j in ico_edges:
        mid_dir = ico_verts[i] + ico_verts[j]
        mid_dir = mid_dir / np.linalg.norm(mid_dir)
        points.append(r2 * mid_dir)
    
    if len(points) >= n_target:
        return np.array(points[:n_target]), min(n_target, len(points))
    
    # Face centers: 20 faces of icosahedron
    # We need to find the 20 triangular faces
    # For a regular icosahedron, each face is a triangle of 3 mutually adjacent vertices
    # Let's find faces by checking triples of vertices that are all mutually connected
    verts = ico_verts
    edge_set = set(ico_edges)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) not in edge_set and (j, i) not in edge_set:
                continue
            for k in range(j + 1, 12):
                if (j, k) not in edge_set and (k, j) not in edge_set:
                    continue
                if (i, k) not in edge_set and (k, i) not in edge_set:
                    continue
                faces.append((i, j, k))
    
    # Scale for face centers
    r2_face = spacing * np.sqrt(3) * 0.8  # approximate
    for face in faces:
        center_dir = (verts[face[0]] + verts[face[1]] + verts[face[2]])
        center_dir = center_dir / np.linalg.norm(center_dir)
        points.append(r2_face * center_dir)
    
    # Trim or pad to n_target
    if len(points) > n_target:
        points = points[:n_target]
    else:
        # Pad with small random perturbations if needed
        while len(points) < n_target:
            p = points[np.random.randint(len(points))] + np.random.normal(0, 0.1, 3)
            points.append(p)
    
    return np.array(points), len(points)

def _generate_defect_icosahedron(n_target, spacing=1.1, rng=None):
    """
    Generate a defective icosahedral cluster by removing some atoms from shell 2
    and adjusting positions.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Start with full Mackay icosahedron
    full_n = 1 + 12 + 42  # 55 atoms
    positions, _ = _generate_mackay_icosahedron(full_n, spacing)
    
    if n_target >= len(positions):
        # Need more atoms - add random atoms near the surface
        while len(positions) < n_target:
            # Place near existing atoms with some offset
            idx = rng.randint(len(positions))
            new_p = positions[idx] + rng.normal(0, 0.3, 3)
            positions = np.vstack([positions, new_p])
        return positions[:n_target]
    
    # Remove atoms from shell 2 (indices 13 onwards) to get n_target
    if n_target <= 13:
        return positions[:n_target]
    
    # Keep core + shell 1 (first 13 atoms)
    keep = positions[:13].copy()
    remaining = n_target - 13
    
    # Randomly select 'remaining' atoms from shell 2
    shell2_indices = list(range(13, len(positions)))
    if remaining <= len(shell2_indices):
        selected = rng.choice(shell2_indices, size=remaining, replace=False)
        selected = sorted(selected)
        for idx in selected:
            keep = np.vstack([keep, positions[idx]])
    else:
        keep = np.vstack([keep, positions[13:]])
    
    return keep[:n_target]

def _basin_hopping(positions, potential, parameter, n_iter=50, max_disp=0.5, rng=None):
    """
    Simple basin hopping: perturb, relax, accept if lower energy.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    energy, current_pos = relax(positions, potential, parameter)
    best_energy = energy
    best_pos = current_pos.copy()
    
    for i in range(n_iter):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        # Random displacement
        displacement = rng.uniform(-max_disp, max_disp, size=current_pos.shape)
        perturbed = current_pos + displacement
        
        # Relax
        try:
            e_new, pos_new = relax(perturbed, potential, parameter)
        except:
            continue
        
        # Metropolis-like acceptance (always accept for simplicity, or use temperature)
        if e_new < best_energy:
            best_energy = e_new
            best_pos = pos_new.copy()
        
        # Accept the move (always for simplicity)
        current_pos = pos_new
    
    return best_energy, best_pos

def _local_minimize(positions, potential, parameter):
    """Local minimization using the provided relax function."""
    energy, positions = relax(positions, potential, parameter)
    return energy, positions

def construct(instance):
    """
    Find low-energy Dzugutov clusters using icosahedral seeding + basin hopping.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple initial candidates
    n_candidates = 8
    candidates = []
    
    # 1. Perfect Mackay icosahedron (trim to size)
    pos, n_actual = _generate_mackay_icosahedron(size, spacing=1.1)
    candidates.append(pos)
    
    # 2. Defective icosahedra with different seeds
    for seed in range(6):
        rng_defect = np.random.default_rng(size * 100 + seed)
        pos_defect = _generate_defect_icosahedron(size, spacing=1.1, rng=rng_defect)
        candidates.append(pos_defect)
    
    # 3. Slightly perturbed icosahedron
    pos, _ = _generate_mackay_icosahedron(size, spacing=1.1)
    pos_perturbed = pos + rng.normal(0, 0.1, pos.shape)
    candidates.append(pos_perturbed)
    
    # For each candidate, do basin hopping
    for idx, cand in enumerate(candidates):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        # First, locally minimize
        try:
            e_local, pos_local = relax(cand, potential, parameter)
        except:
            continue
        
        if e_local < best_energy:
            best_energy = e_local
            best_positions = pos_local.copy()
        
        # Basin hopping from this local minimum
        n_bh_iter = 30 if idx < 3 else 15  # More iterations for better candidates
        e_bh, pos_bh = _basin_hopping(pos_local, potential, parameter, n_iter=n_bh_iter, rng=rng)
        
        if e_bh < best_energy:
            best_energy = e_bh
            best_positions = pos_bh.copy()
    
    # Final safety check
    if best_positions is None:
        # Fallback: random compact cluster
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
        best_positions = np.array(points)
        best_positions, _ = relax(best_positions, potential, parameter)
    
    return best_positions
# EVOLVE-BLOCK-END
