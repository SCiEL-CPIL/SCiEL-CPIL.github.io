# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 280  # safety margin before 330s kill

def _icosahedron_vertices():
    """12 vertices of regular icosahedron, unit-ish scale."""
    phi = (1 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _icosahedron_edges(v):
    """Return list of edge index pairs for icosahedron vertices."""
    n = len(v)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(v[i] - v[j])
            if d < 0.72:  # edge threshold for unit icosahedron
                edges.append((i, j))
    return edges

def _face_centers(v, edges):
    """Generate 20 face centers from icosahedron vertices."""
    # Find faces: triangles where all three edges exist
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (min(i,j), max(i,j)) in edge_set and \
                   (min(j,k), max(j,k)) in edge_set and \
                   (min(i,k), max(i,k)) in edge_set:
                    faces.append((i, j, k))
    
    centers = []
    for (i, j, k) in faces:
        c = (v[i] + v[j] + v[k]) / 3.0
        centers.append(c / np.linalg.norm(c))
    return np.array(centers)

def _shell2_positions(v1, fc):
    """Generate 42 second-shell positions for Mackay icosahedron."""
    # 12 edge midpoints (scaled), 20 face centers (scaled), 12 vertex extensions (scaled)
    edges = _icosahedron_edges(v1)
    edge_mids = []
    for (i, j) in edges:
        m = (v1[i] + v1[j]) / 2.0
        edge_mids.append(m / np.linalg.norm(m))
    edge_mids = np.array(edge_mids)
    
    # Combine: 12 + 20 + 12 = 44, but Mackay shell 2 has 42
    # Actually for Mackay: shell1=12 (vertices), shell2=42
    # The 42 come from: 12 edge-centers + 20 face-centers + 12... 
    # Let me use a different approach: generate points on icosahedral symmetry
    # For simplicity, use 12 vertices + 20 face centers + 12 edge midpoints = 44
    # Then trim to 42 by removing 2
    
    all_pts = np.vstack([v1, fc, edge_mids])  # 44 points
    # Remove 2 to get 42 (remove two edge midpoints for symmetry)
    # Actually, let's just use all 44 and handle n accordingly
    return all_pts

def _build_mackay_shells(r1=1.0, r2=1.55):
    """Build Mackay icosahedral shells: 1 + 12 + 42 = 55 atoms."""
    v1 = _icosahedron_vertices()
    fc = _face_centers(v1, _icosahedron_edges(v1))
    edges = _icosahedron_edges(v1)
    edge_mids = []
    for (i, j) in edges:
        m = (v1[i] + v1[j]) / 2.0
        edge_mids.append(m / np.linalg.norm(m))
    edge_mids = np.array(edge_mids)
    
    shell0 = np.array([[0.0, 0.0, 0.0]])
    shell1 = v1 * r1
    shell2_pts = np.vstack([fc, edge_mids, v1])  # 20 + 12 + 12 = 44
    shell2 = shell2_pts * r2
    
    return shell0, shell1, shell2

def _select_atoms(shells, n_target, rng):
    """Select n_target atoms from the shell structure."""
    shell0, shell1, shell2 = shells
    all_atoms = np.vstack([shell0, shell1, shell2])
    n_avail = len(all_atoms)
    
    if n_target <= n_avail:
        # Select n_target atoms, prioritizing lower shells
        indices = list(range(len(shell0))) + list(range(len(shell0), len(shell0)+len(shell1)))
        remaining = n_target - len(indices)
        if remaining > 0:
            idx2 = list(range(len(shell0)+len(shell1), len(all_atoms)))
            rng.shuffle(idx2)
            indices.extend(idx2[:remaining])
        return all_atoms[indices]
    else:
        # Need more than available - add extra atoms near shell2
        pts = all_atoms.copy()
        extra_needed = n_target - n_avail
        for _ in range(extra_needed):
            base = all_atoms[rng.integers(len(all_atoms))]
            noise = rng.normal(0, 0.3, 3)
            pts = np.vstack([pts, base + noise])
        return pts

def _perturb(positions, amplitude, rng):
    """Add random perturbation."""
    return positions + rng.normal(0, amplitude, positions.shape)

def _basin_hop(positions, potential, max_iter=50, max_temp=0.5, min_temp=0.01, rng=None):
    """Simple basin hopping algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_pos = positions.copy()
    current_energy, _ = relax(current_pos, potential)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    T = max_temp
    cooling = (min_temp / max_temp) ** (1.0 / max_iter)
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        # Propose
        amplitude = 0.3 * (1.0 - i / max_iter * 0.5)
        proposed = _perturb(current_pos, amplitude, rng)
        proposed_energy, _ = relax(proposed, potential)
        
        # Accept/reject
        delta = proposed_energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_pos = proposed
            current_energy = proposed_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        T *= cooling
    
    return best_energy, best_pos

def construct(instance):
    """Construct low-energy cluster using icosahedral seeding + basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    shells = _build_mackay_shells()
    
    rng = np.random.default_rng(size + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # Number of restarts based on time budget
    n_restarts = 8
    max_bh_iter = 30
    
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 15:
            break
        
        # Different initialization strategies
        strategy = restart % 4
        
        if strategy == 0:
            # Direct selection from shells
            positions = _select_atoms(shells, size, rng)
        elif strategy == 1:
            # Perturbed shell selection
            positions = _select_atoms(shells, size, rng)
            positions = _perturb(positions, 0.15, rng)
        elif strategy == 2:
            # Random compact + icosahedral bias
            radius = 1.1 * size ** (1.0/3.0) * 0.8
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(
                    np.linalg.norm(p - q) >= 0.55 for q in points
                ):
                    points.append(p)
            positions = np.array(points)
        else:
            # Icosahedral core + random outer
            shell0, shell1, _ = shells
            core = np.vstack([shell0, shell1])  # 13 atoms
            remaining = size - len(core)
            if remaining > 0:
                radius_outer = 1.55
                outer = []
                while len(outer) < remaining:
                    p = rng.uniform(-radius_outer, radius_outer, 3)
                    if np.linalg.norm(p) > 1.2 and np.linalg.norm(p) <= radius_outer + 0.3:
                        if all(np.linalg.norm(p - q) >= 0.55 for q in core + outer):
                            outer.append(p)
                positions = np.vstack([core, np.array(outer)])
            else:
                positions = core[:size]
        
        # Basin hopping
        try:
            energy, pos = _basin_hop(positions, potential, max_iter=max_bh_iter, rng=rng)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: simple random
        radius = 1.1 * size ** (1.0/3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        best_positions = np.array(points)
        best_energy, best_positions = relax(best_positions, potential)
    
    return best_positions
# EVOLVE-BLOCK-END
