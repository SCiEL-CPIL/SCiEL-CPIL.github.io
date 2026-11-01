"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
# Mackay icosahedron vertex directions (12 vertices of a regular icosahedron)
# Using the standard construction: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = golden ratio
PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _icosahedron_vertices():
    """Return 12 unit vectors pointing to icosahedron vertices."""
    verts = [
        (0, 1, PHI), (0, -1, PHI), (0, 1, -PHI), (0, -1, -PHI),
        (1, PHI, 0), (-1, PHI, 0), (1, -PHI, 0), (-1, -PHI, 0),
        (PHI, 0, 1), (-PHI, 0, 1), (PHI, 0, -1), (-PHI, 0, -1),
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

_ICO_VERTS = _icosahedron_vertices()

def _lj_energy(positions):
    """Compute total Lennard-Jones energy: 4*sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    # Mask out self-interactions
    np.fill_diagonal(dist_sq, np.inf)
    r6 = 1.0 / dist_sq ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_and_grad(positions):
    """Compute LJ energy and gradient."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    r6 = 1.0 / dist_sq ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    # Gradient: dE/dr_i = 4 * sum_j d(r_ij^-12 - r_ij^-6)/dr_i
    # d(r^-12)/dr = -12 r^-13 * dr/dr_i; d(r^-6)/dr = -6 r^-7 * dr/dr_i
    # r^-13 = r^-12 / r = r^-12 / sqrt(r^2); r^-7 = r^-6 / sqrt(r^2)
    r = np.sqrt(dist_sq)
    np.fill_diagonal(r, np.inf)
    r13 = r12 / r
    r7 = r6 / r
    # Force on atom i from atom j: 4 * (-12*r13 + 6*r7) * (r_i - r_j) / r
    # Actually: dE/dr_i = sum_j 4 * (-12*r13_ij + 6*r7_ij) * (r_i - r_j)/r_ij
    # = sum_j 4 * (-12*r13_ij + 6*r7_ij) * diff_ij / r_ij
    # where diff_ij = r_i - r_j
    coeff = 4.0 * (-12.0 * r13 + 6.0 * r7)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _mckay_icosahedron(r_inner, r_outer):
    """
    Generate positions for a 55-atom Mackay icosahedron.
    Shell 0: 1 atom at center (radius 0)
    Shell 1: 12 atoms at icosahedron vertices * r_inner
    Shell 2: 42 atoms: 30 at edge midpoints + 12 at face centers, scaled by r_outer
    """
    # Center atom
    positions = [np.zeros(3)]
    
    # Shell 1: 12 icosahedron vertices
    shell1 = _ICO_VERTS * r_inner
    positions.extend(shell1.tolist())
    
    # Shell 2: 42 atoms
    # 30 edge midpoints of icosahedron
    # An icosahedron has 30 edges. We need to identify them.
    # Two vertices are connected if their dot product is 1/phi (for unit icosahedron)
    # Actually for unit icosahedron, adjacent vertices have dot product = 1/phi^2? Let me compute.
    # For the standard icosahedron with vertices at (0,±1,±φ) etc., the edge length is 2.
    # Dot product of adjacent vertices: e.g., (0,1,φ)·(1,φ,0) = φ. Norms are sqrt(1+φ²)=sqrt(1+φ+1)=sqrt(2+φ)
    # Actually sqrt(1+φ²) = sqrt(1 + (1+2/φ)) ... let me just compute numerically.
    
    verts = _ICO_VERTS
    n_v = 12
    # Find edges: pairs of vertices with the smallest distance
    dists = np.zeros((n_v, n_v))
    for i in range(n_v):
        for j in range(n_v):
            if i != j:
                dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    
    # The 30 edges are the pairs with minimum distance
    min_dist = np.min(dists[dists > 0])
    edges = []
    for i in range(n_v):
        for j in range(i+1, n_v):
            if abs(dists[i, j] - min_dist) < 1e-9:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        edge_mids.append((verts[i] + verts[j]) / 2.0)
    edge_mids = np.array(edge_mids)
    edge_mids = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True) * r_outer
    positions.extend(edge_mids.tolist())
    
    # 12 face centers
    # An icosahedron has 20 faces. Each face is a triangle of 3 mutually adjacent vertices.
    # Face centers: average of 3 vertices, normalized.
    # Find all triangles where all 3 pairs are edges.
    edge_set = set(edges)
    face_centers = []
    for i in range(n_v):
        for j in range(i+1, n_v):
            if (i, j) in edge_set or (j, i) in edge_set:
                for k in range(j+1, n_v):
                    if ((j, k) in edge_set or (k, j) in edge_set) and ((i, k) in edge_set or (k, i) in edge_set):
                        fc = (verts[i] + verts[j] + verts[k]) / 3.0
                        fc = fc / np.linalg.norm(fc) * r_outer
                        face_centers.append(fc)
    
    face_centers = np.array(face_centers)
    positions.extend(face_centers.tolist())
    
    positions = np.array(positions, dtype=float)
    return positions

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    # Start from Mackay icosahedron with known approximate radii
    # For LJ, the minimum is at r = 2^(1/6) ≈ 1.12246
    # Standard Mackay icosahedron radii (in units of nearest-neighbor distance):
    # Inner shell: ~1.0, Outer shell: ~1.93 (these are approximate)
    
    best_positions = None
    best_energy = np.inf
    
    # Initial guess: use radii that give reasonable spacing
    # The icosahedron edge length for unit vertices is 2.
    # If inner radius = r1, the distance between two adjacent inner atoms = r1 * 2 (edge length of icosahedron)
    # We want this to be ~1.12246, so r1 ≈ 0.5612
    # Outer shell: distance from center to face center direction... 
    # Let's try r_inner = 0.56, r_outer = 1.08 (rough estimates)
    
    initial_r_inner = 0.56
    initial_r_outer = 1.08
    
    # Try a few initial radii
    for r_in, r_out in [(0.56, 1.08), (0.55, 1.10), (0.57, 1.05), (0.54, 1.12)]:
        if time.time() > deadline:
            break
        
        positions = _mckay_icosahedron(r_in, r_out)
        
        # Check minimum distance constraint
        if len(positions) < N:
            continue
        
        # Compute energy
        energy = _lj_energy(positions)
        
        # Relax with L-BFGS
        try:
            result = minimize(
                _lj_energy,
                positions,
                jac=lambda x: _lj_energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            relaxed = result.x
            relaxed_energy = _lj_energy(relaxed)
            
            if relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_positions = relaxed
        except Exception:
            pass
    
    # If we still have time, do a final polish with tighter tolerances
    if best_positions is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _lj_energy,
                best_positions,
                jac=lambda x: _lj_energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            polished = result.x
            polished_energy = _lj_energy(polished)
            if polished_energy < best_energy:
                best_positions = polished
                best_energy = polished_energy
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_positions = np.array(points[:N], dtype=float) * 1.12
    
    return best_positions
# EVOLVE-BLOCK-END
