"""Lennard-Jones cluster global minimum, 55 atoms: two-shell Mackay icosahedron via parameterized manifold."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, minimize_scalar

N = 55

# Icosahedron vertex directions (unit vectors)
def _icosahedron_vertices():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

_ICO_VERTS = _icosahedron_vertices()  # shape (12, 3)

def _lj_energy(positions):
    """Compute total LJ energy for N atoms. positions: (N,3) array."""
    n = len(positions)
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    # Mask out self-interactions
    mask = ~np.eye(n, dtype=bool)
    r2 = r2[mask]
    r2 = np.maximum(r2, 1e-12)  # avoid division by zero
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return energy

def _build_positions_from_params(params):
    """
    Build 55-atom positions from parameterized manifold.
    params layout:
      [0:3]   = translation (tx, ty, tz)
      [3:6]   = rotation angles (rx, ry, rz) - small perturbations
      [6:18]  = 12 inner shell radii (one per icosahedron vertex direction)
      [18:30] = 12 outer shell radii (one per icosahedron vertex direction)
    
    Inner shell: 12 atoms at icosahedron vertex directions with radii params[6:18]
    Outer shell: 42 atoms. The Mackay icosahedron outer shell has 42 atoms:
      - 12 at the icosahedron vertex directions (same as inner)
      - 30 at edge-midpoint directions (between adjacent vertices)
    Actually, let's use the standard two-shell Mackay icosahedron:
      Shell 1 (inner): 12 atoms (icosahedron vertices)
      Shell 2 (outer): 42 atoms = 12 (vertex directions) + 30 (face-center directions)
    
    For the 30 face-center directions, we need the 20 faces of the icosahedron.
    Let me compute the face normals.
    """
    tx, ty, tz = params[0:3]
    rx, ry, rz = params[3:6]
    
    # Build rotation matrix from small angles (approximate for small angles)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    inner_radii = params[6:18]
    outer_radii = params[18:30]
    
    # Inner shell: 12 atoms
    inner_positions = _ICO_VERTS * inner_radii[:, np.newaxis]
    
    # Outer shell: 42 atoms
    # 12 at vertex directions with outer_radii
    outer_vertex_positions = _ICO_VERTS * outer_radii[:, np.newaxis]
    
    # 30 face-center directions: compute from icosahedron faces
    # The 20 faces of the regular icosahedron can be found from the vertex indices.
    # Standard icosahedron faces (using the vertex ordering above):
    faces = [
        (0, 1, 5), (0, 5, 2), (0, 2, 11), (0, 11, 1), (0, 1, 9),
        (1, 11, 4), (1, 4, 9), (1, 9, 8), (1, 8, 10), (1, 10, 3),
        (2, 5, 4), (2, 4, 11), (2, 11, 6), (2, 6, 10), (2, 10, 3),
        (3, 8, 7), (3, 7, 10), (3, 10, 4), (3, 4, 6), (3, 6, 7),
    ]
    
    face_normals = []
    for f in faces:
        v1, v2, v3 = _ICO_VERTS[f[0]], _ICO_VERTS[f[1]], _ICO_VERTS[f[2]]
        n = np.cross(v2 - v1, v3 - v1)
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-10:
            n /= n_norm
            # Ensure outward pointing (dot with centroid should be positive)
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            face_normals.append(n)
    
    face_normals = np.array(face_normals)  # shape (20, 3)
    
    # For the 30 face-center positions in the Mackay icosahedron,
    # they are actually at the midpoints of edges, not face centers.
    # Let me reconsider: the Mackay icosahedron with 55 atoms:
    # Shell 1: 12 (vertices)
    # Shell 2: 42 = 12 (vertices) + 30 (edge midpoints)
    
    # Edge midpoints: 30 edges of the icosahedron
    # Find all edges from the faces
    edges_set = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            if a > b:
                a, b = b, a
            edges_set.add((a, b))
    
    edges = list(edges_set)
    # Should have 30 edges
    edge_dirs = []
    for a, b in edges:
        d = (_ICO_VERTS[a] + _ICO_VERTS[b]) / 2.0
        d_norm = np.linalg.norm(d)
        if d_norm > 1e-10:
            d /= d_norm
            edge_dirs.append(d)
    
    edge_dirs = np.array(edge_dirs)  # shape (30, 3)
    
    # For the 30 edge-midpoint atoms, we need 30 radial parameters.
    # But our params only have 12 outer_radii. Let's use a single outer radius for all 30 edge atoms,
    # plus the 12 vertex outer radii.
    # Actually, let's simplify: use a uniform outer shell radius for the 30 edge atoms.
    # We'll add one more parameter for the edge shell radius.
    
    # Wait, our params vector is fixed. Let me restructure:
    # params[6:18] = 12 inner radii
    # params[18:30] = 12 outer vertex radii
    # We need an extra parameter for the 30 edge atoms. Let's use params[30] as the edge shell radius.
    
    edge_radius = params[30]
    edge_positions = edge_dirs * edge_radius
    
    # Combine all positions
    all_positions = np.vstack([inner_positions, outer_vertex_positions, edge_positions])
    
    # Apply rotation and translation
    all_positions = all_positions @ R.T + np.array([tx, ty, tz])
    
    return all_positions

def _energy_from_params(params):
    """Compute LJ energy from parameter vector."""
    positions = _build_positions_from_params(params)
    return _lj_energy(positions)

def _minimize_params(params0, max_iter=200):
    """Minimize energy with respect to parameters using L-BFGS-B."""
    result = minimize(
        _energy_from_params,
        params0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: standard Mackay icosahedron radii
    # Inner shell radius ~ 1.12, outer shell radius ~ 2.06 (for 55-atom Mackay icosahedron)
    inner_r0 = 1.12
    outer_vertex_r0 = 2.06
    edge_r0 = 2.06
    
    # Build initial parameter vector
    params0 = np.zeros(31)
    params0[6:18] = inner_r0
    params0[18:30] = outer_vertex_r0
    params0[30] = edge_r0
    
    best_energy = np.inf
    best_params = params0.copy()
    
    # Try multiple starting points with small random perturbations
    np.random.seed(42)
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            p = params0.copy()
        else:
            p = params0.copy()
            p[6:18] += np.random.uniform(-0.1, 0.1, size=12)
            p[18:30] += np.random.uniform(-0.1, 0.1, size=12)
            p[30] += np.random.uniform(-0.1, 0.1)
            p[3:6] += np.random.uniform(-0.05, 0.05, size=3)
        
        try:
            result = _minimize_params(p, max_iter=300)
            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            result = _minimize_params(best_params, max_iter=500)
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Build final positions
    positions = _build_positions_from_params(best_params)
    
    # Verify minimum distance constraint (atoms must be at least 0.5 apart)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    mask = ~np.eye(N, dtype=bool)
    min_dist = np.sqrt(np.min(r2[mask]))
    
    if min_dist < 0.5:
        # Fallback: use the seed program's cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        positions = np.array(points[:N], dtype=float) * 1.12
    
    return positions
