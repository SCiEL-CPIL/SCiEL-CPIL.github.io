"""Lennard-Jones 55-atom cluster: multi-start global optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _get_icosahedron_data():
    """Return 12 vertices, 30 edge midpoints, 20 face centers as unit vectors."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if np.isclose(d1, edge_len, atol=1e-10) and np.isclose(d2, edge_len, atol=1e-10) and np.isclose(d3, edge_len, atol=1e-10):
                    faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    return verts, midpoints, face_centers

_VERTS, _MIDPOINTS, _FACE_CENTERS = _get_icosahedron_data()

def _make_mackay(r1, r2, r3, which_faces=None):
    """Create Mackay icosahedron: 1 + 12 + 30 + 12 = 55.
    r1: inner shell (12 vertices)
    r2: edge midpoints (30)
    r3: face centers (12 of 20)
    which_faces: which 12 of the 20 faces to use (indices)
    """
    if which_faces is None:
        which_faces = list(range(12))
    
    pos = np.zeros((N, 3))
    pos[0] = 0.0
    pos[1:13] = _VERTS * r1
    pos[13:43] = _MIDPOINTS * r2
    pos[43:55] = _FACE_CENTERS[which_faces] * r3
    return pos

def _make_decahedron(r1, r2, r3):
    """Create a decahedral structure for 55 atoms.
    Decahedron has 12 vertices, 30 edges, 20 faces.
    Structure: 1 center + 12 (vertices) + 30 (edge midpoints) + 12 (face centers) = 55
    But decahedron geometry is different from icosahedron.
    For simplicity, use the same icosahedral directions but with different radii.
    """
    return _make_mackay(r1, r2, r3)

def _relax(pos, maxiter=3000, gtol=1e-12, ftol=1e-14):
    """L-BFGS-B relaxation."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True,
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_pos = None
    best_e = np.inf
    
    # Candidate 1: Mackay icosahedron with standard radii
    candidates = []
    
    # Various radius combinations for Mackay
    radius_sets = [
        (1.04, 1.96, 2.80),
        (1.00, 1.90, 2.70),
        (1.06, 2.00, 2.90),
        (1.02, 1.94, 2.75),
        (1.08, 2.05, 2.95),
        (0.98, 1.85, 2.60),
        (1.10, 2.10, 3.00),
    ]
    
    for r1, r2, r3 in radius_sets:
        if time.time() > deadline - 25:
            break
        pos = _make_mackay(r1, r2, r3)
        candidates.append(pos)
    
    # Also try different face selections
    all_faces = list(range(20))
    import itertools
    # Try a few different 12-face subsets
    face_subsets = [
        list(range(12)),
        list(range(8)) + [12, 13, 14, 15],
        list(range(4)) + [8, 9, 10, 11, 12, 13, 14, 15],
        list(range(6)) + [10, 11, 12, 13, 14, 15],
    ]
    
    for fs in face_subsets:
        if time.time() > deadline - 20:
            break
        pos = _make_mackay(1.04, 1.96, 2.80, which_faces=fs)
        candidates.append(pos)
    
    # Random cluster
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        pos = rng.uniform(-3, 3, size=(N, 3))
        candidates.append(pos)
    
    # Process candidates
    for pos in candidates:
        if time.time() > deadline - 10:
            break
        pos_relaxed, e = _relax(pos)
        if e < best_e:
            best_e = e
            best_pos = pos_relaxed.copy()
    
    # Final polish on best
    if time.time() < deadline - 5 and best_pos is not None:
        pos_final, e_final = _relax(best_pos, maxiter=5000, gtol=1e-14, ftol=1e-16)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    
    if best_pos is None:
        best_pos = _make_mackay(1.04, 1.96, 2.80)
    
    return best_pos
