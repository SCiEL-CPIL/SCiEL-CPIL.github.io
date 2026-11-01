"""Lennard-Jones 55-atom cluster: correct two-shell Mackay icosahedron + high-precision relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
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

def _get_icosahedron_directions():
    """Return 12 vertex, 30 edge-midpoint, and 20 face-center unit vectors of a regular icosahedron."""
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

def make_mackay_icosahedron(r_inner=1.04, r_outer=1.96):
    """Generate a two-shell Mackay icosahedral cluster with 55 atoms.
    
    Structure: 1 (center) + 12 (inner shell at r_inner) + 42 (outer shell at r_outer)
    Outer shell = 30 edge midpoints + 12 face centers (first 12 of 20).
    Total: 1 + 12 + 42 = 55.
    """
    verts, midpoints, face_centers = _get_icosahedron_directions()
    
    inner = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r_inner
    # Use all 30 edge midpoints and 12 face centers for the outer shell
    outer_dirs = np.vstack([midpoints, face_centers[:12]])
    shell2 = outer_dirs * r_outer
    
    pos = np.vstack([inner, shell1, shell2])
    return pos

def make_cubic_lattice():
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def make_random_cluster(seed):
    """Generate a random cluster."""
    rng = np.random.RandomState(seed)
    pos = rng.randn(N, 3) * 2.5
    return pos

def make_perturbed_mackay(seed, r_inner=1.04, r_outer=1.96, scale=0.05):
    """Generate a perturbed Mackay icosahedron."""
    base = make_mackay_icosahedron(r_inner, r_outer)
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def relax(pos, maxiter=500, gtol=1e-10, ftol=1e-12):
    """Relax using L-BFGS-B with gradient."""
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
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Start with the correct two-shell Mackay icosahedron
    best_pos = make_mackay_icosahedron(r_inner=1.04, r_outer=1.96)
    best_e = lj_energy(best_pos)
    
    # First: high-precision relaxation of the base Mackay structure
    pos_relaxed, e_relaxed = relax(best_pos, maxiter=1000, gtol=1e-12, ftol=1e-14)
    if e_relaxed < best_e:
        best_e = e_relaxed
        best_pos = pos_relaxed.copy()
    
    # Try a few nearby radial configurations to see if we can improve
    radial_configs = [
        (1.02, 1.94), (1.04, 1.96), (1.06, 1.98),
        (1.00, 1.92), (1.08, 2.00), (1.04, 1.90), (1.04, 2.02)
    ]
    
    for r_in, r_out in radial_configs:
        if time.time() > deadline - 20:
            break
        pos_try = make_mackay_icosahedron(r_inner=r_in, r_outer=r_out)
        e_try = lj_energy(pos_try)
        if e_try < best_e:
            best_e = e_try
            best_pos = pos_try.copy()
        # Relax this configuration
        if time.time() < deadline - 25:
            pos_r, e_r = relax(pos_try, maxiter=500, gtol=1e-11, ftol=1e-13)
            if e_r < best_e:
                best_e = e_r
                best_pos = pos_r.copy()
    
    # Try a few perturbed versions to escape local minima
    for s in range(5):
        if time.time() > deadline - 15:
            break
        pos_pert = make_perturbed_mackay(seed=s, r_inner=1.04, r_outer=1.96, scale=0.03 + 0.02 * s)
        e_pert = lj_energy(pos_pert)
        if e_pert < best_e:
            best_e = e_pert
            best_pos = pos_pert.copy()
        if time.time() < deadline - 20:
            pos_r, e_r = relax(pos_pert, maxiter=500, gtol=1e-11, ftol=1e-13)
            if e_r < best_e:
                best_e = e_r
                best_pos = pos_r.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        pos_final, e_final = relax(best_pos, maxiter=2000, gtol=1e-13, ftol=1e-15)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    
    return best_pos

# EVOLVE-BLOCK-END
