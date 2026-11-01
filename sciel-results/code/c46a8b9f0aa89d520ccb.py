"""Lennard-Jones 55-atom cluster: optimize Mackay radii + multi-start stochastic search."""
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
    outer_dirs = np.vstack([midpoints, face_centers[:12]])
    shell2 = outer_dirs * r_outer
    
    pos = np.vstack([inner, shell1, shell2])
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

def optimize_radii(r_init_in=1.0, r_init_out=2.0, deadline=None):
    """Optimize the two shell radii using Nelder-Mead on the relaxed energy."""
    def energy_from_radii(radii):
        r_in, r_out = radii[0], radii[1]
        if r_in < 0.5 or r_out < 1.0 or r_out < r_in + 0.3:
            return 1e10
        pos = make_mackay_icosahedron(r_inner=r_in, r_outer=r_out)
        # Quick check: ensure minimum distance > 0.5
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(r2.min())
        if min_r < 0.5:
            return 1e10
        e = lj_energy(pos)
        return e
    
    def relaxed_energy_from_radii(radii):
        r_in, r_out = radii[0], radii[1]
        if r_in < 0.5 or r_out < 1.0 or r_out < r_in + 0.3:
            return 1e10
        pos = make_mackay_icosahedron(r_inner=r_in, r_outer=r_out)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(r2.min())
        if min_r < 0.5:
            return 1e10
        # Relax to get a better estimate
        pos_r, e_r = relax(pos, maxiter=200, gtol=1e-8, ftol=1e-10)
        # Check min distance after relaxation
        diff = pos_r[:, np.newaxis, :] - pos_r[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(r2.min())
        if min_r < 0.5:
            return 1e10
        return e_r
    
    x0 = np.array([r_init_in, r_init_out])
    
    try:
        result = minimize(relaxed_energy_from_radii, x0, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
        if result.fun < energy_from_radii(x0):
            return result.x[0], result.x[1], result.fun
    except Exception:
        pass
    
    return x0[0], x0[1], energy_from_radii(x0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Phase 1: Optimize radii using Nelder-Mead
    r_in_opt, r_out_opt, e_radial = optimize_radii(r_init_in=1.0, r_init_out=2.0, deadline=deadline)
    
    # Make the optimized Mackay structure
    pos_base = make_mackay_icosahedron(r_inner=r_in_opt, r_outer=r_out_opt)
    
    # Relax the base structure
    pos_relaxed, e_relaxed = relax(pos_base, maxiter=1000, gtol=1e-12, ftol=1e-14)
    best_pos = pos_relaxed.copy()
    best_e = e_relaxed
    
    # Phase 2: Multi-start stochastic search with varied perturbation scales
    rng = np.random.RandomState(42)
    n_trials = 0
    while time.time() < deadline - 15:
        n_trials += 1
        # Vary the perturbation scale
        scale = 0.02 + 0.03 * (n_trials % 10)
        pos_pert = pos_base + rng.randn(N, 3) * scale
        
        # Check minimum distance
        diff = pos_pert[:, np.newaxis, :] - pos_pert[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(r2.min())
        if min_r < 0.5:
            continue
        
        # Relax
        pos_r, e_r = relax(pos_pert, maxiter=300, gtol=1e-10, ftol=1e-12)
        
        # Check validity after relaxation
        diff = pos_r[:, np.newaxis, :] - pos_r[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(r2.min())
        if min_r < 0.5:
            continue
        
        if e_r < best_e:
            best_e = e_r
            best_pos = pos_r.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        pos_final, e_final = relax(best_pos, maxiter=2000, gtol=1e-13, ftol=1e-15)
        # Validate
        diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(r2.min())
        if min_r >= 0.5 and e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    
    return best_pos

# EVOLVE-BLOCK-END
