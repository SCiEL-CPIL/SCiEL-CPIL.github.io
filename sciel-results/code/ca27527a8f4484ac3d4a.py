"""Lennard-Jones 55-atom cluster: Mackay icosahedron with optimized radii + multi-start relaxation."""
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

def get_icosahedron_unit_vectors():
    """Return 12 unit vectors pointing to icosahedron vertices."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def get_edge_midpoint_unit_vectors():
    """Return 30 unit vectors pointing to icosahedron edge midpoints."""
    verts = get_icosahedron_unit_vectors()
    edge_len = np.linalg.norm(verts[0] - verts[1])
    midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                midpoints.append(mid)
    return np.array(midpoints)

def make_mackay_icosahedron(r1=1.0, r2=2.0, r3=2.8):
    """Generate a Mackay icosahedral cluster with 55 atoms.
    
    Structure: 1 center + 12 (icosahedron vertices at r1) + 30 (edge midpoints at r2) + 12 (icosahedron vertices at r3)
    """
    ico_verts = get_icosahedron_unit_vectors()
    edge_mids = get_edge_midpoint_unit_vectors()
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = ico_verts * r1
    shell2 = edge_mids * r2
    shell3 = ico_verts * r3
    
    pos = np.vstack([center, shell1, shell2, shell3])
    return pos

def make_cubic_lattice():
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                points.append((i, j, k))
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def relax(pos, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with tight tolerance."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True,
                         options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def optimize_radii_and_relax(start_time, deadline):
    """Optimize the three radii of the Mackay icosahedron, then relax."""
    ico_verts = get_icosahedron_unit_vectors()
    edge_mids = get_edge_midpoint_unit_vectors()
    
    # Precompute direction arrays
    center = np.array([[0.0, 0.0, 0.0]])
    
    def objective_and_relax(radii):
        r1, r2, r3 = radii
        pos = np.vstack([center, ico_verts * r1, edge_mids * r2, ico_verts * r3])
        
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2_arr = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2_arr, np.inf)
        min_dist = np.sqrt(np.min(r2_arr))
        if min_dist < 0.5:
            return 1e10
        
        e, g = lj_energy_grad(pos)
        
        # Relax from this configuration
        pos_relaxed, e_relaxed = relax(pos, maxiter=100, gtol=1e-6)
        
        # Check constraint on relaxed
        diff_r = pos_relaxed[:, np.newaxis, :] - pos_relaxed[np.newaxis, :, :]
        r2_r = np.einsum('ijk,ijk->ij', diff_r, diff_r)
        np.fill_diagonal(r2_r, np.inf)
        min_dist_r = np.sqrt(np.min(r2_r))
        if min_dist_r < 0.5:
            return 1e10
        
        return e_relaxed
    
    # Grid search over radii
    best_e = 1e10
    best_radii = (1.0, 2.0, 2.8)
    
    r1_range = np.linspace(0.8, 1.4, 5)
    r2_range = np.linspace(1.8, 2.6, 5)
    r3_range = np.linspace(2.4, 3.4, 5)
    
    for r1 in r1_range:
        for r2 in r2_range:
            for r3 in r3_range:
                if time.time() > deadline - 10:
                    break
                e = objective_and_relax((r1, r2, r3))
                if e < best_e:
                    best_e = e
                    best_radii = (r1, r2, r3)
    
    # Refine around best with finer grid
    r1_c, r2_c, r3_c = best_radii
    r1_fine = np.linspace(max(0.5, r1_c - 0.1), r1_c + 0.1, 5)
    r2_fine = np.linspace(max(0.5, r2_c - 0.1), r2_c + 0.1, 5)
    r3_fine = np.linspace(max(0.5, r3_c - 0.1), r3_c + 0.1, 5)
    
    for r1 in r1_fine:
        for r2 in r2_fine:
            for r3 in r3_fine:
                if time.time() > deadline - 5:
                    break
                e = objective_and_relax((r1, r2, r3))
                if e < best_e:
                    best_e = e
                    best_radii = (r1, r2, r3)
    
    # Final high-precision relaxation
    pos_best = make_mackay_icosahedron(*best_radii)
    pos_final, e_final = relax(pos_best, maxiter=500, gtol=1e-10)
    
    return pos_final, e_final

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_pos = make_mackay_icosahedron(1.0, 2.0, 2.8)
    best_e = lj_energy(best_pos)
    
    # Strategy 1: Optimize radii of Mackay icosahedron
    if time.time() < deadline - 30:
        pos_opt, e_opt = optimize_radii_and_relax(start_time, deadline)
        if e_opt < best_e:
            best_e = e_opt
            best_pos = pos_opt.copy()
    
    # Strategy 2: Multi-start relaxation from various initial configurations
    candidates = []
    
    # Cubic lattice
    candidates.append(make_cubic_lattice())
    
    # Random clusters
    rng = np.random.RandomState(42)
    for s in range(5):
        candidates.append(rng.randn(N, 3) * 2.0)
    
    # Perturbed Mackay with various scales
    base = make_mackay_icosahedron(1.0, 2.0, 2.8)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        candidates.append(base + rng.randn(N, 3) * scale)
    
    # Perturbed optimized Mackay
    base_opt = make_mackay_icosahedron(1.0, 2.0, 2.8)
    for scale in [0.05, 0.1]:
        candidates.append(base_opt + rng.randn(N, 3) * scale)
    
    for pos in candidates:
        if time.time() > deadline - 10:
            break
        
        e_raw = lj_energy(pos)
        if e_raw < best_e:
            best_e = e_raw
            best_pos = pos.copy()
        
        if time.time() < deadline - 5:
            pos_relaxed, e_relaxed = relax(pos, maxiter=150, gtol=1e-7)
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        pos_polished, e_polished = relax(best_pos, maxiter=500, gtol=1e-10)
        if e_polished < best_e:
            best_e = e_polished
            best_pos = pos_polished
    
    return best_pos

# EVOLVE-BLOCK-END
