# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

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
    # grad_i = sum_{j != i} 4 * (-12 * r_ij^(-14) + 6 * r_ij^(-8)) * (pos_i - pos_j)
    # In terms of r2: r^(-14) = r2^(-7), r^(-8) = r2^(-4)
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def get_mackay_directions():
    """Returns directions for the Mackay icosahedron shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron (distance between adjacent vertices)
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
    
    return verts, midpoints

def make_mackay_55():
    """
    Create a 55-atom cluster based on Mackay icosahedron geometry.
    Uses a 54D radial manifold: each non-center atom sits on its icosahedral
    direction with an independently optimized radial distance.
    """
    verts, mids = get_mackay_directions()
    # 54 directions: 12 verts (inner) + 30 mids + 12 verts (outer)
    dirs = np.vstack([verts, mids, verts])  # shape (54, 3)
    
    # Initial radii: inner shell ~1.0, middle ~1.8, outer ~2.6
    r0 = np.concatenate([
        np.full(12, 1.0),
        np.full(30, 1.8),
        np.full(12, 2.6)
    ])
    
    def radial_energy_and_grad(r):
        """Energy and gradient w.r.t. radial parameters."""
        # Clip to avoid division by zero / negative radii
        r_safe = np.maximum(r, 0.5)
        pos = np.vstack([np.array([[0.0, 0.0, 0.0]]), dirs * r_safe[:, np.newaxis]])
        e, grad_pos = lj_energy_grad(pos)
        # Chain rule: dE/dr_i = dot(grad_pos[i], dirs[i]) for i=1..54
        # grad_pos[0] is gradient w.r.t. center atom (not used)
        grad_r = np.einsum('ij,ij->i', grad_pos[1:], dirs)
        return e, grad_r
    
    # Optimize on radial manifold using L-BFGS-B
    res = minimize(radial_energy_and_grad, r0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-12})
    
    r_opt = np.maximum(res.x, 0.5)
    
    pos = np.vstack([np.array([[0.0, 0.0, 0.0]]), dirs * r_opt[:, np.newaxis]])
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
    pos = rng.randn(N, 3) * 2.0
    return pos

def make_perturbed_mackay(seed, scale=0.1):
    """Generate a perturbed Mackay icosahedron."""
    base = make_mackay_55()
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def relax(pos, maxiter=500, gtol=1e-6):
    """Relax using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Generate the optimized Mackay base (54D radial optimization)
    base = make_mackay_55()
    best_pos = base.copy()
    best_e = lj_energy(best_pos)
    
    candidates = []
    
    # 1. The optimized Mackay base itself
    candidates.append(("mackay_base", base.copy()))
    
    # 2. Small perturbations (most likely to improve)
    for s in range(20):
        candidates.append((f"pert_s_{s}", make_perturbed_mackay(s, scale=0.03)))
    
    # 3. Medium perturbations
    for s in range(15):
        candidates.append((f"pert_m_{s}", make_perturbed_mackay(s, scale=0.08)))
    
    # 4. Larger perturbations
    for s in range(10):
        candidates.append((f"pert_l_{s}", make_perturbed_mackay(s, scale=0.15)))
    
    # 5. Cubic lattice
    candidates.append(("cubic", make_cubic_lattice()))
    
    # 6. Random
    for s in range(8):
        candidates.append((f"random_{s}", make_random_cluster(s)))

    # Relax candidates in order of priority
    for name, pos in candidates:
        if time.time() > deadline - 25:
            break
            
        e_raw = lj_energy(pos)
        if e_raw < best_e:
            best_e = e_raw
            best_pos = pos.copy()
        
        if time.time() < deadline - 15:
            pos_relaxed, e_relaxed = relax(pos, maxiter=800, gtol=1e-7)
            
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()

    # Final high-precision polish
    if time.time() < deadline - 5:
        pos_final, e_final = relax(best_pos, maxiter=5000, gtol=1e-10)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
        
        # Extra polish pass if time permits
        if time.time() < deadline - 2:
            pos_final2, e_final2 = relax(best_pos, maxiter=3000, gtol=1e-12)
            if e_final2 < best_e:
                best_e = e_final2
                best_pos = pos_final2
            
    return best_pos

# EVOLVE-BLOCK-END
