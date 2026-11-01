"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + aggressive L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_icosahedral_directions():
    """Return the 55 unit directions for a Mackay icosahedron (center + 12 + 30 + 12)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron (12 vertices, 30 edges)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    center = np.array([[0.0, 0.0, 0.0]])
    all_dirs = np.vstack([center, verts_unit, edge_mids, verts_unit])
    return all_dirs  # shape (55, 3)

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 edge-midpoints + 12 vertex-like = 55
    """
    dirs = _get_icosahedral_directions()
    radii = np.concatenate([
        [0.0],  # center
        np.full(12, r1),
        np.full(30, r2),
        np.full(12, r3)
    ])
    return dirs * radii[:, np.newaxis]

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def _relax_science(pos, deadline=None):
    """Try the available science_chemistry.relax as an additional path."""
    try:
        import science_chemistry
        relaxed = science_chemistry.relax(pos.copy())
        if relaxed is not None:
            e = _lj_energy(relaxed)
            if e < _lj_energy(pos):
                return relaxed, e
    except Exception:
        pass
    return pos, _lj_energy(pos)

def _polish(pos, deadline=None):
    """High-precision polish pass with progressively tighter tolerances."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 3000, 'gtol': 1e-12, 'maxls': 300, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_energy

def _check_min_dist(pos, threshold=0.5):
    """Check if all pairwise distances are >= threshold. Return min distance."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _fix_min_dist(pos, threshold=0.5):
    """If min distance is below threshold, scale up and relax."""
    min_d = _check_min_dist(pos, threshold)
    if min_d < threshold:
        scale = threshold / min_d * 1.01
        pos = pos * scale
    return pos

def _optimize_per_atom_radial(dirs, r_init, deadline=None):
    """Optimize per-atom radial coordinates (54 free params, center fixed at 0)."""
    # dirs shape (55,3), r_init shape (55,) with r_init[0]=0
    # Free params: indices 1..54 (54 params)
    
    def radial_obj_grad(x):
        r = np.zeros(55)
        r[0] = 0.0
        r[1:] = x
        pos = dirs * r[:, np.newaxis]
        e, g_cart = _lj_energy_grad(pos)
        # dE/dr_i = dot(g_cart[i], dirs[i])
        g_radial = np.sum(g_cart * dirs, axis=1)[1:]
        return e, g_radial
    
    x0 = r_init[1:].copy()
    
    try:
        result = minimize(
            radial_obj_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 100}
        )
        r_opt = np.zeros(55)
        r_opt[0] = 0.0
        r_opt[1:] = result.x
        pos_opt = dirs * r_opt[:, np.newaxis]
        e_opt = _lj_energy(pos_opt)
        return pos_opt, e_opt
    except Exception:
        return dirs * r_init[:, np.newaxis], _lj_energy(dirs * r_init[:, np.newaxis])

def construct():
    """
    Construct 55-atom LJ cluster near global minimum.
    Strategy:
    1. Multi-start stochastic search: perturb exact Mackay geometry with Gaussian noise,
       relax each with L-BFGS, keep best.
    2. Also try radial manifold optimization as a structured alternative.
    3. Final high-precision polish on the best candidate.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    dirs = _get_icosahedral_directions()
    
    best_pos = None
    best_energy = np.inf
    
    # --- Phase 1: Multi-start stochastic search ---
    # Start from the exact Mackay icosahedron and add Gaussian perturbations
    base_pos = _make_mackay_icosahedron(1.05, 1.75, 1.95)
    
    # Fixed seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Try several perturbation scales
    perturbation_scales = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    for sigma in perturbation_scales:
        if time.time() > deadline - 30.0:
            break
        
        if sigma == 0.0:
            # Unperturbed Mackay icosahedron
            pos = base_pos.copy()
            pos = _fix_min_dist(pos)
            pos_relaxed, e_relaxed = _relax_lbfgs(
                pos, deadline=deadline, maxiter=300, gtol=1e-10
            )
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = pos_relaxed.copy()
        else:
            # Multiple random perturbations for each scale
            n_trials = 3
            for trial in range(n_trials):
                if time.time() > deadline - 25.0:
                    break
                
                noise = rng.normal(0, sigma, size=base_pos.shape)
                # Don't move the center atom (index 0)
                noise[0] = 0.0
                pos = base_pos + noise
                pos = _fix_min_dist(pos)
                
                pos_relaxed, e_relaxed = _relax_lbfgs(
                    pos, deadline=deadline, maxiter=200, gtol=1e-8
                )
                
                if e_relaxed < best_energy:
                    best_energy = e_relaxed
                    best_pos = pos_relaxed.copy()
    
    # --- Phase 2: Per-atom radial manifold optimization ---
    if time.time() < deadline - 35.0:
        radius_inits = [
            np.array([0.0] + [1.05]*12 + [1.75]*30 + [1.95]*12),
            np.array([0.0] + [1.04]*12 + [1.76]*30 + [1.98]*12),
            np.array([0.0] + [1.00]*12 + [1.70]*30 + [1.90]*12),
            np.array([0.0] + [1.06]*12 + [1.74]*30 + [1.96]*12),
        ]
        
        for r_init in radius_inits:
            if time.time() > deadline - 30.0:
                break
            
            pos_radial, e_radial = _optimize_per_atom_radial(dirs, r_init, deadline=deadline)
            pos_radial = _fix_min_dist(pos_radial)
            
            pos_relaxed, e_relaxed = _relax_lbfgs(
                pos_radial, deadline=deadline, maxiter=300, gtol=1e-10
            )
            
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = pos_relaxed.copy()
    
    # --- Phase 3: Try science_chemistry.relax on best candidate ---
    if best_pos is not None and time.time() < deadline - 20.0:
        pos_science, e_science = _relax_science(best_pos, deadline=deadline)
        if e_science < best_energy:
            best_energy = e_science
            best_pos = pos_science.copy()
    
    # Fallback if nothing worked
    if best_pos is None:
        best_pos = _make_mackay_icosahedron(1.05, 1.75, 1.95)
        best_pos = _fix_min_dist(best_pos)
        best_energy = _lj_energy(best_pos)
    
    # --- Phase 4: Final high-precision polish ---
    pos_final, e_final = _polish(best_pos, deadline=deadline)
    
    # Final constraint check
    min_d = _check_min_dist(pos_final)
    if min_d < 0.5:
        pos_final = _fix_min_dist(pos_final)
        pos_final, e_final = _relax_lbfgs(
            pos_final, deadline=deadline, maxiter=100, gtol=1e-10
        )
    
    return pos_final
# EVOLVE-BLOCK-END
