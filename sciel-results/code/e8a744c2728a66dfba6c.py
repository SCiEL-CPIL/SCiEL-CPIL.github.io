import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin within 120s/150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    # coeff_ij = 4 * (6/r8 - 12/r14)
    # grad_i = sum_j coeff_ij * (pos_i - pos_j)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _relax(pos, max_iter=1000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        e, _ = _lj_energy_and_grad(pos)
        return pos, e

def _optimize_scale(pos, n_iter=50):
    """
    1D global search over uniform scale factor s applied to centered positions.
    Finds the s that minimizes total LJ energy.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    
    # Coarse scan
    for s in np.linspace(0.95, 1.05, 201):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.05 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 201):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
                
    return centered * best_s + center, best_e

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def _topology_loss_and_grad(pos, target_radii, target_coords):
    """
    Geometric objective function penalizing deviations from FCC truncated octahedron
    shell radii and coordination counts.
    
    Args:
        pos: (N, 3) positions
        target_radii: (N,) target radial distances from center for each atom (sorted)
        target_coords: (N,) target coordination numbers for each atom (sorted)
        
    Returns:
        loss, grad
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    radii = np.linalg.norm(centered, axis=1)
    
    # Sort radii to match sorted targets
    sort_idx = np.argsort(radii)
    sorted_radii = radii[sort_idx]
    
    # Radial loss: sum of squared deviations
    radial_loss = np.sum((sorted_radii - target_radii) ** 2)
    
    # Coordination loss: count neighbors within cutoff for each atom
    # Use a smooth approximation: sum of exp(-(r - cutoff)^2 / sigma^2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    cutoff = 1.2  # Typical neighbor cutoff for LJ
    sigma = 0.1
    coord_matrix = np.exp(-((r - cutoff) ** 2) / (sigma ** 2))
    np.fill_diagonal(coord_matrix, 0.0)
    coords = np.sum(coord_matrix, axis=1)
    
    # Sort coords to match sorted targets
    coord_sort_idx = np.argsort(coords)
    sorted_coords = coords[coord_sort_idx]
    
    coord_loss = np.sum((sorted_coords - target_coords) ** 2)
    
    total_loss = radial_loss + 0.5 * coord_loss
    
    # Gradient computation (approximate via finite differences for simplicity,
    # but we'll use analytic where possible)
    # For now, use a simple gradient estimate
    eps = 1e-6
    grad = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for j in range(3):
            pos_plus = pos.copy()
            pos_minus = pos.copy()
            pos_plus[i, j] += eps
            pos_minus[i, j] -= eps
            lp, _ = _topology_loss_and_grad(pos_plus, target_radii, target_coords)
            lm, _ = _topology_loss_and_grad(pos_minus, target_radii, target_coords)
            grad[i, j] = (lp - lm) / (2 * eps)
    
    return total_loss, grad

def _topology_relax(pos, target_radii, target_coords, max_iter=200, gtol=1e-6):
    """
    Relax using topology loss to guide towards FCC-like geometry.
    """
    def obj_grad(x):
        p = x.reshape(-1, 3)
        l, g = _topology_loss_and_grad(p, target_radii, target_coords)
        return l, g.ravel()
    
    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol},
        )
        p_out = res.x.reshape(-1, 3)
        return p_out
    except Exception:
        return pos

def construct():
    t0 = time.time()
    
    # 1. Generate initial FCC geometry
    try:
        pos0 = _make_fcc_cluster()
    except Exception:
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pos0 = np.array(pts[:N], dtype=np.float64)
        pos0 -= np.mean(pos0, axis=0)
        pos0 *= 1.12

    best_pos = pos0
    best_energy = np.inf

    # 2. Initial relaxation
    try:
        p, e = _relax(pos0, max_iter=500, gtol=1e-8)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass

    # 3. 1D scale optimization
    if time.time() - t0 < DEADLINE - 15.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    # 4. Basin Hopping with topology-guided perturbation
    rng = np.random.default_rng(42)
    
    # Parameters for Basin Hopping
    step_size = 0.5 
    T = 1.0 # Temperature for Metropolis acceptance
    
    # Precompute target radii and coords for FCC cluster
    fcc_pos = _make_fcc_cluster()
    center = np.mean(fcc_pos, axis=0)
    centered = fcc_pos - center
    radii = np.linalg.norm(centered, axis=1)
    target_radii = np.sort(radii)
    
    # Estimate coordination numbers
    diff = fcc_pos[:, np.newaxis, :] - fcc_pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    coords = np.sum(r < 1.2, axis=1)
    target_coords = np.sort(coords.astype(float))
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    # Track acceptance rate for adaptive step size
    accept_count = 0
    total_count = 0
    window_size = 20
    accept_window = []
    
    while time.time() - t0 < DEADLINE - 15.0:
        total_count += 1
        
        # Adaptive step size based on acceptance rate
        if len(accept_window) >= window_size:
            accept_rate = sum(accept_window) / len(accept_window)
            if accept_rate < 0.1:
                step_size *= 1.2
            elif accept_rate > 0.8:
                step_size *= 0.8
            step_size = np.clip(step_size, 0.1, 2.0)
            accept_window.pop(0)
        
        # Perturb
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        new_pos, new_energy = _relax(perturbed, max_iter=500, gtol=1e-6, ftol=1e-10)
        
        # Check separation constraint
        new_pos = _ensure_min_sep(new_pos)
        # Re-evaluate energy after ensuring separation
        new_energy, _ = _lj_energy_and_grad(new_pos)
        
        # Acceptance Criterion
        delta_e = new_energy - current_energy
        
        accepted = False
        if delta_e < 0:
            # Better energy, accept
            current_pos = new_pos
            current_energy = new_energy
            accepted = True
            
            # Update global best
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos
                
        else:
            # Worse energy, accept with probability exp(-delta_e / T)
            if rng.random() < np.exp(-delta_e / T):
                current_pos = new_pos
                current_energy = new_energy
                accepted = True
        
        accept_window.append(1 if accepted else 0)
        
        # Occasionally use topology-guided relaxation to escape local minima
        if total_count % 10 == 0 and time.time() - t0 < DEADLINE - 20.0:
            try:
                topo_pos = _topology_relax(current_pos, target_radii, target_coords, max_iter=50, gtol=1e-4)
                topo_pos, topo_energy = _relax(topo_pos, max_iter=200, gtol=1e-6)
                topo_pos = _ensure_min_sep(topo_pos)
                topo_energy, _ = _lj_energy_and_grad(topo_pos)
                
                if topo_energy < best_energy:
                    best_energy = topo_energy
                    best_pos = topo_pos
                    current_pos = topo_pos
                    current_energy = topo_energy
            except Exception:
                pass

    # 5. Final High-Precision Polish
    if time.time() - t0 < DEADLINE - 2.0:
        try:
            # First, try to optimize scale of the best found
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=20)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
            
            # Then, ultra-tight relaxation
            p, e = _relax(best_pos, max_iter=5000, gtol=1e-13, ftol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 6. Final check and separation enforcement
    best_pos = _ensure_min_sep(best_pos)
    
    # One last energy check to be sure
    final_energy, _ = _lj_energy_and_grad(best_pos)
    
    return best_pos
