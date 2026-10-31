"""
LJ38 cluster global minimum search.

Approach:
1. Generate FCC core (19 atoms) and outer shell candidates (24 atoms).
2. Multiple starting configurations: deterministic subsets, random subsets, icosahedral.
3. L-BFGS relaxation from each start.
4. Symmetry-adapted parameterization (shell breathing) for refined search in FCC basin.
5. Basin hopping from best results to escape local minima.
6. Final high-precision polish.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _generate_fcc_core_and_outer():
    """
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms.
    Using proper FCC primitive vectors: a1=(0,1,1), a2=(1,0,1), a3=(1,1,0)
    """
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 20.0:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    # Core: distances squared <= 4 (includes 0, 2, 4)
    core_pts = pts[pts[:, 3] <= 4.1]
    # Outer: distances squared in (4, 6]
    outer_pts = pts[(pts[:, 3] > 4.1) & (pts[:, 3] <= 6.1)]
    
    return core_pts[:, :3], outer_pts[:, :3]

def _make_icosahedral_structure():
    """
    Icosahedral starting geometry.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    pts.append(np.array([0.25, 0.25, 0.25]))
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale
    return pts

def _relax(pos, max_iter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _optimize_scale(pos):
    """
    Optimize the global scale of the cluster to minimize energy.
    """
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    res = minimize_scalar(energy_at_scale, bounds=(0.8, 1.5), method='bounded', options={'xatol': 1e-10})
    if res.success:
        optimal_scale = res.x
        optimal_pos = pos * optimal_scale
        optimal_energy = res.fun
        return optimal_pos, optimal_energy
    else:
        return pos, energy_at_scale(1.0)

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def _symmetry_adapted_relax(core, outer, idx_outer, max_iter=300, gtol=1e-10):
    """
    Symmetry-adapted parameterization with shell breathing.
    Parameters:
      - core_scale: scale for the 19 core atoms
      - outer_scale: scale for the 19 outer atoms  
      - 3 distortion parameters (anisotropic scaling)
      Total: 5 parameters, keeping FCC topology.
    """
    base_pos = np.vstack([core, outer[idx_outer]])
    base_pos -= np.mean(base_pos, axis=0)
    
    # Core atoms: indices 0-18, Outer atoms: indices 19-37
    n_core = 19
    n_outer = 19
    
    def param_to_pos(params):
        core_scale = params[0]
        outer_scale = params[1]
        dist = params[2:5]
        
        # Apply anisotropic distortion
        dist_matrix = np.diag([1.0 + dist[0], 1.0 + dist[1], 1.0 + dist[2]])
        distorted = base_pos @ dist_matrix.T
        
        # Apply shell-specific scaling
        pos = np.zeros_like(distorted)
        pos[:n_core] = distorted[:n_core] * core_scale
        pos[n_core:] = distorted[n_core:] * outer_scale
        
        # Center
        pos -= np.mean(pos, axis=0)
        
        return pos
    
    def obj_grad(params):
        pos = param_to_pos(params)
        e, g = _lj_energy_and_grad(pos)
        
        # Analytical gradient via chain rule
        # pos = f(params), so dE/dparams = (dE/dpos)^T * (dpos/dparams)
        # For simplicity, use finite differences (robust, 5D is cheap)
        grad = np.zeros_like(params)
        eps = 1e-7
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            e_plus, _ = _lj_energy_and_grad(param_to_pos(params_plus))
            grad[i] = (e_plus - e) / eps
        
        return e, grad
    
    # Initial parameters: start from unit scale, small distortion
    params0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    
    res = minimize(
        obj_grad,
        params0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    
    pos_out = param_to_pos(res.x)
    e_out, _ = _lj_energy_and_grad(pos_out)
    return pos_out, e_out

def _basin_hop(pos, rng, max_iters=50, T_start=0.15, T_end=0.02):
    """
    Basin hopping: perturb, relax, accept/reject by Metropolis criterion.
    """
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy

    for i in range(max_iters):
        # Perturbation: add Gaussian noise with decreasing amplitude
        T_curr = T_start * (T_end / T_start) ** (i / max_iters)
        perturbed = current_pos + rng.normal(0, T_curr, size=current_pos.shape)
        
        # Relax
        relaxed, e_relaxed = _relax(perturbed, max_iter=150, gtol=1e-9)
        e_relaxed, _ = _lj_energy_and_grad(relaxed)
        
        # Metropolis acceptance
        if e_relaxed < current_energy:
            accept = True
        else:
            delta_e = e_relaxed - current_energy
            accept = rng.random() < np.exp(-delta_e / T_curr)
        
        if accept:
            current_pos = relaxed.copy()
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    starts = []
    random_idx_list = []
    
    # 1. Deterministic "38 closest" (first 19 outer atoms)
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    diff = det_start[:, np.newaxis, :] - det_start[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        det_start *= scale
    starts.append(det_start)
    
    # 2. Random subsets of outer atoms
    num_random_starts = 80
    for i in range(num_random_starts):
        if time.time() > deadline - 40.0:
            break
        idx = rng.choice(len(outer), size=19, replace=False)
        random_idx_list.append(idx)
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        
        diff = start_pos[:, np.newaxis, :] - start_pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        nn = np.sqrt(np.min(r2, axis=1))
        avg_nn = np.mean(nn)
        if avg_nn > 1e-12:
            scale = (2.0 ** (1.0 / 6.0)) / avg_nn
            start_pos *= scale
            
        starts.append(start_pos)
        
    # 3. Icosahedral start
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    best_pos = None
    best_energy = np.inf

    # Phase 1: Standard L-BFGS relaxation from all starts
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            p, e = _relax(start, max_iter=200)
            # Apply scale optimization
            p, e = _optimize_scale(p)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Phase 2: Symmetry-adapted relaxation on best candidates
    if time.time() < deadline - 40.0 and best_pos is not None:
        # Try symmetry-adapted on deterministic and top random subsets
        parametric_starts = [None] + random_idx_list[:8]
        for idx in parametric_starts:
            if time.time() > deadline - 30.0:
                break
            try:
                if idx is None:
                    p, e = _symmetry_adapted_relax(core, outer, np.arange(19), max_iter=200)
                else:
                    p, e = _symmetry_adapted_relax(core, outer, idx, max_iter=200)
                p, e = _optimize_scale(p)
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            except Exception:
                continue

    # Phase 3: Basin hopping from best result
    if time.time() < deadline - 20.0 and best_pos is not None:
        try:
            p, e = _basin_hop(best_pos, rng, max_iters=30, T_start=0.12, T_end=0.01)
            p, e = _optimize_scale(p)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Final polish: high-precision relaxation
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=3000, gtol=1e-13)
            p, e = _optimize_scale(p)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
