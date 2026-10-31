"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total LJ energy for positions of shape (N, 3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    # Sum over i < j
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    return energy

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions of shape (N, 3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    
    # Gradient: dE/dx_i = sum_j 4*(-12/r^14 + 6/r^8) * (x_i - x_j) / r^2
    # In terms of r2: 4*(-12*r2^-7 + 6*r2^-4) * (x_i - x_j)
    coeff = 4.0 * (-12.0 * r2 ** -7 + 6.0 * r2 ** -4)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _generate_fcc_38(a, dist_x=1.0, dist_y=1.0, dist_z=1.0):
    """Generate 38 atoms in FCC truncated octahedron with anisotropic distortion."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    pts = np.array([i * dist_x, j * dist_y, k * dist_z], dtype=float) * (a / 2.0)
                    points.append(pts)
    points = np.array(points)
    dists = np.sum(points * points, axis=1)
    idx = np.argsort(dists)
    selected = points[idx[:N]]
    selected = selected - np.mean(selected, axis=0)
    return selected

def _param_to_pos(params):
    """Convert 6 parameters [a, dx, dy, dz, tx, ty] to positions."""
    a = params[0]
    dx = params[1]
    dy = params[2]
    dz = params[3]
    tx = params[4]
    ty = params[5]
    pos = _generate_fcc_38(a, dx, dy, dz)
    # Apply small translation to break symmetry
    pos += np.array([tx, ty, 0.0])
    return pos

def _param_energy(params):
    """Energy as function of 6 parameters."""
    pos = _param_to_pos(params)
    # Check minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    if np.min(r2) < 0.25:
        return 1e10
    return _lj_energy(pos)

def _relax_full(pos, time_limit=50.0):
    """Full 114D relaxation using L-BFGS-B."""
    start_time = time.time()
    
    def energy_func(x):
        p = x.reshape(-1, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        if np.min(r2) < 0.25:
            return 1e10
        return _lj_energy(p)
    
    def gradient_func(x):
        p = x.reshape(-1, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        r2 = np.maximum(r2, 1e-12)
        coeff = 4.0 * (-12.0 * r2 ** -7 + 6.0 * r2 ** -4)
        np.fill_diagonal(coeff, 0.0)
        forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        return -forces.flatten()
    
    x0 = pos.flatten()
    try:
        result = minimize(
            energy_func, x0, jac=gradient_func,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        pos_out = result.x.reshape(-1, 3)
        e = _lj_energy(pos_out)
        return pos_out, e
    except Exception:
        return pos, _lj_energy(pos)

def _polish(pos, time_limit=30.0):
    """High-precision polish pass with very tight tolerances."""
    start_time = time.time()
    
    def energy_func(x):
        p = x.reshape(-1, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        if np.min(r2) < 0.25:
            return 1e10
        return _lj_energy(p)
    
    def gradient_func(x):
        p = x.reshape(-1, 3)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        r2 = np.maximum(r2, 1e-12)
        coeff = 4.0 * (-12.0 * r2 ** -7 + 6.0 * r2 ** -4)
        np.fill_diagonal(coeff, 0.0)
        forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        return -forces.flatten()
    
    x0 = pos.flatten()
    try:
        result = minimize(
            energy_func, x0, jac=gradient_func,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        pos_out = result.x.reshape(-1, 3)
        e = _lj_energy(pos_out)
        return pos_out, e
    except Exception:
        return pos, _lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin
    
    # Phase 1: Optimize 6 global parameters (scale + anisotropic distortion)
    # Initial guess: a=2^(1/6), no distortion
    a0 = 2.0 ** (1.0 / 6.0)
    params0 = np.array([a0, 1.0, 1.0, 1.0, 0.0, 0.0])
    
    # Optimize parameters
    try:
        result_param = minimize(
            _param_energy, params0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        best_params = result_param.x
    except Exception:
        best_params = params0
    
    # Convert to positions
    pos_param = _param_to_pos(best_params)
    
    # Ensure minimum distance
    diff = pos_param[:, np.newaxis, :] - pos_param[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    if min_dist < 0.5:
        scale = 0.5 / min_dist * 1.01
        pos_param = pos_param * scale
    
    # Phase 2: Full relaxation from parameter-optimized structure
    if time.time() < deadline - 40.0:
        pos_relaxed, e_relaxed = _relax_full(pos_param, time_limit=35.0)
    else:
        pos_relaxed, e_relaxed = pos_param, _lj_energy(pos_param)
    
    # Phase 3: Also try relaxing from the standard FCC at 2^(1/6)
    pos_fcc = _generate_fcc_38(a0)
    diff = pos_fcc[:, np.newaxis, :] - pos_fcc[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    if min_dist < 0.5:
        scale = 0.5 / min_dist * 1.01
        pos_fcc = pos_fcc * scale
    
    if time.time() < deadline - 35.0:
        pos_fcc_relaxed, e_fcc = _relax_full(pos_fcc, time_limit=30.0)
    else:
        pos_fcc_relaxed, e_fcc = pos_fcc, _lj_energy(pos_fcc)
    
    # Choose the better result
    if e_relaxed <= e_fcc:
        best_pos = pos_relaxed
        best_energy = e_relaxed
    else:
        best_pos = pos_fcc_relaxed
        best_energy = e_fcc
    
    # Phase 4: High-precision polish
    if time.time() < deadline - 25.0:
        best_pos, best_energy = _polish(best_pos, time_limit=20.0)
    
    # Final validity check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    
    if min_dist < 0.5:
        # Fallback: scale up
        scale = 0.5 / min_dist * 1.01
        best_pos = best_pos * scale
    
    if time.time() > deadline:
        return best_pos
    
    return best_pos
# EVOLVE-BLOCK-END
