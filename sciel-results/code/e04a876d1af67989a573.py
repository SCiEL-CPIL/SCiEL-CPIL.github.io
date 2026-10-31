import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.optimize import minimize as _minimize

N = 38
DEADLINE = 142.0
START_TIME = time.time()

def lj_energy(positions):
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = float(np.dot(rij, rij))
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0

def check_validity(positions, min_dist=0.5):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            if np.dot(rij, rij) < min_dist * min_dist:
                return False
    return True

def _get_fcc_shell_structure(n_atoms=38):
    """Return base FCC positions and shell assignments for LJ38 truncated octahedron."""
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 8
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    # Assign shells based on radial distance
    radii = np.linalg.norm(coords, axis=1)
    # Cluster radii into shells
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    return coords, shell_ids, unique_r

def make_fcc_truncated_octahedron(n_atoms=38):
    coords, _, _ = _get_fcc_shell_structure(n_atoms)
    coords *= 2.0 ** (1.0 / 6.0)
    return coords

def _symmetry_adapted_energy_and_grad(params):
    """
    params: [global_scale, shell_breathing_0, shell_breathing_1, ...]
    Returns energy and gradient w.r.t. params.
    """
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    
    # Apply shell breathing: each atom's position is scaled by (1 + breathing[shell])
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    # Compute energy
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient w.r.t. global_scale
    # dE/d(scale) = sum over pairs of dE/dr * dr/dscale
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    # r = |pos_i - pos_j|, dr/dscale = r/scale (since pos ~ scale * base)
    # Actually: pos_i = scale * breath_i * base_i
    # dr/d(scale) for pair (i,j): d/dscale |scale*breath_i*base_i - scale*breath_j*base_j|
    # = d/dscale |scale * (breath_i*base_i - breath_j*base_j)|
    # = |breath_i*base_i - breath_j*base_j| / |breath_i*base_i - breath_j*base_j| * (breath_i*base_i - breath_j*base_j)
    # = (breath_i*base_i - breath_j*base_j) / r
    # So dE/dscale = sum_pairs coeff_ij * (breath_i*base_i - breath_j*base_j) / r * scale
    # Wait, let me think more carefully.
    # pos_i = scale * b_i * base_i where b_i = 1 + shell_breathing[shell_i]
    # diff_ij = pos_i - pos_j = scale * (b_i * base_i - b_j * base_j)
    # r_ij = |diff_ij| = scale * |b_i * base_i - b_j * base_j|
    # dE/dscale = sum_ij dE/dr_ij * dr_ij/dscale
    # dr_ij/dscale = |b_i * base_i - b_j * base_j| = r_ij / scale
    # So dE/dscale = sum_ij coeff_ij * r_ij / scale
    # But coeff already has the right sign... let me just use numerical grad for simplicity
    # Actually let's do it properly with chain rule
    
    # For global_scale: dE/dscale = sum over pairs of (dE/dr) * (dr/dscale)
    # dr/dscale = r/scale (homogeneous of degree 1)
    grad_scale = np.sum(coeff * r2) / (2.0 * global_scale)  # since r2 = r^2, and sum coeff*r = sum coeff*r2/r... 
    # Actually: sum_ij coeff_ij * r_ij = sum_ij coeff_ij * sqrt(r2_ij)
    # Let me just compute numerically for robustness
    
    # Numerical gradient for robustness
    eps = 1e-6
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_plus[k] += eps
        e_plus = _symmetry_adapted_energy(params_plus)
        params_minus = params.copy()
        params_minus[k] -= eps
        e_minus = _symmetry_adapted_energy(params_minus)
        grad[k] = (e_plus - e_minus) / (2 * eps)
    
    return energy, grad

def _symmetry_adapted_energy(params):
    """Compute energy for symmetry-adapted parameters."""
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0

def optimize_symmetry_adapted(initial_scale=None, maxiter=500):
    """Optimize the low-dimensional symmetry-adapted parameterization."""
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    # Initial guess: scale = 2^(1/6), all breathing = 0
    if initial_scale is None:
        initial_scale = 2.0 ** (1.0/6.0)
    
    x0 = np.array([initial_scale] + [0.0] * n_shells)
    
    try:
        result = _minimize(_symmetry_adapted_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': 1e-12, 'maxls': 150, 'ftol': 1e-16})
        return result.x
    except Exception:
        return x0

def optimize_global_scale(positions, maxiter=200):
    """Optimize only the global scale of a given position array."""
    def scale_energy(scale):
        p = positions * scale
        return lj_energy_vectorized(p)
    
    x0 = np.array([1.0])
    try:
        result = _minimize(scale_energy, x0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-14})
        return result.x[0]
    except Exception:
        return 1.0

def make_fcc_with_noise(n_atoms=38, noise_scale=0.05, seed=42):
    base = make_fcc_truncated_octahedron(n_atoms)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise_scale, base.shape)

def _lj_energy_and_grad(pos_flat):
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    return energy, grad_flat

def _make_initial_clusters():
    clusters = []
    # Primary: FCC truncated octahedron
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            clusters.append(fcc)
    except Exception:
        pass

    # Symmetry-adapted optimization from default scale
    try:
        params_opt = optimize_symmetry_adapted(initial_scale=2.0 ** (1.0/6.0), maxiter=300)
        base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
        global_scale = params_opt[0]
        shell_params = params_opt[1:]
        positions = np.zeros((N, 3))
        for i in range(N):
            s = shell_ids[i]
            if s < len(shell_params):
                breath = 1.0 + shell_params[s]
            else:
                breath = 1.0
            positions[i] = base_coords[i] * breath * global_scale
        if check_validity(positions):
            clusters.append(positions)
    except Exception:
        pass

    # Symmetry-adapted with slightly different initial scales
    for scale_try in [0.99 * 2.0 ** (1.0/6.0), 1.01 * 2.0 ** (1.0/6.0)]:
        try:
            params_opt = optimize_symmetry_adapted(initial_scale=scale_try, maxiter=200)
            base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
            global_scale = params_opt[0]
            shell_params = params_opt[1:]
            positions = np.zeros((N, 3))
            for i in range(N):
                s = shell_ids[i]
                if s < len(shell_params):
                    breath = 1.0 + shell_params[s]
                else:
                    breath = 1.0
                positions[i] = base_coords[i] * breath * global_scale
            if check_validity(positions):
                clusters.append(positions)
        except Exception:
            pass

    # Small-noise FCC variants
    for s in (1, 2, 3, 4, 5, 6):
        try:
            c = make_fcc_with_noise(N, noise_scale=0.04, seed=s)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass

    # Larger-noise FCC variants
    for s in (7, 8, 9, 10):
        try:
            c = make_fcc_with_noise(N, noise_scale=0.10, seed=s)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass

    # BCC-like fragment: cubic lattice scaled
    try:
        side = int(math.ceil(N ** (1.0/3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append((i - side//2, j - side//2, k - side//2))
        pts.sort(key=lambda p: p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
        pts = pts[:N]
        arr = np.array(pts, dtype=float)
        arr *= 1.28 * 2.0 ** (1.0/6.0)
        if check_validity(arr):
            clusters.append(arr)
    except Exception:
        pass

    # Icosahedral-inspired: two concentric shells with icosahedral symmetry
    try:
        ic = _make_icosahedral(N)
        if check_validity(ic):
            clusters.append(ic)
    except Exception:
        pass

    return clusters

def _make_icosahedral(n_atoms=38):
    # Build an icosahedron + inner core
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    # icosahedron has 12 vertices
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    # scale outer shell
    outer_r = 2.8
    inner_r = 1.3
    # place 12 on outer, 12 on inner rotated, 1 at center, plus extras
    outer = verts * outer_r
    inner = verts * inner_r
    # rotate inner by 30 deg around z
    ang = math.pi / 6.0
    R = np.array([[math.cos(ang), -math.sin(ang), 0],
                  [math.sin(ang), math.cos(ang), 0],
                  [0, 0, 1]])
    inner = inner @ R.T
    center = np.array([[0.0, 0.0, 0.0]])
    pts = np.vstack([center, inner, outer])
    if len(pts) >= n_atoms:
        pts = pts[:n_atoms]
    else:
        # add more random
        rng = np.random.default_rng(0)
        extra = rng.normal(0, 0.5, (n_atoms - len(pts), 3))
        pts = np.vstack([pts, extra])
    # scale overall
    pts *= 2.0 ** (1.0 / 6.0) / 1.4
    return pts

def relax(positions, maxiter=400, gtol=1e-9):
    x0 = positions.flatten()
    try:
        result = _minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 60, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=2000, gtol=1e-14):
    x0 = positions.flatten()
    try:
        result = _minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-16})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, n_iter=15, step_size=0.4, maxiter=120, temperature=0.05):
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(99)

    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 12:
            break
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        try:
            result = _minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': maxiter, 'gtol': 1e-7, 'maxls': 60})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue

        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy

    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()

    best_energy = float('inf')
    best_positions = None

    candidates = _make_initial_clusters()

    # First pass: relax all candidates
    for cand in candidates:
        if time.time() - START_TIME > DEADLINE - 45:
            break
        try:
            relaxed = relax(cand, maxiter=300, gtol=1e-9)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass

    # Second pass: global scale optimization
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 40:
        try:
            optimal_scale = optimize_global_scale(best_positions, maxiter=300)
            scaled = best_positions * optimal_scale
            if check_validity(scaled):
                energy = lj_energy_vectorized(scaled)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = scaled
        except Exception:
            pass

    # Third pass: basin hopping with moderate perturbation
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 35:
        try:
            bh_res, bh_e = basin_hopping(best_positions, n_iter=20, step_size=0.3, maxiter=120, temperature=0.05)
            if check_validity(bh_res) and bh_e < best_energy:
                best_energy = bh_e
                best_positions = bh_res
        except Exception:
            pass

    # Fourth pass: basin hopping with larger perturbation to escape local minima
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 28:
        try:
            bh_res2, bh_e2 = basin_hopping(best_positions, n_iter=14, step_size=0.6, maxiter=100, temperature=0.1)
            if check_validity(bh_res2) and bh_e2 < best_energy:
                best_energy = bh_e2
                best_positions = bh_res2
        except Exception:
            pass

    # Fifth pass: symmetry-adapted re-optimization from best found
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            # Project best positions back to symmetry-adapted manifold
            base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
            # Estimate effective scale and breathing from best_positions
            radii_best = np.linalg.norm(best_positions, axis=1)
            radii_base = np.linalg.norm(base_coords, axis=1)
            # Least squares for scale
            mask = radii_base > 1e-6
            scale_est = np.sum(radii_best[mask] * radii_base[mask]) / np.sum(radii_base[mask] * radii_base[mask])
            # Estimate breathing per shell
            shell_params_est = np.zeros(len(unique_r))
            for s, ur in enumerate(unique_r):
                shell_mask = (shell_ids == s) & (radii_base > 1e-6)
                if np.any(shell_mask):
                    shell_params_est[s] = np.mean(radii_best[shell_mask] / (scale_est * radii_base[shell_mask])) - 1.0
            
            params_init = np.array([scale_est] + list(shell_params_est))
            params_opt = optimize_symmetry_adapted(initial_scale=scale_est, maxiter=400)
            
            global_scale = params_opt[0]
            shell_params = params_opt[1:]
            positions = np.zeros((N, 3))
            for i in range(N):
                s = shell_ids[i]
                if s < len(shell_params):
                    breath = 1.0 + shell_params[s]
                else:
                    breath = 1.0
                positions[i] = base_coords[i] * breath * global_scale
            
            if check_validity(positions):
                energy = lj_energy_vectorized(positions)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Sixth pass: final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 15:
        try:
            polished = polish(best_positions, maxiter=5000, gtol=1e-16)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # Seventh pass: short basin hop after polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            bh_final, bh_final_e = basin_hopping(best_positions, n_iter=10, step_size=0.15, maxiter=80, temperature=0.01)
            if check_validity(bh_final) and bh_final_e < best_energy:
                best_energy = bh_final_e
                best_positions = bh_final
                # Polish again
                if time.time() - START_TIME < DEADLINE - 5:
                    polished2 = polish(best_positions, maxiter=3000, gtol=1e-16)
                    if check_validity(polished2):
                        energy2 = lj_energy_vectorized(polished2)
                        if energy2 < best_energy:
                            best_energy = energy2
                            best_positions = polished2
        except Exception:
            pass

    if best_positions is None:
        try:
            best_positions = make_fcc_truncated_octahedron(N)
        except Exception:
            side = int(math.ceil(N ** (1/3)))
            pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
            best_positions = np.array(pts[:N], dtype=float) * 1.3

    return best_positions
