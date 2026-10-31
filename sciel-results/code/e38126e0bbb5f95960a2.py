import numpy as np
import time
import math
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

def make_fcc_truncated_octahedron(n_atoms=38):
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
    coords *= 2.0 ** (1.0 / 6.0)
    return coords

def make_fcc_with_noise(n_atoms=38, noise_scale=0.05, seed=42):
    base = make_fcc_truncated_octahedron(n_atoms)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise_scale, base.shape)

def _lj_energy_and_grad(pos_flat):
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=1)
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
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    outer_r = 2.8
    inner_r = 1.3
    outer = verts * outer_r
    inner = verts * inner_r
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
        rng = np.random.default_rng(0)
        extra = rng.normal(0, 0.5, (n_atoms - len(pts), 3))
        pts = np.vstack([pts, extra])
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

def polish(positions, maxiter=3000, gtol=1e-15):
    x0 = positions.flatten()
    try:
        result = _minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 300, 'ftol': 1e-18})
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
        if time.time() - START_TIME > DEADLINE - 15:
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

def _scale_optimize(positions, scale_bounds=(0.9, 1.1), n_steps=20):
    """1D optimization of global scale factor to minimize energy."""
    best_scale = 1.0
    best_energy = lj_energy_vectorized(positions)
    
    # First, try a coarse scan
    scales = np.linspace(scale_bounds[0], scale_bounds[1], n_steps)
    for s in scales:
        scaled = positions * s
        if not check_validity(scaled):
            continue
        e = lj_energy_vectorized(scaled)
        if e < best_energy:
            best_energy = e
            best_scale = s
    
    # Then refine around the best scale found
    if best_scale > scale_bounds[0] and best_scale < scale_bounds[1]:
        fine_scales = np.linspace(best_scale - 0.01, best_scale + 0.01, 20)
        for s in fine_scales:
            if s < scale_bounds[0] or s > scale_bounds[1]:
                continue
            scaled = positions * s
            if not check_validity(scaled):
                continue
            e = lj_energy_vectorized(scaled)
            if e < best_energy:
                best_energy = e
                best_scale = s
    
    return positions * best_scale, best_energy

def _radial_density_optimize(positions, n_shells=3):
    """
    Optimize by adjusting radial positions of atoms grouped by shell.
    This is a low-dimensional optimization that preserves the angular structure.
    """
    # Compute distances from center
    center = np.mean(positions, axis=0)
    rel_pos = positions - center
    dists = np.linalg.norm(rel_pos, axis=1)
    
    # Group atoms into shells based on distance
    shell_thresholds = np.percentile(dists, np.linspace(0, 100, n_shells + 1)[1:-1])
    shells = []
    for i in range(n_shells):
        if i == 0:
            mask = dists <= shell_thresholds[0]
        elif i == n_shells - 1:
            mask = dists > shell_thresholds[-1]
        else:
            mask = (dists > shell_thresholds[i-1]) & (dists <= shell_thresholds[i])
        if np.any(mask):
            shells.append(np.where(mask)[0])
    
    def _shell_energy_and_grad(params):
        # params: scale factors for each shell
        new_pos = positions.copy()
        for idx, shell_idx in enumerate(shells):
            if idx < len(params):
                scale = 1.0 + params[idx]  # small perturbation
                new_pos[shell_idx] = center + (rel_pos[shell_idx] * scale)
        
        if not check_validity(new_pos):
            return 1e20, np.zeros(len(params))
        
        e, _ = _lj_energy_and_grad(new_pos.flatten())
        
        # Numerical gradient
        grad = np.zeros(len(params))
        eps = 1e-6
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            new_pos_plus = positions.copy()
            for idx, shell_idx in enumerate(shells):
                if idx < len(params_plus):
                    scale = 1.0 + params_plus[idx]
                    new_pos_plus[shell_idx] = center + (rel_pos[shell_idx] * scale)
            if check_validity(new_pos_plus):
                e_plus, _ = _lj_energy_and_grad(new_pos_plus.flatten())
            else:
                e_plus = 1e20
            grad[i] = (e_plus - e) / eps
        
        return e, grad
    
    x0 = np.zeros(len(shells))
    try:
        result = _minimize(_shell_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50})
        best_params = result.x
        new_pos = positions.copy()
        for idx, shell_idx in enumerate(shells):
            if idx < len(best_params):
                scale = 1.0 + best_params[idx]
                new_pos[shell_idx] = center + (rel_pos[shell_idx] * scale)
        if check_validity(new_pos):
            e = lj_energy_vectorized(new_pos)
            return new_pos, e
    except Exception:
        pass
    
    return positions, lj_energy_vectorized(positions)

def construct():
    global START_TIME
    START_TIME = time.time()

    best_energy = float('inf')
    best_positions = None

    candidates = _make_initial_clusters()

    # First pass: relax all candidates
    for cand in candidates:
        if time.time() - START_TIME > DEADLINE - 50:
            break
        try:
            relaxed = relax(cand, maxiter=300, gtol=1e-8)
            if check_validity(relaxed):
                # Try scale optimization
                scaled, scale_e = _scale_optimize(relaxed, scale_bounds=(0.95, 1.05), n_steps=15)
                if scale_e < lj_energy_vectorized(relaxed):
                    relaxed = scaled
                
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass

    # Second pass: radial density optimization on the best candidate
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 40:
        try:
            rad_opt, rad_e = _radial_density_optimize(best_positions, n_shells=3)
            if check_validity(rad_opt) and rad_e < best_energy:
                best_energy = rad_e
                best_positions = rad_opt
        except Exception:
            pass

    # Third pass: basin hopping with moderate perturbation
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            bh_res, bh_e = basin_hopping(best_positions, n_iter=12, step_size=0.3, maxiter=100, temperature=0.05)
            if check_validity(bh_res):
                # Scale optimize the basin hopping result
                scaled, scale_e = _scale_optimize(bh_res, scale_bounds=(0.95, 1.05), n_steps=10)
                if scale_e < bh_e:
                    bh_res = scaled
                    bh_e = scale_e
                if bh_e < best_energy:
                    best_energy = bh_e
                    best_positions = bh_res
        except Exception:
            pass

    # Fourth pass: basin hopping with larger perturbation to escape local minima
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            bh_res2, bh_e2 = basin_hopping(best_positions, n_iter=8, step_size=0.6, maxiter=80, temperature=0.1)
            if check_validity(bh_res2):
                scaled, scale_e = _scale_optimize(bh_res2, scale_bounds=(0.95, 1.05), n_steps=10)
                if scale_e < bh_e2:
                    bh_res2 = scaled
                    bh_e2 = scale_e
                if bh_e2 < best_energy:
                    best_energy = bh_e2
                    best_positions = bh_res2
        except Exception:
            pass

    # Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 15:
        try:
            polished = polish(best_positions, maxiter=3000, gtol=1e-15)
            if check_validity(polished):
                # One more scale optimization
                scaled, scale_e = _scale_optimize(polished, scale_bounds=(0.98, 1.02), n_steps=10)
                if scale_e < lj_energy_vectorized(polished):
                    polished = scaled
                
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # One more short basin hop after polish to see if we can improve further
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            bh_final, bh_final_e = basin_hopping(best_positions, n_iter=4, step_size=0.2, maxiter=60, temperature=0.02)
            if check_validity(bh_final):
                scaled, scale_e = _scale_optimize(bh_final, scale_bounds=(0.98, 1.02), n_steps=10)
                if scale_e < bh_final_e:
                    bh_final = scaled
                    bh_final_e = scale_e
                if bh_final_e < best_energy:
                    best_energy = bh_final_e
                    best_positions = bh_final
                    # Polish again
                    if time.time() - START_TIME < DEADLINE - 5:
                        polished2 = polish(best_positions, maxiter=2000, gtol=1e-15)
                        if check_validity(polished2):
                            scaled2, scale_e2 = _scale_optimize(polished2, scale_bounds=(0.98, 1.02), n_steps=10)
                            if scale_e2 < lj_energy_vectorized(polished2):
                                polished2 = scaled2
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
