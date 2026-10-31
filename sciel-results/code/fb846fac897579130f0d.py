import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 147.0
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < min_dist * min_dist:
                return False
    return True

def _get_fcc_shell_structure(n_atoms=38):
    """
    Return base FCC positions (unscaled, nearest neighbor = 1/sqrt(2)*sqrt(2)=1 in lattice units)
    and shell assignments.
    
    The FCC truncated octahedron for 38 atoms has the following shell structure:
    - Shell 0: 1 atom at origin (distance 0)
    - Shell 1: 12 atoms at distance 1 (nearest neighbors in FCC, <100> type)
    - Shell 2: 24 atoms at distance sqrt(2) (next-nearest, <110> type)
    - Shell 3: 1 atom at distance 2 (from the next shell, to make 38 total)
    
    Wait, let's re-examine. FCC lattice points with i+j+k even, scaled by 1/sqrt(2):
    Origin: (0,0,0) -> r=0
    (1,1,0)/sqrt(2) etc -> 12 points at r=1
    (2,0,0)/sqrt(2) etc -> 6 points at r=sqrt(2)
    (2,2,0)/sqrt(2) etc -> 12 points at r=2
    (2,2,2)/sqrt(2) etc -> 8 points at r=sqrt(6)
    (4,0,0)/sqrt(2) etc -> 6 points at r=2*sqrt(2)
    
    So cumulative: 1, 13, 19, 31, 39...
    For 38 atoms: 1 + 12 + 6 + 12 + 7 = 38. 
    So shells:
    - Shell 0: 1 atom at r=0
    - Shell 1: 12 atoms at r=1
    - Shell 2: 6 atoms at r=sqrt(2)
    - Shell 3: 12 atoms at r=2
    - Shell 4: 7 atoms at r=sqrt(6) (out of 8 possible)
    
    Total: 1+12+6+12+7 = 38. Correct.
    """
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
    
    # Sort by distance from origin, then by coordinates for determinism
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    
    # Select the first n_atoms points
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Assign shells based on radial distance
    radii = np.linalg.norm(coords, axis=1)
    
    # Define shell boundaries
    # Shell 0: r < 0.5
    # Shell 1: 0.5 <= r < 1.5
    # Shell 2: 1.5 <= r < 2.0 (sqrt(2) ≈ 1.414)
    # Shell 3: 2.0 <= r < 2.5
    # Shell 4: r >= 2.5 (sqrt(6) ≈ 2.449)
    
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        if r < 0.5:
            shell_ids[i] = 0
        elif r < 1.5:
            shell_ids[i] = 1
        elif r < 2.0:
            shell_ids[i] = 2
        elif r < 2.5:
            shell_ids[i] = 3
        else:
            shell_ids[i] = 4
    
    # Verify shell counts
    # shell 0: 1, shell 1: 12, shell 2: 6, shell 3: 12, shell 4: 7
    
    return coords, shell_ids

def make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    This is the known global minimum structure for LJ38.
    """
    coords, _ = _get_fcc_shell_structure(n_atoms)
    
    # Scale to equilibrium distance 2^(1/6)
    coords = coords * 2.0 ** (1.0 / 6.0)
    
    return coords

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    
    # Avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) / 2 (because double counted in matrix)
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient:
    # F_i = sum_j (-24 r^-14 + 12 r^-8) * (r_i - r_j)
    # Term = (-24 * inv_r12 * inv_r2 + 12 * inv_r6 * inv_r2) * diff
    # inv_r12 * inv_r2 = inv_r14
    # inv_r6 * inv_r2 = inv_r8
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    # coeff shape (N, N), diff shape (N, N, 3)
    # We want sum over j for each i
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=500, gtol=1e-10):
    """High-precision final polish pass with analytic gradient."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def ultra_polish(positions, maxiter=2000, gtol=1e-14, ftol=1e-16):
    """Ultra-high-precision polish pass for final energy minimization."""
    x0 = positions.flatten()
    
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 200})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def _symmetry_adapted_energy(params):
    """
    Compute energy for symmetry-adapted parameters.
    params: [scale_0, scale_1, scale_2, scale_3, scale_4]
    where scale_k is the radial scale factor for shell k.
    """
    base_coords, shell_ids = _get_fcc_shell_structure(N)
    
    # Apply shell-specific scaling
    positions = base_coords.copy()
    for s in range(5):
        mask = shell_ids == s
        if np.any(mask):
            positions[mask] *= params[s]
    
    return lj_energy_vectorized(positions)

def _symmetry_adapted_grad(params):
    """
    Compute gradient for symmetry-adapted parameters using numerical differentiation.
    """
    eps = 1e-7
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_plus[k] += eps
        e_plus = _symmetry_adapted_energy(params_plus)
        params_minus = params.copy()
        params_minus[k] -= eps
        e_minus = _symmetry_adapted_energy(params_minus)
        grad[k] = (e_plus - e_minus) / (2 * eps)
    return grad

def optimize_symmetry_adapted(initial_scales=None, maxiter=1000):
    """
    Optimize the 5 shell scale parameters to minimize energy.
    This reduces the search space from 114 dimensions to 5 dimensions.
    """
    if initial_scales is None:
        initial_scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * 2.0 ** (1.0 / 6.0)
    
    x0 = initial_scales.copy()
    
    try:
        result = minimize(_symmetry_adapted_energy, x0, method='L-BFGS-B',
                         jac=_symmetry_adapted_grad,
                         options={'maxiter': maxiter, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-14})
        if result.success or result.fun < lj_energy_vectorized(make_fcc_truncated_octahedron()):
            return result.x
    except Exception:
        pass
    
    return x0

def make_fcc_from_scales(scales):
    """Create FCC positions from shell scale parameters."""
    base_coords, shell_ids = _get_fcc_shell_structure(N)
    positions = base_coords.copy()
    for s in range(5):
        mask = shell_ids == s
        if np.any(mask):
            positions[mask] *= scales[s]
    return positions

def basin_hopping(positions, n_iter=10, step_size=0.5, maxiter=100, temperature=0.05):
    """Perform basin hopping optimization with analytic gradient."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    
    rng = np.random.default_rng(123)
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # Perturb
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        
        # Relax
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                            jac=True,
                            options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 50})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue
        
        # Metropolis acceptance
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
    
    # 1. Construct the known FCC truncated octahedron
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            energy = lj_energy_vectorized(fcc)
            if energy < best_energy:
                best_energy = energy
                best_positions = fcc
    except Exception:
        pass
    
    # 2. Symmetry-adapted optimization (5 parameters)
    if time.time() - START_TIME < DEADLINE - 20:
        try:
            scales = optimize_symmetry_adapted(maxiter=1000)
            sym_pos = make_fcc_from_scales(scales)
            if check_validity(sym_pos):
                energy = lj_energy_vectorized(sym_pos)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = sym_pos
        except Exception:
            pass
    
    # 3. Relax the best structure
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            relaxed = relax(best_positions, maxiter=300, gtol=1e-9)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    # 4. Basin hopping from the best result
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=15, step_size=0.2, maxiter=60, temperature=0.02)
            if check_validity(bh_result) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_result
        except Exception:
            pass
    
    # 5. Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=500, gtol=1e-12)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass
    
    # 6. Ultra-precision final polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 5:
        try:
            ultra_polished = ultra_polish(best_positions, maxiter=2000, gtol=1e-14, ftol=1e-16)
            if check_validity(ultra_polished):
                energy = lj_energy_vectorized(ultra_polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = ultra_polished
        except Exception:
            pass
    
    # Final check and return
    if best_positions is None:
        # Fallback
        try:
            best_positions = make_fcc_truncated_octahedron(N)
            if not check_validity(best_positions):
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions
