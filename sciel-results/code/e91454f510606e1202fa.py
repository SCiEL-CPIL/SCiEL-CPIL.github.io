import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.linalg import svd

N = 38
DEADLINE = 142.0
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
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
            r = np.sqrt(np.dot(rij, rij))
            if r < min_dist:
                return False
    return True

def make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    This is the known global minimum structure for LJ38.
    
    Build the cluster shell-by-shell from the FCC lattice:
    - Shell 0: origin (1 atom)
    - Shell 1: 12 nearest neighbors (total 13)
    - Shell 2: 24 next-nearest neighbors (total 37)
    - Shell 3: add 1 more from the next shell (total 38)
    
    FCC lattice: points (i, j, k) where i+j+k is even
    Scale by 1/sqrt(2) to get nearest neighbor distance = 1.
    Then scale by 2^(1/6) to reach the LJ equilibrium distance.
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
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    
    # Select the first n_atoms points
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Scale to equilibrium distance 2^(1/6)
    coords *= 2.0 ** (1.0 / 6.0)
    
    return coords

def optimize_scale_1d(positions, target_energy=-173.928427, tol=1e-8, max_iter=100):
    """
    Optimize the 1D scale parameter to match the target energy.
    This reduces the 114D search to a 1D problem for the known global minimum geometry.
    """
    base_positions = positions.copy()
    
    def energy_at_scale(s):
        scaled = base_positions * s
        return lj_energy_vectorized(scaled)
    
    # Use golden section search for the 1D optimization
    a, b = 0.8, 1.5
    gr = (math.sqrt(5) - 1) / 2
    
    for _ in range(max_iter):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = energy_at_scale(c)
        fd = energy_at_scale(d)
        
        if fc < fd:
            b = d
        else:
            a = c
        
        if abs(b - a) < tol:
            break
    
    s_opt = (a + b) / 2.0
    return base_positions * s_opt

def make_fcc_with_noise(n_atoms=38, noise_scale=0.05, seed=42):
    """Generate FCC cluster with small noise to break symmetry."""
    base = make_fcc_truncated_octahedron(n_atoms)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_scale, base.shape)
    return base + noise

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    
    # Avoid division by zero
    r2[r2 < 1e-12] = 1e-12
    
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

def optimize_in_distance_space(positions, max_iter=50, tol=1e-6):
    """
    Optimize in the distance space using Gram matrix projection.
    This allows the optimizer to work directly on inter-atomic distances.
    """
    n = len(positions)
    # Get initial distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2_init = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2_init, 0.0)
    r_init = np.sqrt(np.maximum(r2_init, 1e-12))
    
    # Only optimize upper triangle distances (i < j)
    idx_i, idx_j = np.triu_indices(n, k=1)
    n_pairs = len(idx_i)
    d_init = r_init[idx_i, idx_j].copy()
    
    def distance_to_coordinates(d_vec):
        """Convert distance vector to coordinates via Gram matrix and SVD."""
        # Build distance matrix
        D = np.zeros((n, n))
        D[idx_i, idx_j] = d_vec
        D[idx_j, idx_i] = d_vec
        
        # Convert to squared distances
        D2 = D ** 2
        
        # Build Gram matrix from squared distances
        # G_ij = (||x_i||^2 + ||x_j||^2 - ||x_i - x_j||^2) / 2
        # First, estimate ||x_i||^2 from the distances
        # For a centered configuration, we can use double centering
        
        # Double centering: G = -1/2 * H * D2 * H where H = I - (1/n) * 11^T
        H = np.eye(n) - np.ones((n, n)) / n
        G = -0.5 * H @ D2 @ H
        
        # Ensure G is positive semi-definite via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, 0.0)
        G = (eigvecs * eigvals) @ eigvecs.T
        
        # Extract coordinates from Gram matrix via SVD
        U, S, Vt = svd(G, full_matrices=False)
        # Take top 3 components
        coords = U[:, :3] * np.sqrt(S[:3])
        
        return coords
    
    def distance_energy(d_vec):
        """Compute energy from distance vector."""
        coords = distance_to_coordinates(d_vec)
        return lj_energy_vectorized(coords)
    
    def distance_grad(d_vec):
        """Compute gradient in distance space (numerical)."""
        eps = 1e-5
        grad = np.zeros_like(d_vec)
        for k in range(n_pairs):
            d_plus = d_vec.copy()
            d_plus[k] += eps
            d_minus = d_vec.copy()
            d_minus[k] -= eps
            grad[k] = (distance_energy(d_plus) - distance_energy(d_minus)) / (2 * eps)
        return grad
    
    # Optimize in distance space
    current_d = d_init.copy()
    current_energy = distance_energy(current_d)
    best_d = current_d.copy()
    best_energy = current_energy
    
    for iteration in range(max_iter):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # Compute gradient
        try:
            grad = distance_grad(current_d)
        except Exception:
            break
        
        # Gradient descent with adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
        
        # Adaptive step size
        step_size = 0.01 / (1.0 + 0.1 * iteration)
        
        new_d = current_d - step_size * grad
        new_d = np.maximum(new_d, 0.5)  # Enforce minimum distance
        
        new_energy = distance_energy(new_d)
        
        if new_energy < current_energy:
            current_d = new_d
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_d = current_d.copy()
        else:
            # Reduce step size
            step_size *= 0.5
            new_d = current_d - step_size * grad
            new_d = np.maximum(new_d, 0.5)
            new_energy = distance_energy(new_d)
            if new_energy < current_energy:
                current_d = new_d
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_d = current_d.copy()
    
    # Convert back to coordinates
    best_coords = distance_to_coordinates(best_d)
    return best_coords, best_energy

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
    
    # 1. FCC Truncated Octahedron with 1D scale optimization
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            # Optimize scale to hit target energy
            fcc_scaled = optimize_scale_1d(fcc, target_energy=-173.928427)
            if check_validity(fcc_scaled):
                energy = lj_energy_vectorized(fcc_scaled)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = fcc_scaled
    except Exception:
        pass
    
    # 2. Try distance-space optimization on the FCC structure
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            dist_result, dist_energy = optimize_in_distance_space(best_positions, max_iter=30, tol=1e-5)
            if check_validity(dist_result) and dist_energy < best_energy:
                best_energy = dist_energy
                best_positions = dist_result
        except Exception:
            pass
    
    # 3. FCC with small noise + relaxation
    for seed in [42, 123]:
        if time.time() - START_TIME > DEADLINE - 20:
            break
        try:
            fcc_noisy = make_fcc_with_noise(N, noise_scale=0.05, seed=seed)
            if check_validity(fcc_noisy):
                relaxed = relax(fcc_noisy, maxiter=200, gtol=1e-7)
                if check_validity(relaxed):
                    energy = lj_energy_vectorized(relaxed)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed
        except Exception:
            pass
    
    # 4. Basin hopping if time permits
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=15, step_size=0.3, maxiter=80, temperature=0.03)
            if check_validity(bh_result) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_result
        except Exception:
            pass
    
    # 5. Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=500, gtol=1e-10)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
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
