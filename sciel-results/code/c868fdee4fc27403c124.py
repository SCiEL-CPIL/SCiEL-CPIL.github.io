"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """
    Compute log energy sum_{i<j} log(1/d) and its tangent-space gradient.
    
    Energy: E = sum_{i<j} -log(d_ij)
    Gradient at x_i: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    Note: The gradient is already in the tangent space because x_i is unit length
    and x_j - x_i is orthogonal to x_i? No, x_j - x_i is not necessarily orthogonal to x_i.
    Wait, let's check: <x_i, x_j - x_i> = <x_i, x_j> - 1. This is not 0.
    However, the force vector F_i = sum (x_j - x_i)/d^2.
    The Riemannian gradient is the projection of the Euclidean gradient onto the tangent space.
    Proj_{T_x S^2} (v) = v - <v, x> x.
    
    Let's compute the Euclidean gradient first:
    g_i_euclid = sum_{j!=i} (x_j - x_i) / ||x_j - x_i||^2
    
    Then project:
    g_i_tangent = g_i_euclid - <g_i_euclid, x_i> x_i
    
    Actually, for the log potential, the force is central.
    Let's verify the projection term.
    <g_i_euclid, x_i> = sum_{j!=i} <x_j - x_i, x_i> / d^2 = sum_{j!=i} (<x_j, x_i> - 1) / d^2.
    This is generally non-zero. So we MUST project.
    """
    n = pts.shape[0]
    # diff[i, j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid div by 0, will mask later
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    # Filter out very small distances to avoid log(0)
    valid = d > 1e-15
    if not np.any(valid):
        return 0.0, np.zeros_like(pts)
    d_valid = d[valid]
    energy = -math.fsum(np.log(d_valid).tolist())
    
    # Gradient
    # We need sum_{j!=i} (x_j - x_i) / d_ij^2
    # Note: diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    # g_i = sum_j (-diff[i,j]) / dist_sq[i,j]
    
    # Mask diagonal
    mask = dist_sq > 1e-30
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Euclidean gradient: g_i = - sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff is (N, N, 3). inv_dist_sq is (N, N).
    # g_euclid[i] = - sum_j diff[i,j] * inv_dist_sq[i,j]
    g_euclid = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    # g_tangent_i = g_euclid_i - <g_euclid_i, x_i> x_i
    proj_coeff = np.sum(g_euclid * pts, axis=1, keepdims=True)
    g_tangent = g_euclid - proj_coeff * pts
    
    return energy, g_tangent


def _energy_only(pts):
    """Compute log energy only."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    return -math.fsum(np.log(d).tolist())


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _exp_map(pts, tangent):
    """
    Exponential map on the sphere.
    Moves pts along tangent vectors.
    exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|)
    """
    norm_v = np.linalg.norm(tangent, axis=1, keepdims=True)
    safe_norm = np.where(norm_v > 1e-15, norm_v, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm_v)
    s = np.sin(norm_v)
    # For norm_v ~ 0, c~1, s~0, result ~ x
    out = c * pts + s * u
    return _project_to_sphere(out)


def _riemannian_lbfgs(pts, time_limit=5.0, tight=False):
    """
    Riemannian L-BFGS optimization.
    We optimize the tangent vector t such that new_pts = exp(pts, t).
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    
    def objective(t):
        if time.time() - start_time > time_limit:
            return np.inf
        tv = t.reshape(-1, 3)
        new_pts = _exp_map(pts, tv)
        e, _ = _energy_and_grad(new_pts)
        return e

    def gradient(t):
        if time.time() - start_time > time_limit:
            return np.zeros(len(t))
        tv = t.reshape(-1, 3)
        new_pts = _exp_map(pts, tv)
        _, g = _energy_and_grad(new_pts)
        # The gradient of the energy w.r.t. the tangent vector t at t=0 is the Riemannian gradient.
        # For t != 0, the gradient in t-space is the pullback of the gradient at new_pts.
        # Since exp is an isometry locally, the gradient in t-space is essentially the Riemannian gradient at new_pts.
        return g.flatten()

    t0 = np.zeros(3 * N)
    
    if tight:
        options = {'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
    else:
        options = {'maxiter': 1000, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}

    try:
        result = minimize(objective, t0, jac=gradient, method='L-BFGS-B', options=options)
        if result.success or result.fun < _energy_only(pts):
            t_opt = result.x
            new_pts = _exp_map(pts, t_opt.reshape(-1, 3))
            new_energy = _energy_only(new_pts)
            if new_energy < _energy_only(pts) and np.all(np.isfinite(new_pts)):
                return new_pts, new_energy
    except Exception:
        pass
        
    return pts, _energy_only(pts)


def _riemannian_adam(pts, time_limit=10.0, lr=0.05):
    """Riemannian Adam optimizer."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Update momentum
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Step in tangent space
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Exponential map update
        pts = _exp_map(pts, step)
        
        if t % 100 == 0:
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-12:
                break
                
    return best_pts, best_energy


def _riemannian_basin_hopping(pts, time_limit=20.0, seed=42):
    """
    Basin Hopping with Riemannian perturbations.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = 0.05
    T_min = 1e-6
    cooling = 0.995
    rng = np.random.RandomState(seed)
    
    while time.time() - start_time < time_limit:
        # Perturbation in tangent space
        # Scale noise by T
        noise_scale = T * 0.2
        tangent_noise = rng.randn(n, 3) * noise_scale
        new_pts = _exp_map(pts, tangent_noise)
        
        # Local polish
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining > 3.0:
            polish_limit = min(2.0, remaining * 0.4)
            polished_pts, polished_energy = _riemannian_lbfgs(new_pts, time_limit=polish_limit)
        else:
            polished_pts = new_pts
            polished_energy = _energy_only(new_pts)
            
        # Metropolis acceptance
        delta_e = polished_energy - current_energy
        if delta_e < 0 or (T > T_min and rng.rand() < math.exp(-delta_e / T)):
            pts = polished_pts
            current_energy = polished_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
                
        T *= cooling
        if T < T_min:
            break
            
    return best_pts, best_energy


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)

    best_pts = None
    best_energy = np.inf

    # Generate diverse starting points
    starts = []
    starts.append(_fibonacci_sphere(N))
    starts.append(_fibonacci_sphere(N, seed=1))
    starts.append(_fibonacci_sphere(N, seed=7))
    for s in range(2, 6):
        starts.append(_random_sphere(N, seed=s * 100))

    # Phase 1: Riemannian L-BFGS from each diverse start
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        time_limit = min(5.0, deadline - elapsed - 20)
        if time_limit < 1.0:
            break
        candidate_pts, candidate_energy = _riemannian_lbfgs(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts.copy()

    # Phase 2: Riemannian Adam from best point
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            time_limit = min(8.0, deadline - elapsed - 20)
            candidate_pts, candidate_energy = _riemannian_adam(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 3: Riemannian Basin Hopping
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 25:
            time_limit = min(15.0, deadline - elapsed - 10)
            candidate_pts, candidate_energy = _riemannian_basin_hopping(
                best_pts, time_limit=time_limit, seed=42
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 4: Another round of Basin Hopping with different seed
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            time_limit = min(10.0, deadline - elapsed - 5)
            candidate_pts, candidate_energy = _riemannian_basin_hopping(
                best_pts, time_limit=time_limit, seed=123
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 5: Final high-precision Riemannian L-BFGS polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            time_limit = min(5.0, deadline - elapsed - 1)
            candidate_pts, candidate_energy = _riemannian_lbfgs(best_pts, time_limit=time_limit, tight=True)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Final projection and validation
    if best_pts is None:
        return _fibonacci_sphere(N)
        
    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        # Points too close, fallback
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
