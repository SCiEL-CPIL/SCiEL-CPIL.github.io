"""Logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_fast(pts):
    """Fast energy using dot-product distance computation."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and tangent-space gradient using dot-product distances."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))

    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)

    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _langevin_verlet(pts, deadline, rng, max_iter=3000, dt=0.003, gamma=0.5, T_init=0.5, T_min=1e-6):
    """
    Langevin dynamics with Velocity Verlet integrator on the sphere.
    Uses a slower cooling schedule for broader exploration.
    """
    current = _normalize(pts.copy())
    velocity = np.zeros_like(current)
    best_e = _energy_fast(current)
    best_pts = current.copy()
    n = N
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Slower cooling: power-law cooling that decays more slowly
        frac = it / max_iter
        T = T_init * (T_min / T_init) ** (frac ** 0.4)
        
        # 1. Half step velocity update with force
        e, g = _energy_grad(current)
        F = -g
        velocity += 0.5 * dt * F
        
        # 2. Full step position update
        new_pts = current + dt * velocity
        new_pts = _normalize(new_pts)
        
        # 3. Compute new force
        new_e, new_g = _energy_grad(new_pts)
        new_F = -new_g
        
        # 4. Half step velocity update with new force
        velocity += 0.5 * dt * new_F
        
        # 5. Apply friction and noise (Langevin kick)
        # Project velocity to tangent space of new_pts
        dot_v = np.sum(velocity * new_pts, axis=1, keepdims=True)
        velocity = velocity - dot_v * new_pts
        
        # Damping
        velocity *= (1.0 - gamma * dt)
        
        # Noise
        if T > T_min:
            noise_scale = np.sqrt(2.0 * gamma * T * dt)
            velocity += noise_scale * rng.standard_normal((N, 3))
            
            # Re-project after adding noise
            dot_v = np.sum(velocity * new_pts, axis=1, keepdims=True)
            velocity = velocity - dot_v * new_pts
        
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            
        current = new_pts
        
    return best_pts, best_e


def _gradient_descent(pts, deadline, max_iter=5000, lr_init=0.05, lr_min=1e-14):
    """Adaptive gradient descent with line search."""
    current = _normalize(pts.copy())
    best_e = _energy_fast(current)
    best_pts = current.copy()
    lr = lr_init
    n = N
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break
        
        # Adaptive step size
        step = lr / max(g_norm, 1e-10)
        new_pts = _normalize(current - step * g)
        
        new_e = _energy_fast(new_pts)
        
        if new_e < best_e - 1e-14:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            lr = min(lr * 1.05, 0.5)
        elif new_e < best_e:
            current = new_pts
            lr = lr * 0.95
        else:
            lr = lr * 0.5
            if lr < lr_min:
                break
    
    return best_pts, best_e


def _optimize_from_start(pts, deadline):
    """
    Optimize from a single starting configuration using multi-phase L-BFGS-B.
    Returns (best_pts, best_precise_energy).
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    # Phase 1: Coarse L-BFGS-B
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad, x0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_p = _precise_energy(new_pts)
            if e_p < best_energy:
                best_pts = new_pts
                best_energy = e_p
    except Exception:
        pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 2: Medium refinement
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad, x0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': 5000, 'maxfun': 25000, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_p = _precise_energy(new_pts)
            if e_p < best_energy:
                best_pts = new_pts
                best_energy = e_p
    except Exception:
        pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 3: High-precision polish
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad, x0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': 10000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-11}
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_p = _precise_energy(new_pts)
            if e_p < best_energy:
                best_pts = new_pts
                best_energy = e_p
    except Exception:
        pass

    return best_pts, best_energy


def _spectral_init(n, rng, l_max=8):
    """
    Generate a smooth initial configuration by sampling a random band-limited
    spherical function. This provides a structured, smooth starting point
    that avoids the combinatorial explosion of local minima.
    """
    # Generate random spherical harmonic coefficients up to l_max
    # We use a simplified approach: random smooth function on sphere
    # by combining random low-frequency spherical harmonics

    # Simple approach: random smooth perturbation of uniform distribution
    # Use a random smooth function defined by low-frequency components
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = 2.0 * rng.uniform(0, 1, n) - 1.0
    sin_phi = np.sqrt(1.0 - cos_phi ** 2)

    pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])

    # Add smooth low-frequency perturbation
    for l in range(1, l_max + 1):
        # Random amplitudes and phases for each m
        for m in range(-l, l + 1):
            amp = rng.normal(0, 0.01 / (l + 1))
            phase = rng.uniform(0, 2 * np.pi)
            # Simple approximation: modulate points
            # This is a heuristic smooth perturbation
            t = np.arctan2(pts[:, 1], pts[:, 0])
            c = pts[:, 2]
            # Use associated Legendre-like modulation (simplified)
            mod = amp * np.cos(m * t + phase) * (c ** abs(m))
            pts[:, 2] += mod * 0.1
            pts[:, 0] += amp * 0.05 * np.sin(m * t + phase) * sin_phi

    return _normalize(pts)


def _local_pairwise_optimize(pts, deadline, max_passes=10):
    """
    Perform local pairwise optimization.
    Iteratively fix all points except one, and optimize that single point 
    on the sphere to minimize its interaction with the rest.
    """
    n = pts.shape[0]
    current = _normalize(pts.copy())
    best_e = _energy_fast(current)
    best_pts = current.copy()
    
    for p in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            x_i = current[i]
            
            def local_obj(flat_x):
                xi_new = _normalize(flat_x.reshape(1, 3))[0]
                diffs = current - xi_new
                diffs[i] = np.zeros(3)
                dists = np.linalg.norm(diffs, axis=1)
                dists[i] = 1.0
                log_dists = np.log(dists)
                log_dists[i] = 0.0
                return -np.sum(log_dists)
            
            x0 = x_i.copy()
            
            try:
                res = minimize(
                    local_obj, x0, 
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                new_x_i = _normalize(res.x.reshape(1, 3))[0]
                
                temp_pts = current.copy()
                temp_pts[i] = new_x_i
                new_e = _energy_fast(temp_pts)
                
                if new_e < best_e - 1e-12:
                    current[i] = new_x_i
                    best_e = new_e
                    best_pts = current.copy()
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
            
    return best_pts, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    rng = np.random.default_rng(12345)

    # Generate diverse initializations
    initializations = []

    # 1. Fibonacci spiral (best structured start)
    initializations.append(_fibonacci_spiral(N))

    # 2. Spectral/smooth initializations
    for seed in range(5):
        r = np.random.default_rng(seed * 100 + 7)
        initializations.append(_spectral_init(N, r, l_max=6))

    # 3. Random initializations
    for i in range(10):
        initializations.append(_random_sphere(N, rng))

    # 4. Perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for scale in [0.005, 0.01, 0.02, 0.05]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))

    # 5. Additional structured: shifted golden spiral
    for offset in [0.0, 0.15, 0.3, 0.5]:
        initializations.append(_fibonacci_spiral(N, offset))

    # Sort by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _energy_fast(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    # First, try Langevin on the best few initializations to escape local minima
    for e_quick, init_pts in scored_inits[:5]:
        if time.time() > deadline - 30:
            break
        
        try:
            lang_pts, lang_e = _langevin_verlet(init_pts, deadline - 10, rng, 
                                                 max_iter=1000, dt=0.002, 
                                                 gamma=0.5, T_init=0.3, T_min=1e-6)
            polished_pts, polished_e = _optimize_from_start(lang_pts, deadline)
            # Add local pairwise optimization
            if time.time() < deadline - 15:
                lp_pts, lp_e = _local_pairwise_optimize(polished_pts, deadline, max_passes=5)
                if lp_e < polished_e:
                    polished_pts = lp_pts
                    polished_e = lp_e
            if polished_e < best_energy:
                best_energy = polished_e
                best_pts = polished_pts
        except Exception:
            pass

    # Now do standard optimization on all initializations
    for e_quick, init_pts in scored_inits:
        if time.time() > deadline - 20:
            break

        optimized_pts, e_precise = _optimize_from_start(init_pts, deadline)
        
        # Add local pairwise optimization
        if time.time() < deadline - 15:
            lp_pts, lp_e = _local_pairwise_optimize(optimized_pts, deadline, max_passes=5)
            if lp_e < e_precise:
                optimized_pts = lp_pts
                e_precise = lp_e

        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    # Final high-precision polish on the best configuration
    if time.time() < deadline - 15:
        try:
            final_pts, final_e = _optimize_from_start(best_pts, deadline)
            # Add local pairwise optimization to final result
            if time.time() < deadline - 5:
                lp_pts, lp_e = _local_pairwise_optimize(final_pts, deadline, max_passes=10)
                if lp_e < final_e:
                    final_pts = lp_pts
                    final_e = lp_e
            if final_e < best_energy:
                best_pts = final_pts
                best_energy = final_e
        except Exception:
            pass

    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _normalize(_fibonacci_spiral(N))

    return best_pts
