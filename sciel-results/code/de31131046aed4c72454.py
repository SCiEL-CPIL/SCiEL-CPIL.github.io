"""Logarithmic energy minimization for N points on the unit sphere.
Direct approach: use structured initialization followed by iterative gradient descent
with adaptive step sizes, exploiting the symmetry and smoothness of the energy landscape."""
import time
import math
import numpy as np

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-16))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of -sum log(d_ij) with respect to points
    # d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    # Sum over j != i: -sum_j (x_i - x_j)/d_ij^2 = -sum_j x_i/d_ij^2 + sum_j x_j/d_ij^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j = -W.sum(axis=1)*x_i + W @ x
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
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


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_like(n, seed=0):
    """Generate well-distributed points using a simple quasi-random approach."""
    rng = np.random.default_rng(seed)
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        t = (i + 0.5) / n
        z = 1.0 - 2.0 * t
        angle = 2.0 * np.pi * i * (1.0 + 1.0 / golden_ratio) + rng.standard_normal() * 0.1
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        points.append([r * np.cos(angle), r * np.sin(angle), z])
    return _project(np.array(points))


def _gradient_descent(pts, deadline, max_iter=5000, lr_init=0.01, lr_min=1e-12):
    """Adaptive gradient descent with line search."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    lr = lr_init
    n = N
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        # Adaptive step size
        step = lr / max(g_norm, 1e-10)
        new_pts = _project(current + step * g * (-1))  # negative gradient
        
        new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-14:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            lr = min(lr * 1.1, 0.1)  # increase step
        elif new_e < best_e:
            current = new_pts
            lr = lr * 0.9
        else:
            lr = lr * 0.5
            if lr < lr_min:
                break
    
    return best_pts, best_e


def _langevin_verlet(pts, deadline, rng, max_iter=2000, dt=0.005, gamma=0.3, T_init=1.0, T_min=1e-6):
    """
    Langevin dynamics with Velocity Verlet integrator on the sphere.
    Uses a slower cooling schedule for broader exploration.
    """
    current = pts.copy()
    velocity = np.zeros_like(current)
    best_e = _energy(current)
    best_pts = current.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Slower cooling: keep temperature high longer for broader exploration
        # Use a power-law cooling that decays more slowly
        frac = it / max_iter
        T = T_init * (T_min / T_init) ** (frac ** 0.5)
        
        # 1. Half step velocity update with force
        e, g = _energy_grad(current)
        F = -g
        velocity += 0.5 * dt * F
        
        # 2. Full step position update
        new_pts = current + dt * velocity
        new_pts = _project(new_pts)
        
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

def _newton_like(pts, deadline, max_iter=2000, tol=1e-12):
    """Second-order method using Hessian approximation via finite differences."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    
    # Simple momentum-based gradient descent with adaptive learning rate
    velocity = np.zeros_like(current)
    beta = 0.9
    lr = 0.05
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            break
        
        velocity = beta * velocity - lr * g / max(g_norm, 1e-10)
        new_pts = _project(current + velocity)
        
        new_e = _energy(new_pts)
        
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            current = new_pts
            lr = min(lr * 1.05, 0.5)
        else:
            current = best_pts.copy()
            velocity *= 0.5
            lr *= 0.7
    
    return best_pts, best_e


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals with wider range
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(15):
        jitter_scale = 0.01 + 0.015 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(15):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Sobol-like points
    for s in range(8):
        initializations.append(_sobol_like(N, seed=s))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: quick evaluation of all initializations
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        
        e_quick = _energy(init_pts)
        if e_quick < best_energy:
            best_energy = e_quick
            best_pts = init_pts.copy()
    
    # Second pass: optimize the top candidates
    if best_pts is not None and time.time() < deadline - 20.0:
        # Sort initializations by quick energy and take top ones
        scored_inits = []
        for init_pts in initializations:
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 12 candidates
        top_candidates = scored_inits[:12]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 15.0:
                break
            
            # Gradient descent
            gd_pts, gd_e = _gradient_descent(init_pts, deadline, max_iter=2000, lr_init=0.01)
            if gd_e < best_energy:
                best_energy = gd_e
                best_pts = gd_pts
            
            # Momentum-based
            nl_pts, nl_e = _newton_like(init_pts, deadline, max_iter=1000)
            if nl_e < best_energy:
                best_energy = nl_e
                best_pts = nl_pts

            # Langevin dynamics with broader exploration
            if time.time() < deadline - 10.0:
                lv_pts, lv_e = _langevin_verlet(init_pts, deadline, rng, max_iter=800, dt=0.005, T_init=1.0)
                if lv_e < best_energy:
                    best_energy = lv_e
                    best_pts = lv_pts
    
    # Final polish on the global best - multiple refinement passes
    if best_pts is not None and time.time() < deadline - 8.0:
        refined, ref_e = _gradient_descent(best_pts, deadline, max_iter=3000, lr_init=0.005, lr_min=1e-14)
        if ref_e < best_energy:
            best_energy = ref_e
            best_pts = refined
    
    if best_pts is not None and time.time() < deadline - 5.0:
        refined, ref_e = _newton_like(best_pts, deadline, max_iter=2000, tol=1e-14)
        if ref_e < best_energy:
            best_energy = ref_e
            best_pts = refined

    # Final Langevin polish with low temperature for fine-tuning
    if best_pts is not None and time.time() < deadline - 2.0:
        lv_pts, lv_e = _langevin_verlet(best_pts, deadline, rng, max_iter=500, dt=0.003, T_init=0.05, T_min=1e-7)
        if lv_e < best_energy:
            best_energy = lv_e
            best_pts = lv_pts
    
    # Final high-precision gradient descent polish
    if best_pts is not None and time.time() < deadline - 0.5:
        refined, ref_e = _gradient_descent(best_pts, deadline, max_iter=2000, lr_init=0.001, lr_min=1e-16)
        if ref_e < best_energy:
            best_energy = ref_e
            best_pts = refined
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
