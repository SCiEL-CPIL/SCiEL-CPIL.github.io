"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START

def _project(X):
    """Project points onto the unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space) using vectorized ops."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of -sum log(d_ij) with respect to points
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy (as required by evaluator)."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points uniformly on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project(pts)


def _gradient_descent(pts, deadline, max_iter=8000, lr_init=0.01, lr_min=1e-14):
    """
    Adaptive gradient descent with backtracking line search and Riemannian exponential map.
    Uses vectorized energy/gradient for speed.
    """
    current = _project(pts.copy())
    best_e = _energy_grad(current)[0]
    best_pts = current.copy()
    lr = lr_init
    c1 = 0.1
    rho = 0.5
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break
        
        # Backtracking line search
        step = lr
        found = False
        for bt in range(15):
            v = -step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            
            new_e = _energy_grad(new_pts)[0]
            
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**15, 1e-16)
            v = -step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            new_e = _energy_grad(new_pts)[0]
        
        if new_e < best_e - 1e-15:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            no_improve_count = 0
            lr = min(step * 1.3, 0.05)
        elif new_e < best_e:
            current = new_pts
            no_improve_count += 1
            lr = step
        else:
            lr = step * 0.5
            no_improve_count += 1
            if lr < lr_min or no_improve_count > 40:
                break
    
    return best_pts, best_e


def _simulated_annealing(pts, deadline, max_iter=3000, T_init=0.05, T_min=1e-7, cooling=0.998):
    """Simulated annealing on the sphere using vectorized energy for speed."""
    current = _project(pts.copy())
    current_e = _energy_grad(current)[0]
    best_e = current_e
    best_pts = current.copy()
    
    T = T_init
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = np.random.randint(N)
        direction = np.random.standard_normal(3)
        direction -= np.dot(direction, current[idx]) * current[idx]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction /= dir_norm
        
        step_size = 0.05 * math.sqrt(T)
        new_pts = current.copy()
        new_pts[idx] += step_size * direction
        new_pts = _project(new_pts)
        
        new_e = _energy_grad(new_pts)[0]
        delta = new_e - current_e
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pts = current.copy()
        
        T = max(T * cooling, T_min)
    
    return best_pts, best_e


def _polish(pts, deadline, max_iter=20000, tol=1e-15):
    """Final high-precision polish with very tight tolerance."""
    return _gradient_descent(pts, deadline, max_iter=max_iter, lr_init=0.005, lr_min=1e-16)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Golden spiral
    initial_configs.append(_fibonacci_spiral(N))
    
    # 2. Random configurations
    for seed in range(5):
        r = np.random.RandomState(seed)
        initial_configs.append(_random_sphere(N, r))
    
    # 3. Perturbed golden spiral
    gs = _fibonacci_spiral(N)
    for scale in [0.1, 0.3]:
        perturbed = gs + scale * np.random.standard_normal((N, 3))
        initial_configs.append(_project(perturbed))
    
    best_pts = None
    best_energy = float('inf')
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 5.0:
            break
        
        # Gradient descent
        refined_pts, energy = _gradient_descent(init_pts, deadline, max_iter=4000)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = refined_pts
        
        # If we have time, also try simulated annealing from this result
        if time.time() < deadline - 15.0:
            sa_pts, sa_energy = _simulated_annealing(refined_pts, deadline, max_iter=2000)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 3.0:
        best_pts, best_energy = _polish(best_pts, deadline, max_iter=15000, tol=1e-15)
    
    # Ensure validity
    best_pts = _project(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                diff = best_pts[i] - best_pts[j]
                if np.linalg.norm(diff) < 1e-15:
                    diff = np.array([1.0, 0.0, 0.0])
                diff /= np.linalg.norm(diff)
                best_pts[i] += 1e-8 * diff
                best_pts[j] -= 1e-8 * diff
                best_pts = _project(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
