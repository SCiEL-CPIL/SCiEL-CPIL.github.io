"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and Riemannian gradient on the sphere.
    E = sum_{i<j} log(1/|x_i - x_j|)
    grad_E = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    Projected to tangent space: grad_T = grad_E - (grad_E . x_i) x_i
    """
    n = len(pts)
    # Vectorized pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy calculation
    # We only need upper triangle for sum
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    if np.any(d < 1e-9):
        return 1e18, np.zeros_like(pts)
    energy = float(np.sum(np.log(1.0 / d)))
    
    # Gradient calculation
    # grad[i] = -sum_j (x_i - x_j) / d_ij^2
    # Note: dist_sq[i,j] = |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad_raw = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    grad_dot_x = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tan = grad_raw - grad_dot_x * pts
    
    return energy, grad_tan

def _project(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _retract(pts, v, alpha):
    """Retraction on the sphere."""
    new_pts = pts + alpha * v
    return _project(new_pts)

def _riemannian_lbfgs(pts, max_iter=2000, tol=1e-10, deadline=None):
    """
    Riemannian L-BFGS optimization.
    """
    if deadline and time.time() > deadline - 5:
        return pts, _energy_and_grad(pts)[0]
        
    pts = pts.copy()
    E, g = _energy_and_grad(pts)
    
    # L-BFGS memory
    m = 20
    s_hist = []
    y_hist = []
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
            
        # Compute inverse Hessian approximation H_k
        rho = []
        q = g.copy()
        alphas = []
        
        for i in range(len(s_hist) - 1, -1, -1):
            s_i = s_hist[i]
            y_i = y_hist[i]
            rho_i = 1.0 / np.sum(s_i * y_i)
            alphas.append(rho_i * np.sum(s_i * q))
            q = q - alphas[-1] * y_i
        alphas.reverse()
        
        # Initial step direction
        # gamma_k = s_{k-1}^T y_{k-1} / ||y_{k-1}||^2
        if s_hist:
            s_last = s_hist[-1]
            y_last = y_hist[-1]
            gamma = np.sum(s_last * y_last) / (np.sum(y_last * y_last) + 1e-15)
        else:
            gamma = 1.0
            
        r = -gamma * q
        
        for i in range(len(s_hist)):
            s_i = s_hist[i]
            y_i = y_hist[i]
            rho_i = 1.0 / np.sum(s_i * y_i)
            beta = rho_i * np.sum(y_i * r)
            r = r + s_i * (alphas[i] - beta)
        
        # Project direction to tangent space
        r_dot_x = np.sum(r * pts, axis=1, keepdims=True)
        d = r - r_dot_x * pts
        
        # Line search (Backtracking)
        alpha = 1.0
        c1 = 1e-4
        found_step = False
        best_E = E
        best_pts = pts
        
        for ls in range(30):
            new_pts = _retract(pts, d, alpha)
            new_E, new_g = _energy_and_grad(new_pts)
            
            if new_E < best_E:
                best_E = new_E
                best_pts = new_pts
                
            if new_E <= E + c1 * alpha * np.sum(g * d):
                found_step = True
                break
            alpha *= 0.5
            
        if not found_step:
            # If no step found, try a smaller step or break
            if alpha < 1e-6:
                break
            # Use the best point found even if Armijo not satisfied
            pts = best_pts
            E = best_E
            _, g = _energy_and_grad(pts)
            
        if not found_step:
             # If we didn't satisfy Armijo, we still updated to best_pts
             # But we need to update history correctly? 
             # Actually, if we didn't find a step, we might just break to be safe.
             # Let's stick to the standard: if no step, break.
             break

        # Update L-BFGS history
        s_new = alpha * d
        y_new = new_g - g
        s_hist.append(s_new)
        y_hist.append(y_new)
        
        if len(s_hist) > m:
            s_hist.pop(0)
            y_hist.pop(0)
            
        pts = new_pts
        E = new_E
        g = new_g
        
        if np.linalg.norm(g) < tol:
            break
            
    return pts, E

def _thiele_initialization():
    """
    Thiele's algorithm for maximizing minimum distance.
    """
    pts = np.zeros((N, 3))
    # Start with north pole
    pts[0] = [0, 0, 1]
    
    for i in range(1, N):
        best_dist = -1
        best_pt = None
        # Sample many random points
        rng = np.random.default_rng(i)
        for _ in range(1000):
            p = rng.standard_normal(3)
            p = p / np.linalg.norm(p)
            # Calculate min distance to existing points
            dists = np.linalg.norm(pts[:i] - p, axis=1)
            min_d = np.min(dists)
            if min_d > best_dist:
                best_dist = min_d
                best_pt = p
        pts[i] = best_pt
        
    return _project(pts)

def _random_initialization(seed):
    """Random points on sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project(pts)

def _fibonacci_initialization():
    """Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def construct():
    start_time = time.time()
    best_pts = None
    best_E = 1e18
    
    def _update_best(pts, E):
        nonlocal best_pts, best_E
        if E < best_E:
            best_E = E
            best_pts = pts.copy()
            
    # Generate diverse initializations
    initializations = [
        _fibonacci_initialization(),
        _thiele_initialization(),
        _random_initialization(1),
        _random_initialization(42),
        _random_initialization(123),
        _random_initialization(999),
        _random_initialization(2024),
    ]
    
    # Optimize each initialization
    for i, init_pts in enumerate(initializations):
        if time.time() > _DEADLINE - 10:
            break
            
        # Use L-BFGS for optimization
        opt_pts, E = _riemannian_lbfgs(init_pts, max_iter=1000, tol=1e-10, deadline=_DEADLINE)
        _update_best(opt_pts, E)
        
        # If we have time, do a second pass with tighter tolerance
        if time.time() < _DEADLINE - 20:
            opt_pts2, E2 = _riemannian_lbfgs(opt_pts, max_iter=500, tol=1e-12, deadline=_DEADLINE)
            _update_best(opt_pts2, E2)
            
    # If we have significant time left, try a simple simulated annealing / random restart
    if best_pts is not None and time.time() < _DEADLINE - 30:
        rng = np.random.default_rng(555)
        current_best = best_pts.copy()
        current_E = best_E
        
        # Simple SA-like exploration
        T = 0.1
        for sa_iter in range(100):
            if time.time() > _DEADLINE - 10:
                break
                
            # Perturb current best
            noise = rng.standard_normal((N, 3)) * T
            noise = noise - np.sum(noise * current_best, axis=1, keepdims=True) * current_best
            new_pts = _retract(current_best, noise, 1.0)
            
            # Quick polish
            polished_pts, E_pol = _riemannian_lbfgs(new_pts, max_iter=100, tol=1e-8, deadline=_DEADLINE)
            
            if E_pol < current_E:
                current_E = E_pol
                current_best = polished_pts
                _update_best(polished_pts, E_pol)
                
            # Cool down
            T *= 0.95
            
    return _project(best_pts) if best_pts is not None else _fibonacci_initialization()

# EVOLVE-BLOCK-END
