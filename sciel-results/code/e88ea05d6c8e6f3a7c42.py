"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import time
import math

N = 100

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    # Compute pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so (x_i - x_j)/|x_i-x_j|^2 = diff[i,j] / dists[i,j]^2
    inv_dist_sq = 1.0 / (dists * dists)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_tangent(grad, pts):
    """Project gradient onto tangent space of the sphere at each point."""
    # For unit sphere, tangent projection: g_tan = g - (g·x)x
    dot = np.sum(grad * pts, axis=1)
    return grad - dot[:, np.newaxis] * pts


def _lbfgs_minimize(pts, max_iter=500, tol=1e-12, deadline=None):
    """Minimize energy using L-BFGS on the sphere via retraction."""
    pts = np.array(pts, dtype=np.float64)
    n = len(pts)
    
    # L-BFGS history
    m = 10  # memory
    s_hist = []  # list of (s, y) pairs
    W = 1.0  # initial step scale
    
    best_energy, best_pts = _energy_and_grad(pts)[0], pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Project gradient to tangent space
        g_tan = _project_to_tangent(grad, pts)
        
        # Check convergence
        if np.linalg.norm(g_tan) < tol:
            break
        
        # L-BFGS two-loop recursion
        # r = -g_tan
        r = -g_tan
        alphas = np.zeros(m)
        betas = np.zeros(m)
        
        # First loop (backward)
        for k in range(len(s_hist) - 1, -1, -1):
            s_k, y_k = s_hist[k]
            ys = np.sum(s_k * y_k)
            sy = np.sum(s_k * s_k)
            alphas[k] = np.sum(s_k * r) / ys
            r = r - alphas[k] * y_k
        
        # Scale by initial Hessian approximation
        r = r * W
        
        # Second loop (forward)
        for k in range(len(s_hist)):
            s_k, y_k = s_hist[k]
            ys = np.sum(s_k * y_k)
            sy = np.sum(s_k * s_k)
            betas[k] = np.sum(y_k * s_k) / ys
            r = r + (alphas[k] - betas[k] * ys) * s_k
        
        # Line search with backtracking
        step = r
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break
        
        # Normalize step direction for line search
        step_dir = step / step_norm
        
        # Armijo line search
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        
        for ls_iter in range(30):
            # Move along tangent direction, then retract to sphere
            new_pts = pts + alpha * step_dir
            # Retract: normalize to unit sphere
            norms = np.sqrt(np.sum(new_pts * new_pts, axis=1))
            norms = np.maximum(norms, 1e-15)
            new_pts = new_pts / norms[:, np.newaxis]
            
            new_energy, _ = _energy_and_grad(new_pts)
            
            if new_energy < energy + c1 * alpha * np.sum(g_tan * step_dir):
                break
            alpha *= rho
        
        # Update L-BFGS history
        new_pts = pts + alpha * step_dir
        norms = np.sqrt(np.sum(new_pts * new_pts, axis=1))
        norms = np.maximum(norms, 1e-15)
        new_pts = new_pts / norms[:, np.newaxis]
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        new_g_tan = _project_to_tangent(new_grad, new_pts)
        
        s = new_pts - pts
        y = new_g_tan - g_tan
        
        sy = np.sum(s * y)
        if sy > 1e-15:
            s_hist.append((s, y))
            if len(s_hist) > m:
                s_hist.pop(0)
            W = np.sum(y * y) / sy
        
        pts = new_pts
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = pts.copy()
    
    return best_pts, best_energy


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.sqrt(np.sum(pts * pts, axis=1))
    return pts / norms[:, np.newaxis]


def _icosahedron_base(n):
    """Generate points starting from icosahedron vertices, then fill with Fibonacci."""
    # 12 icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)
    norms = np.sqrt(np.sum(ico * ico, axis=1))
    ico = ico / norms[:, np.newaxis]
    
    if n <= 12:
        return ico[:n]
    
    # Fill remaining with Fibonacci, avoiding close points
    remaining = n - 12
    fib = _fibonacci_sphere(remaining + 20)  # extra to have choices
    chosen = []
    for p in fib:
        if len(chosen) >= remaining:
            break
        # Check minimum distance to existing points
        d2 = np.sum((ico + np.array(chosen) - p[np.newaxis, :]) ** 2, axis=1)
        if np.min(d2) > 0.01:
            chosen.append(p)
    
    # If we couldn't find enough, just add remaining Fibonacci points
    while len(chosen) < remaining:
        p = fib[len(chosen) % len(fib)]
        chosen.append(p)
    
    return np.vstack([ico, np.array(chosen)])


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple diverse starting configurations
    starts = []
    
    # 1. Fibonacci sphere
    starts.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron-based
    starts.append(_icosahedron_base(N))
    
    # 3. Random starts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        starts.append(_random_sphere(N, seed=seed))
    
    # 4. Perturbed Fibonacci
    rng = np.random.RandomState(7)
    fib = _fibonacci_sphere(N)
    perturbed = fib + 0.05 * rng.randn(N, 3)
    norms = np.sqrt(np.sum(perturbed * perturbed, axis=1))
    starts.append(perturbed / norms[:, np.newaxis])
    
    # 5. Another perturbed version
    rng2 = np.random.RandomState(13)
    perturbed2 = fib + 0.1 * rng2.randn(N, 3)
    norms2 = np.sqrt(np.sum(perturbed2 * perturbed2, axis=1))
    starts.append(perturbed2 / norms2[:, np.newaxis])
    
    best_energy = np.inf
    best_pts = None
    
    # Optimize each starting configuration
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 10.0:  # Need at least 10s for final polish
            break
        
        # Run L-BFGS optimization
        try:
            opt_pts, opt_energy = _lbfgs_minimize(
                start_pts, max_iter=800, tol=1e-13, deadline=deadline - 5.0
            )
        except Exception:
            continue
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < deadline - 2.0:
        try:
            polished_pts, polished_energy = _lbfgs_minimize(
                best_pts, max_iter=500, tol=1e-14, deadline=deadline - 1.0
            )
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pts = polished_pts
        except Exception:
            pass
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final normalization to ensure unit sphere
    norms = np.sqrt(np.sum(best_pts * best_pts, axis=1))
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms[:, np.newaxis]
    
    # Verify no points are too close
    # (The optimizer should handle this, but just in case)
    
    return best_pts
