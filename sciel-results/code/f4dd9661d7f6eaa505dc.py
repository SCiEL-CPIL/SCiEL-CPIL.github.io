"""Improved: vectorized Riemannian CG with Armijo line search + Riemannian L-BFGS polish."""
import time
import math
import numpy as np

N = 150
DEADLINE = 115.0  # safety margin before 120s limit


def _pairwise_dists(pts):
    """Compute pairwise distances matrix."""
    sq = np.sum(pts * pts, axis=1)
    sq_d = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.maximum(sq_d, 0.0, out=sq_d)
    return np.sqrt(sq_d)


def _log_energy(pts):
    """Compute logarithmic energy."""
    d = _pairwise_dists(pts)
    np.fill_diagonal(d, 1.0)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(d[iu]))


def _energy_and_grad(pts):
    """Compute log energy and Riemannian gradient (tangent space)."""
    d = _pairwise_dists(pts)
    np.fill_diagonal(d, 1.0)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(d[iu]))
    
    # Gradient: for each i, sum over j!=i of (p_j - p_i) / |p_i - p_j|^2
    # Vectorized: grad[i] = sum_j (p_j - p_i) / d[i,j]^2
    inv_d2 = np.zeros_like(d)
    # Only fill off-diagonal entries to avoid division by zero on diagonal
    off_diag = ~np.eye(N, dtype=bool)
    inv_d2[off_diag] = 1.0 / (d[off_diag] * d[off_diag])
    
    # grad[i] = sum_j inv_d2[i,j] * (p_j - p_i)
    #         = sum_j inv_d2[i,j] * p_j - (sum_j inv_d2[i,j]) * p_i
    row_sums = inv_d2.sum(axis=1)
    grad = (inv_d2 @ pts) - row_sums[:, None] * pts
    
    # Project to tangent space
    g_dot_p = (grad * pts).sum(axis=1)
    grad = grad - g_dot_p[:, None] * pts
    
    return energy, grad


def _normalize(pts):
    """Normalize to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exp_map(pts, delta):
    """Exponential map on sphere: move points along tangent vectors."""
    # For small steps, use: new_p = cos(|delta|)*p + sin(|delta|)/|delta| * delta
    # For simplicity and stability, use: new_p = normalize(p + delta)
    return _normalize(pts + delta)


def _riemannian_cg(pts, max_iter=2000, tol=1e-12):
    """Riemannian conjugate gradient with Armijo line search and proper tangent projection."""
    energy, grad = _energy_and_grad(pts)
    d = -grad  # search direction (already in tangent space)
    prev_grad = grad.copy()
    
    for k in range(max_iter):
        # Project direction to tangent space (safety)
        d_dot_p = (d * pts).sum(axis=1)
        d = d - d_dot_p[:, None] * pts
        
        # Armijo line search
        alpha = 0.1
        c1 = 1e-4
        slope = (grad * d).sum()
        
        if slope >= 0:
            # Direction is not descent, reset
            d = -grad
            prev_grad = grad.copy()
            slope = (grad * d).sum()
        
        for ls in range(50):
            new_pts = _exp_map(pts, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy <= energy + c1 * alpha * slope:
                break
            alpha *= 0.5
        
        else:
            # Line search failed, try smaller step
            alpha = 1e-6
            new_pts = _exp_map(pts, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
        
        # Check convergence
        if np.abs(energy - new_energy) < tol and np.linalg.norm(new_grad) < 1e-10:
            return new_pts, new_energy
        
        # Update
        pts = new_pts
        energy = new_energy
        
        # Conjugate gradient update (Hestenes-Stiefel)
        grad = new_grad
        if k > 0:
            denom = np.sum(prev_grad * prev_grad)
            if denom > 1e-30:
                beta = np.sum((grad - prev_grad) * grad) / denom
                beta = max(0.0, beta)
                d = -grad + beta * d
            else:
                d = -grad
        prev_grad = grad
    
    return pts, energy


def _riemannian_lbfgs(pts, max_iter=500, m=20):
    """Riemannian L-BFGS for final polish."""
    energy, grad = _energy_and_grad(pts)
    
    # L-BFGS memory
    S = []  # steps
    Y = []  # gradient differences
    W = []  # weights
    alpha = 0.1
    
    for k in range(max_iter):
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        r = -grad
        if len(S) > 0:
            # Two-loop recursion
            q = r.copy()
            rho = []
            for i in range(len(S) - 1, -1, -1):
                s_i = S[i]
                y_i = Y[i]
                ys = (y_i * s_i).sum()
                if abs(ys) > 1e-30:
                    rho.append(1.0 / ys)
                    q = q - rho[-1] * ys * y_i
                else:
                    rho.append(0.0)
            
            # Initial Hessian approximation
            if len(Y) > 0:
                y_last = Y[-1]
                s_last = S[-1]
                yy = (y_last * y_last).sum()
                sy = (s_last * y_last).sum()
                if abs(yy) > 1e-30 and abs(sy) > 1e-30:
                    gamma = sy / yy
                else:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = gamma * q
            
            for i in range(len(S)):
                s_i = S[i]
                y_i = Y[i]
                ys = (y_i * s_i).sum()
                if abs(ys) > 1e-30:
                    rho_i = 1.0 / ys
                    alpha_i = rho_i * (s_i * q).sum()
                    beta_i = rho_i * (y_i * r).sum()
                    r = r + s_i * (alpha_i - beta_i)
                else:
                    alpha_i = 0.0
                    r = r + s_i * alpha_i
        
        # Project to tangent space
        d_dot_p = (r * pts).sum(axis=1)
        d = r - d_dot_p[:, None] * pts
        
        # Armijo line search
        c1 = 1e-4
        slope = (grad * d).sum()
        
        if slope >= 0:
            d = -grad
            slope = (grad * d).sum()
        
        alpha_step = 0.1
        for ls in range(50):
            new_pts = _exp_map(pts, alpha_step * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy <= energy + c1 * alpha_step * slope:
                break
            alpha_step *= 0.5
        
        else:
            alpha_step = 1e-6
            new_pts = _exp_map(pts, alpha_step * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
        
        # Update L-BFGS memory
        s_k = alpha_step * d
        y_k = new_grad - grad
        
        # Curvature condition
        sy = (s_k * y_k).sum()
        if sy > 1e-16:
            S.append(s_k)
            Y.append(y_k)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
        
        # Check convergence
        if np.abs(energy - new_energy) < 1e-14 and np.linalg.norm(new_grad) < 1e-12:
            return new_pts, new_energy
        
        pts = new_pts
        energy = new_energy
        grad = new_grad
    
    return pts, energy


def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere(n):
    """Generate points using Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        # Map to sphere using spherical coordinates
        theta = 2.0 * np.pi * u
        cos_phi = 2.0 * v - 1.0
        sin_phi = np.sqrt(1.0 - cos_phi * cos_phi)
        points.append([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
    return np.array(points)


def _farthest_point_init(n, max_iter=1000):
    """Farthest-point sampling for well-spread initial points."""
    rng = np.random.RandomState(123)
    pts = [rng.randn(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    
    for _ in range(n - 1):
        # Compute distances from all existing points to a random candidate
        best_dist = -1
        best_pt = None
        
        # Use a grid of candidate points
        for _ in range(200):
            cand = rng.randn(3)
            cand = cand / np.linalg.norm(cand)
            
            # Min distance to existing points
            d2 = np.sum((np.array(pts) - cand) ** 2, axis=1)
            min_d = np.min(d2)
            
            if min_d > best_dist:
                best_dist = min_d
                best_pt = cand
        
        pts.append(best_pt)
    
    return np.array(pts)


def construct():
    start_time = time.time()
    
    # Generate initial points using multiple methods
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
        _farthest_point_init(N),
    ]
    
    # Add some random inits
    rng = np.random.RandomState(42)
    for _ in range(4):
        pts = rng.randn(N, 3)
        inits.append(_normalize(pts))
    
    best_points = None
    best_energy = float('inf')
    
    for init_pts in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Optimize with Riemannian CG
        opt_pts, opt_energy = _riemannian_cg(init_pts, max_iter=1200, tol=1e-13)
        
        if opt_energy < best_energy:
            best_points = opt_pts.copy()
            best_energy = opt_energy
    
    # Riemannian L-BFGS polish on the best result
    if best_points is not None and time.time() - start_time < DEADLINE - 10:
        polished_pts, polished_energy = _riemannian_lbfgs(best_points, max_iter=300, m=20)
        if polished_energy < best_energy:
            best_points = polished_pts
            best_energy = polished_energy
    
    # Final CG pass with tighter tolerance
    if time.time() - start_time < DEADLINE - 3:
        final_pts, final_energy = _riemannian_cg(best_points, max_iter=500, tol=1e-15)
        if final_energy < best_energy:
            best_points = final_pts
            best_energy = final_energy
    
    # Ensure valid output
    best_points = _normalize(best_points)
    
    # Final sanity check: no points too close
    d = _pairwise_dists(best_points)
    iu = np.triu_indices(N, k=1)
    min_d = np.min(d[iu])
    if min_d < 1e-9:
        # Fallback to Fibonacci
        best_points = _fibonacci_sphere(N)
    
    return best_points
