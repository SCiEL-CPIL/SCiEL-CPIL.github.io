import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 115.0  # Safety margin
EPS = 1e-12

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
    Optimized: single dist2 matrix, no full 3D diff array.
    """
    # Squared chordal distances: |p_i - p_j|^2 = 2 - 2 p_i . p_j
    dot = pts @ pts.T
    dist2 = 2.0 - 2.0 * dot
    np.fill_diagonal(dist2, 1.0)
    dist2 = np.maximum(dist2, EPS)
    
    # Energy: sum_{i<j} log(1/dist) = -0.5 * sum_{i!=j} log(dist)
    dist = np.sqrt(dist2)
    log_dist = np.log(dist)
    np.fill_diagonal(log_dist, 0.0)
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient: grad_i = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
    # = sum_j p_j / dist2[i,j] - p_i * sum_j 1/dist2[i,j]
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    weight_sums = np.sum(inv_dist2, axis=1, keepdims=True)
    weighted_pts = inv_dist2 @ pts
    
    grad = weighted_pts - weight_sums * pts
    
    return energy, grad

def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _optimize_sphere(pts0, max_iters=2000, tol=1e-10):
    """
    Riemannian Gradient Descent with Backtracking Line Search.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    for it in range(max_iters):
        # Check convergence
        if np.linalg.norm(g_t) < tol:
            break
            
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            new_pts = _retract(pts, alpha * g_t)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * g_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-12:
                break
            continue
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Slowly decay learning rate
        lr = max(0.001, lr * 0.999)
        
    return best_pts, best_energy

def _fibonacci_spiral(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _golden_spiral_v2(n):
    """Alternative golden-angle spiral with different offset."""
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = (1.0 + np.sqrt(5.0)) * np.pi * (i + 0.3)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _icosahedron_subdivide(n):
    """Start from icosahedron vertices, subdivide and project to get ~n points."""
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide: add midpoints of edges, then project
    pts = [verts]
    # Add midpoints of all edges
    edges = [(0,1),(0,3),(0,4),(0,5),(0,10),(0,11),
             (1,2),(1,4),(1,5),(1,8),(1,9),
             (2,3),(2,6),(2,7),(2,10),(2,11),
             (3,4),(3,5),(3,8),(3,9),(3,10),(3,11),
             (4,5),(4,6),(4,7),(4,8),(4,9),
             (5,6),(5,7),(5,8),(5,9),
             (6,7),(6,8),(6,10),(6,11),
             (7,8),(7,9),(7,10),(7,11),
             (8,9),(8,10),(8,11),
             (9,10),(9,11),
             (10,11)]
    mids = []
    for a, b in edges:
        m = (verts[a] + verts[b]) / 2.0
        m = m / np.linalg.norm(m)
        mids.append(m)
    mids = np.array(mids)
    pts.append(mids)
    
    all_pts = np.vstack(pts)  # 12 + 30 = 42 points
    # If we need more, add more subdivision levels
    if n > 42:
        # Add more midpoints from the 42 points
        all_pts_flat = all_pts
        # Sample additional points by interpolating between close pairs
        rng = np.random.RandomState(7)
        for _ in range(n - len(all_pts_flat)):
            idx1 = rng.randint(len(all_pts_flat))
            idx2 = rng.randint(len(all_pts_flat))
            if idx1 != idx2:
                m = (all_pts_flat[idx1] + all_pts_flat[idx2]) / 2.0
                m = m / np.linalg.norm(m)
                all_pts_flat = np.vstack([all_pts_flat, [m]])
    
    return all_pts_flat[:n]

def _random_points(n, seed=42):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _perturbed_points(base_pts, sigma, seed=0):
    """Perturb base points by adding Gaussian noise and reprojecting."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    pts = base_pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _polish_lbfgs(pts, max_iter=200, gtol=1e-14):
    """Polish using L-BFGS-B with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-16})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def construct():
    t_start = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(('fib', _fibonacci_spiral(N)))
    
    # 2. Golden spiral variant
    inits.append(('golden2', _golden_spiral_v2(N)))
    
    # 3. Icosahedron-based
    inits.append(('icosa', _icosahedron_subdivide(N)))
    
    # 4. Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 5. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.1, 10), (0.05, 11), (0.2, 12)]:
        inits.append((f'pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    # 6. Perturbed golden spiral
    gold_base = _golden_spiral_v2(N)
    for sigma, seed in [(0.08, 20), (0.15, 21)]:
        inits.append((f'gpert{sigma}_{seed}', _perturbed_points(gold_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - t_start
        if elapsed > DEADLINE_SECONDS - 15:
            break
        
        # Riemannian Gradient Descent
        pts_opt, energy_opt = _optimize_sphere(init_pts, max_iters=500)
        
        # L-BFGS-B Polish
        if time.time() - t_start < DEADLINE_SECONDS - 10:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=200, gtol=1e-13)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() - t_start < DEADLINE_SECONDS - 5:
        best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=500, gtol=1e-15)
        
        # Additional manifold Newton-style refinement
        if time.time() - t_start < DEADLINE_SECONDS - 2:
            for _ in range(50):
                e, g = _energy_and_grad(best_pts)
                g_t = _tangent_grad(best_pts, g)
                gnorm = np.linalg.norm(g_t)
                if gnorm < 1e-14:
                    break
                # Adaptive step: use gradient norm to set step size
                step = g_t / gnorm
                alpha = min(0.01, 0.1 / (gnorm + 1e-10))
                new_pts = _retract(best_pts, -alpha * g_t)
                new_e, _ = _energy_and_grad(new_pts)
                if new_e < e:
                    best_pts = new_pts
                    best_energy = new_e
                else:
                    alpha *= 0.5
                    new_pts = _retract(best_pts, -alpha * g_t)
                    new_e, _ = _energy_and_grad(new_pts)
                    if new_e < e:
                        best_pts = new_pts
                        best_energy = new_e
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
