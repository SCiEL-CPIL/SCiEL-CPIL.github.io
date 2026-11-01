"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE_SECONDS = 115.0
EPS = 1e-14

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the Euclidean gradient w.r.t. pts.
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    # Squared distances
    dist2 = np.sum(diff * diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # Safe distances
    dist = np.sqrt(np.where(mask, np.maximum(dist2, EPS), 1.0))
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # We sum over all i!=j and divide by 2
    log_terms = np.where(mask, -np.log(dist), 0.0)
    energy = 0.5 * np.sum(log_terms)
    
    # Gradient:
    # d/dp_i log(1/|p_i - p_j|) = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
    # Note: dist2 is |p_i - p_j|^2
    inv_dist2 = np.where(mask, 1.0 / np.maximum(dist2, EPS), 0.0)
    
    # grad[i] = sum_j (pts[j] - pts[i]) * inv_dist2[i, j]
    # = sum_j pts[j] * inv_dist2[i, j] - pts[i] * sum_j inv_dist2[i, j]
    # Let's compute the sum of weights for each i
    weight_sums = np.sum(inv_dist2, axis=1, keepdims=True)
    # Sum of pts[j] * inv_dist2[i, j]
    # This is inv_dist2 @ pts
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

def _optimize_sphere(pts0, max_iters=3000, tol=1e-12, time_limit=None):
    """
    Riemannian Conjugate Gradient with Backtracking Line Search.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    start_time = time.time()
    
    # Conjugate gradient state
    d = -g_t.copy()  # initial descent direction
    g_t_prev = g_t.copy()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Check convergence
        if np.linalg.norm(g_t) < tol:
            break
            
        # Ensure d is a descent direction
        if np.sum(d * g_t) > 0:
            d = -g_t.copy()
        
        # Backtracking line search
        alpha = lr
        improved = False
        g_t_norm_sq = np.sum(g_t * g_t)
        for _ in range(30):
            new_pts = _retract(pts, alpha * d)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * d):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-14:
                break
            # Reset CG direction
            d = -g_t.copy()
            g_t_prev = g_t.copy()
            continue
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t_new = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Polak-Ribiere update for CG
        g_diff = g_t_new - g_t_prev
        pr_coeff = np.sum(g_t_new * g_diff) / max(np.sum(g_t_prev * g_t_prev), 1e-30)
        if pr_coeff < 0:
            pr_coeff = 0.0
        d = -g_t_new + pr_coeff * d
        g_t_prev = g_t
        g_t = g_t_new
        
        # Slowly decay learning rate
        lr = max(0.0005, lr * 0.999)
        
    return best_pts, best_energy

def _fibonacci_spiral(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _icosahedron_subdivide(n_target=100):
    """Generate points via icosahedron subdivision, then select n_target."""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide each face into 4 sub-faces (midpoint subdivision)
    def subdivide_face(v1, v2, v3, cache):
        def midpoint(p, q):
            key = (min(p, q), max(p, q))
            if key in cache:
                return cache[key]
            mid = (verts[p] + verts[q]) / 2.0
            mid = mid / np.linalg.norm(mid)
            idx = len(verts_list)
            verts_list.append(mid)
            cache[key] = idx
            return idx
        
        verts_list = list(verts)
        # We need a mutable list, so restructure
        return None
    
    # Simpler approach: just use multiple levels of subdivision
    all_pts = [verts]
    for level in range(4):
        new_pts = []
        for i in range(len(all_pts[-1])):
            pass
        # Actually, let's just generate points on the icosahedron surface
        break
    
    # Alternative: use the icosahedron vertices and their midpoints recursively
    # For n=100, we can use icosahedron (12) + edge midpoints (30) + face centers (20) + more
    # Let's just generate a good set using the icosahedron structure
    
    # Use icosahedron vertices
    result = [verts]
    
    # Add edge midpoints
    edges = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i+1)%3]]))
            edges.add(e)
    
    midpoints = []
    for e in edges:
        mid = (verts[e[0]] + verts[e[1]]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    result.append(np.array(midpoints))
    
    # Add face centers
    face_centers = []
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        center = center / np.linalg.norm(center)
        face_centers.append(center)
    result.append(np.array(face_centers))
    
    # Combine all
    all_points = np.vstack(result)
    
    # If we have more than n_target, use k-means-like selection or just take first n_target
    if len(all_points) >= n_target:
        # Use a simple greedy selection for diversity
        selected = [0]  # start with first point
        for _ in range(n_target - 1):
            # Find the point that maximizes minimum distance to selected
            best_idx = -1
            best_min_dist = -1
            for i in range(len(all_points)):
                if i in selected:
                    continue
                min_dist = min(np.linalg.norm(all_points[i] - all_points[s]) for s in selected)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i
            selected.append(best_idx)
        return all_points[selected]
    else:
        # Pad with random points
        rng = np.random.RandomState(42)
        extra = rng.randn(n_target - len(all_points), 3)
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        return np.vstack([all_points, extra])

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

def _polish_lbfgs(pts, max_iter=500, gtol=1e-16):
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
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-18, 'maxls': 50})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def _polish_bfgs(pts, max_iter=200, gtol=1e-16):
    """Polish using BFGS with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='BFGS',
                       options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 50})
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
    start_time = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(('fib', _fibonacci_spiral(N)))
    
    # 2. Icosahedron subdivision
    try:
        ico_pts = _icosahedron_subdivide(N)
        if ico_pts is not None and len(ico_pts) == N:
            inits.append(('ico', ico_pts))
    except Exception:
        pass
    
    # 3. Random with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 4. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.05, 10), (0.1, 11), (0.15, 12), (0.2, 13), (0.02, 14), (0.08, 15)]:
        inits.append((f'pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 15:
            break
        
        # Riemannian Conjugate Gradient
        remaining = DEADLINE_SECONDS - 10 - (time.time() - start_time)
        if remaining > 2:
            pts_opt, energy_opt = _optimize_sphere(init_pts, max_iters=1500)
        else:
            pts_opt, energy_opt = init_pts, _energy_and_grad(init_pts)[0]
        
        # L-BFGS-B Polish
        if time.time() - start_time < DEADLINE_SECONDS - 12:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=300, gtol=1e-15)
        
        # BFGS Polish
        if time.time() - start_time < DEADLINE_SECONDS - 8:
            pts_opt2, energy_opt2 = _polish_bfgs(pts_opt, max_iter=150, gtol=1e-15)
            if energy_opt2 < energy_opt:
                pts_opt, energy_opt = pts_opt2, energy_opt2
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE_SECONDS - 8:
        best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=800, gtol=1e-17)
        # Second polish pass with BFGS
        if time.time() - start_time < DEADLINE_SECONDS - 5:
            best_pts2, best_energy2 = _polish_bfgs(best_pts, max_iter=300, gtol=1e-17)
            if best_energy2 < best_energy:
                best_pts, best_energy = best_pts2, best_energy2
        # Third polish: another L-BFGS pass with very tight tolerance
        if time.time() - start_time < DEADLINE_SECONDS - 3:
            best_pts3, best_energy3 = _polish_lbfgs(best_pts, max_iter=500, gtol=1e-18)
            if best_energy3 < best_energy:
                best_pts, best_energy = best_pts3, best_energy3
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them slightly
                best_pts[j] = best_pts[i] + 1e-8 * best_pts[j]
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
