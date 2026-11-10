"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(pts):
    """Compute covering radius in radians. Returns (radius, worst_facet_index, worst_point)."""
    try:
        hull = ConvexHull(pts)
        n_sites = len(pts)
        max_ang = 0.0
        worst_facet = -1
        worst_pt = None
        
        # Vectorized computation of circumcenters for all facets
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        # Normal vectors to each triangle
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return 0.0, -1, None
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        # Ensure normals point outward (away from origin)
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        # Compute angular distance from each circumcenter to nearest site
        dots = np.clip(pts @ normals.T, -1.0, 1.0)
        dists = np.arccos(dots)  # shape: (n_sites, n_facets)
        min_dists = np.min(dists, axis=0)  # shape: (n_facets,)
        
        max_idx = np.argmax(min_dists)
        max_ang = min_dists[max_idx]
        worst_facet = np.where(valid)[0][max_idx]
        worst_pt = normals[max_idx].copy()
        
        return max_ang, worst_facet, worst_pt
    except Exception:
        return np.pi, -1, None


def _covering_radius_soft(pts, beta=50.0):
    """Compute a smooth surrogate for covering radius using soft-max."""
    try:
        hull = ConvexHull(pts)
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return 0.0
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        dots = np.clip(pts @ normals.T, -1.0, 1.0)
        dists = np.arccos(dots)  # shape: (n_sites, n_facets)
        min_dists = np.min(dists, axis=0)  # shape: (n_facets,)
        
        # Soft-max surrogate: beta * log(sum(exp(beta * x))) / beta
        shifted = min_dists - np.max(min_dists)
        soft_max = np.max(min_dists) + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
        
        return soft_max
    except Exception:
        return np.pi


def _fibonacci_start():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(rng):
    """Generate random initial points on the sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _lloyd_step(pts, n_iter=30):
    """Perform Lloyd's algorithm iterations to minimize covering radius."""
    pts = pts.copy()
    for _ in range(n_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Vectorized circumcenter computation
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            break
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        circumcenters = normals
        
        new_pts = pts.copy()
        for i in range(len(pts)):
            dists = np.linalg.norm(circumcenters - pts[i], axis=1)
            k = min(5, len(circumcenters))
            idx = np.argsort(dists)[:k]
            centroid = circumcenters[idx].mean(axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 1e-10:
                centroid = centroid / nc
                new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
                new_pts[i] /= np.linalg.norm(new_pts[i])
        
        pts = new_pts
    
    return pts


def _local_optimize(pts, n_iter=100, base_alpha=0.05):
    """Local optimization targeting the worst facet with adaptive step size."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        a_idx, b_idx, c_idx = tri[0], tri[1], tri[2]
        
        # Adaptive step size: reduce if improvement is small
        alpha = base_alpha * (0.5 ** (it // 20))
        
        for idx in [a_idx, b_idx, c_idx]:
            new_pt = pts[idx] + alpha * (wpt - pts[idx])
            new_pt /= np.linalg.norm(new_pt)
            pts[idx] = new_pt
    
    return pts


def _gradient_descent_step(pts, n_iter=50, lr=0.02, beta=50.0):
    """Use gradient descent on the soft-max surrogate to improve covering radius."""
    pts = pts.copy()
    
    for it in range(n_iter):
        # Compute current soft-max value
        current_val = _covering_radius_soft(pts, beta)
        
        # Compute gradient numerically for each point
        grad = np.zeros_like(pts)
        eps = 1e-4
        for i in range(len(pts)):
            for d in range(3):
                pts_plus = pts.copy()
                pts_plus[i, d] += eps
                # Renormalize
                nrm = np.linalg.norm(pts_plus[i])
                pts_plus[i] /= nrm
                val_plus = _covering_radius_soft(pts_plus, beta)
                
                pts_minus = pts.copy()
                pts_minus[i, d] -= eps
                nrm = np.linalg.norm(pts_minus[i])
                pts_minus[i] /= nrm
                val_minus = _covering_radius_soft(pts_minus, beta)
                
                grad[i, d] = (val_plus - val_minus) / (2 * eps)
        
        # Project gradient to tangent space (remove radial component)
        for i in range(len(pts)):
            radial = np.dot(grad[i], pts[i]) * pts[i]
            grad[i] -= radial
        
        # Update
        lr_eff = lr * (0.9 ** (it // 10))
        new_pts = pts - lr_eff * grad
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check if improved
        new_val = _covering_radius_soft(new_pts, beta)
        if new_val < current_val:
            pts = new_pts
        else:
            # Try smaller step
            new_pts = pts - lr_eff * 0.1 * grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_val = _covering_radius_soft(new_pts, beta)
            if new_val < current_val:
                pts = new_pts
            else:
                break
    
    return pts


def _polish(pts, n_iter=50):
    """High-precision polish phase with adaptive step-size reduction."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        
        # Try multiple step sizes
        for alpha_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
            alpha = 0.01 * alpha_mult
            trial = pts.copy()
            for idx in tri:
                new_pt = trial[idx] + alpha * (wpt - trial[idx])
                new_pt /= np.linalg.norm(new_pt)
                trial[idx] = new_pt
            
            r_trial, _, _ = _covering_radius(trial)
            if r_trial < radius:
                pts = trial
                break
    
    return pts


def _final_polish(pts, n_iter=100):
    """Ultra-fine polish with very small steps and line search."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        
        # Try very small steps in multiple directions
        best_trial = None
        best_radius = radius
        
        for alpha in [0.005, 0.002, 0.001, 0.0005]:
            for direction in [wpt, -wpt, wpt + 0.1 * np.random.randn(3), wpt - 0.1 * np.random.randn(3)]:
                direction = direction / np.linalg.norm(direction)
                trial = pts.copy()
                for idx in tri:
                    new_pt = trial[idx] + alpha * (direction - trial[idx])
                    new_pt /= np.linalg.norm(new_pt)
                    trial[idx] = new_pt
                
                r_trial, _, _ = _covering_radius(trial)
                if r_trial < best_radius:
                    best_radius = r_trial
                    best_trial = trial.copy()
        
        if best_trial is not None:
            pts = best_trial
        else:
            # No improvement, try moving only one vertex at a time
            improved = False
            for idx in tri:
                for alpha in [0.003, 0.001, 0.0005]:
                    trial = pts.copy()
                    new_pt = trial[idx] + alpha * (wpt - trial[idx])
                    new_pt /= np.linalg.norm(new_pt)
                    trial[idx] = new_pt
                    
                    r_trial, _, _ = _covering_radius(trial)
                    if r_trial < best_radius:
                        best_radius = r_trial
                        pts = trial
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate multiple initial configurations
    best_pts = None
    best_radius = np.pi
    
    # Fibonacci start
    pts_fib = _fibonacci_start()
    r_fib, _, _ = _covering_radius(pts_fib)
    if r_fib < best_radius:
        best_radius = r_fib
        best_pts = pts_fib.copy()
    
    # Several random starts
    for _ in range(8):
        if time.time() > deadline - 60:
            break
        pts_rand = _random_start(rng)
        r_rand, _, _ = _covering_radius(pts_rand)
        if r_rand < best_radius:
            best_radius = r_rand
            best_pts = pts_rand.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_start()
        best_radius, _, _ = _covering_radius(best_pts)
    
    pts = best_pts.copy()
    
    # Phase 2: Lloyd's algorithm for global improvement
    while time.time() < deadline - 40:
        pts_lloyd = _lloyd_step(pts, n_iter=20)
        r_lloyd, _, _ = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
            pts = pts_lloyd.copy()
        else:
            pts_lloyd = _lloyd_step(pts, n_iter=5)
            r_lloyd, _, _ = _covering_radius(pts_lloyd)
            if r_lloyd < best_radius:
                best_radius = r_lloyd
                best_pts = pts_lloyd.copy()
                pts = pts_lloyd.copy()
            else:
                break
    
    # Phase 3: Local optimization targeting worst facets
    while time.time() < deadline - 15:
        pts_local = _local_optimize(pts, n_iter=50)
        r_local, _, _ = _covering_radius(pts_local)
        if r_local < best_radius:
            best_radius = r_local
            best_pts = pts_local.copy()
            pts = pts_local.copy()
        else:
            pts_local = _local_optimize(pts, n_iter=20)
            r_local, _, _ = _covering_radius(pts_local)
            if r_local < best_radius:
                best_radius = r_local
                best_pts = pts_local.copy()
                pts = pts_local.copy()
            else:
                break
    
    # Phase 3.5: Gradient descent on soft-max surrogate
    if time.time() < deadline - 10:
        pts_grad = _gradient_descent_step(pts, n_iter=30, lr=0.01, beta=50.0)
        r_grad, _, _ = _covering_radius(pts_grad)
        if r_grad < best_radius:
            best_radius = r_grad
            best_pts = pts_grad.copy()
            pts = pts_grad.copy()
    
    # Phase 4: High-precision polish
    while time.time() < deadline - 5:
        pts_polished = _polish(pts, n_iter=30)
        r_polished, _, _ = _covering_radius(pts_polished)
        if r_polished < best_radius:
            best_radius = r_polished
            best_pts = pts_polished.copy()
            pts = pts_polished.copy()
        else:
            break
    
    # Phase 5: Final ultra-fine polish
    if time.time() < deadline - 2:
        pts_final = _final_polish(pts, n_iter=50)
        r_final, _, _ = _covering_radius(pts_final)
        if r_final < best_radius:
            best_radius = r_final
            best_pts = pts_final.copy()
    
    # Ensure output is valid
    pts_out = best_pts.copy()
    norms = np.linalg.norm(pts_out, axis=1)
    pts_out = pts_out / norms[:, np.newaxis]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts_out[i] - pts_out[j]) < 1e-8:
                pts_out[j] += np.random.randn(3) * 0.01
                pts_out[j] /= np.linalg.norm(pts_out[j])
    
    return pts_out
# EVOLVE-BLOCK-END
