"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 130
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def _remaining_time():
    return max(0.0, TIME_LIMIT - (time.time() - START_TIME))

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts, n_samples=20000):
    """Estimate covering radius by random sampling on sphere."""
    pts = _normalize(pts)
    # Generate random points on sphere
    rng = np.random.default_rng(42)
    u = rng.uniform(-1, 1, (n_samples, 3))
    # Rejection sample
    norms = np.linalg.norm(u, axis=1)
    mask = norms > 0
    u = u[mask]
    while len(u) < n_samples:
        extra = rng.uniform(-1, 1, (1000, 3))
        en = np.linalg.norm(extra, axis=1)
        em = en > 0
        u = np.vstack([u, extra[em]])
    u = u[:n_samples]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    
    # For each random point, find nearest config point
    # cos of angle = dot product
    dots = u @ pts.T  # (n_samples, N)
    max_cos = np.max(dots, axis=1)
    # Clip to [-1, 1]
    max_cos = np.clip(max_cos, -1.0, 1.0)
    angles = np.arccos(max_cos)
    return np.max(angles) * 180.0 / np.pi

def _covering_radius_fine(pts, n_samples=100000):
    """Higher accuracy covering radius estimate."""
    return _covering_radius(pts, n_samples)

def _objective_and_grad(pts_flat):
    """Objective: covering radius (in radians), with gradient via subgradient."""
    pts = _normalize(pts_flat.reshape(N, 3))
    
    # Sample points to estimate covering radius
    rng = np.random.default_rng(12345)
    n_samp = 50000
    u = rng.uniform(-1, 1, (n_samp, 3))
    norms = np.linalg.norm(u, axis=1)
    mask = norms > 0
    u = u[mask]
    while len(u) < n_samp:
        extra = rng.uniform(-1, 1, (5000, 3))
        en = np.linalg.norm(extra, axis=1)
        em = en > 0
        u = np.vstack([u, extra[em]])
    u = u[:n_samp]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    
    dots = u @ pts.T  # (n_samp, N)
    max_cos = np.max(dots, axis=1)
    max_cos_clipped = np.clip(max_cos, -1.0, 1.0)
    angles = np.arccos(max_cos_clipped)
    
    idx_max = np.argmax(angles)
    radius = angles[idx_max]
    
    # Subgradient: derivative of arccos(cos) w.r.t. cos is -1/sqrt(1-cos^2)
    # The point y that maximizes the distance determines the subgradient
    y = u[idx_max]
    cos_val = max_cos_clipped[idx_max]
    sin_val = np.sqrt(max(0.0, 1.0 - cos_val**2))
    
    if sin_val < 1e-10:
        # Degenerate case
        return radius, np.zeros_like(pts_flat)
    
    # d(radius)/d(cos) = -1/sin_val
    # d(cos)/d(pts[i]) = y for the argmax point
    # So d(radius)/d(pts[i]) = -y/sin_val for the argmax point
    nearest_idx = np.argmax(dots[idx_max])
    grad = np.zeros_like(pts_flat)
    # The gradient w.r.t. the nearest point is -y/sin_val
    # But we need to be careful: the objective is max over y of arccos(max_i y·p_i)
    # Subgradient w.r.t. p_j (j = argmax) is -y/sin(angle)
    grad[nearest_idx*3:(nearest_idx+1)*3] = -y / sin_val
    
    # Normalize gradient to unit sphere (project)
    # Actually, for L-BFGS on sphere, we just need the gradient in ambient space
    # and the optimizer handles the constraint
    
    return radius, grad

def _lloyd_iteration(pts, n_iterations=50):
    """Perform Lloyd's algorithm on spherical Voronoi cells."""
    pts = _normalize(pts)
    
    for it in range(n_iterations):
        if _remaining_time() < 5.0:
            break
        
        # For each point, find its Voronoi cell centroid
        # Approximate by sampling within a cap around each point
        new_pts = np.zeros_like(pts)
        
        for i in range(N):
            # Sample points near p_i to estimate cell centroid
            # Use a small cap around p_i
            p = pts[i]
            n_local = 200
            # Generate random directions
            rng = np.random.default_rng(i * 1000 + it)
            v = rng.normal(size=(n_local, 3))
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            
            # Only keep points where p_i is the nearest
            dots = v @ pts.T  # (n_local, N)
            nearest = np.argmax(dots, axis=1)
            mask = nearest == i
            if np.sum(mask) > 0:
                cell_pts = v[mask]
                # Weighted centroid (spherical mean)
                centroid = np.sum(cell_pts, axis=0)
                norm_c = np.linalg.norm(centroid)
                if norm_c > 1e-10:
                    new_pts[i] = centroid / norm_c
                else:
                    new_pts[i] = p
            else:
                new_pts[i] = p
        
        pts = _normalize(new_pts)
    
    return pts

def _bubble_inflation(pts, n_steps=100):
    """Bubble inflation: iteratively find largest uncovered gap and move nearest point."""
    pts = _normalize(pts)
    rng = np.random.default_rng(777)
    
    for step in range(n_steps):
        if _remaining_time() < 5.0:
            break
        
        # Find the largest uncovered region by random sampling
        n_samp = 30000
        u = rng.uniform(-1, 1, (n_samp, 3))
        norms = np.linalg.norm(u, axis=1)
        mask = norms > 0
        u = u[mask]
        while len(u) < n_samp:
            extra = rng.uniform(-1, 1, (5000, 3))
            en = np.linalg.norm(extra, axis=1)
            em = en > 0
            u = np.vstack([u, extra[em]])
        u = u[:n_samp]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        
        dots = u @ pts.T
        max_cos = np.max(dots, axis=1)
        max_cos_clipped = np.clip(max_cos, -1.0, 1.0)
        angles = np.arccos(max_cos_clipped)
        
        idx_max = np.argmax(angles)
        gap_point = u[idx_max]
        nearest_idx = np.argmax(dots[idx_max])
        
        # Move the nearest point toward the gap
        # Blend: new_pos = normalize(alpha * old + (1-alpha) * gap)
        alpha = 0.3
        new_pos = alpha * pts[nearest_idx] + (1 - alpha) * gap_point
        norm_new = np.linalg.norm(new_pos)
        if norm_new > 1e-10:
            pts[nearest_idx] = new_pos / norm_new
    
    return _normalize(pts)

def _gradient_descent(pts, n_steps=200, lr=0.01):
    """Gradient descent with line search on sphere."""
    pts = _normalize(pts)
    
    for step in range(n_steps):
        if _remaining_time() < 5.0:
            break
        
        val, grad = _objective_and_grad(pts.flatten())
        
        # Project gradient to tangent space
        grad_3d = grad.reshape(N, 3)
        proj_grad = grad_3d - np.sum(grad_3d * pts, axis=1, keepdims=True) * pts
        
        # Line search
        best_val = val
        best_pts = pts.copy()
        for lr_try in [lr, lr*0.5, lr*0.25, lr*0.1]:
            new_pts = _normalize(pts - lr_try * proj_grad)
            new_val = _covering_radius(new_pts, n_samples=5000)
            if new_val < best_val:
                best_val = new_val
                best_pts = new_pts
        
        pts = best_pts
    
    return pts

def construct():
    """Construct 130 points on unit sphere minimizing covering radius."""
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius_fine(pts)
    
    # Phase 1: Bubble inflation
    if _remaining_time() > 20.0:
        pts_bi = _bubble_inflation(pts.copy(), n_steps=150)
        r_bi = _covering_radius_fine(pts_bi)
        if r_bi < best_radius:
            best_radius = r_bi
            best_pts = pts_bi.copy()
        pts = pts_bi
    
    # Phase 2: Lloyd's algorithm
    if _remaining_time() > 15.0:
        pts_lloyd = _lloyd_iteration(pts.copy(), n_iterations=30)
        r_lloyd = _covering_radius_fine(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
        pts = pts_lloyd
    
    # Phase 3: Gradient descent with multiple restarts
    if _remaining_time() > 10.0:
        # Try from current best
        pts_gd = _gradient_descent(best_pts.copy(), n_steps=100, lr=0.02)
        r_gd = _covering_radius_fine(pts_gd)
        if r_gd < best_radius:
            best_radius = r_gd
            best_pts = pts_gd.copy()
    
    # Phase 4: More restarts with perturbation
    if _remaining_time() > 15.0:
        rng = np.random.default_rng(999)
        for restart in range(5):
            if _remaining_time() < 5.0:
                break
            
            # Perturb best points
            perturb_scale = 0.1
            noisy_pts = best_pts + rng.normal(scale=perturb_scale, size=(N, 3))
            noisy_pts = _normalize(noisy_pts)
            
            # Lloyd + gradient
            pts_r = _lloyd_iteration(noisy_pts, n_iterations=20)
            pts_r = _gradient_descent(pts_r, n_steps=50, lr=0.01)
            r_r = _covering_radius_fine(pts_r)
            if r_r < best_radius:
                best_radius = r_r
                best_pts = pts_r.copy()
    
    # Phase 5: Final fine-tuning with L-BFGS
    if _remaining_time() > 8.0:
        # Use scipy L-BFGS-B with sphere constraint via parameterization
        # Parameterize each point by 2 angles (theta, phi) to stay on sphere
        def _sphere_params_to_pts(params):
            theta = params[0::2]  # polar angle
            phi = params[1::2]    # azimuthal angle
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            return np.column_stack([x, y, z])
        
        def _obj_and_grad(params):
            pts = _normalize(_sphere_params_to_pts(params))
            # Use a differentiable approximation
            n_samp = 20000
            rng2 = np.random.default_rng(42)
            u = rng2.uniform(-1, 1, (n_samp, 3))
            norms = np.linalg.norm(u, axis=1)
            mask = norms > 0
            u = u[mask]
            while len(u) < n_samp:
                extra = rng2.uniform(-1, 1, (3000, 3))
                en = np.linalg.norm(extra, axis=1)
                em = en > 0
                u = np.vstack([u, extra[em]])
            u = u[:n_samp]
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            
            dots = u @ pts.T
            max_cos = np.max(dots, axis=1)
            max_cos_clipped = np.clip(max_cos, -1.0, 1.0)
            angles = np.arccos(max_cos_clipped)
            val = np.max(angles)
            
            # Numerical gradient (expensive but reliable)
            eps = 1e-6
            grad = np.zeros_like(params)
            for j in range(len(params)):
                params_plus = params.copy()
                params_plus[j] += eps
                pts_plus = _normalize(_sphere_params_to_pts(params_plus))
                dots_plus = u @ pts_plus.T
                max_cos_plus = np.max(dots_plus, axis=1)
                max_cos_plus_clipped = np.clip(max_cos_plus, -1.0, 1.0)
                angles_plus = np.arccos(max_cos_plus_clipped)
                val_plus = np.max(angles_plus)
                grad[j] = (val_plus - val) / eps
            
            return val, grad
        
        # Start from best_pts
        # Invert: convert pts to (theta, phi) params
        theta = np.arccos(np.clip(best_pts[:, 2], -1.0, 1.0))
        phi = np.arctan2(best_pts[:, 1], best_pts[:, 0])
        params0 = np.zeros(2 * N)
        params0[0::2] = theta
        params0[1::2] = phi
        
        try:
            result = minimize(
                _obj_and_grad, params0, method='L-BFGS-B',
                jac=True, options={'maxiter': 100, 'ftol': 1e-10}
            )
            if result.success or result.fun < best_radius:
                pts_opt = _normalize(_sphere_params_to_pts(result.x))
                r_opt = _covering_radius_fine(pts_opt)
                if r_opt < best_radius:
                    best_radius = r_opt
                    best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return _normalize(best_pts)
