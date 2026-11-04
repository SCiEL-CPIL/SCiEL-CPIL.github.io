import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        mask: (M,) boolean array of valid facets
    """
    M = len(simplices)
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = np.where(dist > np.pi / 2, np.pi - dist, dist)
    
    return radii, centers, valid

def _covering_radius_rad(points):
    """
    Compute the covering radius of a set of points on the unit sphere (in radians).
    """
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return np.pi
        return float(np.max(radii[valid]))
    except Exception:
        return np.pi

def _covering_radius_deg(points):
    return _covering_radius_rad(points) * 180.0 / np.pi

def _smooth_max_objective_and_grad(points, beta=40.0):
    """
    Compute the smooth max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros_like(points)
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    M = len(radii)
    
    if M == 0 or not np.any(valid):
        return 0.0, np.zeros_like(points)
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros_like(points)
    
    r_max = np.max(r_valid)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(points)
    
    shifted = beta * (r_valid - r_max)
    # Numerical stability for log-sum-exp
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(points)
    
    for k in range(len(f_valid)):
        w_k = weights[k]
        if w_k < 1e-8:
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        for idx in tri:
            p = points[idx]
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                grad_p = -(c - dot_cp * p) / sin_r
                grad_p -= np.dot(grad_p, p) * p
                grad[idx] += w_k * grad_p
    
    return float(log_sum_exp), grad

def _lbfgs_polish(points, time_limit=20.0, beta_init=40.0, max_iter=2000):
    """
    Polish using L-BFGS-B on a smooth surrogate of the covering radius.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    beta = beta_init
    
    def objective_and_grad(x_flat):
        if time.time() - start > time_limit:
            return np.pi, np.zeros_like(x_flat)
        x = x_flat.reshape(-1, 3)
        new_points = _normalize(x)
        if not np.all(np.isfinite(new_points)):
            return np.pi, np.zeros_like(x_flat)
        obj, grad = _smooth_max_objective_and_grad(new_points, beta)
        # Project gradient to tangent space
        grad_tan = np.zeros_like(new_points)
        for i in range(N):
            g = grad[i]
            gt = g - np.dot(g, new_points[i]) * new_points[i]
            grad_tan[i] = gt
        return obj, grad_tan.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
        if result.fun < best_radius:
            x_opt = result.x.reshape(-1, 3)
            best_points = _normalize(x_opt)
            best_radius = _covering_radius_rad(best_points)
    except Exception:
        pass
    
    return best_points

def _nelder_mead_polish(points, max_facets=50, max_iter=300, time_limit=10.0):
    """
    Polish using Nelder-Mead on the vertices of the largest facets.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return best_points
    except Exception:
        return best_points
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return best_points
    
    top_k = min(max_facets, len(r_valid))
    top_indices = np.argsort(r_valid)[-top_k:]
    
    vertex_indices = set()
    for idx in top_indices:
        for v in f_valid[idx]:
            vertex_indices.add(v)
    
    vertex_indices = list(vertex_indices)
    
    if len(vertex_indices) == 0:
        return best_points
    
    def objective(x_flat):
        if time.time() - start > time_limit:
            return np.pi
        x = x_flat.reshape(-1, 3)
        new_points = points.copy()
        for i, idx in enumerate(vertex_indices):
            new_points[idx] = x[i]
        new_points = _normalize(new_points)
        if not np.all(np.isfinite(new_points)):
            return np.pi
        return _covering_radius_rad(new_points)
    
    x0 = points[vertex_indices].flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_radius:
            x_opt = result.x.reshape(-1, 3)
            best_points = points.copy()
            for i, idx in enumerate(vertex_indices):
                best_points[idx] = x_opt[i]
            best_points = _normalize(best_points)
            best_radius = result.fun
    except Exception:
        pass
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    
    # Generate initializations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = _normalize(fib_pts)
    
    # Perturbed Fibonacci
    fib_pert = fib_pts + rng.normal(0, 0.05, fib_pts.shape)
    fib_pert = _normalize(fib_pert)
    
    inits.append(fib_pts)
    inits.append(fib_pert)
    
    # 2. Random points
    for _ in range(2):
        pts = rng.normal(0, 1, (N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # 3. Greedy initialization
    greedy_pts = np.empty((N, 3))
    greedy_pts[0] = rng.normal(0, 1, 3)
    greedy_pts[0] = greedy_pts[0] / np.linalg.norm(greedy_pts[0])
    for i in range(1, N):
        candidates = rng.normal(0, 1, (1000, 3))
        candidates = _normalize(candidates)
        D = np.einsum('ij,kj->ik', candidates, greedy_pts[:i])
        D = np.clip(D, -1, 1)
        min_dist = np.arccos(D).min(axis=1)
        best_idx = np.argmax(min_dist)
        greedy_pts[i] = candidates[best_idx]
    inits.append(greedy_pts)
    
    # 4. Electrostatic refinement
    def electrostatic_refine(pts, n_iters=150, lr=0.005):
        pts = pts.copy()
        for _ in range(n_iters):
            D = pts @ pts.T
            chord2 = 2.0 - 2.0 * D
            np.fill_diagonal(chord2, 1.0)
            W = 1.0 / chord2
            np.fill_diagonal(W, 0.0)
            row_sum = W.sum(axis=1)
            forces = pts * row_sum[:, None] - (W @ pts)
            forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            forces /= norms
            pts = pts + lr * forces
            pts = _normalize(pts)
        return pts
    
    if time_left() > 15.0:
        elec_pts = electrostatic_refine(fib_pts, n_iters=100, lr=0.005)
        inits.append(elec_pts)

    # Evaluate initializations and pick the best
    best_points = None
    best_radius = np.pi
    
    for init in inits:
        if time_left() < 10.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_radius:
            best_radius = rad
            best_points = init.copy()
    
    if best_points is None:
        best_points = fib_pts
        best_radius = _covering_radius_rad(fib_pts)
    
    # Main optimization loop with Lloyd-style moves
    points = best_points.copy()
    
    use_gradient = False
    gradient_step = 0.02
    beta = 30.0
    lloyd_step = 0.03
    min_step = 1e-7
    no_improve_count = 0
    max_no_improve = 20
    
    while time_left() > 30.0:
        try:
            hull = ConvexHull(points)
            radii, centers, valid = _facet_circumradii(points, hull.simplices)
            if not np.any(valid):
                break
        except Exception:
            kick = rng.normal(0, 0.02, points.shape)
            points += kick
            points = _normalize(points)
            continue
        
        r_valid = radii[valid]
        c_valid = centers[valid]
        f_valid = hull.simplices[valid]
        
        if use_gradient:
            obj, grad = _smooth_max_objective_and_grad(points, beta)
            
            grad_tangent = np.zeros_like(points)
            for i in range(N):
                g = grad[i]
                g_tan = g - np.dot(g, points[i]) * points[i]
                norm_g = np.linalg.norm(g_tan)
                if norm_g > 1e-12:
                    grad_tangent[i] = g_tan / norm_g
            
            step = gradient_step
            new_points = points - step * grad_tangent
            new_points = _normalize(new_points)
            current_radius = _covering_radius_rad(new_points)
            
            while current_radius >= best_radius and step > min_step:
                step *= 0.5
                new_points = points - step * grad_tangent
                new_points = _normalize(new_points)
                current_radius = _covering_radius_rad(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
                gradient_step = min(step * 1.5, 0.05)
                beta = min(beta * 1.03, 150.0)
            else:
                gradient_step = max(step * 0.5, min_step)
                no_improve_count += 1
        else:
            point_max_radius = np.zeros(N)
            point_max_center = np.zeros((N, 3))
            
            for k in range(len(f_valid)):
                r_k = r_valid[k]
                if r_k <= 0:
                    continue
                c_k = c_valid[k]
                tri = f_valid[k]
                for idx in tri:
                    if r_k > point_max_radius[idx]:
                        point_max_radius[idx] = r_k
                        point_max_center[idx] = c_k
            
            max_move = np.zeros_like(points)
            
            for i in range(N):
                if point_max_radius[i] > 0:
                    v = point_max_center[i] - points[i]
                    dot_vp = np.dot(v, points[i])
                    tangent_v = v - dot_vp * points[i]
                    norm_tv = np.linalg.norm(tangent_v)
                    if norm_tv > 1e-12:
                        max_move[i] = (tangent_v / norm_tv) * point_max_radius[i]
            
            new_points = points + lloyd_step * max_move
            new_points = _normalize(new_points)
            
            current_radius = _covering_radius_rad(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
                lloyd_step = min(lloyd_step * 1.04, 0.08)
            else:
                lloyd_step = max(lloyd_step * 0.82, min_step)
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            if not use_gradient:
                use_gradient = True
                gradient_step = 0.02
                beta = 30.0
            else:
                kick = rng.normal(0, 0.04, points.shape)
                points += kick
                points = _normalize(points)
                lloyd_step = 0.02
                gradient_step = 0.01
                beta = 20.0
                no_improve_count = 0
                current_radius = _covering_radius_rad(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
            no_improve_count = 0
    
    # Phase 1: Nelder-Mead polish on worst facets
    if time_left() > 15.0:
        polished = _nelder_mead_polish(best_points, max_facets=50, max_iter=300, time_limit=10.0)
        pol_radius = _covering_radius_rad(polished)
        if pol_radius < best_radius:
            best_radius = pol_radius
            best_points = polished
    
    # Phase 2: L-BFGS polish with smooth surrogate
    if time_left() > 15.0:
        lbfgs_pts = _lbfgs_polish(best_points, time_limit=min(20.0, time_left() - 5.0), beta_init=40.0, max_iter=2000)
        lbfgs_radius = _covering_radius_rad(lbfgs_pts)
        if lbfgs_radius < best_radius:
            best_radius = lbfgs_radius
            best_points = lbfgs_pts
    
    # Phase 3: Final high-precision worst-facet polish
    points = best_points.copy()
    polish_step = 0.002
    
    while time_left() > 5.0:
        try:
            hull = ConvexHull(points)
            radii, centers, valid = _facet_circumradii(points, hull.simplices)
            if not np.any(valid):
                break
        except Exception:
            break
        
        r_valid = radii[valid]
        c_valid = centers[valid]
        f_valid = hull.simplices[valid]
        
        max_idx = np.argmax(r_valid)
        max_r = r_valid[max_idx]
        worst_facet = f_valid[max_idx]
        worst_center = c_valid[max_idx]
        
        for idx in worst_facet:
            v = worst_center - points[idx]
            dot_vp = np.dot(v, points[idx])
            tangent_v = v - dot_vp * points[idx]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-12:
                points[idx] += polish_step * tangent_v / norm_tv
        
        points = _normalize(points)
        
        current_radius = _covering_radius_rad(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            polish_step *= 0.5
            if polish_step < 1e-8:
                break
            points = best_points.copy()
    
    # Phase 4: One more L-BFGS pass if time allows
    if time_left() > 10.0:
        final_pts = _lbfgs_polish(best_points, time_limit=min(10.0, time_left() - 3.0), beta_init=60.0, max_iter=1000)
        final_radius = _covering_radius_rad(final_pts)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_pts
    
    return best_points
