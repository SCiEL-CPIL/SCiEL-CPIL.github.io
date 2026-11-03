import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_covering_radius(points):
    """Compute exact covering radius via convex hull facet circumradii (in radians)."""
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    faces = hull.simplices
    max_circ = 0.0
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        if circumradius > max_circ:
            max_circ = circumradius
    
    return max_circ

def _soft_max_circumradii(points, beta=20.0):
    """
    Compute a smooth approximation to the max circumradius using log-sum-exp.
    Returns (soft_max_value, weights, faces, circumradii, voronoi_vertices).
    """
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.array([]), np.array([], dtype=int), np.array([]), np.array([])
    
    faces = hull.simplices
    circumradii = []
    voronoi_vertices = []
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        circumradii.append(circumradius)
        voronoi_vertices.append(o)
    
    if len(circumradii) == 0:
        return np.pi, np.array([]), faces, np.array([]), np.array([])
    
    circumradii = np.array(circumradii)
    voronoi_vertices = np.array(voronoi_vertices)
    
    max_r = np.max(circumradii)
    shifted = circumradii - max_r
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted))) + max_r
    soft_max_value = log_sum_exp / beta
    
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    return soft_max_value, weights, faces, circumradii, voronoi_vertices

def _gradient_optimize(points, max_iters=300, deadline=None, beta=10.0):
    """
    Optimize using gradient descent with soft-max weighting on facet circumradii.
    Uses a smooth surrogate for the nonsmooth max circumradius objective.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    step = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        soft_max_val, weights, faces, circumradii, voronoi_vertices = _soft_max_circumradii(points, beta)
        
        if len(circumradii) == 0:
            break
        
        current_radius = np.max(circumradii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        total_grad = np.zeros((N, 3))
        
        for k in range(len(circumradii)):
            w = weights[k]
            face = faces[k]
            o = voronoi_vertices[k]
            
            for i, idx in enumerate(face):
                v = points[idx]
                dot_vo = np.dot(v, o)
                dot_vo = np.clip(dot_vo, -1.0, 1.0)
                sin_r = np.sqrt(1 - dot_vo**2)
                if sin_r < 1e-10:
                    grad = np.zeros(3)
                else:
                    grad = -o / sin_r
                
                grad_tangent = grad - np.dot(grad, v) * v
                total_grad[idx] += w * grad_tangent
        
        grad_norms = np.linalg.norm(total_grad, axis=1, keepdims=True)
        grad_norms[grad_norms < 1e-10] = 1.0
        total_grad = total_grad / grad_norms
        
        new_points = points - step * total_grad
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < current_radius:
            points = new_points
            step *= 1.1
        else:
            step *= 0.8
            if step < 1e-6:
                break
        
        step = np.clip(step, 1e-5, 0.1)
    
    return best_points, best_radius

def _compute_facet_circumradii_and_centers(points):
    """
    For each point, compute the maximum circumradius among its adjacent facets,
    and the direction (toward the Voronoi vertex of the worst facet) to move.
    """
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.ones(N) * np.pi, np.zeros((N, 3))
    
    faces = hull.simplices
    max_circ_per_point = np.zeros(N)
    move_dirs = np.zeros((N, 3))
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        for idx, angle in zip(face, angles):
            if angle >= max_circ_per_point[idx]:
                max_circ_per_point[idx] = angle
                move_dirs[idx] = o
    
    norms = np.linalg.norm(move_dirs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    move_dirs = move_dirs / norms
    
    return max_circ_per_point, move_dirs

def _facet_based_optimize(points, max_iters=200, deadline=None):
    """
    Lloyd-type optimization: move each point toward the Voronoi vertex of its worst facet.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        max_circ, move_dirs = _compute_facet_circumradii_and_centers(points)
        current_radius = np.max(max_circ)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        step = 0.1 * np.sqrt(current_radius)
        step = np.clip(step, 0.001, 0.1)
        
        new_points = points + step * move_dirs
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        if new_radius < current_radius:
            points = new_points
        else:
            new_points = points + 0.5 * step * move_dirs
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < current_radius:
                points = new_points
            else:
                new_points = points + 0.25 * step * move_dirs
                new_points = new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
                new_radius = _compute_covering_radius(new_points)
                if new_radius < current_radius:
                    points = new_points
                else:
                    break
    
    return best_points, best_radius

def _nelder_mead_polish(points, max_iters=50, deadline=None):
    """
    Polish the configuration using Nelder-Mead on the exact covering radius.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return best_radius
        pts = x.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        return _compute_covering_radius(pts)
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if res.fun < best_radius:
            new_points = res.x.reshape(N, 3)
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
    except Exception:
        pass
    
    return best_points, best_radius

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _facet_force_optimize(points, max_iters=300, lr=0.01, deadline=None):
    """
    Facet-based force balance optimization.
    Points in large-circumradius facets are pulled toward the facet center;
    points in small-circumradius facets are pushed away to equalize.
    """
    P = points.copy()
    best_P = P.copy()
    best_radius = _compute_covering_radius(P)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(P)
        except Exception:
            break
        
        facet_normals = []
        facet_circumradii = []
        facet_points = []
        
        for simplex in hull.simplices:
            verts = P[simplex]
            if len(verts) < 3:
                continue
            v1, v2, v3 = verts[0], verts[1], verts[2]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            if np.dot(normal, v1) < 0:
                normal = -normal
            
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            d = np.dot(n, v1)
            t = d / (n_norm ** 2)
            o = t * n
            o_norm = np.linalg.norm(o)
            if o_norm < 1e-10:
                continue
            o = o / o_norm
            
            dots = np.array([np.dot(v1, o), np.dot(v2, o), np.dot(v3, o)])
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            circumradius = np.max(angles)
            
            facet_normals.append(normal)
            facet_circumradii.append(circumradius)
            facet_points.append(list(simplex))
        
        if not facet_points:
            break
        
        point_facets = [[] for _ in range(N)]
        for idx, pts in enumerate(facet_points):
            for p_idx in pts:
                point_facets[p_idx].append(idx)
        
        forces = np.zeros((N, 3))
        avg_cr = np.mean(facet_circumradii)
        
        for i in range(N):
            if not point_facets[i]:
                continue
            
            p = P[i]
            force = np.zeros(3)
            
            for f_idx in point_facets[i]:
                facet_normal = facet_normals[f_idx]
                circumradius = facet_circumradii[f_idx]
                
                center_dir = facet_normal / (np.linalg.norm(facet_normal) + 1e-12)
                
                deviation = circumradius - avg_cr
                
                force += deviation * center_dir
            
            for j in range(N):
                if i == j:
                    continue
                d = np.dot(P[i], P[j])
                if d > 0.999:
                    force -= 0.1 * P[j]
            
            dot_fp = np.dot(force, p)
            force_tangent = force - dot_fp * p
            
            forces[i] = force_tangent
        
        P_new = P + lr * forces
        P_new = P_new / np.linalg.norm(P_new, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(P_new)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_P = P_new.copy()
            P = P_new
        else:
            lr *= 0.95
        
        if lr < 1e-6:
            break
    
    return best_P, best_radius

def _pairwise_repulsion_init(n, seed=42):
    """Initialize points with pairwise repulsion to spread them out."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add repulsion forces to spread points
    for it in range(100):
        forces = np.zeros((n, 3))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.dot(points[i], points[j])
                if d > 0.9:
                    forces[i] -= (d - 0.9) * points[j]
        forces = forces - np.dot(forces, points, axis=1, keepdims=True) * points
        forces = forces / (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-10)
        points = points + 0.01 * forces
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

def _anneal_optimize(points, max_iters=500, deadline=None, beta_start=5.0, beta_end=50.0):
    """
    Simulated annealing style optimization with increasing beta for soft-max.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Anneal beta
        progress = it / max_iters
        beta = beta_start + (beta_end - beta_start) * progress
        
        soft_max_val, weights, faces, circumradii, voronoi_vertices = _soft_max_circumradii(points, beta)
        
        if len(circumradii) == 0:
            break
        
        current_radius = np.max(circumradii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        total_grad = np.zeros((N, 3))
        
        for k in range(len(circumradii)):
            w = weights[k]
            face = faces[k]
            o = voronoi_vertices[k]
            
            for i, idx in enumerate(face):
                v = points[idx]
                dot_vo = np.dot(v, o)
                dot_vo = np.clip(dot_vo, -1.0, 1.0)
                sin_r = np.sqrt(1 - dot_vo**2)
                if sin_r < 1e-10:
                    grad = np.zeros(3)
                else:
                    grad = -o / sin_r
                
                grad_tangent = grad - np.dot(grad, v) * v
                total_grad[idx] += w * grad_tangent
        
        grad_norms = np.linalg.norm(total_grad, axis=1, keepdims=True)
        grad_norms[grad_norms < 1e-10] = 1.0
        total_grad = total_grad / grad_norms
        
        # Adaptive step size
        step = 0.05 * np.exp(-0.5 * progress)
        
        new_points = points - step * total_grad
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        # Accept with probability based on improvement
        if new_radius < current_radius:
            points = new_points
        else:
            # Accept worse solutions with decreasing probability
            delta = new_radius - current_radius
            temp = 0.01 * np.exp(-progress)
            if np.random.rand() < np.exp(-delta / (temp + 1e-10)):
                points = new_points
    
    return best_points, best_radius

def _swap_optimize(points, max_iters=100, deadline=None):
    """
    Try swapping pairs of points to see if it improves the covering radius.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                # Try swapping points i and j
                new_points = points.copy()
                new_points[i], new_points[j] = points[j].copy(), points[i].copy()
                new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_radius = _compute_covering_radius(new_points)
                if new_radius < best_radius - 1e-8:
                    points = new_points
                    best_radius = new_radius
                    best_points = points.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_radius = np.pi
    
    # Multiple initializations
    inits = [_fibonacci_spiral(N)]
    
    rng = np.random.RandomState(42)
    for seed in [42, 123, 456, 789, 101112]:
        perturbed = inits[0].copy()
        noise = rng.randn(N, 3) * 0.05
        perturbed = perturbed + noise
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        inits.append(perturbed)
    
    # Add a repulsion-based init
    inits.append(_pairwise_repulsion_init(N, seed=42))
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        current_points = init.copy()
        current_radius = _compute_covering_radius(current_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
        
        # Phase 1: Facet-based Lloyd optimization
        if time.time() < deadline - 25.0:
            points, radius = _facet_based_optimize(current_points, max_iters=100, deadline=deadline - 25.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 2: Annealed gradient optimization
        if time.time() < deadline - 20.0:
            points, radius = _anneal_optimize(current_points, max_iters=200, deadline=deadline - 20.0, beta_start=5.0, beta_end=50.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 3: Facet force balance optimization
        if time.time() < deadline - 15.0:
            points, radius = _facet_force_optimize(current_points, max_iters=100, lr=0.01, deadline=deadline - 15.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 4: Nelder-Mead polish
        if time.time() < deadline - 10.0:
            points, radius = _nelder_mead_polish(current_points, max_iters=50, deadline=deadline - 10.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 5: Swap optimization
        if time.time() < deadline - 5.0:
            points, radius = _swap_optimize(current_points, max_iters=50, deadline=deadline - 5.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
    
    # Final polish on the best configuration
    if best_points is not None and time.time() < deadline - 5.0:
        points, radius = _gradient_optimize(best_points, max_iters=100, deadline=deadline - 5.0, beta=20.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
            
        points, radius = _nelder_mead_polish(best_points, max_iters=30, deadline=deadline - 2.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    if best_points is not None:
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    else:
        best_points = _fibonacci_spiral(N)
    
    return best_points
