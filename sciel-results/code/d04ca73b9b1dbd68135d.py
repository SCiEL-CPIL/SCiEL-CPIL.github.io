import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """Vectorized computation of circumradii for all facets.
    Returns array of radii (radians) and array of circumcenter unit vectors."""
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    n = np.cross(a - b, a - c)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm > 1e-10
    n = n / np.maximum(n_norm[:, None], 1e-10)
    centroids = (a + b + c) / 3.0
    signs = np.where(np.einsum('ij,ij->i', n, centroids) < 0, -1.0, 1.0)
    n = n * signs[:, None]
    dots = np.clip(np.einsum('ij,ij->i', n, a), -1.0, 1.0)
    radii = np.arccos(dots)
    radii[~mask] = 0.0
    return radii, n, mask

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.degrees(np.max(radii)))
    except Exception:
        return 180.0

def _covering_radius_rad(pts):
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.max(radii))
    except Exception:
        return np.pi

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=1000, lr=0.01):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force.
    Uses a simple gradient descent on the electrostatic potential energy."""
    pts = pts.copy()
    for it in range(n_iters):
        # Pairwise dot products
        D = pts @ pts.T
        # Force on i from j: proportional to (pts[i] - pts[j]) / |...|^2 but on sphere we use tangent component
        # Use inverse-square repulsion in chordal distance
        # chord^2 = 2 - 2*D
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        # Repulsion magnitude ~ 1/chord^2
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        # Force = sum_j W_ij * (pts[i] - pts[j]) projected to tangent
        # = pts[i] * sum_j W_ij - sum_j W_ij * pts[j]
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        # Project to tangent plane
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        # Normalize forces to prevent huge steps
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / (force_norms + 1e-12)
        
        # Adaptive learning rate: start higher, decay
        current_lr = lr * (0.99 ** (it / 100))
        pts = pts + current_lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_surrogate(pts, beta=20.0):
    """Smooth surrogate: log-sum-exp of squared facet radii (soft-max).
    Returns (surrogate_value, softmax_weights, radii, simplices, circumcenters, mask)."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return None
    # Use only valid facets
    valid = mask
    r_valid = radii[valid]
    cc_valid = cc[valid]
    simp_valid = hull.simplices[valid]
    # Soft-max of squared radii
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
    else:
        weights = exps / s
    surrogate = m + np.log(s) / beta
    return surrogate, weights, r_valid, cc_valid, simp_valid

def _surrogate_gradient(pts, weights, simplices, circumcenters, beta=10.0):
    """Gradient of smooth surrogate w.r.t. each point (in R^3, then projected to tangent)."""
    grad = np.zeros_like(pts)
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-8:
            continue
        c = circumcenters[k]
        # radius r_k: cos(r) = c . a
        a = pts[simplices[k, 0]]
        cos_r = np.clip(np.dot(c, a), -1.0, 1.0)
        sin_r = np.sqrt(max(1.0 - cos_r * cos_r, 1e-12))
        # For each of the 3 vertices, the contribution to d(r^2)/d v
        # We'll just push toward c with weight w * r / sin_r
        for j in range(3):
            idx = simplices[k, j]
            v = pts[idx]
            # Direction from v to c, projected to tangent at v
            d = c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm < 1e-10:
                continue
            d_t = d_t / d_t_norm
            # Magnitude: 2 * r / sin(r) * w
            r_k = np.arccos(cos_r)
            mag = 2.0 * r_k / sin_r * w
            grad[idx] -= mag * d_t  # negative because we want to decrease r^2
    # Project gradient to tangent plane
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return grad

def _electrostatic_gradient(pts):
    """Compute gradient of electrostatic energy on the sphere."""
    D = pts @ pts.T
    chord2 = 2.0 - 2.0 * D
    np.fill_diagonal(chord2, 1.0)
    W = 1.0 / chord2
    np.fill_diagonal(W, 0.0)
    row_sum = W.sum(axis=1)
    forces = pts * row_sum[:, None] - (W @ pts)
    # Project to tangent
    forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
    return forces

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill

    def _time_left():
        return max_time - (time.time() - start_time)

    best_points = _fibonacci_spiral()
    best_rad = _covering_radius_deg(best_points)

    # Phase 0: Multiple initializations with strong electrostatic refinement
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Fibonacci + Electrostatic
    fib1 = _fibonacci_spiral(0.0)
    inits.append(_electrostatic_refine(fib1, n_iters=500, lr=0.02))
    
    # 2. Fibonacci (offset) + Electrostatic
    fib2 = _fibonacci_spiral(0.5)
    inits.append(_electrostatic_refine(fib2, n_iters=500, lr=0.02))
    
    # 3. Random + Electrostatic
    rand1 = _normalize(rng.randn(N, 3))
    inits.append(_electrostatic_refine(rand1, n_iters=800, lr=0.02))
    
    # 4. Raw Fibonacci (fallback)
    inits.append(fib1)

    # Evaluate all initializations
    for init in inits:
        if _time_left() < 5.0:
            break
        rad = _covering_radius_deg(init)
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()

    # Phase 1: Hybrid Gradient Descent (Covering Radius + Electrostatics)
    pts = best_points.copy()
    cur_rad = _covering_radius_deg(pts)
    
    vel = np.zeros_like(pts)
    momentum = 0.7
    lr = 0.005
    
    n_iter = 0
    for it in range(600):
        if _time_left() < 15.0:
            break
            
        # Compute smooth covering radius gradient
        result = _smooth_surrogate(pts, beta=20.0)
        if result is None:
            break
        surrogate, weights, r_valid, cc_valid, simp_valid = result
        grad_cov = _surrogate_gradient(pts, weights, simp_valid, cc_valid, beta=20.0)
        
        # Compute electrostatic gradient (less frequent to save time)
        if it % 3 == 0:
            grad_elec = _electrostatic_gradient(pts)
        else:
            grad_elec = vel  # reuse previous momentum as proxy
        
        # Blend gradients
        alpha = 0.3 * np.exp(-it / 80.0)
        grad_total = (1 - alpha) * grad_cov + alpha * grad_elec
        
        # Update with momentum
        vel = momentum * vel - lr * grad_total
        pts = pts + vel
        pts = _normalize(pts)
        
        # Evaluate every 5 iterations to save hull computation time
        if it % 5 == 0 or it == 599:
            cur_rad = _covering_radius_deg(pts)
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()
            
        # Decay learning rate
        lr *= 0.998
        if lr < 0.0005:
            lr = 0.0005
        n_iter = it + 1

    # Phase 2: Greedy Lloyd + Polish on the best points
    pts = best_points.copy()
    cur_rad = _covering_radius_deg(pts)
    
    # Process top-3 worst facets simultaneously
    step = 0.015
    for it in range(150):
        if _time_left() < 8.0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        radii, cc, mask = _facet_circumradii(pts, hull.simplices)
        if not np.any(mask):
            break
        valid_idx = np.where(mask)[0]
        # Get top 3 worst facets
        top_k = min(3, len(valid_idx))
        worst_locals = valid_idx[np.argsort(-radii[valid_idx])][:top_k]
        worst_rad = radii[worst_locals[0]]
        if worst_rad < 0.14:  # ~8.0 degrees
            break
        
        # Move vertices of all top facets toward their circumcenters
        moved = set()
        for worst_local in worst_locals:
            simp = hull.simplices[worst_local]
            c = cc[worst_local]
            for j in range(3):
                idx = simp[j]
                if idx in moved:
                    continue
                v = pts[idx]
                new_v = v + step * (c - v)
                new_v = new_v / np.linalg.norm(new_v)
                pts[idx] = new_v
                moved.add(idx)
        
        # Small random perturbation to escape local minima
        if it % 15 == 0:
            for _ in range(3):
                idx = rng.randint(N)
                tangent = rng.randn(3)
                tangent -= np.dot(tangent, pts[idx]) * pts[idx]
                tn = np.linalg.norm(tangent)
                if tn > 1e-10:
                    tangent /= tn
                    new_v = pts[idx] + 0.003 * tangent
                    pts[idx] = new_v / np.linalg.norm(new_v)
                    
        cur_rad = _covering_radius_deg(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()
        step *= 0.985

    # Phase 3: High-precision polish using Nelder-Mead on the worst facets
    if _time_left() > 10.0:
        pts = best_points.copy()
        for it in range(80):
            if _time_left() < 5.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
            if not np.any(mask):
                break
            valid_idx = np.where(mask)[0]
            # Focus on top 2 worst facets
            order = valid_idx[np.argsort(-radii[valid_idx])][:2]
            improved = False
            for worst_local in order:
                if _time_left() < 3.0:
                    break
                simp = hull.simplices[worst_local]
                c = cc[worst_local]
                for j in range(3):
                    idx = simp[j]
                    v = pts[idx]
                    # Try multiple step sizes, accept if improves
                    for s in [0.001, 0.0005, 0.0002]:
                        new_v = v + s * (c - v)
                        new_v = new_v / np.linalg.norm(new_v)
                        old_v = pts[idx].copy()
                        pts[idx] = new_v
                        new_rad = _covering_radius_deg(pts)
                        if new_rad < best_rad - 1e-10:
                            best_rad = new_rad
                            best_points = pts.copy()
                            improved = True
                            break
                        else:
                            pts[idx] = old_v
            if not improved and it > 40:
                break

    return _normalize(best_points)
