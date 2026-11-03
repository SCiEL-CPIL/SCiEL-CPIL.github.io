<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _facet_circumradius(a, b, c):
    """
    Compute the angular circumradius of the spherical triangle (a, b, c).
    Returns (radius, circumcenter) where circumcenter is the unit vector
    equidistant from a, b, c (the one with smaller angular distance).
    """
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.pi, np.zeros(3)
    n = n / nn
    
    best_r = np.pi
    best_c = n
    for s in (1.0, -1.0):
        p = s * n
        da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
        dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
        if abs(da - db) < 1e-4 and abs(da - dc) < 1e-4:
            if da < best_r:
                best_r = da
                best_c = p
    return best_r, best_c

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) via convex hull facet circumradii.
    Returns (radius, worst_circumcenter) or (pi, None) if invalid.
    """
    pts = _to_sphere(pts)
    
    # Check for coincident points
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None
    
    if len(hull.simplices) == 0:
        return np.pi, None
    
    max_r = 0.0
    worst_c = None
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        r, cc = _facet_circumradius(a, b, c)
        if r > max_r:
            max_r = r
            worst_c = cc
    
    return max_r, worst_c

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n):
    """Generate n random points on the sphere."""
    pts = np.random.randn(n, 3)
    return _to_sphere(pts)

def _lloyd_relax(pts, max_iter=100, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Iteratively moves each point toward the circumcenter of the worst facet
    (the one with the largest circumradius), weighted by how much it exceeds
    the average, to balance the covering radius.
    """
    pts = _to_sphere(pts.copy())
    current_cr, _ = _covering_radius(pts)
    
    for it in range(max_iter):
        if time.time() > _deadline - 5.0:
            break
        
        # Compute all facet circumradii and identify the worst
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, compute a target direction based on the facets it belongs to
        # Weighted by the circumradius of those facets
        targets = np.zeros_like(pts)
        weights = np.zeros(len(pts))
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            r, cc = _facet_circumradius(a, b, c)
            
            # The circumcenter cc is the point on the sphere farthest from the facet
            # We want to move the vertices of this facet toward cc to reduce the radius
            # Weight by r^2 to emphasize the worst facets
            w = r * r
            for idx in simplex:
                targets[idx] += w * cc
                weights[idx] += w
        
        # Normalize targets and blend with current positions
        improved = False
        for idx in range(len(pts)):
            if weights[idx] > 1e-12:
                target = targets[idx] / np.linalg.norm(targets[idx])
                # Move a fraction toward the target
                new_pt = 0.7 * pts[idx] + 0.3 * target
                new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
                pts[idx] = new_pt
                improved = True
        
        if not improved:
            break
        
        new_cr, _ = _covering_radius(pts)
        if new_cr < current_cr - 1e-10:
            current_cr = new_cr
        else:
            # Revert if no improvement
            break
        
        if current_cr < tol:
            break
    
    return pts, current_cr

def _local_refine(pts, max_iter=50, step0=0.05):
    """
    Local greedy refinement: nudge the nearest point to the worst circumcenter
    toward it to reduce the covering radius.
    """
    pts = _to_sphere(pts.copy())
    best_cr, worst_c = _covering_radius(pts)
    step = step0
    
    for it in range(max_iter):
        if time.time() > _deadline - 5.0:
            break
        
        if worst_c is None:
            break
        
        # Find the point nearest to the worst circumcenter
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        
        # Move that point toward the circumcenter
        new_pt = pts[k] + step * worst_c
        new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
        
        cand = pts.copy()
        cand[k] = new_pt
        cr, wc = _covering_radius(cand)
        
        if cr < best_cr - 1e-9:
            pts = cand
            best_cr = cr
            worst_c = wc
            step = min(step * 1.1, 0.2)
        else:
            step *= 0.8
            if step < 1e-5:
                break
    
    return pts, best_cr

def _simulated_annealing(pts, max_iter=200, T0=0.3, T_end=0.01):
    """
    Simulated annealing for covering radius minimization.
    """
    current_pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if time.time() > _deadline - 5.0:
            break
        T *= cooling
        
        # Perturb a random point
        idx = np.random.randint(len(current_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.1 * direction
        new_pts[idx] = _to_sphere(new_pts[idx].reshape(1, 3))[0]
        
        new_val, _ = _covering_radius(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
    
    return best_pts, best_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses spherical Lloyd's relaxation, local refinement, and simulated annealing.
    """
    global _deadline
    _deadline = time.time() + 110.0  # safety margin vs 120 s kill
    
    rng = np.random.seed(12345)
    
    best_pts = None
    best_val = np.pi
    
    # Start with Fibonacci spiral
    starts = [_fibonacci_spiral(N)]
    
    # Add a few random starts
    for _ in range(4):
        starts.append(_random_points(N))
    
    for st in starts:
        if time.time() > _deadline - 20.0:
            break
        
        # Apply Lloyd's relaxation
        lloyd_pts, lloyd_val = _lloyd_relax(st, max_iter=80)
        
        # Apply local refinement
        ref_pts, ref_val = _local_refine(lloyd_pts, max_iter=40, step0=0.04)
        
        # Apply simulated annealing for final polish
        sa_pts, sa_val = _simulated_annealing(ref_pts, max_iter=150, T0=0.2, T_end=0.01)
        
        if sa_val < best_val:
            best_val = sa_val
            best_pts = sa_pts.copy()
    
    # Final local refinement on the best result
    if best_pts is not None and time.time() < _deadline - 5.0:
        final_pts, final_val = _local_refine(best_pts, max_iter=30, step0=0.02)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _facet_circumradius(a, b, c):
    """
    Compute the angular circumradius of the spherical triangle (a, b, c).
    Returns (radius, circumcenter) where circumcenter is the unit vector
    equidistant from a, b, c (the one with smaller angular distance).
    """
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.pi, np.zeros(3)
    n = n / nn
    
    best_r = np.pi
    best_c = n
    for s in (1.0, -1.0):
        p = s * n
        da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
        dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
        if abs(da - db) < 1e-4 and abs(da - dc) < 1e-4:
            if da < best_r:
                best_r = da
                best_c = p
    return best_r, best_c

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) via convex hull facet circumradii.
    Returns (radius, worst_circumcenter) or (pi, None) if invalid.
    """
    pts = _to_sphere(pts)
    
    # Check for coincident points
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None
    
    if len(hull.simplices) == 0:
        return np.pi, None
    
    max_r = 0.0
    worst_c = None
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        r, cc = _facet_circumradius(a, b, c)
        if r > max_r:
            max_r = r
            worst_c = cc
    
    return max_r, worst_c

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n):
    """Generate n random points on the sphere."""
    pts = np.random.randn(n, 3)
    return _to_sphere(pts)

def _reptile_search_collapse(n, iters=200, step=0.1):
    """
    Reptile search method (RSM): particles move toward the best position,
    then collapse toward the centroid. Good for global initialization.
    """
    pts = _random_points(n)
    best_pts = pts.copy()
    best_val, _ = _covering_radius(pts)
    
    for it in range(iters):
        if time.time() > _deadline - 10.0:
            break
        
        # Sort by current covering radius contribution (approximate by distance to centroid)
        centroid = np.mean(pts, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        
        # Move each point toward centroid with step
        new_pts = pts.copy()
        for i in range(n):
            new_pts[i] = pts[i] + step * (centroid - pts[i])
            new_pts[i] = _to_sphere(new_pts[i].reshape(1, 3))[0]
        
        # Collapse: move toward mean of new positions
        mean_pos = np.mean(new_pts, axis=0)
        mean_pos = mean_pos / (np.linalg.norm(mean_pos) + 1e-12)
        collapsed = new_pts.copy()
        for i in range(n):
            collapsed[i] = new_pts[i] + 0.5 * step * (mean_pos - new_pts[i])
            collapsed[i] = _to_sphere(collapsed[i].reshape(1, 3))[0]
        
        val, _ = _covering_radius(collapsed)
        if val < best_val:
            best_val = val
            best_pts = collapsed.copy()
        pts = collapsed
    
    return best_pts, best_val

def _smooth_surrogate(pts, beta=100.0):
    """
    Compute a smooth surrogate objective: soft-max of facet circumradii.
    This is differentiable and suitable for gradient-based optimization.
    Returns (surrogate_value, gradient_wrt_pts).
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros_like(pts)
    
    if len(hull.simplices) == 0:
        return np.pi, np.zeros_like(pts)
    
    # Collect all facet circumradii
    radii = []
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        r, _ = _facet_circumradius(a, b, c)
        radii.append(r)
    
    radii = np.array(radii)
    
    # Soft-max with temperature 1/beta
    max_r = np.max(radii)
    exp_vals = np.exp(beta * (radii - max_r))
    exp_sum = np.sum(exp_vals)
    if exp_sum < 1e-15:
        return max_r, np.zeros_like(pts)
    
    weights = exp_vals / exp_sum
    surrogate = max_r - np.log(exp_sum) / beta
    
    # Compute gradient: for each point, sum over facets containing it
    grad = np.zeros_like(pts)
    for i, simplex in enumerate(hull.simplices):
        w = weights[i]
        if w < 1e-10:
            continue
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        r, cc = _facet_circumradius(a, b, c)
        
        # Gradient of r w.r.t. each vertex: move vertex toward circumcenter reduces r
        # d(r)/d(vertex) ≈ -cc (in tangent space)
        for idx in simplex:
            # Project gradient to tangent space at pts[idx]
            tangent_grad = cc - np.dot(cc, pts[idx]) * pts[idx]
            grad[idx] += w * tangent_grad
    
    return surrogate, grad

def _gradient_optimize(pts, max_iter=300, lr=0.01, beta=100.0):
    """
    Gradient descent on the smooth surrogate objective.
    Uses adaptive learning rate and momentum.
    """
    pts = _to_sphere(pts.copy())
    current_val, _ = _covering_radius(pts)
    best_pts = pts.copy()
    best_val = current_val
    
    momentum = np.zeros_like(pts)
    beta_m = 0.9
    
    for it in range(max_iter):
        if time.time() > _deadline - 10.0:
            break
        
        surrogate_val, grad = _smooth_surrogate(pts, beta=beta)
        
        # Adaptive learning rate
        lr_adapt = lr * (1.0 / (1.0 + it * 0.01))
        
        # Update momentum
        momentum = beta_m * momentum - lr_adapt * grad
        
        # Apply update in tangent space
        new_pts = pts.copy()
        for i in range(len(pts)):
            # Project momentum to tangent space
            m_tangent = momentum[i] - np.dot(pmomentum[i], pts[i]) * pts[i]
            new_pts[i] = pts[i] + m_tangent
            new_pts[i] = _to_sphere(new_pts[i].reshape(1, 3))[0]
        
        new_cr, _ = _covering_radius(new_pts)
        if new_cr < best_val:
            best_val = new_cr
            best_pts = new_pts.copy()
        
        # Accept if surrogate improved
        if surrogate_val < current_val - 1e-12:
            current_val = surrogate_val
            pts = new_pts
        else:
            # Revert and reduce learning rate
            lr *= 0.95
            if lr < 1e-6:
                break
    
    return best_pts, best_val

def _local_refine(pts, max_iter=50, step0=0.05):
    """
    Local greedy refinement: nudge the nearest point to the worst circumcenter
    toward it to reduce the covering radius.
    """
    pts = _to_sphere(pts.copy())
    best_cr, worst_c = _covering_radius(pts)
    step = step0
    
    for it in range(max_iter):
        if time.time() > _deadline - 5.0:
            break
        
        if worst_c is None:
            break
        
        # Find the point nearest to the worst circumcenter
        d = np.sum((pts - worst_c[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        
        # Move that point toward the circumcenter
        new_pt = pts[k] + step * worst_c
        new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
        
        cand = pts.copy()
        cand[k] = new_pt
        cr, wc = _covering_radius(cand)
        
        if cr < best_cr - 1e-9:
            pts = cand
            best_cr = cr
            worst_c = wc
            step = min(step * 1.1, 0.2)
        else:
            step *= 0.8
            if step < 1e-5:
                break
    
    return pts, best_cr

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses reptile search for initialization, gradient-based optimization on
    a smooth surrogate, and local greedy refinement.
    """
    global _deadline
    _deadline = time.time() + 110.0  # safety margin vs 120 s kill
    
    np.random.seed(12345)
    
    best_pts = None
    best_val = np.pi
    
    # Multiple diverse initializations
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Jittered Fibonacci
    fib = _fibonacci_spiral(N).copy()
    for i in range(N):
        fib[i] += 0.05 * np.random.randn(3)
        fib[i] = _to_sphere(fib[i].reshape(1, 3))[0]
    starts.append(fib)
    
    # 3. Reptile search from random start
    reptile_pts, reptile_val = _reptile_search_collapse(N, iters=100, step=0.1)
    starts.append(reptile_pts)
    
    # 4. Another random start with reptile
    reptile_pts2, reptile_val2 = _reptile_search_collapse(N, iters=100, step=0.08)
    starts.append(reptile_pts2)
    
    for st in starts:
        if time.time() > _deadline - 25.0:
            break
        
        # Gradient-based optimization on smooth surrogate
        grad_pts, grad_val = _gradient_optimize(st, max_iter=200, lr=0.005, beta=150.0)
        
        # Local greedy refinement for final polish
        ref_pts, ref_val = _local_refine(grad_pts, max_iter=40, step0=0.03)
        
        if ref_val < best_val:
            best_val = ref_val
            best_pts = ref_pts.copy()
    
    # Final local refinement on the best result
    if best_pts is not None and time.time() < _deadline - 5.0:
        final_pts, final_val = _local_refine(best_pts, max_iter=30, step0=0.02)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I notice there's a typo in the gradient optimization function. Let me fix it:

