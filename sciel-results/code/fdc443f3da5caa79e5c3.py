"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# EVOLVE-BLOCK-START

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _fast_facets(points):
    """
    Compute convex hull and return facet data.
    Returns a list of tuples: (radius, circumcenter, simplex_indices)
    where radius is the angular covering radius of that facet,
    circumcenter is the outward unit normal to the facet plane,
    and simplex_indices are the 3 vertex indices.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        out.append((r, v, simplex))
    return out

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    points = _normalize(points)
    facets = _fast_facets(points)
    if not facets:
        return 180.0
    max_r = max(f[0] for f in facets)
    return max_r * 180.0 / np.pi

def _repulsion_energy(points, s=20.0):
    """
    Compute Riesz s-energy: sum over pairs of (1 - dot(p_i, p_j))^(-s/2)
    This pushes points apart. Higher s means stronger repulsion at close range.
    Vectorized implementation.
    """
    points = _normalize(points)
    n = len(points)
    # Compute pairwise dot products
    dots = np.dot(points, points.T)
    # 1 - dot is proportional to squared chord distance
    one_minus = np.maximum(1.0 - dots, 1e-15)
    # Set diagonal to a large value so it doesn't contribute
    np.fill_diagonal(one_minus, 1.0)
    # Energy = sum of (1-dot)^(-s/2) for i < j
    energy = np.sum(one_minus ** (-s/2.0)) / 2.0  # divide by 2 since we count both i,j and j,i
    return energy

def _repulsion_energy_grad(points, s=20.0):
    """
    Compute gradient of Riesz s-energy w.r.t. each point.
    Vectorized implementation.
    """
    points = _normalize(points)
    n = len(points)
    # Compute pairwise dot products
    dots = np.dot(points, points.T)
    one_minus = np.maximum(1.0 - dots, 1e-15)
    np.fill_diagonal(one_minus, 1.0)
    
    # coeff[i,j] = (s/2) * (1 - dot(i,j))^(-s/2 - 1)
    coeff = (s/2.0) * one_minus ** (-s/2.0 - 1.0)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * (p_i - p_j)
    # = sum_j coeff[i,j] * p_i - sum_j coeff[i,j] * p_j
    # = p_i * sum_j coeff[i,j] - (coeff @ points)[i]
    coeff_row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = coeff_row_sum * points - coeff.dot(points)
    
    # Project to tangent space
    grad = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    return grad

def _repulsion_refine(points, max_time=20.0, start_time=None, s=20.0):
    """
    Refine points using L-BFGS-B to minimize Riesz s-energy.
    This maximizes mutual separation, which helps with covering.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    
    if start_time is None:
        start_time = time.time()
    
    def objective_and_grad(x_flat):
        if time.time() - start_time > max_time:
            return 1e6, np.zeros_like(x_flat)
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        energy = _repulsion_energy(x, s=s)
        grad = _repulsion_energy_grad(x, s=s)
        return energy, grad.flatten()
    
    x0 = points.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
    )
    
    x_opt = result.x.reshape(-1, 3)
    x_opt = _normalize(x_opt)
    cr_opt = _covering_radius(x_opt)
    
    if cr_opt < best_cr:
        best = x_opt.copy()
        best_cr = cr_opt
    
    return best, best_cr

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute the smooth covering radius (log-sum-exp over facet radii) and its
    gradient with respect to the input points.
    Uses epigraph formulation with auxiliary variable t for better conditioning.
    """
    points = _normalize(points)
    facets = _fast_facets(points)
    if not facets:
        return 0.0, np.zeros_like(points)

    radii = np.array([f[0] for f in facets])
    centers = np.array([f[1] for f in facets])
    simplices = [f[2] for f in facets]

    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    
    # Log-sum-exp for numerical stability
    val = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta

    w = np.exp(beta * (radii - m))
    w /= w.sum()

    grad = np.zeros_like(points)
    for i in range(len(radii)):
        if w[i] < 1e-10:
            continue
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            # Gradient of arccos(dot(center, p)) w.r.t p
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan

    return val, grad

def _lbfgs_refine(points, max_time=30.0, start_time=None, beta=50.0):
    """
    Refine points using L-BFGS-B on a smooth log-sum-exp surrogate.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        if time.time() - start_time > max_time:
            return 1e6, np.zeros_like(x_flat)
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        # Project gradient to tangent space
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 2000}
    )

    x_opt = result.x.reshape(-1, 3)
    x_opt = _normalize(x_opt)
    cr_opt = _covering_radius(x_opt)

    if cr_opt < best_cr:
        best = x_opt.copy()
        best_cr = cr_opt

    return best, best_cr

def _spherical_lloyd(points, max_iters=200, time_limit=None, start_time=None):
    """
    Lloyd-style: each point moves to the average circumcenter of adjacent facets.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        facets = _fast_facets(points)
        if not facets:
            break
        
        centers_per_point = [[] for _ in range(len(points))]
        for r, center, simplex in facets:
            for idx in simplex:
                centers_per_point[idx].append(center)
        
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            if not centers_per_point[i]:
                new_points[i] = points[i]
            else:
                c = np.mean(centers_per_point[i], axis=0)
                cn = np.linalg.norm(c)
                if cn < 1e-12:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / cn
        
        alpha = 0.5
        blended = (1 - alpha) * points + alpha * new_points
        blended = _normalize(blended)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended = _normalize(blended)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break

    return best, best_cr

def _fibonacci_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _jittered_fibonacci_points(jitter=0.1, seed=42):
    rng = np.random.RandomState(seed)
    pts = _fibonacci_points()
    noise = rng.normal(0, jitter, pts.shape)
    return _normalize(pts + noise)

def _random_points(seed=0):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.normal(0, 1, (N, 3))
    return _normalize(pts)

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0  # Safety margin before 150s kill
    start_time = time.time()
    
    # Stage 1: Multi-start with repulsion energy to get good initial configs
    best_points = None
    best_cr = 180.0
    
    # Try several different initializations
    inits = [
        _fibonacci_points(),
        _jittered_fibonacci_points(0.1, 42),
        _jittered_fibonacci_points(0.2, 123),
        _jittered_fibonacci_points(0.05, 999),
        _random_points(1),
        _random_points(2),
        _random_points(7),
        _random_points(13),
    ]
    
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > total_time_limit - 45:
            break
        remaining = total_time_limit - elapsed
        if remaining < 8:
            break
            
        # First, use repulsion energy to spread points
        pts_rep, cr_rep = _repulsion_refine(init, max_time=min(4, remaining - 4), start_time=start_time, s=20.0)
        
        # Then Lloyd's
        pts_lloyd, cr_lloyd = _spherical_lloyd(pts_rep, max_iters=40, time_limit=min(6, remaining - 4), start_time=start_time)
        
        if cr_lloyd < best_cr:
            best_cr = cr_lloyd
            best_points = pts_lloyd.copy()

    if best_points is None:
        best_points = _fibonacci_points()
        best_cr = _covering_radius(best_points)

    # Stage 2: Multi-beta L-BFGS refinement
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 35:
        # Beta cascade: start low, increase
        for beta in [10.0, 30.0, 100.0, 300.0, 1000.0]:
            elapsed = time.time() - start_time
            if elapsed > total_time_limit - 12:
                break
            remaining = total_time_limit - elapsed
            if remaining < 6:
                break
            pts_lbfgs, cr_lbfgs = _lbfgs_refine(best_points, max_time=min(12, remaining - 2), start_time=start_time, beta=beta)
            if cr_lbfgs < best_cr:
                best_cr = cr_lbfgs
                best_points = pts_lbfgs.copy()

    # Stage 3: Final Lloyd polish
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 15:
        remaining = total_time_limit - elapsed
        pts_final, cr_final = _spherical_lloyd(best_points, max_iters=150, time_limit=min(15, remaining - 5), start_time=start_time)
        if cr_final < best_cr:
            best_cr = cr_final
            best_points = pts_final.copy()

    # Stage 4: Final high-precision L-BFGS polish
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 5:
        remaining = total_time_limit - elapsed
        if remaining > 5:
            pts_final2, cr_final2 = _lbfgs_refine(best_points, max_time=min(8, remaining - 2), start_time=start_time, beta=1000.0)
            if cr_final2 < best_cr:
                best_cr = cr_final2
                best_points = pts_final2.copy()

    return _normalize(best_points)

# EVOLVE-BLOCK-END
