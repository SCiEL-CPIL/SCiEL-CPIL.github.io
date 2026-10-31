"""Heilbronn triangles: place 11 points in equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
# Equilateral triangle vertices: A=(0,0), B=(1,0), C=(0.5, S3/2)
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, S3 / 2.0])
V = np.array([A, B, C])

# Area of the big triangle
BIG_AREA = 0.5 * S3 / 2.0  # = S3/4

def bary_to_cart(b):
    """Convert barycentric coords (u,v,w) with u+v+w=1 to Cartesian.
    We use (u, v) as free variables, w = 1-u-v."""
    u = b[:, 0]
    v = b[:, 1]
    w = 1.0 - u - v
    x = u * A[0] + v * B[0] + w * C[0]
    y = u * A[1] + v * B[1] + w * C[1]
    return np.column_stack([x, y])

def cart_to_bary(p):
    """Convert Cartesian to barycentric (u,v) where point = u*A + v*B + w*C, w=1-u-v."""
    # p = v*B + w*C  (since A is origin)
    # p_x = v + 0.5*w
    # p_y = (S3/2)*w
    w = p[:, 1] / (S3 / 2.0)
    v = p[:, 0] - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def triangle_area(p1, p2, p3):
    """Area of triangle formed by 3 points."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def min_triangle_area_bary(b):
    """Compute the minimum triangle area among all C(11,3) triangles, using barycentric coords.
    Returns (min_area, min_area_squared) for optimization."""
    p = bary_to_cart(b)
    n = len(p)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = triangle_area(p[i], p[j], p[k])
                if a < min_a:
                    min_a = a
    return min_a

def objective_neg_min(b):
    """Negative of min triangle area (for minimization)."""
    p = bary_to_cart(b)
    n = len(p)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = triangle_area(p[i], p[j], p[k])
                if a < min_a:
                    min_a = a
    return -min_a

def soft_min_area(b, beta):
    """Soft minimum of all triangle areas using log-sum-exp."""
    p = bary_to_cart(b)
    n = len(p)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas.append(triangle_area(p[i], p[j], p[k]))
    areas = np.array(areas)
    # Log-sum-exp for soft min: -logsumexp(-areas/beta) * beta
    max_neg = np.max(-areas / beta)
    logsum = max_neg + np.log(np.sum(np.exp(-areas / beta - max_neg)))
    return -beta * logsum

def objective_soft(b, beta):
    return soft_min_area(b, beta)

def grad_soft(b, beta):
    """Gradient of soft min area w.r.t. barycentric coords."""
    p = bary_to_cart(b)
    n = len(p)
    areas = np.zeros((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas[idx] = triangle_area(p[i], p[j], p[k])
                idx += 1
    
    # weights for soft min
    neg_a = -areas / beta
    max_neg = np.max(neg_a)
    exp_neg = np.exp(neg_a - max_neg)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # weights sum to 1
    
    # Gradient of -beta * log(sum(exp(-a_i/beta))) = sum_i weights_i * a_i
    # So objective = sum_i weights_i * a_i
    # We need gradient of this w.r.t. b (barycentric)
    
    # Each area a_i = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # For points in a triangle, if they're in order, the sign is consistent
    # Let's compute gradient numerically for safety
    
    eps = 1e-7
    grad = np.zeros_like(b)
    for d in range(b.size):
        bp = b.copy()
        bp.flat[d] += eps
        bm = b.copy()
        bm.flat[d] -= eps
        grad.flat[d] = (objective_soft(bp, beta) - objective_soft(bm, beta)) / (2*eps)
    return grad

def hexagonal_init():
    """Initialize points on a hexagonal-like grid inside the triangle."""
    # Use a structured grid in barycentric coordinates
    # For 11 points, use a pattern: corners + edge midpoints + interior
    b = np.zeros((N, 2))
    # Corners
    b[0] = [1.0, 0.0]  # A
    b[1] = [0.0, 1.0]  # B
    b[2] = [0.0, 0.0]  # C
    # Edge midpoints
    b[3] = [0.5, 0.5]  # AB mid
    b[4] = [0.5, 0.0]  # AC mid
    b[5] = [0.0, 0.5]  # BC mid
    # Interior points - use a small hexagonal pattern
    # Center
    b[6] = [1/3, 1/3]
    # Surrounding 4 points
    b[7] = [0.4, 0.4]
    b[8] = [0.6, 0.4]
    b[9] = [0.4, 0.6]
    b[10] = [0.6, 0.6]
    
    # Ensure all points are inside the triangle (barycentric coords >= 0, sum = 1)
    # Clamp to be safe
    b = np.clip(b, 0.0, 1.0)
    # Renormalize: we store (u, v), w = 1-u-v must be >= 0
    # So u+v <= 1
    s = b[:, 0] + b[:, 1]
    mask = s > 1.0
    if np.any(mask):
        b[mask] = b[mask] / s[mask, np.newaxis]
    return b

def random_bary_init(rng):
    """Generate random points in barycentric coords inside the triangle."""
    b = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                b[i] = [u, v]
                break
    return b

def refine_from_init(b0, deadline, rng):
    """Refine an initial configuration using multi-stage L-BFGS-B."""
    best_b = b0.copy()
    best_val = min_triangle_area_bary(b0)
    
    # Multi-stage with increasing beta
    betas = [0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        # Use L-BFGS-B on the soft objective
        def obj(b):
            return objective_soft(b, beta)
        
        # Bounds: u >= 0, v >= 0, u+v <= 1
        # We parameterize as (u, v) with constraints
        # L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # u
            bounds.append((0.0, 1.0))  # v
        
        x0 = best_b.flatten()
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < obj(x0):
                new_b = res.x.reshape(N, 2)
                # Check validity
                if np.all(new_b[:, 0] + new_b[:, 1] <= 1.0 + 1e-6) and np.all(new_b >= -1e-6):
                    new_b = np.clip(new_b, 0, 1)
                    s = new_b[:, 0] + new_b[:, 1]
                    mask = s > 1.0
                    if np.any(mask):
                        new_b[mask] = new_b[mask] / s[mask, np.newaxis]
                    val = min_triangle_area_bary(new_b)
                    if val > best_val:
                        best_val = val
                        best_b = new_b.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
    
    return best_b, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    best_b = None
    best_val = -1.0
    
    # Try multiple initializations
    inits = []
    
    # 1. Hexagonal structured init
    inits.append(hexagonal_init())
    
    # 2. Several random inits
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        r = np.random.RandomState(seed)
        b = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = r.random()
                v = r.random()
                if u + v < 1.0:
                    b[i] = [u, v]
                    break
        inits.append(b)
    
    # 3. Perturbed hexagonal
    for pert in [0.05, 0.1, 0.15]:
        b = hexagonal_init() + rng.normal(0, pert, (N, 2))
        b = np.clip(b, 0, 1)
        s = b[:, 0] + b[:, 1]
        mask = s > 1.0
        if np.any(mask):
            b[mask] = b[mask] / s[mask, np.newaxis]
        inits.append(b)
    
    for b0 in inits:
        if time.time() > deadline - 10:
            break
        b, val = refine_from_init(b0, deadline, rng)
        if val > best_val:
            best_val = val
            best_b = b
    
    # Final local refinement with very small beta
    if best_b is not None and time.time() < deadline - 5:
        for beta in [0.001, 0.0005]:
            if time.time() > deadline - 3:
                break
            def obj(b):
                return objective_soft(b, beta)
            bounds = [(0.0, 1.0)] * (2 * N)
            x0 = best_b.flatten()
            try:
                res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < obj(x0):
                    new_b = res.x.reshape(N, 2)
                    if np.all(new_b[:, 0] + new_b[:, 1] <= 1.0 + 1e-6) and np.all(new_b >= -1e-6):
                        new_b = np.clip(new_b, 0, 1)
                        s = new_b[:, 0] + new_b[:, 1]
                        mask = s > 1.0
                        if np.any(mask):
                            new_b[mask] = new_b[mask] / s[mask, np.newaxis]
                        val = min_triangle_area_bary(new_b)
                        if val > best_val:
                            best_val = val
                            best_b = new_b.copy()
            except Exception:
                pass
    
    if best_b is None:
        # Fallback: return hexagonal init
        best_b = hexagonal_init()
    
    # Convert to Cartesian
    p = bary_to_cart(best_b)
    
    # Ensure points are strictly inside or on the boundary
    # Clip to be safe
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, S3 / 2.0)
    
    return p
