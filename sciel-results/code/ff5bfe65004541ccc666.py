"""Rewritten program: L-BFGS-B on log-sum-exp surrogate for sphere covering."""
import numpy as np
from scipy.optimize import minimize
import time

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize rows of pts to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """
    Compute the covering radius in degrees for a set of unit vectors.
    Uses the convex hull approach: the covering radius is the maximum circumradius
    of the facets (great-circle distances) of the convex hull of the points.
    For each facet (triangle), compute the circumradius of the spherical triangle.
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
        max_circ = 0.0
        for tri in facets:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            # Compute the circumradius of the spherical triangle
            # The circumradius R satisfies: cos(R) = (cos(ab)*cos(bc)*cos(ca) + 1) / (cos(ab)+cos(bc)+cos(ca) - 1) ... 
            # Actually, for a spherical triangle with sides a,b,c (angular), the circumradius R is:
            # tan(R) = (4*sqrt(s*(s-a)*(s-b)*(s-c))) / (sin(a)*sin(b)*sin(c))  where s=(a+b+c)/2
            # But simpler: find the point on the sphere equidistant from a,b,c (the circumcenter).
            # The circumcenter of three points on the sphere is along the cross product direction.
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            # The circumcenter is at +/- n. Choose the one that is on the same side as the facet normal.
            # The facet normal for convex hull points outward.
            fa = a + b + c
            fa_norm = np.linalg.norm(fa)
            if fa_norm < 1e-12:
                continue
            fa = fa / fa_norm
            if np.dot(n, fa) < 0:
                n = -n
            # The angular distance from n to a is the circumradius
            cos_r = np.clip(np.dot(n, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_circ:
                max_circ = r
        return np.degrees(max_circ)
    except Exception:
        return 180.0


def _lse_objective_and_grad(flat, beta):
    """
    Log-sum-exp surrogate for the covering problem.
    We want to minimize the max over pairs of -dot(p_i, p_j) (i.e., maximize min dot product).
    Actually, we want to minimize the covering radius. A good surrogate is:
    f = (1/beta) * log(sum_{i,j} exp(-beta * dot(p_i, p_j)))
    This is a smooth approximation to -min_{i,j} dot(p_i, p_j) = max_{i,j} (-dot(p_i,p_j)).
    Minimizing this encourages all pairwise dot products to be large (points spread out).
    
    However, this maximizes minimum separation, which is related but not identical to minimizing
    covering radius. For well-distributed points they align.
    
    flat: 390-dim vector
    beta: temperature parameter
    """
    pts = flat.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute all pairwise dot products
    # D[i,j] = dot(p_i, p_j)
    D = pts @ pts.T
    # We want to maximize the minimum off-diagonal dot product
    # Minimize: (1/beta) * log(sum_{i != j} exp(-beta * D[i,j]))
    
    # For efficiency, we can use the fact that we only need the sum of exp(-beta*D)
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    vals = -beta * D[mask]
    
    # Log-sum-exp for numerical stability
    max_val = np.max(vals)
    if max_val > 100:
        # Overflow protection
        logsumexp = max_val + np.log(np.sum(np.exp(vals - max_val)))
    else:
        logsumexp = np.log(np.sum(np.exp(vals)))
    
    f = logsumexp / beta
    
    # Gradient with respect to flat
    # d/dp_i [ log(sum_{j!=i} exp(-beta * dot(p_i, p_j))) ]
    # = sum_{j!=i} (-beta * p_j * exp(-beta*dot(p_i,p_j))) / sum_{k!=i} exp(-beta*dot(p_i,p_k))
    
    grad_pts = np.zeros_like(pts)
    for i in range(N):
        # For point i, the contribution from all other points
        dots_i = -beta * D[i]
        dots_i[i] = -np.inf  # mask self
        max_v = np.max(dots_i)
        exps = np.exp(dots_i - max_v)
        sum_exps = np.sum(exps)
        if sum_exps < 1e-300:
            continue
        weights = exps / sum_exps
        # grad contribution: (1/beta) * sum_j weights_j * (-beta * p_j) = -sum_j weights_j * p_j
        grad_pts[i] = -np.sum(pts * weights[:, None], axis=0)
    
    grad_flat = grad_pts.reshape(-1)
    
    return f, grad_flat


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Multiple restarts with different initial configurations
    best_pts = None
    best_score = np.inf
    
    # Generate several initial configurations
    def fibonacci_spiral(n, offset=0.0):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def random_init(n, seed):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, 3)
        return _normalize(pts)
    
    def icosahedral_seed(n):
        """Start from icosahedron and subdivide."""
        # 12 vertices of icosahedron
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        verts = _normalize(verts)
        # Add more points by perturbing and adding random ones
        rng = np.random.RandomState(42)
        extra = rng.randn(n - 12, 3)
        extra = _normalize(extra)
        return _normalize(np.vstack([verts, extra]))
    
    inits = [
        fibonacci_spiral(N, 0.0),
        fibonacci_spiral(N, 1.0),
        fibonacci_spiral(N, 2.5),
        random_init(N, 0),
        random_init(N, 1),
        icosahedral_seed(N),
    ]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        flat = init_pts.reshape(-1).copy()
        
        # Anneal beta from low to high
        betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            # L-BFGS-B optimization
            try:
                res = minimize(
                    lambda x: _lse_objective_and_grad(x, beta)[0],
                    flat,
                    jac=lambda x: _lse_objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 50}
                )
                flat = res.x
            except Exception:
                break
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        # Evaluate actual covering radius
        final_pts = _normalize(flat.reshape(N, 3))
        score = _covering_radius_deg(final_pts)
        
        if score < best_score:
            best_score = score
            best_pts = final_pts.copy()
        
        if time.time() > deadline - 10:
            break
    
    if best_pts is None:
        # Fallback
        best_pts = fibonacci_spiral(N, 0.0)
    
    return best_pts

# EVOLVE-BLOCK-END
