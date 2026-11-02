"""Heilbronn-11: Differentiable bottom-k soft-min with Adam optimization."""
import numpy as np
import time
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 130.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _bary_to_cart(uv_flat):
    """
    Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2.
    Since V0=(0,0), P = v*(1,0) + w*(0.5, S3H) = (v + 0.5w, w*S3H).
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """
    Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v.
    Points must be inside the triangle.
    """
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        w = y / S3H
        v = x - 0.5 * w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv


def _vectorized_min_area_cart(pts):
    """Vectorized computation of min triangle area from Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _bottom_k_soft_min(areas, k=10, beta=100.0):
    """
    Differentiable approximation to the minimum using a bottom-k soft-min.
    This focuses optimization on the k smallest areas, providing a smoother
    gradient signal than the full log-sum-exp soft-min.
    """
    # Get the k smallest areas
    k = min(k, len(areas))
    # Use partition to find k smallest (O(n) on average)
    indices = np.argpartition(areas, k)[:k]
    small_areas = areas[indices]
    
    # Soft-min over just these k elements
    max_a = np.max(small_areas)
    shifted = -beta * (small_areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_cartesian(pts):
    """Project points into the triangle if they are slightly outside."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _project_to_simplex(uv_flat):
    """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(-1, 2).copy()
    for i in range(len(uv)):
        u, v = uv[i]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        if u + v > 1.0:
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        uv[i] = [u, v]
    return uv.flatten()


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _vectorized_min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Initial configuration: High symmetry guess
    init_cart = np.array([
        [0.0, 0.0],          # Corner 0
        [1.0, 0.0],          # Corner 1
        [0.5, S3H],          # Corner 2
        [0.5, 0.0],          # Mid bottom
        [0.75, S3H / 2],     # Mid right
        [0.25, S3H / 2],     # Mid left
        [1/3, S3H/3],        # Centroid
        [0.333, 0.0],        # Bottom edge
        [0.667, 0.0],        # Bottom edge
        [0.5, S3H/3],        # Median
        [0.2, 0.1],          # Interior
    ], dtype=float)
    
    init_cart = _project_cartesian(init_cart)
    _update_best(init_cart)
    
    init_uv = _cart_to_bary(init_cart)
    
    def _adam_step(uv_curr, beta, k, lr, steps=200):
        """
        Adam optimizer step for maximizing the bottom-k soft-min area.
        Returns updated uv.
        """
        # Adam hyperparameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m = np.zeros_like(uv_curr)
        v = np.zeros_like(uv_curr)
        t = 0
        
        for step in range(steps):
            if not _elapsed_ok(5):
                break
            
            uv = uv_curr.reshape(-1, 2)
            u = uv[:, 0]
            v_bary = uv[:, 1]
            w = 1.0 - u - v_bary
            
            pts = _bary_to_cart(uv_curr)
            
            # Compute areas
            pi = pts[_T_I]
            pj = pts[_T_J]
            pk = pts[_T_K]
            dx1 = pj[:, 0] - pi[:, 0]
            dy1 = pj[:, 1] - pi[:, 1]
            dx2 = pk[:, 0] - pi[:, 0]
            dy2 = pk[:, 1] - pi[:, 1]
            cross = dx1 * dy2 - dy1 * dx2
            areas = np.abs(cross) * 0.5
            
            # Compute bottom-k soft-min
            soft_min_val = _bottom_k_soft_min(areas, k, beta)
            
            # Compute gradient of soft_min w.r.t. areas
            k_actual = min(k, len(areas))
            indices = np.argpartition(areas, k_actual)[:k_actual]
            small_areas = areas[indices]
            max_a = np.max(small_areas)
            shifted = -beta * (small_areas - max_a)
            exp_vals = np.exp(shifted)
            sum_exp = np.sum(exp_vals)
            weights = exp_vals / sum_exp
            
            # Gradient of soft_min w.r.t. each area
            grad_areas = np.zeros_like(areas)
            grad_areas[indices] = -weights
            
            # Gradient of areas w.r.t. points
            # area = 0.5 * |cross|, cross = dx1*dy2 - dy1*dx2
            # d(area)/d(cross) = 0.5 * sign(cross)
            sign_cross = np.sign(cross)
            d_area_d_cross = 0.5 * sign_cross
            
            # d(cross)/d(pts) for each triplet
            # cross = (pj-pi) x (pk-pi)
            # d(cross)/dpi = -(pj-pi) x (pk-pi) ... actually need to be careful
            # Let a = pj - pi, b = pk - pi
            # cross = a.x * b.y - a.y * b.x
            # d(cross)/dpi = -a.x * (db/dpi).y + ... this is complex
            # Simpler: use finite differences or analytic gradient
            
            # For efficiency, compute gradient analytically
            # d(cross)/dpi = -(dy1 * dx2 - dy2 * dx1) ... wait
            # cross = dx1*dy2 - dy1*dx2
            # d(cross)/dpi.x = -dy2 + dy1 = -(dy2 - dy1)
            # d(cross)/dpi.y = -dx2 + dx1 = -(dx2 - dx1)
            # d(cross)/dpj.x = dy2
            # d(cross)/dpj.y = -dx2
            # d(cross)/dpk.x = -dy1
            # d(cross)/dpk.y = dx1
            
            grad_pts = np.zeros((N, 2))
            for idx in range(len(_TRIPLETS)):
                gi = _T_I[idx]
                gj = _T_J[idx]
                gk = _T_K[idx]
                
                ga = grad_areas[idx] * d_area_d_cross[idx]
                
                # Gradient w.r.t. pi
                grad_pts[gi, 0] += ga * (-(dy2[idx] - dy1[idx]))
                grad_pts[gi, 1] += ga * (-(dx2[idx] - dx1[idx]))
                
                # Gradient w.r.t. pj
                grad_pts[gj, 0] += ga * dy2[idx]
                grad_pts[gj, 1] += ga * (-dx2[idx])
                
                # Gradient w.r.t. pk
                grad_pts[gk, 0] += ga * (-dy1[idx])
                grad_pts[gk, 1] += ga * dx1[idx]
            
            # Convert gradient from Cartesian to barycentric
            # x = v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
            # y = w*S3H = (1-u-v)*S3H
            # d(x)/du = -0.5, d(x)/dv = 0.5
            # d(y)/du = -S3H, d(y)/dv = -S3H
            # grad_uv = J^T * grad_pts where J is Jacobian of (x,y) w.r.t. (u,v)
            # J = [[-0.5, 0.5], [-S3H, -S3H]]
            # grad_u = -0.5*grad_x - S3H*grad_y
            # grad_v = 0.5*grad_x - S3H*grad_y
            
            grad_uv = np.zeros_like(uv_curr)
            grad_uv[0::2] = -0.5 * grad_pts[:, 0] - S3H * grad_pts[:, 1]
            grad_uv[1::2] = 0.5 * grad_pts[:, 0] - S3H * grad_pts[:, 1]
            
            # We want to MAXIMIZE soft_min, so gradient ascent
            # Adam update
            t += 1
            m = beta1 * m + (1 - beta1) * grad_uv
            v = beta2 * v + (1 - beta2) * (grad_uv ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            uv_curr = uv_curr + lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Project to simplex
            uv_curr = _project_to_simplex(uv_curr)
        
        return uv_curr
    
    # Phase 1: Multi-start with Adam optimization
    n_starts = 30
    beta_schedule = [(50, 5), (100, 10), (200, 15)]
    
    for s in range(n_starts):
        if not _elapsed_ok(15):
            break
        
        if s == 0:
            uv_start = init_uv.copy()
        else:
            uv_start = init_uv + rng.normal(0, 0.15, size=(N, 2))
            uv_start = _project_to_simplex(uv_start)
        
        # Run Adam with different beta and k settings
        for beta, k in beta_schedule:
            if not _elapsed_ok(10):
                break
            
            lr = 0.01
            steps = 300
            uv_opt = _adam_step(uv_start, beta, k, lr, steps)
            
            pts_opt = _bary_to_cart(uv_opt)
            pts_opt = _project_cartesian(pts_opt)
            _update_best(pts_opt)
            
            # Use the result as start for next beta
            uv_start = _cart_to_bary(pts_opt)
    
    # Phase 2: Final polish with high beta and small steps
    if best_pts is not None and _elapsed_ok(20):
        uv_best = _cart_to_bary(best_pts)
        for beta, k in [(300, 10), (500, 5)]:
            if not _elapsed_ok(10):
                break
            uv_polished = _adam_step(uv_best, beta, k, 0.005, 500)
            pts_polished = _bary_to_cart(uv_polished)
            pts_polished = _project_cartesian(pts_polished)
            _update_best(pts_polished)
            uv_best = _cart_to_bary(pts_polished)
    
    # Final sanity check
    if best_pts is None:
        best_pts = init_cart
        best_area = _vectorized_min_area_cart(best_pts)
    else:
        best_pts = _project_cartesian(best_pts)
        
    return best_pts
