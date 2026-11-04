import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 14
_TRIPLE_INDICES = np.array(list(combinations(range(N), 3)), dtype=int)
_N_TRIPLES = _TRIPLE_INDICES.shape[0]

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(points):
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min(areas, beta):
    """Smooth min via log-sum-exp. Returns (value, weights)."""
    m = np.min(areas)
    if m < 1e-15:
        return m, np.ones(len(areas)) / len(areas)
    a = areas - m
    neg_beta_a = -beta * a
    # Shift for numerical stability
    max_val = np.max(neg_beta_a)
    exp_vals = np.exp(neg_beta_a - max_val)
    log_sum = max_val + np.log(np.sum(exp_vals))
    val = m - (1.0 / beta) * log_sum
    weights = exp_vals / np.sum(exp_vals)
    return val, weights

def _soft_score(points, beta=15.0):
    """Smooth score: soft_min(areas) / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas(points)
    sm, _ = _soft_min(areas, beta)
    return sm / hull_area

def _neg_soft_score(points, beta=15.0):
    return -_soft_score(points, beta)

def _neg_soft_score_grad(points, beta=15.0):
    """
    Gradient of -soft_min(areas)/hull_area.
    d(soft_min)/d(points) = sum_k w_k * d(areas[k])/d(points)
    d(areas[k])/d(p_i) for triple (i,j,k):
      area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
      For non-degenerate: d(area)/d(p_i) = 0.5 * cross(p_k - p_i, p_j - p_i) ... 
      Actually let's use: area = 0.5 * |det([p_j-p_i, p_k-p_i])|
      d/dp_i of det = -0.5 * (det basis) ... 
      
      Simpler: area_k = 0.5 * |cross(v1, v2)| where v1 = p_j - p_i, v2 = p_k - p_i
      d(area_k)/d(p_i) = 0.5 * d|cross(v1,v2)|/d(p_i)
      cross(v1,v2) = (p_j - p_i) x (p_k - p_i)
      d cross / d p_i = -(p_k - p_i) x ... no.
      
      Let's use the signed area: A = 0.5 * ((x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i))
      dA/dx_i = 0.5 * (-(y_k - y_i) + (y_j - y_i)) = 0.5 * (y_j - y_k)
      dA/dy_i = 0.5 * ((x_k - x_i) - (x_j - x_i)) = 0.5 * (x_k - x_j)
      dA/dx_j = 0.5 * (y_k - y_i)
      dA/dy_j = 0.5 * (-(x_k - x_i)) = 0.5 * (x_i - x_k)
      dA/dx_k = 0.5 * (-(y_j - y_i)) = 0.5 * (y_i - y_j)
      dA/dy_k = 0.5 * (x_j - x_i)
      
      For |A|, the gradient is sign(A) * dA/dp (when A != 0).
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return np.zeros(2 * N)
    
    areas = _triangle_areas(points)
    sm, weights = _soft_min(areas, beta)
    
    # Gradient of soft_min w.r.t. each point
    grad_soft = np.zeros(2 * N)
    
    # Signed areas for sign determination
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    signed_cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
                   (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    signs = np.sign(signed_cross)
    # Avoid zero sign
    signs[signs == 0] = 1.0
    
    # d(signed_area)/d(points) for each triple
    # dA/dx_i = 0.5*(y_j - y_k), dA/dy_i = 0.5*(x_k - x_j)
    # dA/dx_j = 0.5*(y_k - y_i), dA/dy_j = 0.5*(x_i - x_k)
    # dA/dx_k = 0.5*(y_i - y_j), dA/dy_k = 0.5*(x_j - x_i)
    
    dA_dx_i = 0.5 * (p_j[:, 1] - p_k[:, 1])
    dA_dy_i = 0.5 * (p_k[:, 0] - p_j[:, 0])
    dA_dx_j = 0.5 * (p_k[:, 1] - p_i[:, 1])
    dA_dy_j = 0.5 * (p_i[:, 0] - p_k[:, 0])
    dA_dx_k = 0.5 * (p_i[:, 1] - p_j[:, 1])
    dA_dy_k = 0.5 * (p_j[:, 0] - p_i[:, 0])
    
    # Weighted sum with sign
    w = weights * signs  # w_k * sign(A_k)
    
    # Accumulate gradients
    # For each triple, add w * dA/dp to the respective point
    grad_soft[i_idx * 2 + 0] += w * dA_dx_i
    grad_soft[i_idx * 2 + 1] += w * dA_dy_i
    grad_soft[j_idx * 2 + 0] += w * dA_dx_j
    grad_soft[j_idx * 2 + 1] += w * dA_dy_j
    grad_soft[k_idx * 2 + 0] += w * dA_dx_k
    grad_soft[k_idx * 2 + 1] += w * dA_dy_k
    
    # Now d(soft_min)/d(points) = grad_soft
    # d(soft_min/hull_area)/d(points) = grad_soft / hull_area - (soft_min / hull_area^2) * d(hull_area)/d(points)
    # For simplicity, ignore hull_area gradient (it's relatively smooth and we normalize at the end)
    # Actually, let's include it approximately by just using grad_soft / hull_area
    # The hull_area gradient is complex; for now, use the approximation
    
    neg_grad = -grad_soft / hull_area
    
    return neg_grad

def _generate_ring_config(n_outer, r_inner, n_inner, theta_offset=0.0, r_outer=1.0):
    """Generate a two-ring configuration."""
    outer_angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    outer_pts = r_outer * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_angles = theta_offset + np.linspace(0, 2*np.pi, n_inner, endpoint=False)
    inner_pts = r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])

def _generate_diverse_configs():
    """Generate diverse starting configurations."""
    configs = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Two-ring configs
    for n_out in [7, 8, 6]:
        n_in = N - n_out
        for r_in in [0.3, 0.5, 0.7]:
            for theta_off in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                configs.append(_generate_ring_config(n_out, r_in, n_in, theta_off))
    
    # Three-ring: 5-5-4
    for r1 in [0.4, 0.6]:
        for r2 in [0.7, 0.85]:
            a0 = np.linspace(0, 2*np.pi, 5, endpoint=False)
            a1 = np.pi/5 + np.linspace(0, 2*np.pi, 5, endpoint=False)
            a2 = np.linspace(0, 2*np.pi, 4, endpoint=False)
            c0 = np.column_stack([np.cos(a0), np.sin(a0)])
            c1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
            c2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
            configs.append(np.vstack([c0, c1, c2]))
    
    # Random configs
    rng = np.random.RandomState(123)
    for _ in range(8):
        pts = rng.randn(N, 2) * 0.8
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / np.maximum(norms, 0.1)[:, np.newaxis]
        # Scale to fit in unit disk
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 1.0:
            pts *= 1.0 / max_norm
        configs.append(pts)
    
    # Jittered regular polygon
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for _ in range(5):
        jitter = rng.randn(N, 2) * 0.1
        configs.append(base + jitter)
    
    return configs

def _optimize_from(points, start_time, deadline, beta=15.0):
    """Run L-BFGS-B on the soft objective, then Nelder-Mead polish."""
    x0 = points.flatten()
    
    # L-BFGS-B with gradient
    best_x = x0.copy()
    best_val = _neg_soft_score(points, beta)
    
    try:
        res = minimize(
            _neg_soft_score,
            x0,
            args=(beta,),
            jac=_neg_soft_score_grad,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8,
                     'maxls': 20, 'iprint': 0}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Nelder-Mead polish on the raw (non-smooth) objective
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-10,
                         'adaptive': True, 'initial_simplex': 0.01}
            )
            if -res2.fun > -best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    return best_x.reshape(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    configs = _generate_diverse_configs()
    
    # Score all configs with the true objective
    scored = []
    for pts in configs:
        s = _score(pts)
        scored.append((s, pts))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored[0][0] if scored else 0.0
    best_points = scored[0][1].copy() if scored else np.zeros((N, 2))
    
    # Optimize top candidates with gradient-based methods
    top_k = min(10, len(scored))
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
        
        s_init, pts_init = scored[i]
        
        # Try different beta values for soft-min
        for beta in [10.0, 20.0, 40.0]:
            if time.time() > deadline - 10:
                break
            
            optimized = _optimize_from(pts_init, start_time, deadline, beta=beta)
            s_opt = _score(optimized)
            
            if s_opt > best_score:
                best_score = s_opt
                best_points = optimized.copy()
        
        # Also try Nelder-Mead directly on the raw objective from this starting point
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    lambda x: -_score(x.reshape(N, 2)),
                    pts_init.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-11,
                             'adaptive': True, 'initial_simplex': 0.05}
                )
                s_nm = -res.fun
                if s_nm > best_score:
                    best_score = s_nm
                    best_points = res.x.reshape(N, 2)
            except Exception:
                pass
    
    # Final polish: Nelder-Mead on the best found
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12,
                         'adaptive': True, 'initial_simplex': 0.02}
            )
            s_final = -res.fun
            if s_final > best_score:
                best_score = s_final
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass
    
    return best_points
