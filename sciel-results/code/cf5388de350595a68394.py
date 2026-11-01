import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE_FRAC = 145.0  # wall-clock guard (evaluator kills at 150s)

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _hull_area_grad(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        n = len(points)
        grad = np.zeros_like(points)
        hv = hull.vertices
        m = len(hv)
        for i in range(m):
            idx = hv[i]
            prev_idx = hv[(i - 1) % m]
            next_idx = hv[(i + 1) % m]
            grad[idx, 0] += 0.5 * (points[prev_idx, 1] - points[next_idx, 1])
            grad[idx, 1] += 0.5 * (points[next_idx, 0] - points[prev_idx, 0])
        return grad
    except Exception:
        return np.zeros_like(points)

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_ring_heptagon(r, phi):
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r, phi)
    return np.vstack([outer, inner])

def _three_ring_config(r1, r2, phi1, phi2):
    outer = _regular_polygon(5, 1.0, 0.0)
    mid = _regular_polygon(4, r2, phi2)
    inner = _regular_polygon(5, r1, phi1)
    return np.vstack([outer, mid, inner])

def _soft_min_area_and_grad(points, beta):
    """
    Soft-min approximation of minimum triangle area with gradient.
    Uses log-sum-exp formulation: -beta * log(sum(exp(-A_i / beta)))
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for numerical stability
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / beta)
    sum_exp = np.sum(exp_terms)
    lse_val = max_a + beta * np.log(sum_exp)
    
    # Weights for gradient
    weights = exp_terms / sum_exp
    
    # Gradient of soft-min w.r.t. points
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    coeff = 0.5 * signs * weights
    
    grad = np.zeros_like(points)
    
    # d(soft_min)/dp_i for each triangle
    gi = coeff[:, np.newaxis] * np.column_stack([-v2[:, 1], v2[:, 0]])
    np.add.at(grad, _I, gi)
    
    gj = coeff[:, np.newaxis] * np.column_stack([v2[:, 1], -v2[:, 0]])
    np.add.at(grad, _J, gj)
    
    gk = coeff[:, np.newaxis] * np.column_stack([-v1[:, 1], v1[:, 0]])
    np.add.at(grad, _K, gk)
    
    return lse_val, grad

def _smooth_neg_score_grad(flat, beta):
    """
    Smooth approximation of -min_area/hull_area with gradient.
    """
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros_like(flat)
    
    lse_val, grad_lse = _soft_min_area_and_grad(pts, beta)
    grad_ha = _hull_area_grad(pts)
    
    # f = -lse_val / ha
    # grad f = -(grad_lse * ha - lse_val * grad_ha) / ha^2
    #        = -grad_lse / ha + lse_val * grad_ha / ha^2
    grad_f = -grad_lse / ha + (lse_val / (ha * ha)) * grad_ha
    
    return -lse_val / ha, grad_f.flatten()

def _anneal_beta_optimization(pts0, beta_start, beta_end, n_steps):
    """
    Perform gradient-based optimization with adaptive beta annealing.
    Returns optimized points.
    """
    current = pts0.copy()
    beta = beta_start
    
    for step in range(n_steps):
        # Anneal beta
        frac = step / max(1, n_steps - 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        # Run L-BFGS-B with current beta
        try:
            res = scipy.optimize.minimize(
                lambda x: _smooth_neg_score_grad(x, beta)[0],
                current.flatten(),
                jac=lambda x: _smooth_neg_score_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            current = res.x.reshape(-1, 2)
        except Exception:
            pass
    
    return current

def _build_diverse_starts():
    """Build diverse initial configurations."""
    starts = []
    
    # Two-ring heptagons with varied parameters
    for r in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for phi in [0.0, np.pi/14, np.pi/7, 2*np.pi/14, 3*np.pi/14]:
            starts.append(_two_ring_heptagon(r, phi))
    
    # Three-ring configurations
    for r1 in [0.3, 0.4, 0.5]:
        for r2 in [0.6, 0.7, 0.8]:
            for phi1 in [0.0, np.pi/5]:
                for phi2 in [0.0, np.pi/4]:
                    starts.append(_three_ring_config(r1, r2, phi1, phi2))
    
    # Regular polygon with perturbations
    base = _regular_polygon(N, 1.0, 0.0)
    for _ in range(5):
        perturbed = base + 0.05 * np.random.randn(N, 2)
        starts.append(perturbed)
    
    # Elliptical configurations
    for a, b in [(1.0, 0.8), (1.0, 0.85), (1.0, 0.9), (1.05, 0.95)]:
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        starts.append(np.column_stack([x, y]))
    
    return starts

def construct():
    t0 = time.time()
    def left():
        return DEADLINE_FRAC - (time.time() - t0)
    
    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))
    
    # Phase 1: Grid search over two-ring manifold
    if left() > 20.0:
        r_grid = np.linspace(0.3, 0.8, 40)
        phi_grid = np.linspace(0, np.pi/7, 40)
        
        best_sym_score = 0.0
        best_r = 0.5
        best_phi = 0.0
        
        for r in r_grid:
            for phi in phi_grid:
                pts = _two_ring_heptagon(r, phi)
                sc = _score(pts)
                if sc > best_sym_score:
                    best_sym_score = sc
                    best_r = r
                    best_phi = phi
        
        # Refine with Nelder-Mead on 2D parameter space
        if best_sym_score > 0 and left() > 15.0:
            def obj_sym(params):
                r, phi = params
                if r < 0.1 or r > 1.5:
                    return 1e10
                pts = _two_ring_heptagon(r, phi)
                return -_score(pts)
            
            try:
                res_sym = scipy.optimize.minimize(
                    obj_sym, np.array([best_r, best_phi]),
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 3000}
                )
                if res_sym.fun < -best_sym_score:
                    r_opt, phi_opt = res_sym.x
                    best_sym_pts = _two_ring_heptagon(r_opt, phi_opt)
                    best_sym_score = _score(best_sym_pts)
            except Exception:
                pass
        
        if best_sym_score > best_score:
            best_score = best_sym_score
            best_pts = _two_ring_heptagon(best_r, best_phi).copy()
    
    # Phase 2: Diverse starts with smooth gradient optimization
    if left() > 40.0:
        starts = _build_diverse_starts()
        
        for i, start in enumerate(starts):
            if left() < 10.0:
                break
            
            # Try smooth optimization with beta annealing
            try:
                opt_pts = _anneal_beta_optimization(
                    start,
                    beta_start=0.05,
                    beta_end=0.001,
                    n_steps=8
                )
                opt_pts = _normalize(opt_pts)
                sc = _score(opt_pts)
                
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
            except Exception:
                pass
            
            # Also try direct Nelder-Mead on this start
            if left() > 5.0:
                try:
                    res = scipy.optimize.minimize(
                        _neg_score, start.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'fatol': 1e-11, 'xatol': 1e-9}
                    )
                    opt_pts = _normalize(res.x.reshape(-1, 2))
                    sc = _score(opt_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = opt_pts.copy()
                except Exception:
                    pass
    
    # Phase 3: Differential evolution for global exploration
    if left() > 25.0:
        def de_obj(flat):
            return -_score(flat.reshape(-1, 2))
        
        bounds = [(-1.5, 1.5)] * (N * 2)
        
        try:
            de_res = scipy.optimize.differential_evolution(
                de_obj, bounds,
                seed=42, maxiter=80, popsize=20,
                tol=1e-5, mutation=(0.5, 1.0), recombination=0.8,
                polish=True
            )
            
            de_pts = _normalize(de_res.x.reshape(-1, 2))
            de_score = _score(de_pts)
            
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
    
    # Phase 4: Final refinement with exact objective
    if left() > 8.0:
        try:
            res = scipy.optimize.minimize(
                _neg_score, best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'fatol': 1e-13, 'xatol': 1e-11}
            )
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            if sc > best_score:
                best_score = sc
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    return _normalize(best_pts)
