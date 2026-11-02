"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Differential Evolution for global search + Analytic Gradient Soft-Min Annealing for local polish + Nelder-Mead final refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 145  # Safety margin from 150s limit

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    bary: shape (N, 2) where u, v >= 0 and u+v <= 1.
    P = u*V0 + v*V1 + (1-u-v)*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v).
    """
    u = (S3H - pts[:, 1]) / S3H
    v = 1.0 - 2.0 * pts[:, 0]
    return np.column_stack([u, v])

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
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

def _soft_min_area_and_grad_cart(pts, beta):
    """
    Soft minimum of triangle areas and its gradient w.r.t. pts (Cartesian).
    Returns (soft_min_value, grad) where grad has shape (N, 2).
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Numerical stability
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    soft_min = max_a - log_sum / beta
    
    # weights w_i = exp(-beta * (a_i - max_a)) / sum_exp
    w = exp_shifted / sum_exp  # shape (M,)
    
    sign_cross = np.sign(cross)
    sign_cross = np.where(np.abs(cross) < 1e-15, 1.0, sign_cross)
    
    grad = np.zeros((N, 2))
    coeff = w * 0.5 * sign_cross  # shape (M,)
    
    # Gradients for pi, pj, pk
    # d(soft_min)/d(pts) = sum_i w_i * d(a_i)/d(pts)
    # d(a_i)/d(pi) = 0.5*sign*(dy1-dy2, dx2-dx1)
    # d(a_i)/d(pj) = 0.5*sign*(dy2, -dx2)
    # d(a_i)/d(pk) = 0.5*sign*(-dy1, dx1)
    
    grad_x_pi = coeff * (dy1 - dy2)
    grad_y_pi = coeff * (dx2 - dx1)
    np.add.at(grad, (_T_I, 0), grad_x_pi)
    np.add.at(grad, (_T_I, 1), grad_y_pi)
    
    grad_x_pj = coeff * dy2
    grad_y_pj = coeff * (-dx2)
    np.add.at(grad, (_T_J, 0), grad_x_pj)
    np.add.at(grad, (_T_J, 1), grad_y_pj)
    
    grad_x_pk = coeff * (-dy1)
    grad_y_pk = coeff * dx1
    np.add.at(grad, (_T_K, 0), grad_x_pk)
    np.add.at(grad, (_T_K, 1), grad_y_pk)
    
    return soft_min, grad

def _cart_to_bary_grad(cart_grad):
    """
    Convert gradient from Cartesian to barycentric coordinates.
    cart_grad: shape (N, 2) gradient w.r.t. Cartesian coords.
    Returns gradient w.r.t. barycentric (u, v).
    """
    # x = 0.5 + 0.5v - 0.5u  => dx/du = -0.5, dx/dv = 0.5
    # y = S3H - S3H*u - S3H*v => dy/du = -S3H, dy/dv = -S3H
    
    # grad_u = grad_x * dx/du + grad_y * dy/du
    # grad_v = grad_x * dx/dv + grad_y * dy/dv
    
    gu = cart_grad[:, 0] * (-0.5) + cart_grad[:, 1] * (-S3H)
    gv = cart_grad[:, 0] * (0.5) + cart_grad[:, 1] * (-S3H)
    return np.column_stack([gu, gv])

def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()

    # Initial symmetric configuration
    base_sym = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.2],
        [0.2, 0.4],
    ])
    _update_best(base_sym)

    # Phase 1: Differential Evolution
    # We optimize in barycentric (u, v) space.
    # Bounds: u in [0, 1], v in [0, 1]. We add a penalty for u+v > 1.
    
    def de_objective(flat_bc):
        bc = flat_bc.reshape(-1, 2)
        u = bc[:, 0]
        v = bc[:, 1]
        
        # Penalty for constraint violation
        penalty = 0.0
        # u, v should be >= 0 (handled by bounds)
        # u + v <= 1
        violation = np.maximum(0, u + v - 1)
        penalty = np.sum(violation ** 2) * 1000.0
        
        bary = np.column_stack([u, v])
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        
        return -area + penalty

    bounds_de = [(0, 1), (0, 1)] * N
    
    if _elapsed_ok(60):
        try:
            de_result = differential_evolution(
                de_objective,
                bounds_de,
                maxiter=50,
                popsize=20,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            if de_result.success or de_result.fun < 1e10: # Check if penalty didn't explode
                de_bary = de_result.x.reshape(N, 2)
                de_bary = _project_bary_to_valid(de_bary)
                _update_best(de_bary)
        except Exception:
            pass

    # Phase 2: Local Polish with Analytic Gradients (L-BFGS-B)
    # We take the best candidate and refine it using the smooth soft-min objective.
    # We try multiple beta values to anneal the soft-min.
    
    if best_bary is not None and _elapsed_ok(40):
        current_bary = best_bary.copy()
        
        # Annealing schedule
        betas = [50.0, 200.0, 1000.0, 5000.0, 20000.0]
        
        for beta in betas:
            if not _elapsed_ok(10):
                break
                
            def objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                val, cart_grad = _soft_min_area_and_grad_cart(cart, beta)
                bary_grad = _cart_to_bary_grad(cart_grad)
                return -val, -bary_grad.flatten()
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                if res.success or res.fun < 1e10:
                    current_bary = res.x.reshape(N, 2)
                    current_bary = _project_bary_to_valid(current_bary)
                    _update_best(current_bary)
            except Exception:
                pass

    # Phase 3: Final Polish with Nelder-Mead on Hard Minimum
    if best_bary is not None and _elapsed_ok(15):
        nm_start = best_bary.flatten()
        
        def nm_objective(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res_nm.success:
                nm_bary = res_nm.x.reshape(N, 2)
                nm_bary = _project_bary_to_valid(nm_bary)
                _update_best(nm_bary)
        except Exception:
            pass

    # Fallback
    if best_cart is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart

# EVOLVE-BLOCK-END
