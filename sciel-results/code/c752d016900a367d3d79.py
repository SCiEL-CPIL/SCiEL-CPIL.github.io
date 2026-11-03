# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, basin_hopping
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 145

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

_TRIPLETS = list(itertools.combinations(range(11), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

N = 11

def _barycentric_to_cartesian(bc):
    """Convert (b, c) barycentric coords to (x,y).
    barycentric: a=1-b-c, point = a*V0 + b*V1 + c*V2
    x = b*1.0 + c*0.5, y = c*S3H
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * S3H
    return np.column_stack([x, y])

def _project_bary(bc):
    """Project barycentric coords (b,c) to valid triangle: b>=0, c>=0, b+c<=1."""
    b = bc[:, 0].copy()
    c = bc[:, 1].copy()
    b = np.maximum(b, 0.0)
    c = np.maximum(c, 0.0)
    s = b + c
    mask = s > 1.0
    if np.any(mask):
        b[mask] /= s[mask]
        c[mask] /= s[mask]
    return np.column_stack([b, c])

def _min_triangle_area(pts):
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

def _soft_min_area(pts, beta):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Log-Sum-Exp trick for stability
    # We want to minimize -min(area), which is approx -logsumexp(-beta * areas) / beta
    # Actually, we want to maximize min(area). 
    # Objective for minimizer: -soft_min.
    # soft_min ~ min.
    # Let's use: - (1/beta) * log( sum(exp(-beta * areas)) )
    # As beta->inf, this approaches -min(areas).
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta

def _soft_min_grad(pts, beta):
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
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp

    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0

    w = 0.5 * weights * sign_cross

    grad = np.zeros((N, 2))
    grad_pi = w[:, None] * np.column_stack([-dy2, dx2])
    grad_pj = w[:, None] * np.column_stack([dy2, -dx2])
    grad_pk = w[:, None] * np.column_stack([-dy1, dx1])

    np.add.at(grad, _T_I, grad_pi)
    np.add.at(grad, _T_J, grad_pj)
    np.add.at(grad, _T_K, grad_pk)

    return grad

def _objective_bary(flat_bc, beta):
    bc = flat_bc.reshape(-1, 2)
    bc = _project_bary(bc)
    pts = _barycentric_to_cartesian(bc)
    return -_soft_min_area(pts, beta)

def _gradient_bary(flat_bc, beta):
    bc = flat_bc.reshape(-1, 2)
    bc = _project_bary(bc)
    pts = _barycentric_to_cartesian(bc)
    val = -_soft_min_area(pts, beta)
    grad_pts = _soft_min_grad(pts, beta)
    
    # Chain rule: x = b + 0.5*c, y = c*S3H
    # d(val)/db = -grad_pts[:,0]
    # d(val)/dc = -(0.5*grad_pts[:,0] + S3H*grad_pts[:,1])
    grad_b = -grad_pts[:, 0]
    grad_c = -(0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1])
    grad_flat = np.column_stack([grad_b, grad_c]).flatten()
    return val, grad_flat

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Initial guess: Symmetric
    init_bc = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.25, 0.5],
        [0.75, 0.5],
        [1/3, 1/3],
        [2/3, 0.0],
        [1/3, 0.0],
        [0.5, 1/3],
        [0.5, 0.5]
    ])
    init_bc = _project_bary(init_bc)
    
    best_pts = _barycentric_to_cartesian(init_bc)
    best_area = _min_triangle_area(best_pts)
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Multi-start Basin Hopping
    n_starts = 10
    for i in range(n_starts):
        if not _elapsed_ok(10):
            break
            
        # Perturb initial guess for diversity
        noise = rng.normal(0, 0.1, size=(N, 2))
        start_bc = _project_bary(init_bc + noise)
        start_flat = start_bc.flatten()
        
        try:
            # Basin Hopping with L-BFGS-B local optimizer
            # We use a moderate beta for the soft-min to guide the optimizer
            beta = 100.0
            
            def local_obj(x):
                return _objective_bary(x, beta)
            
            def local_grad(x):
                return _gradient_bary(x, beta)[1]
                
            def perturbation(x, T, stepsize):
                # Gaussian perturbation
                return x + stepsize * rng.normal(size=x.shape)
                
            res = basin_hopping(
                local_obj, 
                start_flat, 
                minimizer_kwargs={'jac': local_grad, 'method': 'L-BFGS-B'},
                niter=50, 
                stepsize=0.2,
                seed=42+i
            )
            
            # Convert result back
            res_bc = _project_bary(res.x.reshape(-1, 2))
            res_pts = _barycentric_to_cartesian(res_bc)
            _update_best(res_pts)
            
            # Polish with Nelder-Mead on exact min
            if _elapsed_ok(5):
                def exact_obj(x):
                    bc = _project_bary(x.reshape(-1, 2))
                    pts = _barycentric_to_cartesian(bc)
                    return -_min_triangle_area(pts)
                
                nm_res = minimize(
                    exact_obj, 
                    res.x, 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                nm_bc = _project_bary(nm_res.x.reshape(-1, 2))
                nm_pts = _barycentric_to_cartesian(nm_bc)
                _update_best(nm_pts)
                
        except Exception as e:
            pass
            
    # Final safety check
    result = best_pts
    for i in range(N):
        x, y = result[i]
        if y < -1e-10 or y > S3H + 1e-10:
            result[i] = [0.5, 0.0]
        x_min = y / S3
        x_max = 1.0 - y / S3
        if x < x_min - 1e-10 or x > x_max + 1e-10:
            result[i] = [(x_min + x_max) / 2, y]
            
    return result
# EVOLVE-BLOCK-END
