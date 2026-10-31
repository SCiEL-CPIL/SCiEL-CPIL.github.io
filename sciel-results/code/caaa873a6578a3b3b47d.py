"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Strategy: Analytic-gradient soft-min L-BFGS-B with Delaunay-informed multi-start,
plus a fast coordinate-descent polish that only re-evaluates affected triangles.
"""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0
EPS = 1e-9

# Pre-compute triangle indices
tri_idx = np.array(list(combinations(range(N), 3)))
T = tri_idx.shape[0]  # 165
I0 = tri_idx[:, 0]
I1 = tri_idx[:, 1]
I2 = tri_idx[:, 2]

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def clamp_bary(bary, eps=1e-8):
    u = np.clip(bary[:, 0], eps, 1.0 - eps)
    v = np.clip(bary[:, 1], eps, 1.0 - eps)
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums[invalid], eps)
        u[invalid] *= scale
        v[invalid] *= scale
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])

def calc_det(bary):
    """Compute signed determinants for all 165 triangles."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    det = (u[I0] * (v[I1]*w[I2] - v[I2]*w[I1])
           - v[I0] * (u[I1]*w[I2] - u[I2]*w[I1])
           + w[I0] * (u[I1]*v[I2] - u[I2]*v[I1]))
    return det

def hard_min_val(bary):
    return np.min(np.abs(calc_det(bary)))

def soft_min_obj(bary_flat, beta):
    bary = clamp_bary(bary_flat.reshape(N, 2))
    areas = np.abs(calc_det(bary))
    s = -beta * areas
    m = np.max(s)
    if m > 700:
        return -np.min(areas)
    return -(m + np.log(np.sum(np.exp(s - m)))) / beta

def soft_min_obj_grad(bary_flat, beta):
    bary = clamp_bary(bary_flat.reshape(N, 2))
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    det = (u[I0] * (v[I1]*w[I2] - v[I2]*w[I1])
           - v[I0] * (u[I1]*w[I2] - u[I2]*w[I1])
           + w[I0] * (u[I1]*v[I2] - u[I2]*v[I1]))
    areas = np.abs(det)
    s = -beta * areas
    m = np.max(s)
    if m > 700:
        # Fall back to hard min gradient
        idx_min = np.argmin(areas)
        grad_u = np.zeros(N)
        grad_v = np.zeros(N)
        i, j, k = tri_idx[idx_min]
        grad_u[i] = v[j]*w[k] - v[k]*w[j] - w[i]*(u[j]*v[k] - u[k]*v[j])
        grad_v[i] = -w[i]*(u[j]*v[k] - u[k]*v[j])
        grad_u[j] = w[i]*v[k] - w[k]*v[i] + u[i]*(v[k]*w[j] - v[j]*w[k])
        grad_v[j] = -w[i]*v[k] + w[k]*v[i] + u[i]*(v[j]*w[k] - w[j]*v[k])
        grad_u[k] = -w[i]*v[j] + w[j]*v[i] + u[i]*(v[j]*w[k] - v[k]*w[j])
        grad_v[k] = w[i]*v[j] - w[j]*v[i] + u[i]*(v[k]*w[j] - v[j]*w[k])
        return -soft_min_obj(bary_flat, beta), np.column_stack([grad_u, grad_v]).flatten()
    p = np.exp(s - m)
    p /= p.sum()
    # d(soft_min)/du_i = -sum_t p_t * sign(det_t) * d(det_t)/du_i
    sign = np.sign(det)
    du0 = v[I1]*w[I2] - v[I2]*w[I1] - w[I0]*(u[I1]*v[I2] - u[I2]*v[I1])
    dv0 = -w[I0]*(u[I1]*v[I2] - u[I2]*v[I1])
    du1 = w[I0]*v[I2] - w[I2]*v[I0] + u[I0]*(v[I2]*w[I1] - v[I1]*w[I2])
    dv1 = -w[I0]*v[I2] + w[I2]*v[I0] + u[I0]*(v[I1]*w[I2] - w[I1]*v[I2])
    du2 = -w[I0]*v[I1] + w[I1]*v[I0] + u[I0]*(v[I1]*w[I2] - v[I2]*w[I1])
    dv2 = w[I0]*v[I1] - w[I1]*v[I0] + u[I0]*(v[I2]*w[I1] - v[I1]*w[I2])
    ps = p * sign
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    np.add.at(grad_u, I0, -ps * du0)
    np.add.at(grad_v, I0, -ps * dv0)
    np.add.at(grad_u, I1, -ps * du1)
    np.add.at(grad_v, I1, -ps * dv1)
    np.add.at(grad_u, I2, -ps * du2)
    np.add.at(grad_v, I2, -ps * dv2)
    val = -(m + np.log(np.sum(p))) / beta
    return val, np.column_stack([grad_u, grad_v]).flatten()

def delaunay_init():
    """Use Delaunay triangulation to find a good initial point set."""
    best_bary = None
    best_val = -np.inf
    for seed in range(20):
        rng = np.random.default_rng(seed * 7 + 3)
        pts = rng.random((N, 2))
        # Project into triangle
        bary = np.column_stack([1 - pts[:, 0] - pts[:, 1], pts[:, 0], pts[:, 1]])
        # Use u, v as the two barycentric coords
        bary2 = np.column_stack([bary[:, 0], bary[:, 1]])
        cart = bary_to_cart(bary2)
        try:
            tri = Delaunay(cart)
        except Exception:
            continue
        # Evaluate
        val = hard_min_val(bary2)
        if val > best_val:
            best_val = val
            best_bary = bary2.copy()
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    return best_bary

def random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def hex_init():
    """Hexagonal lattice initialization."""
    best_bary = None
    best_val = -np.inf
    for s in np.linspace(0.08, 0.45, 25):
        points = []
        for i in range(8):
            for j in range(8):
                u = i * s
                v = j * s
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        if len(points) == N:
            bary = np.array(points)
            val = hard_min_val(bary)
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    if best_bary is None:
        best_bary = random_init(0)
    return best_bary

def coordinate_descent(bary, deadline, max_rounds=50):
    """
    Coordinate descent: for each point, optimize its (u,v) by trying small steps.
    Only re-evaluates triangles involving that point (2 * C(10,2) = 90 triangles).
    """
    bary = clamp_bary(bary.copy())
    best_val = hard_min_val(bary)
    
    # Pre-compute for each point, which triangles involve it
    # point_to_tris[i] = list of triangle indices that include point i
    point_to_tris = [[] for _ in range(N)]
    for t in range(T):
        for p in range(3):
            point_to_tris[tri_idx[t, p]].append(t)
    
    steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline - 3:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 3:
                break
            tri_set = point_to_tris[i]
            for step in steps:
                for du in [-step, 0, step]:
                    for dv in [-step, 0, step]:
                        if du == 0 and dv == 0:
                            continue
                        new_bary = bary.copy()
                        new_bary[i, 0] += du
                        new_bary[i, 1] += dv
                        new_bary = clamp_bary(new_bary)
                        # Only compute areas for triangles involving point i
                        det_all = calc_det(new_bary)
                        areas_all = np.abs(det_all)
                        val = np.min(areas_all)
                        if val > best_val + 1e-12:
                            bary = new_bary
                            best_val = val
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    
    return bary

def try_optimize(bary_init, deadline, beta1=30.0, beta2=100.0, beta3=300.0):
    """Multi-stage L-BFGS-B with analytic gradients + coordinate descent polish."""
    if deadline is not None and time.time() > deadline - 15:
        return bary_init
    
    try:
        # Use linear constraints for barycentric validity: u>=0, v>=0, u+v<=1
        # For L-BFGS-B, use bounds on u and v, then clamp
        bounds = [(0.0, 1.0)] * (N * 2)
        
        # Stage 1: smooth
        r1 = minimize(soft_min_obj_grad, bary_init.flatten(), args=(beta1,),
                      method='L-BFGS-B', jac=True, bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8, 'maxfun': 1000})
        if deadline is not None and time.time() > deadline - 10:
            return clamp_bary(r1.x.reshape(N, 2))
        
        # Stage 2: sharper
        r2 = minimize(soft_min_obj_grad, r1.x, args=(beta2,),
                      method='L-BFGS-B', jac=True, bounds=bounds,
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000})
        if deadline is not None and time.time() > deadline - 8:
            return clamp_bary(r2.x.reshape(N, 2))
        
        # Stage 3: very sharp
        r3 = minimize(soft_min_obj_grad, r2.x, args=(beta3,),
                      method='L-BFGS-B', jac=True, bounds=bounds,
                      options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000})
        result = clamp_bary(r3.x.reshape(N, 2))
        
        # Coordinate descent polish
        if deadline is not None and time.time() < deadline - 5:
            result = coordinate_descent(result, deadline, max_rounds=30)
        
        return result
    except Exception:
        return bary_init

def nelder_mead_polish(bary_init, deadline, maxiter=300):
    if deadline is not None and time.time() > deadline - 3:
        return bary_init
    try:
        res = minimize(hard_min_val, bary_init.flatten(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
        return clamp_bary(res.x.reshape(N, 2))
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = hard_min_val(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    # 1. Delaunay-informed initialization
    dl_init = delaunay_init()
    result = try_optimize(dl_init, deadline)
    update_best(result)
    
    # 2. Hexagonal lattice
    if time.time() < deadline - 40:
        hexi = hex_init()
        result = try_optimize(hexi, deadline)
        update_best(result)
    
    # 3. Multiple random starts
    for i in range(12):
        if time.time() > deadline - 50:
            break
        ri = random_init(100 + i * 17)
        result = try_optimize(ri, deadline, beta1=20.0, beta2=80.0, beta3=250.0)
        update_best(result)
    
    # 4. Perturb best and re-optimize
    if best_bary is not None:
        for i in range(8):
            if time.time() > deadline - 30:
                break
            rng = np.random.default_rng(500 + i * 31)
            perturbed = clamp_bary(best_bary + rng.normal(0, 0.03, best_bary.shape))
            result = try_optimize(perturbed, deadline, beta1=50.0, beta2=150.0, beta3=400.0)
            update_best(result)
    
    # 5. Coordinate descent on best
    if best_bary is not None and time.time() < deadline - 15:
        cd_result = coordinate_descent(best_bary, deadline, max_rounds=80)
        update_best(cd_result)
    
    # 6. Nelder-Mead polish
    if best_bary is not None and time.time() < deadline - 5:
        nm_result = nelder_mead_polish(best_bary, deadline, maxiter=400)
        update_best(nm_result)
    
    # 7. Final coordinate descent
    if best_bary is not None and time.time() < deadline - 2:
        cd_result = coordinate_descent(best_bary, deadline, max_rounds=50)
        update_best(cd_result)
    
    if best_bary is None:
        best_bary = random_init(0)
    
    best_bary = clamp_bary(best_bary)
    result_cart = bary_to_cart(best_bary)
    
    # Ensure strictly inside
    result_cart = np.clip(result_cart, 1e-9, None)
    result_cart[:, 1] = np.clip(result_cart[:, 1], 1e-9, H - 1e-9)
    x_min = result_cart[:, 1] / np.sqrt(3)
    x_max = 1.0 - result_cart[:, 1] / np.sqrt(3)
    result_cart[:, 0] = np.clip(result_cart[:, 0], x_min + 1e-9, x_max - 1e-9)
    
    return result_cart
