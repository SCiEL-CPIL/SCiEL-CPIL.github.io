import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
AREA_T = np.sqrt(3) / 4.0

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(bary):
    """bary: (N, 2) array of (lam1, lam2). lam0 = 1 - lam1 - lam2."""
    lam1 = bary[:, 0]
    lam2 = bary[:, 1]
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """points: (N, 2) -> bary (N, 2) of (lam1, lam2)"""
    x = points[:, 0]
    y = points[:, 1]
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    return np.column_stack([lam1, lam2])

def _project_to_bary_simplex(bary):
    """Clip barycentric coords to the simplex: lam1>=0, lam2>=0, lam1+lam2<=1."""
    lam1 = np.clip(bary[:, 0], 0.0, 1.0)
    lam2 = np.clip(bary[:, 1], 0.0, 1.0)
    # Ensure lam1 + lam2 <= 1
    total = lam1 + lam2
    mask = total > 1.0
    if np.any(mask):
        scale = 1.0 / total[mask]
        lam1[mask] *= scale
        lam2[mask] *= scale
    return np.column_stack([lam1, lam2])

def _all_dets_from_bary(bary):
    """
    Compute signed double-areas for all triples from barycentric coords.
    Returns array of shape (N_TRIPLES,).
    The signed double-area of triangle (i,j,k) in barycentric coords is:
    det = lam1_i*(lam2_j - lam2_k) + lam1_j*(lam2_k - lam2_i) + lam1_k*(lam2_i - lam2_j)
    """
    l1_i = bary[_IDX_I, 0]
    l2_i = bary[_IDX_I, 1]
    l1_j = bary[_IDX_J, 0]
    l2_j = bary[_IDX_J, 1]
    l1_k = bary[_IDX_K, 0]
    l2_k = bary[_IDX_K, 1]
    dets = l1_i * (l2_j - l2_k) + l1_j * (l2_k - l2_i) + l1_k * (l2_i - l2_j)
    return dets

def _min_area_ratio_from_bary(bary):
    """Compute min triangle area / area of containing triangle."""
    dets = _all_dets_from_bary(bary)
    areas = 0.5 * np.abs(dets)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / AREA_T

def _softmin_objective(params, beta):
    """
    Smooth surrogate: -logsumexp(beta * (areas - t)) - log(beta)
    params: (2N+1,) = [bary_1, bary_2, ..., bary_N, t]
    """
    bary = params[:2*N].reshape(N, 2)
    t = params[2*N]
    
    bary = _project_to_bary_simplex(bary)
    dets = _all_dets_from_bary(bary)
    areas = 0.5 * np.abs(dets)
    
    # Soft-min: -(1/beta) * log(sum(exp(-beta * areas)))
    # To avoid overflow, shift by min
    a_min = np.min(areas)
    if a_min < 1e-300:
        a_min = 1e-300
    shifted = -beta * (areas - a_min)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    logsumexp = np.log(np.sum(np.exp(shifted)))
    soft_min = a_min - np.log(np.sum(np.exp(shifted))) / beta
    
    # Penalize if t > soft_min (we want t <= min_area)
    # Actually we want to maximize t subject to t <= all areas
    # Use: objective = -t + penalty * max(0, t - soft_min)^2
    # But for smoothness, use: -t + (1/beta) * log(1 + exp(beta*(t - soft_min)))
    
    # Simpler: just minimize -soft_min + penalty for t > soft_min
    # Let's use: f = -t + gamma * (t - soft_min)^2 if t > soft_min, else -t
    # Smooth version: f = -t + (1/beta) * log(1 + exp(2*beta*(t - soft_min)))
    
    diff = t - soft_min
    if diff > 20.0 / beta:
        penalty = beta * diff * diff
    elif diff < -20.0 / beta:
        penalty = 0.0
    else:
        penalty = np.log(1.0 + np.exp(2.0 * beta * diff)) / beta
    
    return -t + penalty

def _hard_objective(params):
    """Exact objective: -min_area_ratio."""
    bary = params[:2*N].reshape(N, 2)
    bary = _project_to_bary_simplex(bary)
    return -_min_area_ratio_from_bary(bary)

def _structured_init_fps(rng, n_points=N, n_candidates=300):
    """Farthest Point Sampling in barycentric space."""
    bary = np.zeros((n_points, 2))
    # Start at centroid
    bary[0] = [1.0/3, 1.0/3]
    
    for k in range(1, n_points):
        # Generate random candidates in the triangle
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([r1, r2])
        
        # Distance to existing points
        diffs = candidates[:, np.newaxis, :] - bary[np.newaxis, :k, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        bary[k] = candidates[best_idx]
    
    return bary

def _random_init_bary(rng):
    r1 = rng.uniform(0, 1, N)
    r2 = rng.uniform(0, 1, N)
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    return np.column_stack([r1, r2])

def _l_bfgs_softmin(start_bary, t_init, beta, maxiter=200):
    """Optimize using L-BFGS-B on the softmin objective."""
    def obj_and_grad(params):
        bary = params[:2*N].reshape(N, 2)
        t = params[2*N]
        
        bary_clipped = _project_to_bary_simplex(bary)
        dets = _all_dets_from_bary(bary_clipped)
        areas = 0.5 * np.abs(dets)
        
        a_min = np.min(areas)
        if a_min < 1e-300:
            a_min = 1e-300
        shifted = -beta * (areas - a_min)
        shifted = np.clip(shifted, -700, 700)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        soft_min = a_min - np.log(sum_exp) / beta
        
        # Gradient of soft_min w.r.t. bary
        # d(soft_min)/d(area_m) = -exp(-beta*(area_m - a_min)) / sum_exp
        weights = exp_shifted / sum_exp  # shape (N_TRIPLES,)
        
        # d(soft_min)/dbary = sum_m weights_m * d(area_m)/dbary
        # area_m = 0.5 * |det_m|
        # For simplicity, use subgradient: only the min area contributes
        # Actually let's compute gradient properly for the smooth case
        # d(area_m)/dbary_i = 0.5 * sign(det_m) * d(det_m)/dbary_i
        
        # det_m = l1_i*(l2_j - l2_k) + l1_j*(l2_k - l2_i) + l1_k*(l2_i - l2_j)
        # d(det_m)/d(l1_i) = l2_j - l2_k
        # d(det_m)/d(l2_i) = -(l1_j - l1_k) = l1_k - l1_j
        # d(det_m)/d(l1_j) = l2_k - l2_i
        # d(det_m)/d(l2_j) = -(l1_k - l1_i) = l1_i - l1_k
        # d(det_m)/d(l1_k) = l2_i - l2_j
        # d(det_m)/d(l2_k) = -(l1_i - l1_j) = l1_j - l1_i
        
        grad_bary = np.zeros((N, 2))
        signs = np.sign(dets)
        signs[signs == 0] = 1.0
        
        # Vectorized gradient computation
        l1_i = bary_clipped[_IDX_I, 0]
        l2_i = bary_clipped[_IDX_I, 1]
        l1_j = bary_clipped[_IDX_J, 0]
        l2_j = bary_clipped[_IDX_J, 1]
        l1_k = bary_clipped[_IDX_K, 0]
        l2_k = bary_clipped[_IDX_K, 1]
        
        w = 0.5 * weights * signs
        
        # Accumulate gradients
        grad_bary[_IDX_I, 0] += w * (l2_j - l2_k)
        grad_bary[_IDX_I, 1] += w * (l1_k - l1_j)
        grad_bary[_IDX_J, 0] += w * (l2_k - l2_i)
        grad_bary[_IDX_J, 1] += w * (l1_i - l1_k)
        grad_bary[_IDX_K, 0] += w * (l2_i - l2_j)
        grad_bary[_IDX_K, 1] += w * (l1_j - l1_i)
        
        # Gradient of penalty term
        diff = t - soft_min
        if diff > 20.0 / beta:
            dpenalty_dt = 2.0 * beta * diff
            dpenalty_dsm = -2.0 * beta * diff
        elif diff < -20.0 / beta:
            dpenalty_dt = 0.0
            dpenalty_dsm = 0.0
        else:
            sig = 1.0 / (1.0 + np.exp(-2.0 * beta * diff))
            dpenalty_dt = 2.0 * beta * sig
            dpenalty_dsm = -2.0 * beta * sig
        
        # Total gradient
        grad = np.zeros(2*N + 1)
        grad[:2*N] = -dpenalty_dsm * grad_bary.flatten()
        grad[2*N] = -1.0 + dpenalty_dt
        
        # Objective
        if diff > 20.0 / beta:
            penalty = beta * diff * diff
        elif diff < -20.0 / beta:
            penalty = 0.0
        else:
            penalty = np.log(1.0 + np.exp(2.0 * beta * diff)) / beta
        
        return -t + penalty, grad
    
    x0 = np.zeros(2*N + 1)
    x0[:2*N] = start_bary.flatten()
    x0[2*N] = t_init
    
    bounds = [(0.0, 1.0)] * (2*N) + [(-1.0, 1.0)]
    
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True,
                       bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12})
        return res.x, res.fun
    except Exception:
        return x0, np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_bary = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # FPS initializations
    for i in range(12):
        if time_left() < 50.0:
            break
        inits.append(_structured_init_fps(rng))
    
    # Random initializations
    for i in range(8):
        if time_left() < 50.0:
            break
        inits.append(_random_init_bary(rng))
    
    # Symmetric init: centroid + rings
    center_bary = np.array([1.0/3, 1.0/3])
    pts = [center_bary.copy()]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.25
        pt = center_bary + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.45
        pt = center_bary + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    inits.append(np.array(pts))
    
    # Optimize each initialization
    for init_bary in inits:
        if time_left() < 30.0:
            break
        
        init_bary = _project_to_bary_simplex(init_bary)
        score_init = _min_area_ratio_from_bary(init_bary)
        t_init = score_init
        
        # Multi-stage beta annealing
        for beta in [5.0, 10.0, 20.0, 40.0, 80.0]:
            if time_left() < 20.0:
                break
            try:
                x_opt, _ = _l_bfgs_softmin(init_bary, t_init, beta, maxiter=150)
                opt_bary = _project_to_bary_simplex(x_opt[:2*N].reshape(N, 2))
                score = _min_area_ratio_from_bary(opt_bary)
                if score > best_score:
                    best_score = score
                    best_bary = opt_bary.copy()
                # Use result as next start
                init_bary = opt_bary
                t_init = score
            except Exception:
                pass
    
    # Iterative refinement with perturbations
    if best_bary is not None:
        for scale in [0.03, 0.015, 0.008, 0.004]:
            if time_left() < 20.0:
                break
            for i in range(6):
                if time_left() < 15.0:
                    break
                try:
                    noise = rng.normal(0, scale, size=(N, 2))
                    perturbed = _project_to_bary_simplex(best_bary + noise)
                    score_p = _min_area_ratio_from_bary(perturbed)
                    t_init = score_p
                    
                    for beta in [20.0, 40.0, 80.0]:
                        if time_left() < 10.0:
                            break
                        x_opt, _ = _l_bfgs_softmin(perturbed, t_init, beta, maxiter=100)
                        opt_bary = _project_to_bary_simplex(x_opt[:2*N].reshape(N, 2))
                        score = _min_area_ratio_from_bary(opt_bary)
                        if score > best_score:
                            best_score = score
                            best_bary = opt_bary.copy()
                        perturbed = opt_bary
                        t_init = score
                except Exception:
                    pass
    
    # Final polish with high beta
    if best_bary is not None and time_left() > 15.0:
        try:
            score = _min_area_ratio_from_bary(best_bary)
            for beta in [80.0, 150.0, 300.0]:
                if time_left() < 10.0:
                    break
                x_opt, _ = _l_bfgs_softmin(best_bary, score, beta, maxiter=200)
                opt_bary = _project_to_bary_simplex(x_opt[:2*N].reshape(N, 2))
                score = _min_area_ratio_from_bary(opt_bary)
                if score > best_score:
                    best_score = score
                    best_bary = opt_bary.copy()
        except Exception:
            pass
    
    if best_bary is None:
        best_bary = np.full((N, 2), 1.0/3)
    
    points = _bary_to_cart(best_bary)
    
    # Safety: clip to triangle
    def _clip_to_triangle(pts):
        for i in range(len(pts)):
            x, y = pts[i]
            l2 = 2 * y / np.sqrt(3)
            l1 = x - y / np.sqrt(3)
            l0 = 1 - l1 - l2
            if l0 < 0 or l1 < 0 or l2 < 0:
                l0 = max(0, l0)
                l1 = max(0, l1)
                l2 = max(0, l2)
                s = l0 + l1 + l2
                if s > 0:
                    l0 /= s
                    l1 /= s
                    l2 /= s
                else:
                    l0 = 1; l1 = 0; l2 = 0
                pts[i] = l0 * V0 + l1 * V1 + l2 * V2
        return pts
    
    points = _clip_to_triangle(points)
    return np.array(points)
