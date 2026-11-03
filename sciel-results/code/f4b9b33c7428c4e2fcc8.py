import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # dets = det([r1, r2, r3]) = r1 dot (r2 cross r3)
    # In barycentric coords, area is 0.5 * |det|
    # We compute the signed determinant for gradient purposes
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """
    Project barycentric coordinates to the simplex.
    """
    b = bary.copy()
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    b = np.clip(b, 0, 1)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _softmin_area_bary(p, beta):
    """
    Returns the softmin of the areas and the gradient with respect to p.
    p: (N, 3)
    beta: temperature parameter. Higher beta -> closer to hard min.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # Compute signed determinants
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    # Areas are 0.5 * |det|
    # To make it differentiable, we use the softmin of the areas.
    # However, |det| is not differentiable at 0. 
    # A common trick is to optimize the log-sum-exp of the negative areas? 
    # No, we want to MAXIMIZE the MIN area.
    # So we want to MINIMIZE the -MIN area.
    # -min(areas) = max(-areas).
    # Softmax approximation of max is log-sum-exp.
    # So we can use logsumexp(-beta * areas) / beta as a surrogate for max(-areas)/1?
    # Actually, let's just minimize the softmin of the areas directly? 
    # No, minimizing the softmin of areas is what we want? 
    # Wait. We want to MAXIMIZE min(areas).
    # Equivalent to MINIMIZING -min(areas).
    # -min(areas) is the max of (-areas).
    # logsumexp(-areas * beta) / beta approximates max(-areas) as beta -> inf.
    # So objective = logsumexp(-areas * beta) / beta.
    
    # But areas = 0.5 * |det|.
    # Let's use a differentiable approximation of |det|.
    # |x| approx sqrt(x^2 + eps).
    # Or simply, since we are inside the triangle, determinants can be positive or negative.
    # The area is the absolute value.
    # Let's use smooth_abs = sqrt(det^2 + eps).
    # Then area = 0.5 * smooth_abs.
    # Then we want to minimize logsumexp(-beta * areas) / beta.
    
    eps = 1e-12
    smooth_abs = np.sqrt(dets**2 + eps)
    areas = 0.5 * smooth_abs
    
    # Compute logsumexp(-beta * areas)
    # To avoid overflow/underflow, use the log-sum-exp trick
    x = -beta * areas
    x_max = np.max(x)
    lse = x_max + np.log(np.sum(np.exp(x - x_max)))
    
    soft_min_val = lse / beta
    
    # Gradient of soft_min_val with respect to p
    # d(soft_min_val)/d(p) = (1/beta) * d(lse)/d(p)
    # lse = log(sum(exp(x_i)))
    # d(lse)/dx_i = exp(x_i) / sum(exp(x_j)) = softmax(x)_i
    # So d(lse)/dp = sum_i (softmax(x)_i * dx_i/dp)
    
    weights = np.exp(x - x_max)
    weights /= np.sum(weights)
    
    # dx_i/dp
    # x_i = -beta * 0.5 * sqrt(det_i^2 + eps)
    # dx_i/d(det_i) = -beta * 0.5 * det_i / sqrt(det_i^2 + eps)
    
    grad_det = -beta * 0.5 * dets / smooth_abs
    
    # Now we need d(det_i)/d(p)
    # det_i = r1 dot (r2 cross r3)
    # This is a multilinear form.
    # We can compute the gradient of each det with respect to p.
    # p is (N, 3).
    # det_i depends on p[i1], p[i2], p[i3].
    
    # Let's compute the Jacobian of dets with respect to p.
    # J[k, idx] = d(det_k)/d(p[idx])
    # This is sparse.
    # For each triple k, the gradient of det_k w.r.t p is:
    # d(det)/dp[i1] = r2 cross r3
    # d(det)/dp[i2] = r3 cross r1
    # d(det)/dp[i3] = r1 cross r2
    
    # We can accumulate the gradient of the objective w.r.t p.
    # grad_obj = (1/beta) * sum_k (weights[k] * grad_det[k] * J_k)
    
    grad_p = np.zeros_like(p)
    
    # For each triple k:
    # grad_p[i1] += weights[k] * grad_det[k] * (r2 cross r3)
    # grad_p[i2] += weights[k] * grad_det[k] * (r3 cross r1)
    # grad_p[i3] += weights[k] * grad_det[k] * (r1 cross r2)
    
    # Vectorized approach:
    # We have r1, r2, r3 arrays of shape (M, 3) where M is num triples.
    # We need to scatter the gradients back to p.
    
    # Let's create arrays for the gradients of each term
    g1 = weights[:, np.newaxis] * grad_det[:, np.newaxis] * (r2[:, np.newaxis] * r3[:, np.newaxis]) # Placeholder, actual cross product needed
    # Let's do it properly.
    
    # grad_det is (M,)
    # weights is (M,)
    # coeff = weights * grad_det (M,)
    
    coeff = weights * grad_det
    
    # For each triple k:
    # term1 = coeff[k] * (r2[k] cross r3[k]) -> adds to p[i1[k]]
    # term2 = coeff[k] * (r3[k] cross r1[k]) -> adds to p[i2[k]]
    # term3 = coeff[k] * (r1[k] cross r2[k]) -> adds to p[i3[k]]
    
    # We can use np.add.at or bincount if indices were flat, but here indices are (M, 3)
    # Let's use a loop for clarity and correctness, M=165 is small.
    # Or vectorized with advanced indexing and reduction.
    
    # Let's try vectorized:
    # g_r1 = coeff[:, None] * np.cross(r2, r3) # (M, 3)
    # g_r2 = coeff[:, None] * np.cross(r3, r1) # (M, 3)
    # g_r3 = coeff[:, None] * np.cross(r1, r2) # (M, 3)
    
    # Now scatter:
    # grad_p[_TRIPLES[:, 0]] += g_r1
    # grad_p[_TRIPLES[:, 1]] += g_r2
    # grad_p[_TRIPLES[:, 2]] += g_r3
    
    # np.add.at is safe for duplicate indices (though here i<j<k so no duplicates per triple, 
    # but different triples might share indices, so we must accumulate)
    
    g_r1 = coeff[:, np.newaxis] * np.cross(r2, r3)
    g_r2 = coeff[:, np.newaxis] * np.cross(r3, r1)
    g_r3 = coeff[:, np.newaxis] * np.cross(r1, r2)
    
    np.add.at(grad_p, _TRIPLES[:, 0], g_r1)
    np.add.at(grad_p, _TRIPLES[:, 1], g_r2)
    np.add.at(grad_p, _TRIPLES[:, 2], g_r3)
    
    # The gradient of the objective (which is soft_min_val) is grad_p / beta
    # Wait, soft_min_val = lse / beta.
    # d(soft_min_val)/dp = (1/beta) * d(lse)/dp
    # d(lse)/dp = sum_k (softmax(x)_k * dx_k/dp)
    # We computed coeff = softmax(x) * dx_k/d(det_k)
    # So d(lse)/dp = sum_k (coeff_k * J_k) which is what we accumulated in grad_p.
    # So the final gradient is grad_p / beta.
    
    grad_obj = grad_p / beta
    
    return soft_min_val, grad_obj

def _objective_and_grad(x_flat, beta):
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    val, grad = _softmin_area_bary(bary, beta)
    return val, grad.flatten()

def _generate_inits():
    inits = []
    # Random Dirichlet
    for seed in range(10):
        r = np.random.default_rng(seed)
        inits.append(r.dirichlet([1, 1, 1], size=N))
        
    # Symmetric init
    center = np.array([1/3, 1/3, 1/3])
    # 6 points around center in cartesian, convert to bary?
    # Let's just use a few structured ones
    # 1. Grid
    # 2. Corners
    corners = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9]
    ])
    r = np.random.default_rng(123)
    rest = r.dirichlet([1, 1, 1], size=N-3)
    inits.append(np.vstack([corners, rest]))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_area = 0.0
    best_bary = None
    
    inits = _generate_inits()
    rng = np.random.default_rng(42)
    rng.shuffle(inits)
    
    # Adaptive beta: start low (smooth), increase later (sharp)
    # But since we have limited time, maybe just use a high beta from the start?
    # Or a schedule.
    # Let's try a few betas.
    betas = [10, 50, 200, 1000]
    
    for init in inits:
        if time.time() > deadline - 5.0:
            break
            
        x0 = init.flatten()
        
        # Try different betas
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                res = minimize(
                    _objective_and_grad,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True, # We return (val, grad)
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                refined_bary = _project_bary(res.x.reshape(N, 3))
                # Evaluate the TRUE min area (not softmin) to compare
                true_area = _min_area_bary(refined_bary)
                
                if true_area > best_area:
                    best_area = true_area
                    best_bary = refined_bary.copy()
                    
            except Exception:
                continue
                
    if best_bary is None:
        rng = np.random.default_rng(0)
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
        
    return _bary_to_cart(best_bary)
