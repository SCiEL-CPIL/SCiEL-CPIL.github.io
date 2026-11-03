"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_triangle_area_fast(points):
    """Fully vectorized: compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = points[j] - points[i]  # (2,)
            diff_k = points - points[i]  # (n, 2)
            cross = d[0] * diff_k[:, 1] - d[1] * diff_k[:, 0]  # (n,)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross[mask])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]

    y = np.clip(y, 0, np.sqrt(3) / 2)

    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)

    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)

    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2).
    Barycentric: (a,b,c) means point = a*(0,0) + b*(1,0) + c*(0.5, sqrt(3)/2)
    So x = b + 0.5*c, y = c*sqrt(3)/2"""
    b, c = bary[:, 1], bary[:, 2]
    x = b + 0.5 * c
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    x = points[:, 0]
    y = points[:, 1]
    c = y / (np.sqrt(3) / 2)
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _objective_and_grad_bary(flat):
    """
    Objective and gradient for barycentric optimization.
    flat: (2*N,) array of (b_i, c_i) for i=0..N-1.
    a_i = 1 - b_i - c_i.
    Minimizes -SoftMin(|Det(B_ijk)|) + penalty for a_i < 0.
    """
    bary_bc = flat.reshape(N, 2)
    b = bary_bc[:, 0]
    c = bary_bc[:, 1]
    a = 1.0 - b - c
    
    # Construct full barycentric matrix B (N, 3)
    B = np.column_stack([a, b, c])
    
    # Penalty for a < 0
    penalty = 0.0
    grad_penalty = np.zeros_like(flat)
    mask_neg = a < 0
    if np.any(mask_neg):
        # Use squared penalty for smoothness
        neg_a = a[mask_neg]
        penalty += np.sum(neg_a ** 2) * 100.0
        # d/da (a^2 * 100) = 200 * a
        # d/da (a) = -1 for b, -1 for c
        # So grad_b += 200 * a * (-1) = -200 * a
        # grad_c += 200 * a * (-1) = -200 * a
        grad_penalty[mask_neg, 0] = -200.0 * neg_a
        grad_penalty[mask_neg, 1] = -200.0 * neg_a

    # Compute all triangle determinants and their gradients
    # We want to maximize min |Det|. So minimize -SoftMin(|Det|).
    # SoftMin(x) = -tau * log(sum(exp(-x/tau)))
    # Let x_k = |Det_k|.
    # L = -SoftMin(x) = tau * log(sum(exp(-x_k/tau)))
    # dL/dx_k = (exp(-x_k/tau) / sum(exp(-x_j/tau))) * (-1/tau)
    # dL/dB = sum_k (dL/dx_k * dx_k/dB)
    
    tau = 0.05
    triplets = []
    dets = []
    grads_det = [] # List of (N,3) gradients for each triplet
    
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                # Extract rows
                Bi = B[i]
                Bj = B[j]
                Bk = B[k]
                
                # Determinant of [[a_i, b_i, c_i], [a_j, b_j, c_j], [a_k, b_k, c_k]]
                # det = a_i(b_j c_k - c_j b_k) - b_i(a_j c_k - c_j a_k) + c_i(a_j b_k - b_j a_k)
                det_val = (Bi[0] * (Bj[1] * Bk[2] - Bj[2] * Bk[1]) - 
                           Bi[1] * (Bj[0] * Bk[2] - Bj[2] * Bk[0]) + 
                           Bi[2] * (Bj[0] * Bk[1] - Bj[1] * Bk[0]))
                
                abs_det = np.abs(det_val)
                
                # Gradient of |det| w.r.t. B
                # If det > 0, grad = grad(det). If det < 0, grad = -grad(det).
                sign = 1.0 if det_val > 0 else -1.0
                
                # Cofactors for row i (derivatives w.r.t. Bi)
                # d(det)/d(Bi[0]) = Bj[1]*Bk[2] - Bj[2]*Bk[1]
                # d(det)/d(Bi[1]) = -(Bj[0]*Bk[2] - Bj[2]*Bk[0])
                # d(det)/d(Bi[2]) = Bj[0]*Bk[1] - Bj[1]*Bk[0]
                
                grad_Bi = np.array([
                    Bj[1] * Bk[2] - Bj[2] * Bk[1],
                    -(Bj[0] * Bk[2] - Bj[2] * Bk[0]),
                    Bj[0] * Bk[1] - Bj[1] * Bk[0]
                ])
                
                # Cofactors for row j (derivatives w.r.t. Bj)
                # d(det)/d(Bj[0]) = -(Bi[1]*Bk[2] - Bi[2]*Bk[1])
                # d(det)/d(Bj[1]) = (Bi[0]*Bk[2] - Bi[2]*Bk[0])
                # d(det)/d(Bj[2]) = -(Bi[0]*Bk[1] - Bi[1]*Bk[0])
                
                grad_Bj = np.array([
                    -(Bi[1] * Bk[2] - Bi[2] * Bk[1]),
                    (Bi[0] * Bk[2] - Bi[2] * Bk[0]),
                    -(Bi[0] * Bk[1] - Bi[1] * Bk[0])
                ])
                
                # Cofactors for row k (derivatives w.r.t. Bk)
                # d(det)/d(Bk[0]) = (Bi[1]*Bj[2] - Bi[2]*Bj[1])
                # d(det)/d(Bk[1]) = -(Bi[0]*Bj[2] - Bi[2]*Bj[0])
                # d(det)/d(Bk[2]) = (Bi[0]*Bj[1] - Bi[1]*Bj[0])
                
                grad_Bk = np.array([
                    (Bi[1] * Bj[2] - Bi[2] * Bj[1]),
                    -(Bi[0] * Bj[2] - Bi[2] * Bj[0]),
                    (Bi[0] * Bj[1] - Bi[1] * Bj[0])
                ])
                
                grad_abs_det = np.zeros((N, 3))
                grad_abs_det[i] = sign * grad_Bi
                grad_abs_det[j] = sign * grad_Bj
                grad_abs_det[k] = sign * grad_Bk
                
                triplets.append((i, j, k))
                dets.append(abs_det)
                grads_det.append(grad_abs_det)
    
    dets = np.array(dets)
    
    # Compute SoftMin weights
    # x = dets
    # weights = exp(-x/tau) / sum(exp(-x/tau))
    # To avoid overflow, subtract max
    max_det = np.max(dets)
    exp_terms = np.exp(-(dets - max_det) / tau)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective: tau * log(sum_exp) + tau * log(exp(max_det/tau)) ? 
    # SoftMin(x) = -tau * log(sum(exp(-x/tau)))
    # L = -SoftMin(x) = tau * log(sum(exp(-x/tau)))
    # sum(exp(-x/tau)) = exp(max/tau) * sum(exp(-(x-max)/tau))
    # log(sum) = max/tau + log(sum_exp)
    # L = tau * (max/tau + log(sum_exp)) = max + tau * log(sum_exp)
    # Wait, if x is large, max is large.
    # Let's check: if x = [10, 10], tau=1.
    # SoftMin = -1 * log(2 * exp(-10)) = -log(2) + 10.
    # L = -SoftMin = log(2) - 10.
    # Formula: max + tau * log(sum_exp) = 10 + 1 * log(2 * exp(0)) = 10 + log(2).
    # Sign error?
    # SoftMin is approx min. If x=[10,10], min=10. SoftMin ~ 10.
    # We want to MINIMIZE -min. So we want to MINIMIZE -10.
    # My L formula gave +10.
    # Ah, SoftMin(x) = -tau log(sum exp(-x/tau)).
    # For x=[10,10], sum exp(-10) = 2e-10. log = log(2)-10.
    # SoftMin = -1 * (log(2)-10) = 10 - log(2). Correct.
    # We want to minimize -SoftMin.
    # Obj = -SoftMin = log(2) - 10.
    # Gradient of Obj w.r.t x:
    # d/dx (-SoftMin) = - d/dx SoftMin.
    # d SoftMin / dx_k = -tau * (1/sum) * exp(-x_k/tau) * (-1/tau) = exp(-x_k/tau) / sum = w_k.
    # So d SoftMin / dx = w.
    # d Obj / dx = -w.
    # So we need to subtract the weighted gradients.
    
    obj_softmin = -(max_det + tau * np.log(sum_exp)) # This is -SoftMin? No.
    # SoftMin = max_det + tau * log(sum_exp) ??
    # sum(exp(-x/tau)) = exp(-max/tau) * sum(exp(-(x-max)/tau)).
    # log(sum) = -max/tau + log(sum_exp).
    # SoftMin = -tau * (-max/tau + log(sum_exp)) = max - tau * log(sum_exp).
    # So -SoftMin = -max + tau * log(sum_exp).
    
    obj_value = -max_det + tau * np.log(sum_exp) + penalty
    
    # Gradient of -SoftMin w.r.t B
    # d(-SoftMin)/dB = - sum_k w_k * (d|Det_k|/dB)
    # Note: w_k is the weight for the k-th triplet in the soft-min sum.
    
    grad_B_total = np.zeros((N, 3))
    for idx, g in enumerate(grads_det):
        grad_B_total += -weights[idx] * g
        
    # Chain rule: flat = [b_0, c_0, b_1, c_1, ...]
    # B = [a, b, c]
    # a = 1 - b - c
    # d(B[:,0])/d(b_i) = -1, d(B[:,1])/d(b_i) = 1
    # d(B[:,0])/d(c_i) = -1, d(B[:,2])/d(c_i) = 1
    
    grad_flat = np.zeros_like(flat)
    for i in range(N):
        # grad_flat[2*i] is d/d(b_i)
        # grad_flat[2*i+1] is d/d(c_i)
        
        # Sum contributions from grad_B_total[i]
        # dL/db_i = -grad_B_total[i,0] + grad_B_total[i,1]
        # dL/dc_i = -grad_B_total[i,0] + grad_B_total[i,2]
        
        grad_flat[2*i] = -grad_B_total[i, 0] + grad_B_total[i, 1] + grad_penalty[2*i]
        grad_flat[2*i+1] = -grad_B_total[i, 0] + grad_B_total[i, 2] + grad_penalty[2*i+1]
        
    return obj_value, grad_flat


def _structured_init(seed_idx):
    """Create diverse structured initial configurations based on symmetry."""
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2  # height of triangle

    configs = []

    # Config 0: Symmetric hexagonal-like arrangement
    configs.append(np.array([
        [0.05, 0.05],
        [0.95, 0.05],
        [0.5, h - 0.05],
        [0.5, 0.05],
        [0.25, h / 2],
        [0.75, h / 2],
        [0.5, h / 3],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.5, 0.45],
        [0.4, 0.35],
    ]))

    # Config 1: Three rows of 4-4-3
    configs.append(np.array([
        [0.1, 0.08], [0.35, 0.08], [0.65, 0.08], [0.9, 0.08],
        [0.2, 0.35], [0.4, 0.35], [0.6, 0.35], [0.8, 0.35],
        [0.3, 0.6], [0.5, 0.6], [0.7, 0.6],
    ]))

    # Config 2: Concentric rings
    configs.append(np.array([
        [0.5, 0.05],
        [0.2, 0.25], [0.8, 0.25],
        [0.5, 0.45],
        [0.3, 0.55], [0.7, 0.55],
        [0.1, 0.15], [0.9, 0.15],
        [0.4, 0.7], [0.6, 0.7],
        [0.5, 0.85],
    ]))

    # Config 3: Grid-based
    configs.append(np.array([
        [0.15, 0.1], [0.4, 0.1], [0.65, 0.1], [0.85, 0.1],
        [0.25, 0.35], [0.5, 0.35], [0.75, 0.35],
        [0.35, 0.6], [0.55, 0.6], [0.75, 0.6],
        [0.45, 0.8],
    ]))

    # Config 4: Random but structured
    rng = np.random.default_rng(seed_idx * 1000 + 42)
    pts = []
    # Place 3 near corners
    pts.append([0.08, 0.08])
    pts.append([0.92, 0.08])
    pts.append([0.5, h - 0.08])
    # Place 8 in interior with some structure
    for i in range(8):
        angle = i * 2 * np.pi / 8
        r = 0.2 + 0.1 * (i % 2)
        cx = 0.5 + r * np.cos(angle)
        cy = h / 3 + r * 0.8 * np.sin(angle)
        pts.append([cx, cy])
    configs.append(np.array(pts))

    return configs


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s

    rng = np.random.default_rng(42)

    best_points = None
    best_score = -float('inf')

    # Get structured initial configurations
    inits = _structured_init(0)

    # Add grid-based inits
    for jitter in [0.03, 0.05, 0.07]:
        # Generate random points inside triangle
        pts = []
        for _ in range(N):
            x = rng.uniform(0, 1)
            y = rng.uniform(0, 1)
            if x + y <= 1:
                # Map to triangle
                # This is a rejection sampling in barycentric
                # b = x, c = y, a = 1-x-y
                # x_cart = b + 0.5c, y_cart = c*sqrt(3)/2
                xc = x + 0.5 * y
                yc = y * (np.sqrt(3) / 2)
                pts.append([xc, yc])
        if len(pts) == N:
            pts = np.array(pts)
            pts = pts + rng.normal(0, jitter, size=(N, 2))
            pts = _clip_to_triangle(pts)
            inits.append(pts)

    # Multi-start L-BFGS-B optimization
    for start_idx, points_init in enumerate(inits):
        if time.time() > deadline - 10:
            break

        # Convert to barycentric (b, c)
        bary = _cart_to_bary(points_init)
        b = bary[:, 1]
        c = bary[:, 2]
        flat_init = np.column_stack([b, c]).flatten()

        # Run L-BFGS-B
        try:
            result = minimize(
                _objective_and_grad_bary,
                flat_init,
                method='L-BFGS-B',
                jac=True, # Use analytic gradient
                options={
                    'maxiter': 500,
                    'gtol': 1e-8,
                    'ftol': 1e-10,
                }
            )

            # Convert result back to Cartesian
            bary_opt = np.column_stack([
                1.0 - result.x[0::2] - result.x[1::2], 
                result.x[0::2], 
                result.x[1::2]
            ])
            points_opt = _bary_to_cart(bary_opt)
            points_opt = _clip_to_triangle(points_opt)

            # Evaluate
            score = _min_triangle_area_fast(points_opt)

            if score > best_score:
                best_score = score
                best_points = points_opt.copy()

        except Exception:
            pass

    # If no valid result, use a fallback
    if best_points is None:
        best_points = _clip_to_triangle(np.array([
            [0.05, 0.05], [0.95, 0.05], [0.5, np.sqrt(3) / 2 - 0.05],
            [0.5, 0.05], [0.25, np.sqrt(3) / 4], [0.75, np.sqrt(3) / 4],
            [0.5, np.sqrt(3) / 6], [0.3, 0.3], [0.7, 0.3],
            [0.5, 0.5], [0.4, 0.4]
        ]))
        best_score = _min_triangle_area_fast(best_points)

    # Final refinement: run one more L-BFGS-B from the best point
    if time.time() < deadline - 5:
        try:
            bary_best = _cart_to_bary(best_points)
            flat_best = np.column_stack([bary_best[:, 1], bary_best[:, 2]]).flatten()
            
            result = minimize(
                _objective_and_grad_bary,
                flat_best,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1000,
                    'gtol': 1e-10,
                    'ftol': 1e-12,
                }
            )
            
            bary_final = np.column_stack([
                1.0 - result.x[0::2] - result.x[1::2], 
                result.x[0::2], 
                result.x[1::2]
            ])
            points_final = _bary_to_cart(bary_final)
            points_final = _clip_to_triangle(points_final)
            
            score_final = _min_triangle_area_fast(points_final)
            if score_final > best_score:
                best_points = points_final
                best_score = score_final
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
