# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _shannon_entropy(probs):
    """Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))

def _get_projection_values(n, r):
    """Precompute the projection values x + r*y for all cells."""
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    vals = xs + r * ys
    unique_vals, inverse = np.unique(vals, return_inverse=True)
    return unique_vals, inverse, vals

# Precompute for n=10
N = 10
PROJ_DATA = {}
for r in [-1.0, 0.0, 1.0, 2.0]:
    unique_vals, inverse, vals = _get_projection_values(N, r)
    PROJ_DATA[r] = {
        'unique_vals': unique_vals,
        'inverse': inverse,
        'vals': vals,
        'num_bins': len(unique_vals)
    }

def _compute_proj_probs(P, r):
    """Compute the projected probability distribution for slope r."""
    inverse = PROJ_DATA[r]['inverse']
    num_bins = PROJ_DATA[r]['num_bins']
    proj = np.bincount(inverse.flatten(), weights=P.flatten(), minlength=num_bins)
    return proj

def _entropy_and_gradient(P, r):
    """
    Compute Shannon entropy H and its gradient with respect to P for projection r.
    P is a flattened array of shape (100,).
    """
    proj = _compute_proj_probs(P, r)
    
    mask = proj > 1e-15
    if not np.any(mask):
        return 0.0, np.zeros_like(P)
    
    p = proj[mask]
    H = float(-np.sum(p * np.log2(p)))
    
    grad_proj = np.zeros_like(proj)
    grad_proj[mask] = -(1.0 / np.log(2.0)) * (1.0 + np.log2(p))
    
    inverse = PROJ_DATA[r]['inverse']
    grad_P = grad_proj[inverse.flatten()]
    
    return H, grad_P

def _ratio_from_P(P):
    """Compute D/M ratio from a valid probability array."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return 0.0
    P_norm = P / s
    
    H_m1, _ = _entropy_and_gradient(P_norm.flatten(), -1.0)
    H_0, _ = _entropy_and_gradient(P_norm.flatten(), 0.0)
    H_1, _ = _entropy_and_gradient(P_norm.flatten(), 1.0)
    H_2, _ = _entropy_and_gradient(P_norm.flatten(), 2.0)
    
    D = H_m1
    M = max(H_0, H_1, H_2)
    
    if M < 1e-9 or D < 1e-12:
        return 0.0
    
    return D / M

def _objective_and_grad(P_flat, soft_beta=50.0):
    """
    Objective: -D / M_smooth
    D = H_{-1}
    M_smooth = soft-max of (H_0, H_1, H_2) via log-sum-exp for smoothness
    """
    P = P_flat.reshape(N, N)
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return 1e6, np.zeros_like(P_flat)
    P_norm = P / s
    
    H_m1, grad_m1 = _entropy_and_gradient(P_norm.flatten(), -1.0)
    H_0, grad_0 = _entropy_and_gradient(P_norm.flatten(), 0.0)
    H_1, grad_1 = _entropy_and_gradient(P_norm.flatten(), 1.0)
    H_2, grad_2 = _entropy_and_gradient(P_norm.flatten(), 2.0)
    
    D = H_m1
    
    # Soft-max for smoothness: M = (1/beta) * log(sum(exp(beta * H_i)))
    H_vals = np.array([H_0, H_1, H_2])
    grad_vals = [grad_0, grad_1, grad_2]
    
    if np.max(H_vals) < 1e-9 or D < 1e-12:
        return 1e6, np.zeros_like(P_flat)
    
    # Log-sum-exp for numerical stability
    h_max = np.max(H_vals)
    exp_vals = np.exp(beta * (H_vals - h_max))
    log_sum_exp = h_max + np.log(np.sum(exp_vals)) / beta
    M = log_sum_exp
    
    # Weights for soft-max
    w = exp_vals / np.sum(exp_vals)
    
    val = -D / M
    
    # Gradient of M w.r.t. P
    grad_M = np.zeros_like(P_flat)
    for i in range(3):
        grad_M += w[i] * grad_vals[i]
    
    grad_D = grad_m1
    
    grad_val = (D * grad_M - M * grad_D) / (M * M)
    
    return val, grad_val

def _make_initial_patterns(n):
    """Generate diverse initial patterns targeting high H_{-1} / low max(H_0,H_1,H_2)."""
    patterns = []
    
    # 1. Uniform
    patterns.append(np.ones((n, n)) / (n * n))
    
    # 2. Diagonal x=y (high H_{-1} since x-y=0 for all)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 1.0 / n
    patterns.append(P)
    
    # 3. Anti-diagonal x+y=n-1
    P = np.zeros((n, n))
    for i in range(n):
        P[i, n - 1 - i] = 1.0 / n
    patterns.append(P)
    
    # 4. Concentrated on parallel lines with slope -1 (maximize H_{-1})
    for offset in range(-4, 5):
        P = np.zeros((n, n))
        count = 0
        for y in range(n):
            x = y + offset
            if 0 <= x < n:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            patterns.append(P.copy())
    
    # 5. Horizontal lines (slope 0) - low H_0
    for y0 in range(0, n, 2):
        P = np.zeros((n, n))
        P[y0, :] = 1.0
        P /= P.sum()
        patterns.append(P)
    
    # 6. Diagonal lines (slope 1) - low H_1
    for offset in range(-4, 5, 2):
        P = np.zeros((n, n))
        count = 0
        for y in range(n):
            x = y + offset
            if 0 <= x < n:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            patterns.append(P.copy())
    
    # 7. Lines with slope 2 - low H_2
    for offset in range(-6, 7, 2):
        P = np.zeros((n, n))
        count = 0
        for y in range(n):
            x = 2 * y + offset
            if 0 <= x < n:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            patterns.append(P.copy())
    
    # 8. Multi-anti-diagonal: spread mass across several x-y values
    # This targets high H_{-1} while keeping H_0, H_1, H_2 moderate
    for num_diags in [2, 3, 4, 5]:
        for start_off in range(0, n - num_diags + 1):
            P = np.zeros((n, n))
            count = 0
            for off in range(start_off, start_off + num_diags):
                for y in range(n):
                    x = y + off
                    if 0 <= x < n:
                        P[y, x] = 1.0
                        count += 1
            if count > 0:
                P /= count
                patterns.append(P.copy())
    
    # 9. Checkerboard-like: alternate on anti-diagonals
    for parity in [0, 1]:
        P = np.zeros((n, n))
        count = 0
        for y in range(n):
            for x in range(n):
                if (x - y) % 2 == parity:
                    P[y, x] = 1.0
                    count += 1
        if count > 0:
            P /= count
            patterns.append(P.copy())
    
    # 10. Random sparse
    rng = np.random.RandomState(42)
    for _ in range(8):
        P = np.zeros((n, n))
        k = rng.randint(5, 18)
        indices = rng.choice(n * n, size=k, replace=False)
        P.flatten()[indices] = 1.0
        P /= P.sum()
        patterns.append(P)
        
    # 11. Random dense
    for _ in range(5):
        P = rng.rand(n, n)
        P /= P.sum()
        patterns.append(P)
        
    # 12. Block patterns
    for size in [2, 3, 4]:
        for start_y in range(0, n - size + 1, 2):
            for start_x in range(0, n - size + 1, 2):
                P = np.zeros((n, n))
                P[start_y:start_y+size, start_x:start_x+size] = 1.0
                P /= P.sum()
                patterns.append(P)
                
    # 13. Two-line patterns
    for offset1 in range(-3, 4, 2):
        for offset2 in range(offset1 + 2, 5, 2):
            P = np.zeros((n, n))
            count = 0
            for y in range(n):
                for off in [offset1, offset2]:
                    x = y + off
                    if 0 <= x < n:
                        P[y, x] = 1.0
                        count += 1
            if count > 0:
                P /= count
                patterns.append(P.copy())
    
    # 14. Triangular patterns: weight increases along anti-diagonals
    for direction in [1, -1]:
        P = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                val = x - y
                if direction == 1:
                    P[y, x] = max(0, val + n)
                else:
                    P[y, x] = max(0, n - val)
        s = P.sum()
        if s > 1e-12:
            P /= s
            patterns.append(P.copy())
    
    return patterns

def _coordinate_descent(P, max_iters=200, tol=1e-10, deadline=None):
    """
    Coordinate descent: for each cell, find optimal weight via 1D golden-section search.
    """
    n = P.shape[0]
    P = P.copy()
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return P
    P /= s
    
    best_score = _ratio_from_P(P)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for y in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for x in range(n):
                if deadline is not None and time.time() > deadline:
                    break
                
                old_val = P[y, x]
                rest_sum = 1.0 - old_val
                if rest_sum < 1e-15:
                    continue
                
                # Golden section search for optimal P[y,x] in [0, 1]
                # We need to find t in [0, 1] that maximizes ratio when P[y,x] = t * (1) / (1 + t*rest_sum/(1-old_val))... 
                # Simpler: parametrize as P[y,x] = t, others scaled by (1-t)/(1-old_val)
                # Actually just search t in [0, min(1, 10*old_val+0.01)]
                lo = 0.0
                hi = min(1.0, old_val * 5 + 0.02) if old_val > 0 else 0.02
                
                # Evaluate at a few points
                best_t = old_val
                best_local = best_score
                
                # Golden section
                gr = (np.sqrt(5) - 1) / 2
                a, b = lo, hi
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                P_c = P.copy()
                P_d = P.copy()
                P_c[y, x] = c
                P_d[y, x] = d
                # Renormalize: scale other cells
                if 1 - c > 1e-15:
                    mask = np.ones((n, n), dtype=bool)
                    mask[y, x] = False
                    P_c[mask] *= (1 - c) / (1 - old_val)
                    P_c[y, x] = c
                if 1 - d > 1e-15:
                    mask = np.ones((n, n), dtype=bool)
                    mask[y, x] = False
                    P_d[mask] *= (1 - d) / (1 - old_val)
                    P_d[y, x] = d
                
                fc = _ratio_from_P(P_c)
                fd = _ratio_from_P(P_d)
                
                for _ in range(30):
                    if deadline is not None and time.time() > deadline:
                        break
                    if fc > fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        P_c[y, x] = c
                        mask = np.ones((n, n), dtype=bool)
                        mask[y, x] = False
                        P_c[mask] = P[mask] * (1 - c) / (1 - old_val)
                        fc = _ratio_from_P(P_c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        P_d[y, x] = d
                        mask = np.ones((n, n), dtype=bool)
                        mask[y, x] = False
                        P_d[mask] = P[mask] * (1 - d) / (1 - old_val)
                        fd = _ratio_from_P(P_d)
                
                # Best of endpoints and golden section
                for t in [a, b, c, d, (a+b)/2]:
                    if t < 0 or t > 1:
                        continue
                    P_t = P.copy()
                    P_t[y, x] = t
                    if 1 - t > 1e-15:
                        mask = np.ones((n, n), dtype=bool)
                        mask[y, x] = False
                        P_t[mask] = P[mask] * (1 - t) / (1 - old_val)
                    else:
                        P_t[mask] = 0
                    ft = _ratio_from_P(P_t)
                    if ft > best_local:
                        best_local = ft
                        best_t = t
                
                if best_t != old_val:
                    P[y, x] = best_t
                    if 1 - best_t > 1e-15:
                        mask = np.ones((n, n), dtype=bool)
                        mask[y, x] = False
                        P[mask] = P[mask] * (1 - best_t) / (1 - old_val)
                    else:
                        P[mask] = 0
                    P = np.clip(P, 0, None)
                    s2 = P.sum()
                    if s2 > 1e-15:
                        P /= s2
                    improved = True
        
        if not improved:
            break
    
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 580
    
    initial_patterns = _make_initial_patterns(n)
    
    best_P = initial_patterns[0].copy()
    best_score = _ratio_from_P(best_P)
    
    constraints = [{'type': 'eq', 'fun': lambda P: np.sum(P) - 1.0, 'jac': lambda P: np.ones_like(P)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    # Sort patterns by initial score, take top ones
    scored_patterns = [(p, _ratio_from_P(p)) for p in initial_patterns]
    scored_patterns.sort(key=lambda x: -x[1])
    
    num_starts = min(len(scored_patterns), 20)
    
    for i in range(num_starts):
        if time.time() > deadline - 80:
            break
            
        x0 = scored_patterns[i][0].flatten().copy()
        
        try:
            result = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
            )
            
            P_candidate = result.x.reshape(n, n)
            P_candidate = np.clip(P_candidate, 0, None)
            s = P_candidate.sum()
            if s > 1e-12:
                P_candidate /= s
                score = _ratio_from_P(P_candidate)
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
                    
        except Exception:
            continue
            
    # Coordinate descent polish
    if time.time() < deadline - 40:
        P_cd = _coordinate_descent(best_P, max_iters=100, deadline=deadline - 20)
        score_cd = _ratio_from_P(P_cd)
        if score_cd > best_score:
            best_score = score_cd
            best_P = P_cd.copy()
    
    # Nelder-Mead polish (derivative-free, handles non-smoothness)
    if time.time() < deadline - 30:
        rng = np.random.RandomState(123)
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            
            noise_scale = 0.03
            noise = rng.randn(n, n) * noise_scale
            P_start = best_P + noise
            P_start = np.clip(P_start, 0, None)
            s = P_start.sum()
            if s > 1e-12:
                P_start /= s
            else:
                continue
            
            try:
                result = minimize(
                    lambda x: -_ratio_from_P(x.reshape(n, n)),
                    P_start.flatten().copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                P_candidate = result.x.reshape(n, n)
                P_candidate = np.clip(P_candidate, 0, None)
                s = P_candidate.sum()
                if s > 1e-12:
                    P_candidate /= s
                    score = _ratio_from_P(P_candidate)
                    if score > best_score:
                        best_score = score
                        best_P = P_candidate.copy()
            except Exception:
                continue
    
    # Final high-precision SLSQP polish
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective_and_grad,
                best_P.flatten().copy(),
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            
            P_candidate = result.x.reshape(n, n)
            P_candidate = np.clip(P_candidate, 0, None)
            s = P_candidate.sum()
            if s > 1e-12:
                P_candidate /= s
                score = _ratio_from_P(P_candidate)
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
        except Exception:
            pass
    
    # Final coordinate descent
    if time.time() < deadline - 5:
        P_cd = _coordinate_descent(best_P, max_iters=50, deadline=deadline - 1)
        score_cd = _ratio_from_P(P_cd)
        if score_cd > best_score:
            best_score = score_cd
            best_P = P_cd.copy()
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-12:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= s
    
    return best_P

# EVOLVE-BLOCK-END
