# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _softmax(z):
    z2 = z - np.max(z)
    e = np.exp(z2)
    s = e.sum()
    if s < 1e-300 or not np.isfinite(s):
        return np.full_like(z, 1.0 / len(z))
    return e / s

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _entropy_grad(q):
    """
    Gradient of Shannon entropy H(q) = -sum(q * log2(q)) with respect to q.
    dH/dq_i = -log2(q_i) - 1/ln(2)
    """
    grad = np.zeros_like(q)
    m = q > 1e-300
    grad[m] = -np.log2(q[m]) - 1.0 / np.log(2.0)
    return grad

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _make_f_and_grad(idx_a, idx_b, idx_c, idx_d, smooth_beta=10.0):
    """
    Creates objective function and gradient for optimizing z (log-probs).
    Uses Log-Sum-Exp to smooth the max() operation for differentiability.
    """
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    def f_and_grad(z):
        P = _softmax(z).reshape(10, 10)
        pf = P.ravel()
        
        # Compute pooled distributions
        qa = np.bincount(idx_a, weights=pf, minlength=la)
        qb = np.bincount(idx_b, weights=pf, minlength=lb)
        qc = np.bincount(idx_c, weights=pf, minlength=lc)
        qd = np.bincount(idx_d, weights=pf, minlength=ld)
        
        # Entropies
        D = _entropy(qc)
        Ea = _entropy(qa)
        Eb = _entropy(qb)
        Ed = _entropy(qd)
        
        # Smooth Max: M_smooth = (1/beta) * log(sum(exp(beta * H)))
        # We want to maximize D / M. 
        # Let's optimize log(D) - log(M_smooth) to keep ratio stable, 
        # or just D - lambda * M? 
        # The task says "value is proportional to ratio". 
        # Optimizing D/M directly can be unstable if M->0.
        # Optimizing D - M is a common proxy, but the task specifically asks for ratio.
        # Let's try optimizing D / M directly but with a penalty if M is small.
        # Actually, let's optimize D - M * (1 + small_penalty) ? No.
        
        # Let's stick to D/M but use the smooth max for the gradient of M.
        # M = max(Ea, Eb, Ed)
        # logsumexp(beta * [Ea, Eb, Ed]) / beta
        H_vals = np.array([Ea, Eb, Ed])
        m_scaled = H_vals * smooth_beta
        m_max = np.max(m_scaled)
        lse = m_max + np.log(np.sum(np.exp(m_scaled - m_max)))
        M_smooth = lse / smooth_beta
        
        # Gradient of M_smooth w.r.t H_vals
        # d(M)/d(H_i) = exp(beta * H_i) / sum(exp(beta * H_j))
        weights = np.exp(m_scaled - m_max)
        weights /= np.sum(weights)
        dM_dH = weights
        
        # If M_smooth is too small, we are in a degenerate region.
        if M_smooth < 1e-6:
             return -1e9, np.zeros_like(z)
        
        val = D / M_smooth
        
        # Gradient calculation:
        # d(val)/dz = (1/M) * dD/dz - (D/M^2) * dM/dz
        # dD/dz = sum_k (dD/dqc_k) * (dqc_k/dz)
        # dqc_k/dz = sum_{i in cell k} dP_i/dz
        # dP_i/dz_j = P_i (delta_ij - P_j)
        
        # 1. Gradient of D w.r.t P
        grad_D_qc = _entropy_grad(qc)
        # Map back to P
        # dD/dP_i = grad_D_qc[idx_c[i]]
        grad_D_P = grad_D_qc[idx_c]
        
        # 2. Gradient of M_smooth w.r.t P
        # dM/dP = dM/dEa * dEa/dP + dM/dEb * dEb/dP + dM/dEd * dEd/dP
        grad_Ea_P = dM_dH[0] * _entropy_grad(qa)[idx_a]
        grad_Eb_P = dM_dH[1] * _entropy_grad(qb)[idx_b]
        grad_Ed_P = dM_dH[2] * _entropy_grad(qd)[idx_d]
        grad_M_P = grad_Ea_P + grad_Eb_P + grad_Ed_P
        
        # 3. Chain rule to z
        # dP/dz = P * (I - P^T) -> (dP/dz)_i = P_i * delta_ij - P_i * P_j
        # dF/dz_j = sum_i (dF/dP_i) * (P_i * delta_ij - P_i * P_j)
        #         = P_j * sum_i (dF/dP_i) * P_i - P_j * sum_i (dF/dP_i) * P_i ... wait
        # sum_i (dF/dP_i) * P_i * delta_ij = (dF/dP)_j * P_j
        # sum_i (dF/dP_i) * (-P_i * P_j) = -P_j * sum_i (dF/dP_i) * P_i
        
        dot_D = np.dot(grad_D_P, P.ravel())
        dot_M = np.dot(grad_M_P, P.ravel())
        
        grad_val_P = (1.0 / M_smooth) * grad_D_P - (D / (M_smooth ** 2)) * grad_M_P
        
        grad_val_z = grad_val_P * P.ravel() - P.ravel() * np.dot(grad_val_P, P.ravel())
        
        return -val, -grad_val_z # Minimize negative value
        
    return f_and_grad

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    # recompute M to ensure validity
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    # Multi-start L-BFGS-B
    best_z = None
    best_val = -1e9
    
    # Generate diverse initial points
    rng = np.random.default_rng(42)
    starts = []
    
    # 1. Uniform
    starts.append(np.zeros(100))
    
    # 2. Vertical lines (known ratio 1.0)
    for x0 in range(10):
        z = np.full(100, -10.0)
        for y in range(10):
            z[y*10 + x0] = 0.0
        starts.append(z)
        
    # 3. Random
    for _ in range(20):
        starts.append(rng.normal(0, 1.5, 100))
        
    # 4. Structured random (sparse)
    for _ in range(10):
        z = np.full(100, -10.0)
        mask = rng.random(100) < 0.3
        z[mask] = rng.normal(0, 1.0, 100)[mask]
        starts.append(z)

    f_and_grad = _make_f_and_grad(idx_a, idx_b, idx_c, idx_d, smooth_beta=20.0)
    
    for z0 in starts:
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(
                f_and_grad,
                z0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            z_opt = res.x
            P_opt = _softmax(z_opt).reshape(10, 10)
            val_opt = _value_from_P(P_opt, idx_a, idx_b, idx_c, idx_d)
            
            if val_opt > best_val:
                best_val = val_opt
                best_z = z_opt.copy()
                
        except Exception:
            continue
            
    # If no valid result found, fallback to uniform
    if best_z is None:
        return np.full((10, 10), 1.0 / 100.0)
        
    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P

# EVOLVE-BLOCK-END
