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

def _entropy_and_grad(q, idx, P):
    """
    Compute Shannon entropy of distribution of idx weighted by P,
    and its gradient w.r.t P.
    q is the aggregated probabilities for each bin.
    """
    # q = bincount(idx, weights=P)
    # H = -sum(q * log2(q))
    # dH/dP_i = -log2(q[idx_i]) - 1/ln(2)
    
    # Clip q to avoid log(0)
    q_safe = np.clip(q, 1e-300, None)
    H = -np.sum(q * np.log2(q_safe))
    
    # Gradient w.r.t P
    # grad_H[i] = -log2(q[idx[i]]) - 1/ln(2)
    # We need to map back to P indices
    # q_idx = q[idx]
    # grad_H = -np.log2(q_idx) - 1/np.log(2)
    
    # However, if q[idx[i]] is very small, log is large negative.
    # This is correct.
    
    # Note: If we use softmax, P is never 0.
    # But if a bin has 0 mass, q=0.
    # If P_i > 0, then q[idx[i]] >= P_i > 0.
    # So q[idx[i]] is never 0 for any i with P_i > 0.
    
    q_mapped = q[idx]
    q_mapped_safe = np.clip(q_mapped, 1e-300, None)
    grad_H = -np.log2(q_mapped_safe) - 1.0 / np.log(2.0)
    
    return H, grad_H

def _value_and_grad(z, idx_a, idx_b, idx_c, idx_d):
    P = _softmax(z)
    
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    qa = np.bincount(idx_a, weights=P, minlength=la)
    qb = np.bincount(idx_b, weights=P, minlength=lb)
    qc = np.bincount(idx_c, weights=P, minlength=lc)
    qd = np.bincount(idx_d, weights=P, minlength=ld)
    
    H0, g0 = _entropy_and_grad(qa, idx_a, P)
    H1, g1 = _entropy_and_grad(qb, idx_b, P)
    H2, g2 = _entropy_and_grad(qd, idx_d, P)
    Hm, gm = _entropy_and_grad(qc, idx_c, P) # D = Hm
    
    M = max(H0, H1, H2)
    
    if M < 1e-12:
        return -1e9, np.zeros_like(z)
    
    val = Hm / M
    
    # Gradient of val = Hm / M
    # d(val)/dP = (M * dHm/dP - Hm * dM/dP) / M^2
    # dM/dP is the gradient of the active max component.
    # If H0 is max, dM/dP = g0.
    # If H1 is max, dM/dP = g1.
    # If H2 is max, dM/dP = g2.
    # If tie, we can pick any, or average. Let's pick the one with largest value.
    # To be safe against numerical ties, we can use a soft max or just pick the first max.
    
    if H0 >= H1 and H0 >= H2:
        gM = g0
    elif H1 >= H0 and H1 >= H2:
        gM = g1
    else:
        gM = g2
        
    grad_val_P = (M * gm - Hm * gM) / (M * M)
    
    # Chain rule: d(val)/dz = J_P^T @ grad_val_P
    # J_P = diag(P) - outer(P, P)
    # J_P^T @ v = P * v - P * (P @ v)
    
    Pv = P @ grad_val_P
    grad_val_z = P * grad_val_P - P * Pv
    
    return val, -grad_val_z # We want to maximize val, so minimize -val

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(-np.sum(qa * np.log2(np.clip(qa, 1e-300, None))), 
            -np.sum(qb * np.log2(np.clip(qb, 1e-300, None))), 
            -np.sum(qd * np.log2(np.clip(qd, 1e-300, None))))
    D = -np.sum(qc * np.log2(np.clip(qc, 1e-300, None)))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 550.0  # safety margin
    
    best_val = -1e9
    best_z = None
    
    # Initial guesses
    rng = np.random.default_rng(42)
    initial_points = []
    
    # Uniform
    initial_points.append(np.zeros(100))
    
    # Random
    for _ in range(10):
        initial_points.append(rng.normal(0, 2.0, 100))
        
    # Structured: Concentrate on anti-diagonals to boost D
    # Try to make H(x) small by concentrating on a column
    for x0 in [0, 4, 5, 9]:
        z = np.full(100, -10.0)
        for y in range(10):
            z[y*10 + x0] = 0.0
        initial_points.append(z)
        
    # Try to make H(x+y) small
    for s0 in range(0, 18, 3):
        z = np.full(100, -10.0)
        for y in range(10):
            x = s0 - y
            if 0 <= x < 10:
                z[y*10 + x] = 0.0
        initial_points.append(z)
        
    for z0 in initial_points:
        if time.time() > deadline - 60:
            break
            
        def objective(z):
            v, g = _value_and_grad(z, idx_a, idx_b, idx_c, idx_d)
            return v, g
            
        try:
            # Use L-BFGS-B which is good for large scale
            res = minimize(objective, z0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Evaluate at the result
            val_res, _ = _value_and_grad(res.x, idx_a, idx_b, idx_c, idx_d)
            
            if val_res > best_val:
                best_val = val_res
                best_z = res.x.copy()
                
        except Exception as e:
            # Fallback to Nelder-Mead if gradient fails?
            # Or just skip
            continue
            
    if best_z is None:
        P = np.full((10, 10), 1.0/100.0)
        return P
        
    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P

# EVOLVE-BLOCK-END
