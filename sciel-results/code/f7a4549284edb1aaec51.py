import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Start with uniform
    theta = np.zeros((n, n), dtype=np.float64)
    
    # Precompute indices for projections
    # u = x - y, range -9 to 9 (offset 9 -> 0 to 18)
    # v0 = x, range 0 to 9
    # v1 = x + y, range 0 to 18
    # v2 = x + 2y, range 0 to 27
    
    xs = np.arange(n)
    ys = np.arange(n)
    XX, YY = np.meshgrid(xs, ys, indexing='xy') # Shape (10, 10), YY is y, XX is x? 
    # Wait, P[y, x]. So index [y, x].
    # Let's create arrays of coordinates
    # Y_idx[i, j] = i, X_idx[i, j] = j
    
    Y_idx, X_idx = np.indices((n, n))
    
    U = X_idx - Y_idx + 9 # 0..18
    V0 = X_idx # 0..9
    V1 = X_idx + Y_idx # 0..18
    V2 = X_idx + 2 * Y_idx # 0..27
    
    num_u = 19
    num_v0 = 10
    num_v1 = 19
    num_v2 = 28
    
    # One-hot matrices for pooling? Or just bincount.
    # bincount is faster.
    
    def get_probs(theta):
        # Softmax
        t = theta - np.max(theta)
        p_flat = np.exp(t)
        p_flat /= np.sum(p_flat)
        return p_flat.reshape((n, n))
    
    def get_entropies(P):
        p_flat = P.flatten()
        
        # Project
        pu = np.bincount(U.flatten(), weights=p_flat, minlength=num_u)
        pv0 = np.bincount(V0.flatten(), weights=p_flat, minlength=num_v0)
        pv1 = np.bincount(V1.flatten(), weights=p_flat, minlength=num_v1)
        pv2 = np.bincount(V2.flatten(), weights=p_flat, minlength=num_v2)
        
        def shannon(p):
            p = p[p > 1e-15]
            if len(p) == 0: return 0.0
            return -np.sum(p * np.log2(p))
        
        Hu = shannon(pu)
        Hv0 = shannon(pv0)
        Hv1 = shannon(pv1)
        Hv2 = shannon(pv2)
        
        M = max(Hv0, Hv1, Hv2)
        
        # Determine active projection for gradient
        # If multiple are max, pick one arbitrarily or average? 
        # Let's just pick the first one that is max.
        if Hv0 >= Hv1 and Hv0 >= Hv2:
             active = 0
             H_active = Hv0
        elif Hv1 >= Hv0 and Hv1 >= Hv2:
             active = 1
             H_active = Hv1
        else:
             active = 2
             H_active = Hv2
             
        return Hu, M, active, (pu, pv0, pv1, pv2)

    def compute_grad(theta, P):
        # Gradient of log-likelihood or entropy?
        # We want to maximize D/M.
        # Approximation: Maximize D - (D/M) * M ? No.
        # Let's use the gradient of D - lambda * M_active where lambda = D/M.
        # This is the gradient of D - (D/M)M = 0? No.
        # We want grad of D/M.
        # d(D/M) = (M dD - D dM) / M^2.
        # So we want to maximize M * dD - D * dM.
        
        Hu, M, active, probs = get_entropies(P)
        if M < 1e-9:
             return np.zeros_like(theta)
             
        ratio = Hu / M
        
        # Gradient of Entropy H(p) wrt p_i is -log2(p_i) - 1/ln(2)
        # Let g_p = -log2(p) - 1/ln(2) for p>0, else 0
        
        def grad_entropy(p):
            g = np.zeros_like(p)
            mask = p > 1e-15
            g[mask] = -np.log2(p[mask]) - 1.0 / np.log(2.0)
            return g
            
        g_pu = grad_entropy(probs[0])
        g_pv0 = grad_entropy(probs[1])
        g_pv1 = grad_entropy(probs[2])
        g_pv2 = grad_entropy(probs[3])
        
        # dD = sum(g_pu * dp_u)
        # dM = sum(g_pv_active * dp_v_active)
        
        # dp_u = J_u @ dP
        # dp_v = J_v @ dP
        # So dD = (J_u.T @ g_pu) @ dP
        # dM = (J_v.T @ g_pv) @ dP
        
        # We need J_u.T @ g_pu.
        # J_u[i, k] = 1 if cell i maps to bin k.
        # (J_u.T @ g_pu)[i] = g_pu[U[i]]
        
        grad_D = g_pu[U.flatten()].reshape((n, n))
        
        if active == 0:
             grad_M = g_pv0[V0.flatten()].reshape((n, n))
        elif active == 1:
             grad_M = g_pv1[V1.flatten()].reshape((n, n))
        else:
             grad_M = g_pv2[V2.flatten()].reshape((n, n))
             
        # Objective derivative wrt P:
        # dObj/dP = M * grad_D - D * grad_M
        # Note: This is the derivative of the scalar value wrt P.
        # But we are optimizing theta.
        # dP/dtheta = P * (I - P_vec) ...
        # Actually, simpler: 
        # The gradient of the objective with respect to theta is:
        # grad_theta = P * (grad_P - <grad_P, P>)
        # where grad_P is the derivative of the objective wrt P.
        
        grad_P = M * grad_D - Hu * grad_M
        
        # Project gradient onto tangent space of simplex
        # grad_theta = P * (grad_P - np.sum(grad_P * P))
        
        mean_grad = np.sum(grad_P * P)
        grad_theta = P * (grad_P - mean_grad)
        
        return grad_theta

    # Optimization loop
    start_time = time.time()
    deadline = start_time + 500 # 10s margin
    
    lr = 0.1
    # Add small noise to break symmetry
    theta += np.random.randn(n, n) * 0.1
    
    best_theta = theta.copy()
    best_score = 0.0
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        P = get_probs(theta)
        Hu, M, active, _ = get_entropies(P)
        if M > 1e-9:
            score = Hu / M
        else:
            score = 0.0
            
        if score > best_score:
            best_score = score
            best_theta = theta.copy()
            
        grad = compute_grad(theta, P)
        
        # Adaptive step size? Simple constant for now.
        theta += lr * grad
        
        # Optional: Renormalize theta to prevent overflow (softmax handles it, but keeps numbers small)
        theta -= np.max(theta)
        
    # Return best found
    P_final = get_probs(best_theta)
    # Clip and normalize just in case
    P_final = np.clip(P_final, 0, None)
    s = np.sum(P_final)
    if s > 0:
        P_final /= s
    else:
        P_final = np.ones((n, n)) / (n*n)
        
    return P_final
