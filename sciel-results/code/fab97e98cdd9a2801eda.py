import numpy as np
import time
import sys

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    The baseline is the uniform distribution over the grid.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    # Slopes
    r_inf = -1.0
    r_finite = [0.0, 1.0, 2.0]
    
    # Pre-compute mapping from (y, x) index to value for each slope
    # Index i = y * n + x
    vals = {r: np.zeros((n, n), dtype=int) for r in [r_inf] + r_finite}
    for y in range(n):
        for x in range(n):
            vals[r_inf][y, x] = int(x + r_inf * y)
            for r in r_finite:
                vals[r][y, x] = int(x + r * y)
                
    # Min and max values for each slope to create probability vectors
    min_vals = {r: int(vals[r].min()) for r in vals}
    max_vals = {r: int(vals[r].max()) for r in vals}
    sizes = {r: max_vals[r] - min_vals[r] + 1 for r in vals}
    
    def compute_probs(P_flat):
        """
        Compute the projected probability distributions for all slopes.
        P_flat is a flat array of length 100 (or 99 if using constraint trick, 
        but here we assume full 100 for gradient computation simplicity 
        and handle normalization in the objective).
        Actually, for gradient descent on P with sum=1, we need to be careful.
        Let's use the full P vector and normalize it inside the entropy calculation?
        No, the entropy formula assumes sum P = 1.
        If we optimize P directly, we must enforce sum P = 1.
        
        Alternative: Optimize log-P (unconstrained) and use softmax.
        P = softmax(Z).
        This ensures P > 0 and sum P = 1.
        Gradient of softmax is easy.
        """
        # This function is called inside the objective.
        # We will implement the objective using Z (log-probs) to avoid constraints.
        pass

    def objective_and_grad(Z_flat):
        """
        Z_flat: length 100.
        Returns: (loss, grad)
        Loss = -log(H_inf) + log(max(H_0, H_1, H_2))
        """
        # Softmax
        Z_max = np.max(Z_flat)
        exp_Z = np.exp(Z_flat - Z_max)
        P = exp_Z / np.sum(exp_Z)
        
        # Compute projected distributions
        Qs = {}
        dH_dP = {}
        
        for r in [r_inf] + r_finite:
            Q = np.zeros(sizes[r])
            # Map values to indices
            # val = x + r*y
            # idx = val - min_vals[r]
            for y in range(n):
                for x in range(n):
                    idx = vals[r][y, x] - min_vals[r]
                    Q[idx] += P[y, x]
            
            # Handle zero probabilities for entropy
            mask = Q > 1e-12
            Q_safe = np.where(mask, Q, 1.0)
            
            H = -np.sum(Q[mask] * np.log2(Q[mask]))
            
            # Gradient of H wrt P
            # dH/dQ_j = -log2(Q_j) - 1/ln(2) for Q_j > 0, else 0
            dH_dQ = np.zeros(sizes[r])
            dH_dQ[mask] = -np.log2(Q_safe[mask]) - 1.0 / np.log(2)
            
            # dH/dP_yx = dH_dQ[idx_yx]
            dH_dP[r] = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    idx = vals[r][y, x] - min_vals[r]
                    dH_dP[r][y, x] = dH_dQ[idx]
            
            Qs[r] = Q
            dH_dP[r] = dH_dP[r]
        
        H_inf = -np.sum(Qs[r_inf][Qs[r_inf] > 1e-12] * np.log2(Qs[r_inf][Qs[r_inf] > 1e-12]))
        
        H_0 = -np.sum(Qs[0.0][Qs[0.0] > 1e-12] * np.log2(Qs[0.0][Qs[0.0] > 1e-12]))
        H_1 = -np.sum(Qs[1.0][Qs[1.0] > 1e-12] * np.log2(Qs[1.0][Qs[1.0] > 1e-12]))
        H_2 = -np.sum(Qs[2.0][Qs[2.0] > 1e-12] * np.log2(Qs[2.0][Qs[2.0] > 1e-12]))
        
        M = max(H_0, H_1, H_2)
        
        if H_inf <= 1e-12 or M <= 1e-12:
            return 1e10, np.zeros(100)
            
        loss = -np.log(H_inf) + np.log(M)
        
        # Gradient of loss
        # d(loss)/dP = -1/H_inf * dH_inf/dP + 1/M * dM/dP
        # dM/dP = dH_active/dP where H_active is the max
        
        grad_P = np.zeros((n, n))
        
        # Term 1: -1/H_inf * dH_inf/dP
        grad_P -= (1.0 / H_inf) * dH_dP[r_inf]
        
        # Term 2: 1/M * dH_active/dP
        # Find active slope(s)
        active = []
        if np.isclose(H_0, M): active.append(0.0)
        if np.isclose(H_1, M): active.append(1.0)
        if np.isclose(H_2, M): active.append(2.0)
        
        if not active:
             active = [0.0] # Fallback
            
        for r in active:
             grad_P += (1.0 / (len(active) * M)) * dH_dP[r]
             
        # Backprop to Z
        # P = softmax(Z)
        # dP/dZ = P * (I - P)
        # grad_Z = P * (grad_P - sum(grad_P * P))
        grad_Z = P * (grad_P - np.sum(grad_P * P))
        
        return loss, grad_Z.flatten()

    best_P = np.ones((n, n)) / (n * n)
    best_score = 1.0
    
    # Initializations
    inits = []
    # 1. Uniform
    inits.append(np.zeros(100)) # Z=0 -> P=1/100
    
    # 2. Random
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(rng.normal(0, 1, 100))
        
    # 3. Structured: Anti-diagonal peaks
    # P[y, x] high if x+y is near 9
    Z_struct = np.zeros(100)
    for y in range(n):
        for x in range(n):
             Z_struct[y, x] = -abs(x + y - 9) * 2.0
    inits.append(Z_struct)
    
    # 4. Structured: Main diagonal
    Z_struct2 = np.zeros(100)
    for y in range(n):
        for x in range(n):
             Z_struct2[y, x] = -abs(x - y) * 2.0
    inits.append(Z_struct2)
             
    from scipy.optimize import minimize
    
    for Z0 in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective_and_grad, Z0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
        except Exception:
            continue
            
        Z_opt = res.x
        Z_max = np.max(Z_opt)
        P_opt = np.exp(Z_opt - Z_max)
        P_opt /= np.sum(P_opt)
        
        # Calculate score
        Q_inf = np.zeros(sizes[r_inf])
        for y in range(n):
            for x in range(n):
                Q_inf[vals[r_inf][y, x] - min_vals[r_inf]] += P_opt[y, x]
        mask_inf = Q_inf > 1e-12
        H_inf = -np.sum(Q_inf[mask_inf] * np.log2(Q_inf[mask_inf]))
        
        M = 0
        for r in r_finite:
            Q = np.zeros(sizes[r])
            for y in range(n):
                for x in range(n):
                    Q[vals[r][y, x] - min_vals[r]] += P_opt[y, x]
            mask = Q > 1e-12
            H = -np.sum(Q[mask] * np.log2(Q[mask]))
            if H > M: M = H
            
        if H_inf > 1e-12 and M > 1e-12:
            score = H_inf / M
            if score > best_score:
                best_score = score
                best_P = P_opt.copy()
                
    # Final coordinate descent polish on best_P?
    # Simple local search
    if time.time() < deadline - 10:
        P_curr = best_P.copy()
        for _ in range(500):
            if time.time() > deadline: break
            improved = False
            for i in range(100):
                if time.time() > deadline: break
                y, x = divmod(i, n)
                # Try moving mass from (y,x) to neighbors
                # Simple: perturb P
                orig = P_curr[y, x]
                if orig < 1e-6: continue
                
                # Try moving to random neighbor
                ny, nx = np.random.randint(0, n), np.random.randint(0, n)
                if ny == y and nx == x: continue
                
                delta = orig * 0.1
                P_curr[y, x] -= delta
                P_curr[ny, nx] += delta
                
                # Check validity
                if P_curr[y, x] < -1e-9 or P_curr[ny, nx] < -1e-9:
                    P_curr[y, x] = orig
                    P_curr[ny, nx] -= delta
                    continue
                    
                # Normalize
                P_curr = P_curr / np.sum(P_curr)
                
                # Score
                Q_inf = np.zeros(sizes[r_inf])
                for yy in range(n):
                    for xx in range(n):
                        Q_inf[vals[r_inf][yy, xx] - min_vals[r_inf]] += P_curr[yy, xx]
                mask_inf = Q_inf > 1e-12
                H_inf = -np.sum(Q_inf[mask_inf] * np.log2(Q_inf[mask_inf]))
                
                M = 0
                for r in r_finite:
                    Q = np.zeros(sizes[r])
                    for yy in range(n):
                        for xx in range(n):
                            Q[vals[r][yy, xx] - min_vals[r]] += P_curr[yy, xx]
                    mask = Q > 1e-12
                    H = -np.sum(Q[mask] * np.log2(Q[mask]))
                    if H > M: M = H
                
                if H_inf > 1e-12 and M > 1e-12:
                    score = H_inf / M
                    if score > best_score:
                        best_score = score
                        best_P = P_curr.copy()
                        improved = True
                    else:
                        # Revert
                        P_curr[ny, nx] -= delta
                        P_curr[y, x] += delta
                        P_curr = P_curr / np.sum(P_curr)
                        
    # Ensure output is valid
    best_P = np.clip(best_P, 0, None)
    best_P = best_P / np.sum(best_P)
    
    return best_P

# EVOLVE-BLOCK-END
