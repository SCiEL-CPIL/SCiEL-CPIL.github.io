import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Initial guess: 0.5
    h0 = np.full(N, 0.5)
    
    best_h = h0
    best_val = np.inf
    
    def overlaps(h):
        # Returns array of S_k for k=0..N-1
        S = np.zeros(N)
        for k in range(N):
            # h[:N-k] * (1 - h[k:])
            S[k] = np.sum(h[:N-k] * (1 - h[k:]))
        return S

    def objective(h, beta):
        S = overlaps(h)
        # Log-sum-exp for stability
        m = np.max(S)
        if m == 0:
            return 0.0
        return m + np.log(np.sum(np.exp(beta * (S - m)))) / beta

    # Constraints
    # Sum h_i = N/2
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - N/2})
    bounds = [(0, 1) for _ in range(N)]
    
    # We will run a few iterations with increasing beta
    # SLSQP might be slow, so we limit iterations
    
    # Restart 1: Uniform
    h_curr = h0.copy()
    
    for beta in [1.0, 5.0, 20.0, 100.0]:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective, 
                h_curr, 
                args=(beta,),
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            if res.success:
                h_curr = res.x
                val = np.max(overlaps(h_curr))
                if val < best_val:
                    best_val = val
                    best_h = h_curr.copy()
        except Exception:
            pass
            
    # Try a random restart if time permits
    if time.time() < deadline - 20:
        np.random.seed(42)
        # Generate a valid random h
        h_rand = np.random.rand(N)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        # Adjust sum exactly
        diff = N/2 - np.sum(h_rand)
        # Distribute diff
        mask = h_rand < 1
        if np.sum(mask) > 0:
            h_rand[mask] += diff / np.sum(mask)
            h_rand = np.clip(h_rand, 0, 1)
            # Final fix
            h_rand += (N/2 - np.sum(h_rand)) / N
            
        for beta in [10.0, 50.0]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    objective, 
                    h_rand, 
                    args=(beta,),
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=cons,
                    options={'maxiter': 50, 'ftol': 1e-8}
                )
                if res.success:
                    h_curr = res.x
                    val = np.max(overlaps(h_curr))
                    if val < best_val:
                        best_val = val
                        best_h = h_curr.copy()
            except Exception:
                pass

    # Final check
    if time.time() > deadline - 5:
         return best_h
         
    return best_h
