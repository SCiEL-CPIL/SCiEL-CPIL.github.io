import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_correlations(h):
    # h is shape (N,)
    # 1-h is shape (N,)
    corr = np.correlate(h, 1 - h, mode='full')
    # We only need lags 0 to N-1 (indices N-1 to 2N-2 in full)
    # Actually, max is over all, but symmetry makes it 0..N-1
    # Full output length 2N-1.
    # Lags are -(N-1) ... 0 ... N-1.
    # Index N-1 corresponds to lag 0.
    # We want max over all.
    return np.max(corr)

def objective(x):
    # x[0:N] is h, x[N] is t
    h = x[:N]
    t = x[N]
    # We want to minimize t
    # Subject to C_k <= t
    # In SLSQP, we minimize objective.
    # We can just return t.
    # Constraints handle C_k - t <= 0
    return t

def constraints(x):
    h = x[:N]
    t = x[N]
    
    cons = []
    
    # Sum constraint
    cons.append({'type': 'eq', 'fun': lambda h: np.sum(h) - N/2})
    
    # Box constraints are handled by bounds
    
    # Correlation constraints: C_k - t <= 0
    # C_k = correlate(h, 1-h)
    # We only need to check lags 0..N-1
    # np.correlate gives full.
    # Let's compute C for lags 0..N-1 efficiently or just use full and take max?
    # Using full is O(N^2). N=100 is fine.
    corr = np.correlate(h, 1 - h, mode='full')
    # The max of corr is what we bound.
    # But we need individual constraints?
    # If we just constrain max(corr) <= t, it's one constraint.
    # max(corr) is non-smooth.
    # Better to constrain each C_k <= t.
    # There are 2N-1 constraints.
    # This might be slow for SLSQP.
    # Alternative: use a smooth approximation of max, or just constrain the top few?
    # Or just constrain max(corr) - t <= 0 using a penalty or smooth max?
    # SLSQP handles non-smooth poorly.
    # Let's try constraining all C_k <= t.
    # 199 constraints.
    
    # To speed up, we can just compute the max and constrain that?
    # No, max is not differentiable.
    # Let's add constraints for all lags.
    for k in range(len(corr)):
        cons.append({'type': 'ineq', 'fun': lambda t, ck: t - ck, 'args': (t, corr[k])})
        
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_h = np.full(N, 0.5)
    best_val = compute_correlations(best_h)
    
    # Generate diverse initial points
    inits = []
    inits.append(np.full(N, 0.5))
    
    # Sine waves
    for freq in [1, 2, 5, 10, 20]:
        for phase in [0, np.pi/2, np.pi]:
            t = np.linspace(0, 2 * np.pi * freq, N)
            h = 0.5 + 0.5 * np.sin(t + phase)
            # Clip to [0,1]
            h = np.clip(h, 0, 1)
            # Adjust sum
            diff = N/2 - np.sum(h)
            h = h + diff / N
            h = np.clip(h, 0, 1)
            inits.append(h)
            
    # Random
    for _ in range(20):
        h = np.random.rand(N) * 0.5 + 0.25 # Range [0.25, 0.75]
        diff = N/2 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0, 1)
        inits.append(h)

    # Optimization
    for h0 in inits:
        if time.time() > deadline:
            break
            
        x0 = np.concatenate([h0, [compute_correlations(h0) + 0.1]])
        
        bounds = [(0, 1)] * N + [(0, 100)]
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                h_opt = res.x[:N]
                # Verify
                if np.all(h_opt >= 0) and np.all(h_opt <= 1) and abs(np.sum(h_opt) - N/2) < 1e-6:
                    val = compute_correlations(h_opt)
                    if val < best_val:
                        best_val = val
                        best_h = h_opt
        except Exception:
            continue
            
    return best_h
