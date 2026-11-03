import numpy as np
import time

N = 400

def score_sequence(x):
    # Normalize to sum 1 to remove scale dependency
    s = np.sum(x)
    if np.abs(s) < 1e-12:
        return float('inf')
    x_norm = x / s
    conv = np.convolve(x_norm, x_norm, mode='full')
    max_abs = np.max(np.abs(conv))
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = float('inf')
    best_x = None
    
    # Candidate 1: All ones
    x = np.ones(N)
    sc = score_sequence(x)
    if sc < best_score:
        best_score = sc
        best_x = x
    
    # Candidate 2: U-shape (Cosine squared)
    i = np.arange(N)
    x = np.cos(np.pi * i / (N - 1))**2
    # Ensure non-negative? Task says "signed", so signs are allowed.
    # But U-shape is positive.
    sc = score_sequence(x)
    if sc < best_score:
        best_score = sc
        best_x = x
        
    # Candidate 3: V-shape (Sine squared / Hanning)
    x = np.sin(np.pi * i / (N - 1))**2
    sc = score_sequence(x)
    if sc < best_score:
        best_score = sc
        best_x = x
        
    # Candidate 4: Parametric 1 + a * cos(2*pi*i/N)
    # Try a range of a
    for a in np.linspace(-0.5, 0.5, 100):
        if time.time() > deadline:
            break
        x = 1 + a * np.cos(2 * np.pi * i / N)
        # Avoid negative sums or zeros
        if np.sum(x) == 0: continue
        sc = score_sequence(x)
        if sc < best_score:
            best_score = sc
            best_x = x
            
    # Candidate 5: Simple local optimization around best_x
    # Use scipy if available and time permits
    if time.time() < deadline - 20:
        try:
            from scipy.optimize import minimize
            
            def objective(x_flat):
                x = x_flat.reshape(N)
                return score_sequence(x)
                
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            
            # Start from best_x normalized
            x0 = best_x / np.sum(best_x)
            
            res = minimize(objective, x0, method='SLSQP', constraints=constraints, options={'maxiter': 50, 'ftol': 1e-6})
            if res.success:
                sc = res.fun
                if sc < best_score:
                    best_score = sc
                    best_x = res.x
        except Exception:
            pass
            
    # Return the best sequence found
    # The task asks for "sequence of signed step heights"
    # The score is calculated by the evaluator on the returned sequence.
    # The evaluator calculates max|autoconv| / (sum)^2.
    # We should return the sequence that minimizes this.
    
    return best_x
