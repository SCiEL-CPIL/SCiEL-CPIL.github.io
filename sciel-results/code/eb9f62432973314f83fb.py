import numpy as np
import time
from scipy.optimize import minimize

N = 100

def compute_value(h):
    # h is an array of length N
    # Compute full correlation of h and 1-h
    # np.correlate(h, 1-h, mode='full')
    # The max is taken over the full correlation
    # Value = max / N * 2
    corr = np.correlate(h, 1 - h, mode="full")
    max_corr = np.max(corr)
    return max_corr / N * 2

def _fix_sum(h):
    """Adjust h to satisfy sum(h) = N/2 while keeping h in [0,1] if possible."""
    sum_err = np.sum(h) - N/2
    if abs(sum_err) < 1e-12:
        return h
    # Try simple projection
    h_proj = h - sum_err / N
    if np.all(h_proj >= 0) and np.all(h_proj <= 1):
        return h_proj
    # If projection fails, do a simple iterative fix
    h_fixed = h.copy()
    for _ in range(100):
        sum_err = np.sum(h_fixed) - N/2
        if abs(sum_err) < 1e-12:
            break
        # Adjust elements that have room
        delta = sum_err / N
        for i in range(N):
            if delta > 0 and h_fixed[i] < 1:
                adj = min(delta, 1 - h_fixed[i])
                h_fixed[i] += adj
                delta -= adj
                if delta <= 0:
                    break
            elif delta < 0 and h_fixed[i] > 0:
                adj = max(delta, -h_fixed[i])
                h_fixed[i] += adj
                delta -= adj
                if delta >= 0:
                    break
    return h_fixed

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    x = np.arange(N)
    num_restarts = 30  # More restarts with tighter per-restart budget
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a valid sinusoidal seed with sum exactly N/2
        valid_seed = False
        for attempt in range(20):
            f_int = np.random.randint(1, 8)  # Wider frequency range
            phi = np.random.uniform(0, 2 * np.pi)
            A = np.random.uniform(0.15, 0.48)
            
            h_seed = 0.5 + A * np.sin(2 * np.pi * f_int * x / N + phi)
            sum_err = np.sum(h_seed) - N/2
            h_seed -= sum_err / N
            
            if np.all(h_seed >= 0) and np.all(h_seed <= 1):
                valid_seed = True
                break
        
        if not valid_seed:
            # Fallback: try lower amplitude
            h_seed = 0.5 + 0.1 * np.sin(2 * np.pi * 3 * x / N)
            sum_err = np.sum(h_seed) - N/2
            h_seed -= sum_err / N
            if not (np.all(h_seed >= 0) and np.all(h_seed <= 1)):
                h_seed = np.full(N, 0.5)
        
        h_curr = h_seed.copy()
        curr_val = compute_value(h_curr)
        
        if curr_val < best_val:
            best_val = curr_val
            best_h = h_curr.copy()
        
        # Run SLSQP optimization from this seed
        def objective(h_vec):
            return compute_value(h_vec)
        
        def constraint_sum(h_vec):
            return np.sum(h_vec) - N/2
            
        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        bounds = [(0, 1)] * N
        
        # Run SLSQP with moderate iterations
        try:
            res = minimize(objective, h_curr, method='SLSQP', bounds=bounds, 
                          constraints=constraints, 
                          options={'maxiter': 150, 'ftol': 1e-7})
            
            if res.success:
                # Fix any small sum violations
                h_fixed = _fix_sum(res.x)
                new_val = compute_value(h_fixed)
                
                # Check validity
                if (np.all(h_fixed >= 0) and np.all(h_fixed <= 1) and 
                    abs(np.sum(h_fixed) - N/2) < 1e-9):
                    if new_val < best_val:
                        best_val = new_val
                        best_h = h_fixed.copy()
        except Exception:
            pass
    
    # Final polish: run SLSQP with tighter tolerance from best solution found
    if time.time() < deadline - 5:
        try:
            def objective(h_vec):
                return compute_value(h_vec)
            
            def constraint_sum(h_vec):
                return np.sum(h_vec) - N/2
                
            constraints = [{'type': 'eq', 'fun': constraint_sum}]
            bounds = [(0, 1)] * N
            
            res = minimize(objective, best_h, method='SLSQP', bounds=bounds, 
                          constraints=constraints, 
                          options={'maxiter': 300, 'ftol': 1e-10})
            
            if res.success:
                h_fixed = _fix_sum(res.x)
                new_val = compute_value(h_fixed)
                
                if (np.all(h_fixed >= 0) and np.all(h_fixed <= 1) and 
                    abs(np.sum(h_fixed) - N/2) < 1e-9):
                    if new_val < best_val:
                        best_val = new_val
                        best_h = h_fixed.copy()
        except Exception:
            pass
    
    # Final validity check and fix
    best_h = _fix_sum(best_h)
    # Ensure strict bounds
    best_h = np.clip(best_h, 0, 1)
    # Final sum correction
    sum_err = np.sum(best_h) - N/2
    if abs(sum_err) > 1e-12:
        best_h -= sum_err / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h

# Test locally if needed
if __name__ == "__main__":
    t0 = time.time()
    h = construct()
    t1 = time.time()
    print("Time:", t1 - t0)
    print("Sum:", np.sum(h))
    print("Sum error:", abs(np.sum(h) - N/2))
    print("Min:", np.min(h))
    print("Max:", np.max(h))
    print("Value:", compute_value(h))
    print("Score:", 0.5 / compute_value(h) if compute_value(h) > 0 else float('inf'))
