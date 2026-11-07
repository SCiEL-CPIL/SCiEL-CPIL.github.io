import numpy as np
import time
from scipy.optimize import minimize_scalar

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

def optimize_coordinate(h, i, deadline):
    """
    Optimize h[i] to minimize the max correlation, keeping other h fixed.
    Uses a 1D minimization.
    """
    # To speed up, we can pre-compute parts of the correlation that don't depend on h[i]
    # But for N=100, direct computation in the objective function of minimize_scalar is fast enough.
    
    # We need to check time
    if time.time() > deadline:
        return h[i]
        
    # Define the 1D objective function
    def obj(x):
        if x < 0 or x > 1:
            return float('inf')
        h_temp = h.copy()
        h_temp[i] = x
        return compute_value(h_temp)
        
    # Use minimize_scalar with bounded method
    # This is robust for this 1D problem
    try:
        res = minimize_scalar(obj, bounds=(0, 1), method='bounded', options={'maxiter': 50})
        if res.success:
            return res.x
        else:
            return h[i]
    except:
        return h[i]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit (or 150s)
    
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # We will try several random sinusoidal patterns as seeds for coordinate descent
    # A sinusoid h(x) = 0.5 + A * sin(2*pi*f*x + phi)
    # We need sum(h) = N/2. A sinusoid has mean 0.5 if centered correctly.
    # h[i] = 0.5 + A * sin(2*pi*f*i/N + phi)
    # We must ensure h[i] in [0,1]. So A <= 0.5.
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random parameters
        # Frequency: integer number of cycles to ensure periodicity helps with correlation structure?
        # Or just random real frequency.
        f = np.random.uniform(1, 5) # 1 to 5 cycles
        phi = np.random.uniform(0, 2 * np.pi)
        A = np.random.uniform(0.2, 0.5) # Amplitude
        
        # Generate seed
        x = np.arange(N)
        h_curr = 0.5 + A * np.sin(2 * np.pi * f * x / N + phi)
        
        # Clip to [0,1]
        h_curr = np.clip(h_curr, 0, 1)
        
        # Adjust sum to be exactly N/2?
        # The problem states sum(h) must be N/2 within tolerance.
        # Coordinate descent might drift the sum slightly if we don't constrain it.
        # However, the objective function does NOT explicitly penalize sum deviation, 
        # but the task contract says "A broken bound or sum makes the result invalid".
        # So we MUST maintain the sum constraint.
        
        # Strategy: 
        # 1. During coordinate descent, we optimize h[i] in [0,1].
        # 2. After each full pass or periodically, we might need to re-normalize?
        #    No, re-normalizing breaks the individual [0,1] bounds or the optimality.
        #    Better: Constrain the optimization of h[i] such that the sum remains N/2?
        #    That makes it a coupled problem.
        #    
        # Alternative: 
        # The seed must have sum N/2. 
        # If we start with a clipped sinusoid, the sum is likely not exactly N/2.
        # We should adjust the constant offset or scale to make sum N/2 while keeping h in [0,1].
        # Let h_raw = A * sin(...). Mean is ~0.
        # Let h = 0.5 + c * h_raw.
        # Sum(h) = N/2 + c * Sum(h_raw).
        # If Sum(h_raw) is close to 0, we are good.
        # For a full period sine, sum is 0.
        # If f is integer, sum is 0.
        # Let's enforce integer frequencies to ensure sum(h_raw)=0 (approximately, due to discretization).
        # Actually, discrete sum of sin over integer cycles is 0.
        
        # Let's refine the seed generation to ensure sum is N/2.
        # Use integer frequency f_int
        f_int = np.random.randint(1, 6)
        phi = np.random.uniform(0, 2 * np.pi)
        A = np.random.uniform(0.3, 0.5)
        
        h_seed = 0.5 + A * np.sin(2 * np.pi * f_int * x / N + phi)
        
        # Due to discretization, sum might not be exactly N/2.
        # Calculate error
        sum_err = np.sum(h_seed) - N/2
        # Subtract error/N from all elements
        h_seed -= sum_err / N
        
        # Clip
        h_seed = np.clip(h_seed, 0, 1)
        
        # Re-check sum after clip? Clipping might change sum.
        # If clipping occurred, sum is no longer N/2.
        # To be safe, we can do a small projection step:
        # If any h_i is 0 or 1, we can't adjust them easily without violating bounds.
        # However, with A <= 0.5 and offset 0.5, clipping is rare unless A is close to 0.5 and phase aligns.
        # If clipping happens, we might have invalid sum.
        # Let's just ensure A is small enough, e.g., 0.4, so clipping is unlikely.
        
        # Let's restart seed generation if sum is invalid
        valid_seed = False
        for attempt in range(10):
            f_int = np.random.randint(1, 6)
            phi = np.random.uniform(0, 2 * np.pi)
            A = np.random.uniform(0.2, 0.45)
            
            h_seed = 0.5 + A * np.sin(2 * np.pi * f_int * x / N + phi)
            sum_err = np.sum(h_seed) - N/2
            h_seed -= sum_err / N
            
            if np.all(h_seed >= 0) and np.all(h_seed <= 1):
                valid_seed = True
                break
        
        if not valid_seed:
            # Fallback to uniform
            h_seed = np.full(N, 0.5)
            continue
            
        h_curr = h_seed.copy()
        curr_val = compute_value(h_curr)
        
        if curr_val < best_val:
            best_val = curr_val
            best_h = h_curr.copy()
            
        # Coordinate Descent
        # We need to maintain the sum constraint.
        # Standard coordinate descent on h[i] in [0,1] does NOT maintain sum.
        # To maintain sum, we can parameterize h differently or use a penalty.
        # However, the task requires exact sum.
        # 
        # Idea: Optimize h[i] and h[j] simultaneously? No, too complex.
        # Idea: Use a Lagrange multiplier for the sum constraint?
        # Minimize max(C_k) subject to sum(h)=N/2, 0<=h<=1.
        # 
        # Simpler Idea: 
        # Perform coordinate descent on h[i] in [0,1].
        # After each full sweep, re-project to the hyperplane sum(h)=N/2?
        # Projection: h = h - (sum(h) - N/2)/N * ones.
        # This projection might push values out of [0,1].
        # If it does, we clip and re-project? This is a standard algorithm (POCS).
        # But it might oscillate.
        
        # Alternative: 
        # Since N=100 is small, we can use `scipy.optimize.minimize` with SLSQP 
        # with the sum constraint explicitly, starting from the best seed found so far.
        # SLSQP handles linear equality constraints well.
        
        # Let's try SLSQP from the seed.
        
        def objective(h_vec):
            return compute_value(h_vec)
            
        def constraint_sum(h_vec):
            return np.sum(h_vec) - N/2
            
        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        bounds = [(0, 1)] * N
        
        # Run SLSQP
        # Limit iterations to avoid hanging
        try:
            res = minimize_scalar # placeholder
            from scipy.optimize import minimize
            res = minimize(objective, h_curr, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success:
                new_val = res.fun
                # Check validity
                if np.all(res.x >= 0) and np.all(res.x <= 1) and abs(np.sum(res.x) - N/2) < 1e-8:
                    if new_val < best_val:
                        best_val = new_val
                        best_h = res.x.copy()
        except Exception as e:
            pass # Ignore errors and continue to next restart

    return best_h

# Test locally if needed
if __name__ == "__main__":
    h = construct()
    print("Sum:", np.sum(h))
    print("Min:", np.min(h))
    print("Max:", np.max(h))
    print("Value:", compute_value(h))
