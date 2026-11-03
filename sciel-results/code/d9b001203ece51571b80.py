import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_ratio(h):
    # h is expected to be a numpy array
    # Ensure non-negative
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    
    # Normalize to sum=1 for shape evaluation
    h_norm = h / l1
    
    l2_sq = np.sum(h_norm**2)
    
    # Autoconvolution
    # np.convolve(h_norm, h_norm, mode='full')
    conv = np.convolve(h_norm, h_norm, mode='full')
    l_inf = np.max(conv)
    
    if l_inf == 0:
        return 0.0
        
    ratio = l2_sq / (1.0 * l_inf) # Since l1=1
    return ratio

def objective(x):
    # We want to maximize ratio, so minimize -ratio
    # x are the heights
    h = np.maximum(x, 0)
    return -compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    best_h = None
    best_ratio = -1.0
    
    # Initial guesses
    # 1. Uniform
    h_uniform = np.ones(N) / N
    best_h = h_uniform.copy()
    best_ratio = compute_ratio(h_uniform)
    
    # 2. Delta
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    r = compute_ratio(h_delta)
    if r > best_ratio:
        best_ratio = r
        best_h = h_delta.copy()
        
    # 3. Bi-modal
    h_bi = np.zeros(N)
    h_bi[0] = 0.7
    h_bi[1] = 0.3
    r = compute_ratio(h_bi)
    if r > best_ratio:
        best_ratio = r
        best_h = h_bi.copy()

    # 4. Random starts with local optimization
    np.random.seed(42)
    
    max_iterations = 0
    while time.time() < deadline - 10.0:
        # Generate a random start
        # Mix of uniform and spike
        h_start = np.random.rand(N)
        # Sometimes add a spike
        if np.random.rand() < 0.5:
            idx = np.random.randint(N)
            h_start[idx] += 1.0
        # Normalize
        h_start = h_start / np.sum(h_start)
        
        # Optimize
        # Use Nelder-Mead as it handles non-smoothness (max in conv) better than gradient methods
        res = minimize(objective, h_start, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000})
        
        if res.success or res.fun < -best_ratio:
            h_opt = np.maximum(res.x, 0)
            h_opt = h_opt / np.sum(h_opt)
            r = compute_ratio(h_opt)
            if r > best_ratio:
                best_ratio = r
                best_h = h_opt.copy()
        
        max_iterations += 1
        if max_iterations > 100: # Safety break
            break

    # Return the best heights found
    # Ensure they are non-negative and sum to 1 (or keep raw if evaluator prefers, but sum=1 is safe for "shape")
    # The prompt asks for "sequence of nonnegative step heights".
    # Returning sum=1 is the most standard interpretation for "maximizing a ratio" where scale might be ambiguous in the prompt but critical in the metric.
    return list(best_h)

# EVOLVE-BLOCK-START
def construct():
    # The function above is defined in the global scope, but we need to keep the interface.
    # I will move the logic inside construct or call it.
    # The prompt says "Keep the function names and signatures of the seed program".
    # Seed program:
    # def construct():
    #     return [1.0] * N
    
    # I will replace the body.
    
    start_time = time.time()
    deadline = 110.0 # Keep 10s margin
    
    best_h = np.ones(N) / N
    best_ratio = compute_ratio(best_h)
    
    # Try a few specific shapes
    shapes = [
        np.zeros(N), # Delta
        np.zeros(N), # Bi-modal
        np.ones(N),  # Uniform
    ]
    
    # Set specific shapes
    shapes[0][0] = 1.0
    shapes[1][0] = 0.7
    shapes[1][1] = 0.3
    
    for s in shapes:
        s = s / np.sum(s)
        r = compute_ratio(s)
        if r > best_ratio:
            best_ratio = r
            best_h = s.copy()
            
    # Local search
    np.random.seed(123)
    for _ in range(20): # Fixed number of restarts, fast enough
        if time.time() > deadline:
            break
            
        h_start = np.random.rand(N)
        if np.random.rand() < 0.5:
            idx = np.random.randint(N)
            h_start[idx] += 1.0
        h_start = h_start / np.sum(h_start)
        
        try:
            res = minimize(objective, h_start, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 1000})
            if res.fun < -best_ratio:
                h_opt = np.maximum(res.x, 0)
                h_opt = h_opt / np.sum(h_opt)
                r = compute_ratio(h_opt)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    return list(best_h)
# EVOLVE-BLOCK-END
