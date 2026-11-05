import numpy as np
import scipy.optimize as opt
import time

N = 400

def _compute_value(h):
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / np.sum(h) ** 2)

def _objective_and_grad(x):
    # x is h[1:]
    h = np.zeros(N)
    h[0] = 1.0 - np.sum(x)
    h[1:] = x
    
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    # Gradient of max_val w.r.t h
    # d(max_val)/d(h[i]) = sum_j h[j] * delta_{idx, i+j}
    # This is equivalent to: for each i, if 0 <= idx-i < N, add h[idx-i]
    # This is a correlation of h with a spike at idx.
    # Let's construct the gradient vector g_h
    g_h = np.zeros(N)
    # idx is in [0, 2N-2]
    # i is in [0, N-1]
    # j = idx - i must be in [0, N-1]
    # So i must be in [idx-(N-1), idx]
    i_start = max(0, idx - (N - 1))
    i_end = min(N - 1, idx)
    
    if i_start <= i_end:
        # For i in [i_start, i_end], j = idx - i
        # g_h[i] = h[idx - i]
        # This is h[range(idx-i_end, idx-i_start+1)] reversed?
        # Let's just loop or vectorize carefully.
        # Vectorized:
        # i = i_start ... i_end
        # j = idx - i
        # g_h[i] = h[j]
        # We can do:
        # g_h[i_start:i_end+1] = h[idx - i_end : idx - i_start + 1][::-1]
        
        j_start = idx - i_end
        j_end = idx - i_start
        # h[j_start : j_end + 1]
        # We need to assign to g_h[i_start : i_end + 1]
        # The order of h is j_start, j_start+1, ..., j_end
        # The order of i is i_start, i_start+1, ..., i_end
        # j = idx - i. As i increases, j decreases.
        # So g_h[i_start] = h[idx - i_start] = h[j_end]
        # g_h[i_end] = h[idx - i_end] = h[j_start]
        # So we take h[j_start:j_end+1] and reverse it.
        g_h[i_start:i_end+1] = h[j_start:j_end+1][::-1]
        
    # Now we need gradient w.r.t x
    # h[0] = 1 - sum(x) => dh[0]/dx = -1
    # h[i] = x[i-1] => dh[i]/dx = I
    # So grad_x[0] = -g_h[0] + sum_{i=1}^{N-1} g_h[i] ? No.
    # d(max)/dx_k = sum_i (dh_i/dx_k) * g_h[i]
    # For k=0 (x[0] is h[1]): dh_1/dx_0 = 1, others 0? No.
    # x[0] corresponds to h[1].
    # x[k] corresponds to h[k+1].
    # So dh_{k+1}/dx_k = 1.
    # dh_0/dx_k = -1 for all k.
    
    # grad_x[k] = dh_0/dx_k * g_h[0] + dh_{k+1}/dx_k * g_h[k+1]
    #           = -1 * g_h[0] + 1 * g_h[k+1]
    
    grad_x = np.empty(N - 1)
    for k in range(N - 1):
        grad_x[k] = -g_h[0] + g_h[k + 1]
        
    return max_val, grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_val = float('inf')
    best_h = np.ones(N)
    
    # Try a few initializations
    # 1. All ones (seed)
    # 2. Random normal
    # 3. Alternating signs with small bias
    
    initializations = []
    
    # Seed
    initializations.append(np.ones(N))
    
    # Random normal
    np.random.seed(42)
    for _ in range(5):
        v = np.random.randn(N)
        # Normalize to sum 1? The objective in _objective_and_grad assumes sum(h)=1 via parameterization.
        # But the initialization is for the full h.
        # We need to convert h to x.
        # x = h[1:]
        # h[0] is determined by x.
        # So we just pick x randomly.
        # But we want diverse h.
        # Let's just pick x randomly.
        initializations.append(v)
        
    # Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    # This sums to 0. 
    # We need sum(h) != 0.
    # The parameterization enforces sum(h)=1.
    # So we just pass x = alt[1:].
    # h[0] = 1 - sum(alt[1:]) = 1 - (-1) = 2? 
    # sum(alt) = 1 + 199*(-1) + 200*(-1)? No.
    # N=400.
    # alt[0]=1, alt[1]=-1, ..., alt[399]=-1.
    # Sum = 1 + 199*(-1) + 200*(-1)? No, indices 1..399 are 399 elements.
    # Half are -1, half are -1?
    # 1, -1, 1, -1...
    # 200 ones, 200 minus ones. Sum = 0.
    # So h[0] = 1 - sum(alt[1:]) = 1 - (-1) = 2.
    # h = [2, -1, 1, -1, ...]
    # Sum = 2 - 1 + 1 - 1 ... = 2 + 199*(-1) + 200*(-1)?
    # Let's just use the parameterization.
    initializations.append(alt[1:])
    
    # More random
    for _ in range(5):
        initializations.append(np.random.randn(N - 1))
        
    for x0 in initializations:
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B
            # We don't have bounds, so it's unconstrained in x.
            res = opt.minimize(
                _objective_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            x_opt = res.x
            h_opt = np.zeros(N)
            h_opt[0] = 1.0 - np.sum(x_opt)
            h_opt[1:] = x_opt
            
            val = _compute_value(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception as e:
            pass

    # If we still have time, try to polish the best solution
    if time.time() < deadline - 5:
        x_best = best_h[1:]
        try:
            res = opt.minimize(
                _objective_and_grad, 
                x_best, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_opt = res.x
            h_opt = np.zeros(N)
            h_opt[0] = 1.0 - np.sum(x_opt)
            h_opt[1:] = x_opt
            
            val = _compute_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except:
            pass
            
    return best_h
