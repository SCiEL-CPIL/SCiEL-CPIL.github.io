import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _exact_ratio(h):
    """
    Compute the exact ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _objective(params, beta, lambda_l1, deadline):
    """
    Objective function for minimization.
    We maximize R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    This is equivalent to minimizing -log(R(h)) = log(||h||_1 * ||h*h||_inf) - log(||h||_2^2).
    
    To handle the non-differentiability of max and potential zeros, we use:
    1. Softmax approximation for the max in autoconvolution.
    2. Log-sum-exp or similar for norms if needed, but standard log is fine if h>0.
    3. We parameterize h = exp(params) / sum(exp(params)) to ensure positivity and scale invariance (L1=1).
       Wait, if L1=1, the ratio becomes ||h||_2^2 / ||h*h||_inf.
       Let's stick to the general form to be safe, but normalization helps conditioning.
       
    Actually, let's optimize over log-weights directly.
    h_i = exp(x_i).
    Since the ratio is scale invariant, we can fix sum(h)=1 or just let the optimizer find the scale.
    However, L-BFGS-B works best with bounded variables or well-conditioned ones.
    Let's use h_i = exp(x_i) and normalize internally for the objective to prevent overflow/underflow, 
    but return the unnormalized h? No, the ratio is invariant, so we can assume sum(h)=1.
    
    Let's define h = exp(x) / sum(exp(x)).
    Then L1 = 1.
    Ratio = sum(h^2) / max(conv(h)).
    Minimize: -log(sum(h^2)) + log(max_conv_soft).
    
    max_conv_soft = logsumexp(beta * conv) / beta.
    
    We also want to encourage sparsity? 
    The problem asks for nonnegative heights. It doesn't force sparsity.
    However, the "seed" is uniform. 
    The current best strategy found that sparse structures (Sidon-like) help.
    But maybe a dense, smooth structure is better?
    The inspiration 3 used basis functions.
    
    Let's try a direct continuous optimization with a sparsity-promoting term 
    or just let the dynamics find the structure.
    Actually, adding a penalty for L1 norm of h (which is 1) doesn't make sense.
    Adding a penalty for the number of non-zeros is hard.
    
    Alternative: Minimize -Ratio directly with a smooth max.
    Let's use the log-domain optimization.
    """
    # Check time
    if time.time() > deadline:
        return 1e10  # Penalize timeout
    
    # Reconstruct h from log-weights, normalized to sum=1
    # Use stable softmax
    x = params - np.max(params)
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x)
    if sum_exp < 1e-15:
        return 1e10
    h = exp_x / sum_exp
    
    # Compute L2 squared
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-15:
        return 1e10
    
    # Compute autoconvolution
    conv = np.convolve(h, h, mode='full')
    
    # Smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    # logsumexp(beta * conv) / beta
    # To avoid overflow in exp, subtract m
    max_conv_soft = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    
    if max_conv_soft < 1e-15:
        return 1e10
        
    # Objective: minimize -log(l2_sq / max_conv_soft) = log(max_conv_soft) - log(l2_sq)
    return np.log(max_conv_soft) - np.log(l2_sq)

def _run_optimization(init_x, deadline):
    """
    Run L-BFGS-B optimization starting from init_x (log-weights).
    """
    results = []
    
    # Try different betas for the smooth max
    betas = [10.0, 30.0, 100.0, 300.0]
    
    for beta in betas:
        if time.time() > deadline - 2:
            break
            
        try:
            res = minimize(
                _objective,
                init_x,
                args=(beta, 0.0, deadline),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Reconstruct h
            x_res = res.x
            x_res = x_res - np.max(x_res)
            exp_x = np.exp(x_res)
            sum_exp = np.sum(exp_x)
            if sum_exp > 1e-15:
                h_res = exp_x / sum_exp
                score = _exact_ratio(h_res)
                results.append((score, h_res))
        except Exception:
            pass
            
    if not results:
        return None, 0.0
        
    # Return the best result
    best_score, best_h = max(results, key=lambda x: x[0])
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_h = None
    best_score = 0.0
    
    # Strategy: Multiple random restarts in log-space
    # We want to explore the space of non-negative vectors.
    # Uniform distribution in log-space corresponds to a heavy-tailed distribution in h-space.
    # We will generate random log-weights and optimize from there.
    
    np.random.seed(42)
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate random initial log-weights
        # Mix of uniform and sparse-like initializations
        if i % 2 == 0:
            # Dense random
            init_x = np.random.uniform(-5, 5, size=N)
        else:
            # Sparse-like: few large values, many small
            init_x = np.random.exponential(scale=1.0, size=N)
            # Add some noise
            init_x += np.random.uniform(-2, 2, size=N)
        
        h_opt, score_opt = _run_optimization(init_x, deadline)
        
        if h_opt is not None and score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
            
    # Fallback: Uniform
    h_uniform = np.ones(N) / N
    score_uniform = _exact_ratio(h_uniform)
    if best_h is None or score_uniform > best_score:
        best_h = h_uniform
        best_score = score_uniform
        
    # Final polish: Try to optimize the best found solution further with a high beta
    if best_h is not None and time.time() < deadline - 10:
        # Convert best_h to log-space
        # Add small epsilon to avoid log(0)
        h_safe = np.maximum(best_h, 1e-10)
        init_x = np.log(h_safe)
        
        h_polished, score_polished = _run_optimization(init_x, deadline)
        if h_polished is not None and score_polished > best_score:
            best_h = h_polished
            best_score = score_polished
            
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()
