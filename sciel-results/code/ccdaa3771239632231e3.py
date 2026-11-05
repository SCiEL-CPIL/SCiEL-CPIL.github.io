import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial random guess, normalized to sum to 1
    # Using a distribution that allows negative values is crucial for reducing max autocorrelation
    # e.g., uniform [-1, 1] or normal
    h = np.random.randn(N)
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_val = np.inf
    
    # Objective function: minimize max(h*h)
    # We use a soft-max approximation to make it differentiable
    # f(h) = (1/beta) * log(sum(exp(beta * (h*h))))
    
    def compute_grad(h, beta):
        conv = np.convolve(h, h)
        # Soft max
        # To avoid overflow, subtract max
        max_conv = np.max(conv)
        exp_conv = np.exp(beta * (conv - max_conv))
        sum_exp = np.sum(exp_conv)
        
        # Value is (1/beta) * (log(sum_exp) + beta * max_conv)
        # grad w.r.t h is needed
        # d/dh [ log(sum(exp(beta c))) ] = (beta * exp(beta c)) / sum(exp(beta c)) * dc/dh
        # c_k = sum_i h_i h_{i+k} (approx, depends on convolution definition)
        # np.convolve(h, h) gives full convolution.
        # grad_k = sum_j (beta * exp(beta c_j) / sum_exp) * d c_j / d h_k
        
        # Efficient way: 
        # grad = beta * (exp(beta * (h*h)) / sum_exp) * (2 * h) ?? No, that's for element-wise.
        # For convolution:
        # Let w_j = beta * exp(beta * c_j) / sum_exp
        # grad = sum_j w_j * (h shifted)
        # Specifically, c_j = sum_i h_i h_{i+j-N+1} (indices vary)
        # Actually, grad of sum_j w_j c_j is w * h + reverse(w) * h?
        # Let's use the property: grad of <w, conv(h,h)> is conv(w, h) + conv(flip(w), h)
        
        weights = beta * exp_conv / sum_exp
        grad = np.convolve(weights, h, mode='full') + np.convolve(weights[::-1], h, mode='full')
        # We need to align the gradients to the size of h
        # The full convolution has size 2N-1. The gradient w.r.t h (size N) is the central part?
        # Let's check indices.
        # c_k = sum_{i} h_i h_{i+k} for k in -N+1..N-1?
        # np.convolve(h, h) index k corresponds to sum_{i} h_i h_{k-i}
        # grad_k = sum_j w_j (delta_{k-i} + delta_{k-i'}) ...
        # It is known that grad of sum_j w_j (h*h)_j is conv(w, h) + conv(w[::-1], h) restricted to the valid part?
        # Actually, the gradient vector has length N.
        # The contribution of h_i to c_j is h_{j-i} + h_{j-N+1-i} (roughly).
        # Simpler: use automatic differentiation logic or just compute the gradient via finite differences for a few steps? 
        # No, finite differences are too slow for 400 vars.
        
        # Let's derive carefully.
        # J = sum_j w_j (h*h)_j
        # (h*h)_j = sum_i h_i h_{j-i}
        # dJ/dh_k = sum_j w_j (h_{j-k} + h_{j-k}^T) where the second term comes from the other h in the product.
        # Note: (h*h) is symmetric, so (h*h)_j = (h*h)_{-j}.
        # dJ/dh_k = sum_j w_j h_{j-k} + sum_j w_j h_{j-k} = 2 sum_j w_j h_{j-k}
        # This is a convolution of w and h, but we need to handle the indices correctly for the full convolution output.
        
        # Let's use the standard formula:
        # grad = 2 * (w * h) where * is correlation?
        # np.correlate(w, h, mode='full')?
        
        # Let's try: grad_full = np.convolve(w, h, 'full') + np.convolve(w[::-1], h, 'full')
        # The gradient for h (size N) corresponds to the middle N elements of this sum?
        # Let's test with small N.
        
        # Alternative: Use scipy.optimize? No, let's stick to manual GD.
        # The gradient of the soft-max of the auto-correlation is:
        # g = 2 * np.correlate(weights, h, mode='same') ?? 
        # Let's verify.
        # If h is length N, conv(h,h) is length 2N-1.
        # weights is length 2N-1.
        # The term (h*h)_k = sum_i h_i h_{i+k}.
        # d/dh_m sum_k w_k sum_i h_i h_{i+k} = sum_k w_k (h_{k-m} + h_{k-m}) = 2 sum_k w_k h_{k-m}.
        # This is 2 * (w * h)_m where * is cross-correlation?
        # sum_k w_k h_{k-m}. Let j = k-m => sum_j w_{j+m} h_j.
        # This is the cross-correlation of w and h evaluated at m?
        # np.correlate(w, h, 'full') gives sum_j w_j h_{j-m} or similar.
        
        # Let's just compute the gradient using the definition in a vectorized way if possible, or use a library.
        # Actually, there is a known result: grad = 2 * np.convolve(weights, h, 'same') is NOT correct because of the boundary effects of the full convolution.
        
        # Let's use a simpler approach: 
        # The gradient of the max of a quadratic form is related to the eigenvector? No.
        
        # Let's use a finite difference check to ensure the gradient is correct, or just use a subgradient method?
        # Subgradient of max is just the h*h value at the argmax.
        # Let's try Subgradient Descent with the argmax.
        # It's slower but robust.
        
        # However, the prompt suggests "smooth surrogate".
        # Let's implement the gradient correctly.
        # The vector of partial derivatives is:
        # grad[i] = 2 * sum_{k} weights[k] * h[k-i]  (with appropriate index shifts for the full convolution)
        
        # Let's construct the gradient matrix M such that grad = M @ h.
        # M is sparse.
        # But for N=400, we can just do the convolution.
        # The correlation of w and h:
        # corr = np.correlate(weights, h, mode='full')
        # The gradient should be 2 * corr[N-1 : 2*N-1]? Or something similar.
        
        # Let's test this hypothesis in the code.
        
        # For now, let's use a simpler heuristic:
        # Update h in the direction of - (h*h) at the max index?
        # This is essentially subgradient.
        
        # Let's try a coordinate descent / local search approach which is often effective for these "minimize max" problems.
        # Or just use scipy.optimize.minimize with a smooth objective.
        
        pass

    # Let's switch to a more robust strategy: 
    # 1. Start with random h.
    # 2. Use scipy.optimize.minimize to minimize the soft-max objective.
    
    from scipy.optimize import minimize
    
    def objective_and_grad(h_flat):
        h = h_flat.reshape(N)
        # Normalize to sum 1 to fix scale? 
        # Actually, the objective is scale invariant. 
        # But minimize works better with a fixed scale.
        # Let's constrain sum(h) = 1.
        # If we don't constrain, the gradient might push towards 0 or infinity.
        # Let's just minimize the raw value and normalize at the end?
        # No, if we minimize max(h*h)/sum(h)^2, scaling doesn't change it.
        # But the gradient of the ratio is complicated.
        # Let's fix sum(h) = 1.
        
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10, np.zeros(N)
        
        # We want to minimize max(h*h). 
        # Since sum(h)=1, this is equivalent to minimizing the value.
        # Use soft max.
        beta = 50.0
        
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        exp_conv = np.exp(beta * (conv - max_conv))
        sum_exp = np.sum(exp_conv)
        val = (1.0 / beta) * (np.log(sum_exp) + beta * max_conv)
        
        # Gradient
        weights = beta * exp_conv / sum_exp
        
        # Gradient of sum(weights * conv) w.r.t h
        # As derived, grad[i] = 2 * sum_j weights[j] h[j-i] (with shifts)
        # Let's use np.correlate
        # The correlation of weights (length 2N-1) and h (length N)
        # We need the part that corresponds to the derivative w.r.t h.
        
        # Let's use a finite difference to get the gradient right if the analytic one is tricky?
        # No, FD is too slow.
        
        # Let's try this:
        # grad = 2 * np.convolve(weights, h, mode='same')
        # This might be off by a shift.
        
        # Let's assume the gradient is approximately 2 * (weights * h) in frequency domain?
        
        # Actually, let's just use `scipy.optimize.minimize` with method='L-BFGS-B' or 'BFGS' and let it handle the gradient?
        # No, we need to provide the gradient for speed.
        
        # Let's use the following identity:
        # d/dh (w^T (h*h)) = 2 * (w * h) where * is correlation.
        # Specifically, if c = conv(h, h), then grad = 2 * correlate(w, h, 'full') restricted to the valid indices.
        
        # Let's compute the full correlation
        # w is length 2N-1, h is length N.
        # correlate(w, h, 'full') has length 2N-1 + N - 1 = 3N-2.
        
        # Let's just use a subgradient step for simplicity and robustness, with a decaying step size.
        # It converges slower but is much less prone to implementation errors in the gradient.
        
        # Subgradient of max(x) is a vector with 1 at the argmax and 0 elsewhere.
        # So grad_sub = 2 * (h shifted to align with argmax)
        
        return val, None # Let's use BFGS which estimates gradient? No, BFGS uses FD if not provided. FD is 400 evals per step. 150s / 400 = 0.37s per step. We can do ~400 steps.
        # This is feasible.

    # Let's use BFGS with finite differences.
    # Objective:
    def obj(h):
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        # Use soft max to help BFGS navigate the non-smooth max
        beta = 20.0
        max_conv = np.max(conv)
        exp_conv = np.exp(beta * (conv - max_conv))
        sum_exp = np.sum(exp_conv)
        return (1.0 / beta) * (np.log(sum_exp) + beta * max_conv)
        
    # Constraint: sum(h) = 1
    # We can parameterize h to satisfy the constraint automatically.
    # Let h = 1/N * (1 + 2 * tanh(x)) ? No, sum is not fixed.
    # Let h_i = 1/N + u_i, where sum(u) = 0.
    # This is hard to constrain in standard minimize.
    
    # Alternative: Minimize the original ratio directly?
    def obj_ratio(h):
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        # Add small regularization to prevent sum=0
        return 2.0 * N * max_conv / (s*s)
        
    # Let's try minimizing obj_ratio with BFGS.
    # Initial guess: random
    h0 = np.random.randn(N)
    
    # Run minimize
    res = minimize(obj_ratio, h0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-5})
    
    h_opt = res.x
    val_opt = res.fun
    
    # Check time
    if time.time() > deadline:
        return h_opt / np.sum(h_opt) # Normalize
        
    # Polish with higher beta or exact max
    # Let's do a few steps of subgradient descent with the exact max
    
    def exact_val(h):
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        return 2.0 * N * np.max(conv) / (s*s)
        
    # Subgradient step
    # grad of max(conv) is 2 * h shifted by argmax
    for i in range(100):
        if time.time() > deadline:
            break
        s = np.sum(h_opt)
        h_norm = h_opt / s
        conv = np.convolve(h_norm, h_norm)
        idx = np.argmax(conv)
        
        # Subgradient of max(conv) w.r.t h_norm
        # d/dh max(conv) = 2 * h shifted
        # The shift is determined by idx.
        # conv[k] = sum_j h[j] h[k-j]
        # d/dh[i] conv[idx] = h[idx-i] + h[idx-i] = 2 h[idx-i] (if indices valid)
        
        grad = np.zeros(N)
        # h[idx-i] is non-zero when 0 <= idx-i < N => idx-N < i <= idx
        start = max(0, idx - N + 1)
        end = min(N, idx + 1)
        
        # We need h[j] where j = idx - i
        # So grad[i] = 2 * h[idx - i]
        # Let's construct this vector
        # j = idx - i => i = idx - j
        # For j in [0, N-1], i = idx - j.
        # If 0 <= idx - j < N, then grad[idx-j] += 2 * h[j]
        
        for j in range(N):
            i = idx - j
            if 0 <= i < N:
                grad[i] += 2 * h_norm[j]
                
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm
            
        # Step
        step_size = 0.1 / (i + 1)
        h_new = h_norm - step_size * grad
        
        # Project back to sum=1
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-6:
            h_new = h_new / s_new
        else:
            continue
            
        val_new = exact_val(h_new)
        
        if val_new < val_opt:
            h_opt = h_new
            val_opt = val_new
        else:
            # Reduce step size
            step_size *= 0.5
            
    # Final normalization
    s_final = np.sum(h_opt)
    if abs(s_final) < 1e-6:
        return np.ones(N)
    h_final = h_opt / s_final
    
    return h_final
