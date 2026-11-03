import numpy as np
import time

N = 600

def compute_conv(h):
    # Full convolution of h with h
    # h is length N
    # Result length 2N-1
    return np.convolve(h, h, mode='full')

def score(h):
    # Normalize h to unit L2 norm for scoring calculation
    norm = np.linalg.norm(h)
    if norm == 0:
        return float('inf')
    h_norm = h / norm
    c = compute_conv(h_norm)
    return np.max(c)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = float('inf')
    best_h = None
    
    # Initializations
    # 1. Uniform
    h0 = np.ones(N)
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    h1 = np.exp(-x**2 / (2*0.1**2))
    # 3. Parabolic
    h2 = np.maximum(0, 1 - x**2)
    # 4. Random smooth
    np.random.seed(42)
    h3 = np.random.rand(N)
    # Smooth h3
    h3 = np.convolve(h3, np.ones(10)/10, mode='same')
    
    inits = [h0, h1, h2, h3]
    
    for h_init in inits:
        if time.time() > deadline:
            break
            
        h = h_init.copy()
        # Normalize
        norm = np.linalg.norm(h)
        if norm > 0:
            h = h / norm
            
        best_local = float('inf')
        
        # Gradient Descent
        # Objective: min max(conv(h))
        # Using soft-max surrogate: log(sum(exp(beta * conv)))
        beta = 50.0
        
        for it in range(1000):
            if time.time() > deadline:
                break
            
            c = compute_conv(h)
            # Soft max
            max_c = np.max(c)
            # Subtract max for numerical stability
            exp_c = np.exp(beta * (c - max_c))
            sum_exp = np.sum(exp_c)
            # Gradient of log-sum-exp wrt c is exp_c / sum_exp
            # Gradient wrt h: d/dh log(sum exp(beta c)) = sum (dc/dh * exp_c / sum_exp * beta)
            # c = conv(h, h). dc/dh is conv(e, h) where e is the weight vector?
            # Actually, c_k = sum_i h_i h_{k-i}.
            # d c_k / d h_j = h_{k-j} + h_{j-k} (if we consider symmetry and boundaries carefully)
            # Simpler: use the fact that gradient of ||conv(h)||^2 is 2*conv(conv(h), h).
            # But we want max.
            # Let's use the subgradient of max: vector g where g_k = 1 if c_k is max, else 0.
            # To smooth, use weights w_k = exp(beta(c_k - max_c)) / sum_exp.
            # Then gradient is sum_k w_k * d c_k / d h.
            # d c_k / d h_j = h_{k-j} + h_{j-k} (indices adjusted for full conv)
            # This is effectively conv(w, h) + conv(w[::-1], h)?
            # Let's use a numerical gradient check or a simpler approach:
            # Coordinate descent is O(N^2) per step. N=600, N^2=360k. 1000 steps = 360M ops. Might be too slow in Python loop, but vectorized?
            
            # Let's try a different approach: 
            # Minimize sum(c^2) subject to max(c) <= t? No.
            
            # Let's use the property that for non-negative h, the peak is at the center if h is symmetric.
            # We can enforce symmetry h[i] = h[N-1-i].
            # This reduces variables to N/2.
            
            # Let's try a simple projected gradient descent on the smooth objective:
            # J(h) = log(sum(exp(beta * c)))
            # grad J = beta * conv(w, h) where w = exp(beta*(c-max))/sum(exp(beta*(c-max)))
            # Note: conv(w, h) computes sum_i w_i h_{j-i}.
            # The derivative of c_k = sum_i h_i h_{k-i} wrt h_j is h_{k-j} + h_{j-k} (if k-j and j-k are valid indices).
            # This is equivalent to (w * h) + (w * h)[::-1]? 
            # Let's just use the gradient of the squared norm of the weighted convolution?
            # Actually, let's use a simpler heuristic: 
            # If we minimize the L2 norm of the convolution, we minimize the energy. 
            # But we want to minimize the MAX.
            
            # Let's use a "peak shaving" approach.
            # Identify peaks in c.
            # Reduce h at positions that contribute most to the peaks.
            
            # Alternative: Use scipy.optimize.minimize?
            # It might be too slow for 1000 iterations with N=600.
            
            # Let's stick to a vectorized gradient step.
            # w = softmax(beta * c)
            w = exp_c / sum_exp
            
            # Gradient of J wrt h
            # J = sum_k w_k c_k (approx, if w is constant)
            # dJ/dh_j = sum_k w_k (h_{k-j} + h_{j-k})
            # This is conv(w, h) + conv(w, h) reversed?
            # Let's compute g = conv(w, h)
            # The term h_{k-j} corresponds to g_j?
            # g_j = sum_i w_i h_{j-i}.
            # We need sum_k w_k h_{k-j}. Let m = k-j => k = m+j.
            # sum_m w_{m+j} h_m. This is conv(w, h) but with w shifted?
            # Actually, it's the cross-correlation.
            # grad_h = np.convolve(w, h, mode='full') + np.convolve(w, h, mode='full')[::-1]
            # Wait, c is full convolution.
            # Let's verify with small N.
            
            # For now, let's use a simpler, robust method:
            # Coordinate descent on the "active" set of peaks?
            
            # Let's try this:
            # 1. Compute c
            # 2. Find indices where c is close to max
            # 3. Update h to reduce those peaks
            # This is like subgradient descent.
            
            max_val = np.max(c)
            # Threshold
            thresh = max_val - 0.01 * max_val
            mask = c > thresh
            # Weighted gradient
            # grad = conv(mask, h) + conv(mask, h)[::-1]
            # Normalize mask
            mask_sum = np.sum(mask)
            if mask_sum == 0:
                break
            w_mask = mask / mask_sum
            
            grad = np.convolve(w_mask, h, mode='full')
            grad = grad + grad[::-1]
            # grad is length 2N-1, h is length N.
            # The gradient wrt h_j is the sum of contributions.
            # The central part of grad corresponds to h.
            # Specifically, if h is centered, the gradient for h_j is roughly grad[N-1+j] + ...?
            # Let's align it.
            # c_k = sum_i h_i h_{k-i}.
            # If we change h_j by delta, c_k changes by delta * (h_{k-j} + h_{j-k}).
            # So grad_j = sum_k w_k (h_{k-j} + h_{j-k}).
            # This is exactly the cross-correlation of w and h, summed over k?
            # No, w_k are weights.
            # grad_j = sum_k w_k h_{k-j} + sum_k w_k h_{j-k}
            # Let u = k-j => sum_u w_{u+j} h_u.
            # Let v = j-k => sum_v w_{j-v} h_v.
            # This is conv(w, h) evaluated at j?
            # np.convolve(w, h, 'full')[j] = sum_i w_i h_{j-i}.
            # So term 1 is conv(w, h)[j] ? No, indices are shifted.
            # Let's just use the full gradient vector g_full of length 2N-1.
            # g_full[k] = sum_i w_i h_{k-i} ? No.
            
            # Let's use a simpler update:
            # h_new = h - eta * (conv(w_mask, h) + conv(w_mask, h)[::-1])[N-1 : 2*N-1]
            # This assumes the gradient for h_j is at index N-1+j in the "summed" gradient?
            
            # Let's try a different, very effective method for this specific problem:
            # The optimal non-negative function minimizing the peak of autoconvolution (relative to L2 norm) is known to be related to the "optimal" pulse.
            # However, we can just run a few iterations of a simple heuristic:
            # 1. Compute c.
            # 2. h = h - eta * c_convolved_with_h? No.
            
            # Let's use scipy.optimize.minimize with method='L-BFGS-B' on a subset?
            # No, N=600 is large for L-BFGS-B without good gradient.
            
            # Let's use the "soft max" gradient properly.
            # J = log(sum(exp(beta c)))
            # w = softmax(beta c)
            # grad_J = beta * (conv(w, h) + conv(w, h)[::-1])
            # We need to map the gradient from the convolution domain (length 2N-1) back to h (length N).
            # The relationship is linear.
            # If we define the operator C: h -> conv(h, h), it's quadratic.
            # The gradient is 2 * conv(h, h_w) where h_w is h weighted?
            # Actually, d/dh log(sum exp(beta conv(h,h))) = 2 * beta * conv(h, w) ?
            # Let's check: d/dh_j sum_k w_k c_k = sum_k w_k (h_{k-j} + h_{j-k}) = 2 * (w * h)_j (if w is symmetric).
            # w is symmetric if c is symmetric (which it is).
            # So grad = 2 * beta * conv(w, h, mode='full')[N-1 : 2*N-1]
            # Wait, conv(w, h) has length 2N-1.
            # The term (w * h)_j = sum_i w_i h_{j-i}.
            # This corresponds to index j in the convolution?
            # Yes, if we align them.
            
            grad_full = np.convolve(w, h, mode='full')
            # Extract the part corresponding to h
            # h has indices 0..N-1.
            # The gradient for h_j is related to grad_full[N-1+j]?
            # Let's assume the standard alignment where the center of the convolution corresponds to the center of h.
            grad_h = grad_full[N-1 : 2*N-1]
            # Wait, length of grad_full is 2N-1.
            # Indices 0..2N-2.
            # h indices 0..N-1.
            # If h is centered at N-1, the gradient for h_j is at N-1+j?
            # Let's just take the middle N elements.
            grad_h = grad_full[N-1 : N-1+N]
            
            # Update
            eta = 0.01
            h = h - eta * grad_h
            
            # Project to non-negative
            h = np.maximum(0, h)
            
            # Normalize
            norm = np.linalg.norm(h)
            if norm < 1e-9:
                break
            h = h / norm
            
            # Check improvement
            current_score = np.max(compute_conv(h))
            if current_score < best_local:
                best_local = current_score
                best_h_local = h.copy()
                
            # Convergence check
            if it > 10 and np.abs(current_score - best_local) < 1e-6:
                break
                
        if best_local < best_score:
            best_score = best_local
            best_h = best_h_local.copy()
            
    # Return the heights
    # The problem asks for a sequence of nonnegative step heights.
    # It does not explicitly say to normalize the output, but the score is "normalized maximum".
    # If we return h such that score(h) is minimized, the evaluator will compute the score.
    # If the evaluator computes score = max(conv(h))/norm(h)^2, then any scaling is fine.
    # We should return the h that achieved the best score.
    
    if best_h is None:
        return [1.0] * N
        
    return list(best_h)
