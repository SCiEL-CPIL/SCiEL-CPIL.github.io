import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def neg_log_ratio(h):
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        l2_sq = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 1e10
        return -np.log(l2_sq / max_conv)

    best_h = np.zeros(N)
    best_score = 0.0

    # Strategy: Gradient flow / game-theoretic approach
    # We treat the problem as minimizing the peak of the autoconvolution relative to the energy.
    # We use a gradient descent method with adaptive step size on the probability simplex.
    
    np.random.seed(42)
    
    # Initial guesses
    inits = [
        np.ones(N) / N,
        np.zeros(N), # Will be filled
        np.zeros(N), # Will be filled
    ]
    
    # 1. Uniform
    inits[0] = np.ones(N) / N
    
    # 2. Sparse random
    for _ in range(10):
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0 / k
        inits.append(h)
        
    # 3. Structured (e.g., powers of 2 indices for Sidon-like properties)
    idxs = [2**i for i in range(1, 6) if 2**i < N]
    if len(idxs) > 0:
        h = np.zeros(N)
        h[idxs] = 1.0 / len(idxs)
        inits.append(h)
        
    # 4. Gaussian-like
    x = np.linspace(-1, 1, N)
    h = np.exp(-x**2 * 5)
    h /= np.sum(h)
    inits.append(h)

    # Gradient Descent with projection onto simplex
    # We minimize f(h) = -log( ||h||_2^2 / max_conv(h) )
    # Since the function is homogeneous of degree 0, we can work on the simplex sum(h)=1.
    # However, the max_conv is non-smooth. We use subgradients or numerical gradients.
    # To be efficient, we use a simple projected gradient descent.
    
    # Precompute convolution indices? No, np.convolve is fast enough for N=50.
    
    def project_simplex(v):
        # Project v onto the probability simplex
        n = len(v)
        if np.sum(v) < 1e-10:
            return np.ones(n) / n
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    # We will run gradient flow for each initial guess
    # To handle the non-differentiability of max, we can use a soft-max approximation
    # or just use numerical gradients. Given N=50, numerical gradients are feasible.
    
    # Actually, a better approach for "game-theoretic" or "flow" is to use
    # multiplicative updates or similar. But let's stick to gradient descent with
    # a small number of iterations and multiple restarts, which is effectively
    # what the previous code did but less efficiently.
    
    # The directive suggests a "first-order game-theoretic gradient flow".
    # This often implies updating h_i proportional to exp(-lambda * grad_i).
    # Let's implement a simple gradient descent with adaptive step size.
    
    def optimize_from(h0):
        h = h0.copy()
        h = project_simplex(h)
        
        step_size = 0.05
        patience = 50
        no_improve = 0
        current_score = score(h)
        best_local = h.copy()
        best_local_score = current_score
        
        for i in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            # Compute numerical gradient of neg_log_ratio
            # To save time, we can use a smaller eps or batch
            eps = 1e-5
            grad = np.zeros(N)
            f0 = neg_log_ratio(h)
            
            # Only compute gradient for non-zero entries? 
            # The gradient is 0 for zero entries in many cases, but not strictly due to max_conv.
            # Let's compute for all to be safe, but it's 50 * 50 operations per step.
            # 2000 steps * 50 dims * 2 convs = 200,000 convs. 
            # Each conv is O(N^2) = 2500 ops. Total ~ 5e8 ops. Might be too slow.
            
            # Optimization: Analytical gradient?
            # f = -log( sum(h^2) / max_conv )
            # Let A = sum(h^2), B = max_conv. f = -log(A) + log(B).
            # df/dh = -2h/A + (1/B) * dB/dh.
            # B = max_k conv_k. dB/dh is the gradient of the max convolution bin.
            # conv_k = sum_i h_i h_{k-i}.
            # dB/dh_j = sum_{i: k-i=j} h_i + sum_{i: i=j} h_{k-i} = sum_{i} (h_i * delta_{k-i, j} + h_{k-i} * delta_{i, j})
            # This is the convolution of h with a delta at k, plus delta at k convolved with h.
            # Essentially, if the max is at index k, the gradient contribution from B is
            # (h * delta_k + delta_k * h) / B.
            # delta_k * h is h shifted by k.
            
            conv = np.convolve(h, h, mode='full')
            k_max = np.argmax(conv)
            B = conv[k_max]
            A = np.sum(h**2)
            
            if A < 1e-10 or B < 1e-10:
                break
                
            # Gradient of -log(A) is -2h/A
            grad_A = -2.0 * h / A
            
            # Gradient of log(B)
            # B = h . h_shifted_k + h_shifted_k . h (if k is even/odd considerations?)
            # Let's just use the derivative of the specific bin k_max.
            # conv_k = sum_{i=0}^{N-1} h_i h_{k-i} (with h_j=0 if j<0 or j>=N)
            # d(conv_k)/dh_j = h_{k-j} + h_{k-j} ? No.
            # conv_k = sum_i h_i h_{k-i}
            # d/dh_j = h_{k-j} (from first term) + h_{k-j} (from second term, if we consider symmetry)
            # Wait, conv is defined as sum_{i} h_i h_{j-i} for full mode?
            # np.convolve(h, h, 'full')[k] = sum_{i} h[i] h[k-i]
            # d/dh[j] sum_i h[i] h[k-i] = h[k-j] + (if k-j is in range, the term where i=j is h[j]h[k-j], derivative is h[k-j] + h[k-j]? No.)
            # Let's check: f(h) = h[0]h[k] + h[1]h[k-1] + ... + h[k]h[0]
            # d/dh[j]: 
            # If j is in the first factor: h[k-j]
            # If j is in the second factor: h[k-j] (since the term is h[k-j]h[j])
            # So it is 2 * h[k-j] if 0 <= k-j < N.
            
            grad_B = np.zeros(N)
            idx = k_max - np.arange(N)
            valid = (idx >= 0) & (idx < N)
            grad_B[valid] = 2.0 * h[idx[valid]]
            
            grad_B /= B
            
            grad = grad_A + grad_B
            
            # Gradient Descent Step
            h_new = h - step_size * grad
            h_new = project_simplex(h_new)
            
            # Check improvement
            new_score = score(h_new)
            
            if new_score > current_score:
                h = h_new
                current_score = new_score
                no_improve = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = h.copy()
            else:
                no_improve += 1
                # Adaptive step size
                if no_improve > patience:
                    step_size *= 0.8
                    no_improve = 0
                    if step_size < 1e-5:
                        break
            
        return best_local, best_local_score

    # Run optimization from multiple starts
    for h0 in inits:
        if time.time() > deadline - 10.0:
            break
        h_opt, s_opt = optimize_from(h0)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt

    # Final polish with smaller steps if time permits
    if time.time() < deadline - 5.0 and best_score > 0:
        h0 = best_h.copy()
        h_opt, s_opt = optimize_from(h0) # This will reuse the function
        # The function above has a fixed number of iterations. 
        # Let's do a quick local search with even smaller steps
        # Actually, optimize_from already does adaptive steps.
        # Let's just call it again to see if it improves further?
        # It might get stuck in the same local minimum.
        
        # Try a few random perturbations of the best solution
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            h_pert = best_h + np.random.randn(N) * 0.01
            h_pert = project_simplex(h_pert)
            h_opt, s_opt = optimize_from(h_pert)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt

    return list(best_h)
