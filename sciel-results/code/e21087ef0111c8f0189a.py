"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Uses Subgradient Descent with Adaptive Step Size and Renormalization.
"""
import numpy as np
import time

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    # Initial value for seed
    s_seed = np.sum(best_h)
    conv_seed = np.convolve(best_h, best_h)
    val_seed = abs(2.0 * N * np.max(conv_seed) / (s_seed * s_seed))
    best_value = val_seed

    rng = np.random.RandomState(42)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        return val

    def normalize(h, target_sum=1.0):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return h + 1e-6
        return h * (target_sum / s)

    def subgradient_step(h, eta):
        """
        Perform one subgradient step to minimize max(conv(h,h)) assuming sum(h)=1.
        """
        conv = np.convolve(h, h)
        k_star = np.argmax(conv)
        
        # The derivative of c_k with respect to h_i is:
        # c_k = sum_j h_j h_{j+k}
        # d c_k / d h_i = h_{i-k} + h_{i+k} (if indices valid)
        
        grad = np.zeros(N)
        
        # Term 1: h_{i-k_star}
        # i - k_star >= 0 => i >= k_star
        # i - k_star < N => i < N + k_star (always true for i < N if k_star >= 0)
        if k_star < N:
            # Indices i such that 0 <= i - k_star < N
            # i ranges from k_star to k_star + N - 1
            # But i is in [0, N-1]
            # So i in [k_star, N-1]
            # h_{i-k_star} corresponds to h[0 : N-k_star]
            grad[k_star:] += h[:N-k_star]
            
            # Term 2: h_{i+k_star}
            # i + k_star < N => i < N - k_star
            # i >= 0
            # So i in [0, N-k_star-1]
            # h_{i+k_star} corresponds to h[k_star : N]
            if N - k_star > 0:
                grad[:N-k_star] += h[k_star:]
        
        # Update
        h_new = h - eta * grad
        return h_new

    # Generate diverse initial points
    initial_points = []
    
    # 1. All ones (normalized)
    initial_points.append(np.ones(N))
    
    # 2. Random binary sequences (+1/-1)
    for _ in range(20):
        h = rng.choice([-1.0, 1.0], size=N)
        initial_points.append(h)
        
    # 3. Random Gaussian
    for _ in range(20):
        h = rng.normal(0, 1, N)
        initial_points.append(h)
        
    # 4. Smoothed random
    for _ in range(10):
        h = rng.normal(0, 1, N)
        # Simple moving average smoothing
        kernel = np.ones(5) / 5.0
        h = np.convolve(h, kernel, mode='same')
        initial_points.append(h)
        
    # 5. Low frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        
    # 6. Sparse spikes
    for _ in range(10):
        h = np.zeros(N)
        k = rng.randint(20, 100)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Sort initial points by their current value (normalized)
    def init_sort_key(h):
        h_norm = normalize(h)
        return evaluate(h_norm)

    initial_points.sort(key=init_sort_key)

    # Optimization Loop
    # We use a "chasing" subgradient method with adaptive step size.
    # Since the function is non-smooth, we use a decreasing step size schedule
    # but with random restarts if stuck.
    
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break
            
        h = normalize(init_h.copy())
        current_val = evaluate(h)
        
        if current_val < best_value:
            best_value = current_val
            best_h = h.copy()

        # Local optimization via subgradient descent
        # We want to minimize max(conv). 
        # Step size eta needs to be small enough to avoid oscillation.
        # Typical scale of h is 1/N ~ 0.0025. Grad scale is similar.
        # Let's start with eta = 0.01 and decay.
        
        eta = 0.05
        min_eta = 1e-5
        decay = 0.999
        stuck_count = 0
        max_iter = 2000
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
                
            h_new = subgradient_step(h, eta)
            h_new = normalize(h_new)
            
            new_val = evaluate(h_new)
            
            if new_val < current_val:
                current_val = new_val
                h = h_new
                stuck_count = 0
                if current_val < best_value:
                    best_value = current_val
                    best_h = h.copy()
            else:
                # Reject step, reduce step size
                eta *= decay
                stuck_count += 1
                if stuck_count > 100:
                    # Try a random perturbation to escape local minimum
                    h = h + 0.01 * rng.normal(0, 1, N)
                    h = normalize(h)
                    current_val = evaluate(h)
                    stuck_count = 0
                    eta = 0.01 # Reset step size slightly
            
            if eta < min_eta:
                break
                
    # Final Polish with very small steps
    if time.time() - start_time < DEADLINE - 2:
        h = best_h.copy()
        h = normalize(h)
        eta = 1e-4
        for _ in range(1000):
            if time.time() - start_time > DEADLINE - 1:
                break
            h_new = subgradient_step(h, eta)
            h_new = normalize(h_new)
            new_val = evaluate(h_new)
            if new_val < best_value:
                best_value = new_val
                best_h = h_new.copy()
                h = h_new
            else:
                eta *= 0.95
            if eta < 1e-8:
                break

    return best_h
