import numpy as np

def construct():
    # 1. Initial Guess: Golay-Rudin-Shapiro sequence truncated to 24
    # R-S sequence generation
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    coeffs = np.array(p[:24], dtype=float)
    
    # 2. Evaluation Function
    # We need high accuracy, so we use dense sampling + refinement
    N = 24
    M = 2048 # Power of 2 for FFT? No, we just need dense sampling. 2000 is fine.
    # Actually, for refinement, we need to evaluate specific points.
    
    def eval_score(c):
        # c is a numpy array of shape (24,)
        # Sample the circle densely
        # Using FFT would be faster for the initial dense scan, but N is small.
        # Direct summation is O(N*M). 24*2048 is ~50k ops. Very fast.
        angles = 2 * np.pi * np.arange(M) / M
        # Build the Vandermonde-like matrix Z where Z[i, k] = exp(i * k * angles[i])
        # To avoid recomputing Z every time, we can precompute it outside.
        pass

    # Precompute Z
    angles = 2 * np.pi * np.arange(M) / M
    # Z[i, k] = exp(1j * k * angles[i])
    k_idx = np.arange(N)
    Z = np.exp(1j * np.outer(angles, k_idx))
    
    def get_max_norm(c):
        # c is float array
        # P(z) = Z @ c
        P = Z @ c
        abs_P = np.abs(P)
        
        # Find the global max
        max_val = np.max(abs_P)
        
        # Refine the highest peaks to get 1e-9 accuracy
        # Find indices of the top 10 peaks
        # To avoid refining the same peak multiple times, we can just take the argmax
        # and refine around it. But there might be multiple peaks close in value.
        # Let's take the top 5.
        
        # Get indices of top 5
        top_indices = np.argsort(abs_P)[-5:][::-1]
        
        refined_max = 0
        for idx in top_indices:
            # Interval around idx
            # The sample spacing is 2*pi/M
            # We search in [angle[idx-1], angle[idx+1]] or just around angle[idx]
            # Golden section search on the function f(theta) = |P(e^{i theta})|
            
            # Define the function for refinement
            def f(theta):
                z = np.exp(1j * theta)
                # P(z) = sum c_k z^k
                # We can compute this directly for a single point
                # z_powers = z ** k_idx
                # return np.abs(np.dot(c, z_powers))
                # This is O(N) = 24 ops. Very fast.
                return np.abs(np.dot(c, z ** k_idx))
            
            # Golden section search parameters
            # We need a bracket. The peak is at angles[idx].
            # The next samples are at angles[idx] +/- delta
            delta = 2 * np.pi / M
            a = angles[idx] - delta
            b = angles[idx] + delta
            
            # Ensure we wrap around?
            # Since we are searching locally, and M is large, wrapping is unlikely to be an issue
            # unless idx is near 0 or M-1.
            # For simplicity, we assume the peak is well within the interval.
            # If idx == 0, a is negative. np.exp handles negative angles fine.
            
            # Perform Golden Section Search
            gr = (np.sqrt(5) - 1) / 2 # ~0.618
            x1 = b - gr * (b - a)
            x2 = a + gr * (b - a)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(50): # 50 iterations is plenty for 1e-9 on this scale
                if f1 < f2:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + gr * (b - a)
                    f2 = f(x2)
                else:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = b - gr * (b - a)
                    f1 = f(x1)
                    
            peak_val = max(f1, f2)
            if peak_val > refined_max:
                refined_max = peak_val
                
        return refined_max

    # 3. Optimization: Simulated Annealing
    current_coeffs = coeffs.copy()
    current_score = get_max_norm(current_coeffs)
    
    best_coeffs = current_coeffs.copy()
    best_score = current_score
    
    # Annealing schedule
    T_start = 1.0
    T_end = 0.001
    iterations = 20000
    # Linear cooling or geometric? Geometric is standard.
    # Let's use a simple linear cooling in log space or just linear T.
    
    for i in range(iterations):
        # Temperature
        T = T_start * (T_end / T_start) ** (i / iterations)
        
        # Generate neighbor: flip a random coefficient
        idx = np.random.randint(N)
        new_coeffs = current_coeffs.copy()
        new_coeffs[idx] *= -1
        
        new_score = get_max_norm(new_coeffs)
        
        # Acceptance probability
        delta_score = new_score - current_score
        if delta_score < 0:
            # Improvement
            current_coeffs = new_coeffs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            if np.random.rand() < np.exp(-delta_score / T):
                current_coeffs = new_coeffs
                current_score = new_score
        
    return best_coeffs.astype(int).tolist()
