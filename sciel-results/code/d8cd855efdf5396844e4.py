import numpy as np

def construct():
    # Start with Golay-Rudin-Shapiro polynomial truncated to 24
    p, q = [1], [1]
    while len(p) < 32: 
        p, q = p + q, p + [-c for c in q]
    coeffs = np.array(p[:24], dtype=np.float64)
    
    # Pre-sample points on the unit circle
    num_samples = 2048 # Increased density for better peak detection
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    k_vals = np.arange(24)
    Z = np.exp(1j * np.outer(k_vals, angles)) # Shape (24, num_samples)
    
    def refine_peak(peak_idx, c):
        # Golden section search for local max of |P(z)| around peak_idx
        # The peak is at angles[peak_idx]. 
        # The function is periodic. The local max is near this angle.
        # We search in a window of +/- 1 sample width.
        step = 2 * np.pi / num_samples
        # Golden ratio
        gr = (np.sqrt(5) - 1) / 2
        a = angles[peak_idx] - step
        b = angles[peak_idx] + step
        
        # If peak is at 0 or end, handle wrap-around? 
        # Since we used linspace endpoint=False, 0 is start, 2pi is end (excluded).
        # The function is periodic, so we can just work in real space if we map angles appropriately, 
        # but it's easier to just evaluate P(exp(1j*theta)).
        
        def eval_modulus(theta):
            z = np.exp(1j * theta)
            # P(z) = sum c_k z^k
            # Using precomputed Z is faster for many thetas, but for 1D search, direct eval is ok?
            # Z is (24, N). We don't have Z for arbitrary theta.
            # Let's compute powers directly or use a small precomputed table?
            # 24 coeffs is small. Direct computation is fine.
            powers = z ** k_vals
            val = np.dot(c, powers)
            return np.abs(val)
            
        # Golden section search
        c_val = b - gr * (b - a)
        d_val = a + gr * (b - a)
        fc = eval_modulus(c_val)
        fd = eval_modulus(d_val)
        
        for _ in range(50): # 50 iterations gives high precision
            if fc < fd:
                a = c_val
                c_val = d_val
                fc = fd
                d_val = a + gr * (b - a)
                fd = eval_modulus(d_val)
            else:
                b = d_val
                d_val = c_val
                fd = fc
                c_val = b - gr * (b - a)
                fc = eval_modulus(c_val)
                
        mid = (a + b) / 2
        return eval_modulus(mid)

    def get_max_modulus(c):
        p_vals = Z.T @ c
        mod_vals = np.abs(p_vals)
        # Find top 5 peaks
        top_indices = np.argsort(mod_vals)[-5:]
        max_refined = 0.0
        for idx in top_indices:
            val = refine_peak(idx, c)
            if val > max_refined:
                max_refined = val
        return max_refined

    current_max = get_max_modulus(coeffs)
    
    # Local Search: Hill Climbing with accurate metric
    improved = True
    while improved:
        improved = False
        for i in range(24):
            old_val = coeffs[i]
            coeffs[i] = -old_val
            new_max = get_max_modulus(coeffs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                coeffs[i] = old_val
                
    return [int(x) for x in coeffs]
