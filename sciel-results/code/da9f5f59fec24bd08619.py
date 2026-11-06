import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    n = 24
    # Use a fine grid for local search evaluation
    # 1024 points is a good balance. 2048 might be safer for precision.
    num_points = 1024
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    z = np.exp(1j * thetas)
    
    # Precompute powers matrix Z (num_points x n)
    Z = np.ones((num_points, n), dtype=np.complex128)
    for k in range(1, n):
        Z[:, k] = Z[:, k-1] * z
        
    def get_sup_norm(coeffs):
        # coeffs is a numpy array of 1s and -1s
        vals = Z @ coeffs
        return np.max(np.abs(vals))
        
    def refine_peak(coeffs, theta_init):
        # Refine the peak around theta_init using golden section
        def f(theta):
            z_point = np.exp(1j * theta)
            p_val = 0.0
            zp = 1.0 + 0.0j
            for c in coeffs:
                p_val += c * zp
                zp *= z_point
            return np.abs(p_val)
            
        # Determine a small interval around theta_init
        # The grid spacing is 2*pi/num_points
        delta = 2 * np.pi / num_points
        # Expand interval slightly to ensure the peak is inside
        a = theta_init - delta
        b = theta_init + delta
        
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = f(c)
        fd = f(d)
        
        for _ in range(30): # More iterations for higher precision
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
                
        return f((a + b) / 2)

    # Generate starting candidates
    candidates = []
    
    # 1. Rudin-Shapiro sequence
    # r_0 = 1, r_1 = 1
    # r_{2n} = r_n, r_{2n+1} = -r_n
    # We generate up to 24
    def rudin_shapiro(n):
        if n == 0:
            return np.array([1])
        if n == 1:
            return np.array([1])
        # Recursive generation is slow for large n, but 24 is small.
        # Let's do it iteratively.
        seq = [1]
        for i in range(1, n):
            # Determine bit pattern of i
            # Actually, the standard definition is:
            # r_0 = 1
            # r_{2n} = r_n
            # r_{2n+1} = -r_n
            # This generates the sequence for indices 0..2^k-1.
            # For n=24, we need indices 0..23.
            pass
        
        # Let's just hardcode the first few or use a simpler random start.
        # Actually, let's just use random starts and one all-ones.
        # The local search is powerful.
        return None

    # Start 1: All 1s
    candidates.append(np.ones(n, dtype=np.float64))
    
    # Start 2: Alternating 1, -1
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.float64))
    
    # Start 3-10: Random
    np.random.seed(42)
    for _ in range(8):
        candidates.append(np.random.choice([1, -1], n).astype(np.float64))
        
    # Add Rudin-Shapiro if we can generate it easily
    # r_n based on binary representation of n?
    # r_n = (-1)^{s_2(n)} where s_2(n) is the number of 1s in the binary representation? No, that's Thue-Morse.
    # Rudin-Shapiro is different.
    # Let's stick to the randoms and simple patterns.
    
    best_coeffs = None
    best_score = float('inf')
    
    for start_coeffs in candidates:
        if time.time() > deadline - 10.0:
            break
            
        coeffs = start_coeffs.copy()
        current_max = get_sup_norm(coeffs)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for k in range(n):
                if time.time() > deadline - 5.0:
                    break
                    
                # Flip coefficient k
                old_val = coeffs[k]
                coeffs[k] = -old_val
                
                new_max = get_sup_norm(coeffs)
                
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    # Restart the loop to check other flips
                    # This is a standard steepest descent or first-improvement hill climb.
                    # First improvement is faster.
                    break 
                else:
                    coeffs[k] = old_val
        
        # Refine the best peak
        # Find the index of the max on the grid
        vals = Z @ coeffs
        abs_vals = np.abs(vals)
        idx = np.argmax(abs_vals)
        theta_max = thetas[idx]
        
        refined_max = refine_peak(coeffs, theta_max)
        
        if refined_max < best_score:
            best_score = refined_max
            best_coeffs = coeffs.copy()
            
    # Final refinement on the best candidate
    if best_coeffs is not None:
        vals = Z @ best_coeffs
        abs_vals = np.abs(vals)
        idx = np.argmax(abs_vals)
        theta_max = thetas[idx]
        
        # Do a very precise refinement
        # We can also try shifting the peak slightly in case the grid missed it
        # But golden section around the grid peak is usually sufficient.
        
        # Return coefficients as list of ints
        return best_coeffs.astype(int).tolist()
    else:
        return [1] * n
