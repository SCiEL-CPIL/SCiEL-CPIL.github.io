import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial candidate: Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=np.float64)
    
    # Precompute powers for all k up to 23
    N = 4096
    thetas = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * thetas)
    
    powers = np.ones((24, N), dtype=complex)
    for k in range(1, 24):
        powers[k] = powers[k-1] * z
        
    def fast_eval(coeffs):
        # coeffs is a numpy array or list
        return np.max(np.abs(np.dot(coeffs, powers)))
        
    def eval_at_theta(coeffs, theta):
        """Evaluate P(e^{i*theta}) and return |P|."""
        z_val = np.exp(1j * theta)
        # Compute sum c_k * z_val^k
        val = 0.0 + 0.0j
        zk = 1.0 + 0.0j
        for k in range(24):
            val += coeffs[k] * zk
            zk *= z_val
        return abs(val)
        
    def golden_section_refine(coeffs, theta_lo, theta_hi, tol=1e-12, max_iter=100):
        """
        Find the maximum of |P(z)| on [theta_lo, theta_hi] using golden section search.
        Assumes unimodal in this interval (which should be true for narrow intervals around peaks).
        """
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio ~0.618
        a = theta_lo
        b = theta_hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = eval_at_theta(coeffs, c)
        fd = eval_at_theta(coeffs, d)
        
        for _ in range(max_iter):
            if b - a < tol:
                break
            if fc < fd:
                a = c
                c = d
                d = a + gr * (b - a)
                fc = fd
                fd = eval_at_theta(coeffs, d)
            else:
                b = d
                d = c
                c = b - gr * (b - a)
                fd = fc
                fc = eval_at_theta(coeffs, c)
        
        theta_mid = (a + b) / 2.0
        return eval_at_theta(coeffs, theta_mid), theta_mid
        
    def high_precision_sup_norm(coeffs):
        """
        Compute sup norm with high precision:
        1. Dense sampling to find candidate peaks
        2. Golden-section refinement on the top peaks
        """
        # Dense sampling
        P_vals = np.dot(coeffs, powers)
        abs_vals = np.abs(P_vals)
        
        # Find top 20 peaks from dense sampling
        top_indices = np.argsort(abs_vals)[-20:]
        top_values = abs_vals[top_indices]
        
        # For each top peak, refine using golden section in a small neighborhood
        peak_width = 2 * np.pi / N  # one sample spacing
        best_refined = 0.0
        best_theta = 0.0
        
        for idx in top_indices:
            theta_peak = thetas[idx % N]
            # Search in a window of +/- 2 sample spacings around the peak
            lo = theta_peak - 2 * peak_width
            hi = theta_peak + 2 * peak_width
            refined_val, refined_theta = golden_section_refine(coeffs, lo, hi)
            if refined_val > best_refined:
                best_refined = refined_val
                best_theta = refined_theta
        
        return best_refined
        
    best_score = fast_eval(best_coeffs)
    
    rng = np.random.default_rng(42)
    
    def local_search(coeffs):
        """
        Perform 1-bit, 2-bit, and 3-bit flip hill climbing.
        Returns the improved coeffs and score.
        """
        current_score = fast_eval(coeffs)
        improved = True
        
        # 1-bit flips
        while improved:
            improved = False
            for i in range(24):
                # Flip bit i
                coeffs[i] *= -1
                new_score = fast_eval(coeffs)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    coeffs[i] *= -1
        
        # 2-bit flips (to escape local minima)
        improved = True
        while improved:
            improved = False
            for i in range(23):
                for j in range(i + 1, 24):
                    # Flip bits i and j
                    coeffs[i] *= -1
                    coeffs[j] *= -1
                    new_score = fast_eval(coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        coeffs[i] *= -1
                        coeffs[j] *= -1
        
        # 3-bit flips (to escape deeper local minima)
        improved = True
        while improved:
            improved = False
            for i in range(22):
                for j in range(i + 1, 23):
                    for k in range(j + 1, 24):
                        # Flip bits i, j, k
                        coeffs[i] *= -1
                        coeffs[j] *= -1
                        coeffs[k] *= -1
                        new_score = fast_eval(coeffs)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            # Revert
                            coeffs[i] *= -1
                            coeffs[j] *= -1
                            coeffs[k] *= -1
                        
        return coeffs, current_score

    # Run local search on the initial candidate
    best_coeffs, best_score = local_search(best_coeffs)
    
    # If we are already very close to the known optimum, we might stop early?
    # Known optimum ratio is 1.2583758.
    # Target max_norm = 1.2583758 * sqrt(24) ≈ 6.15633
    
    # List of deterministic seeds to try
    seeds = []
    # Original GRS
    seeds.append(np.array(p[:24], dtype=np.float64))
    
    # Shifted GRS (cyclic shifts)
    for shift in range(1, 24):
        seeds.append(np.roll(p[:24], shift).astype(np.float64))
        
    # Inverted GRS
    seeds.append(-np.array(p[:24], dtype=np.float64))
    
    # Random seeds
    # We will generate random seeds on the fly if time permits.
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        candidate = seed.copy()
        candidate, score = local_search(candidate)
        
        if score < best_score:
            best_score = score
            best_coeffs = candidate
            
        # Check if we found the optimum
        if abs(best_score / np.sqrt(24) - 1.2583758) < 1e-6:
            break

    # If we still have time, do random restarts
    while time.time() < deadline:
        candidate = rng.choice([-1, 1], size=24).astype(np.float64)
        candidate, score = local_search(candidate)
        
        if score < best_score:
            best_score = score
            best_coeffs = candidate
            
        # Check if we found the optimum
        if abs(best_score / np.sqrt(24) - 1.2583758) < 1e-6:
            break

    # Final high-precision polish pass
    if time.time() < deadline - 5.0:
        polished_score = high_precision_sup_norm(best_coeffs)
        # Note: polished_score is the true sup norm; best_score from fast_eval is an approximation
        # We don't change the coefficients, but this confirms the quality
        pass

    # Return the best coefficients found
    # The function must return 24 coefficients in {+1, -1}
    return [int(c) for c in best_coeffs]
