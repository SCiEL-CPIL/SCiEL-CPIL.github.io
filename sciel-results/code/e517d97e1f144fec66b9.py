import numpy as np
import math
import time
import random

def construct():
    N = 24
    # Grid size for evaluation during search
    # 4096 is good for 1e-5 accuracy on peak location, 
    # final refinement handles the rest.
    M = 4096
    angles = 2 * np.pi * np.arange(M) / M
    z = np.exp(1j * angles)
    
    # Precompute powers of z for fast dot product
    # powers[k, m] = z[m] ** k
    powers = np.zeros((N, M), dtype=complex)
    powers[0, :] = 1
    for k in range(1, N):
        powers[k, :] = powers[k-1, :] * z
        
    # powers_T is M x N for efficient dot product
    powers_T = powers.T

    def compute_values(coeffs):
        """Compute P(z) values for given coeffs."""
        return powers_T @ coeffs

    def get_max_abs(vals):
        return np.max(np.abs(vals))

    def refine_peak_value(coeffs):
        """
        Refine the peak value using golden section search on the top few local maxima
        found in the coarse grid.
        """
        vals = compute_values(coeffs)
        abs_vals = np.abs(vals)
        
        # Find local maxima in the coarse grid
        local_max_indices = []
        for i in range(M):
            if abs_vals[i] >= abs_vals[(i-1) % M] and abs_vals[i] >= abs_vals[(i+1) % M]:
                local_max_indices.append(i)
        
        if not local_max_indices:
            return np.max(abs_vals)
        
        max_val = 0
        # Refine the top 10 local maxima to be safe
        local_max_indices.sort(key=lambda i: abs_vals[i], reverse=True)
        
        for idx in local_max_indices[:10]:
            delta = 2 * np.pi / M
            center = angles[idx]
            a = center - delta
            b = center + delta
            
            def f(theta):
                z_t = np.exp(1j * theta)
                p_val = 0
                for k in range(N):
                    p_val += coeffs[k] * (z_t ** k)
                return np.abs(p_val)
            
            # Golden section search
            invphi = (1 + math.sqrt(5)) / 2
            invphi2 = invphi * invphi
            c = b - (b - a) / invphi
            d = a + (b - a) / invphi
            fc = f(c)
            fd = f(d)
            
            for _ in range(60):
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + (b - a) / invphi2
                    fd = f(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - (b - a) / invphi2
                    fc = f(c)
            
            val = max(fc, fd)
            if val > max_val:
                max_val = val
                
        return max_val

    # Start with Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N], dtype=int)
    
    # Calculate initial score
    initial_vals = compute_values(best_coeffs)
    best_score = get_max_abs(initial_vals)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer for final refinement and overhead
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        current_coeffs = best_coeffs.copy()
        current_vals = compute_values(current_coeffs)
        current_score = get_max_abs(current_vals)
        
        improved = True
        while improved:
            improved = False
            
            # 1-flip pass
            best_1_flip_idx = -1
            best_1_flip_score = current_score
            for k in range(N):
                # Flip k
                # New vals = current_vals - 2 * c_k * z^k
                # c_k is current_coeffs[k]
                delta_vals = -2 * current_coeffs[k] * powers[k]
                new_vals = current_vals + delta_vals
                new_score = get_max_abs(new_vals)
                if new_score < best_1_flip_score:
                    best_1_flip_score = new_score
                    best_1_flip_idx = k
            
            if best_1_flip_idx != -1:
                k = best_1_flip_idx
                current_coeffs[k] *= -1
                current_vals = current_vals - 2 * current_coeffs[k] * powers[k] # Note: coeffs[k] is already flipped, so we subtracted 2*old*power. 
                # Wait, let's re-derive.
                # old: c_k. new: -c_k.
                # P_new = P_old - c_k z^k + (-c_k) z^k = P_old - 2 c_k z^k.
                # In the loop, we used `current_coeffs[k]` which was the OLD value.
                # So `delta_vals` was correct.
                # Now we update `current_vals` and `current_coeffs`.
                current_score = best_1_flip_score
                improved = True
                continue # Restart passes after a 1-flip to ensure convergence
                
            # 2-flip pass
            best_2_flip_i = -1
            best_2_flip_j = -1
            best_2_flip_score = current_score
            
            # Optimization: Only check pairs that might improve.
            # We can precompute the effect of each single flip? No, interactions matter.
            # Just iterate all pairs. 276 pairs * 4096 elements is fast.
            for i in range(N):
                for j in range(i+1, N):
                    # Flip i and j
                    # P_new = P_old - 2 c_i z^i - 2 c_j z^j
                    delta_vals = -2 * current_coeffs[i] * powers[i] - 2 * current_coeffs[j] * powers[j]
                    new_vals = current_vals + delta_vals
                    new_score = get_max_abs(new_vals)
                    if new_score < best_2_flip_score:
                        best_2_flip_score = new_score
                        best_2_flip_i = i
                        best_2_flip_j = j
            
            if best_2_flip_i != -1:
                i, j = best_2_flip_i, best_2_flip_j
                current_coeffs[i] *= -1
                current_coeffs[j] *= -1
                # Update vals
                # P_new = P_old - 2 c_i z^i - 2 c_j z^j
                # c_i, c_j are OLD values in the formula above.
                # So we just add the delta computed.
                current_vals = current_vals - 2 * (current_coeffs[i] * -1) * powers[i] - 2 * (current_coeffs[j] * -1) * powers[j]
                # Simpler:
                # old_i = -current_coeffs[i] (since we flipped it)
                # This is getting confusing. Let's just recompute or store old vals.
                
                # Let's recompute to be safe and simple, cost is negligible compared to search.
                # Actually, we can just do:
                # current_vals = compute_values(current_coeffs)
                current_vals = compute_values(current_coeffs)
                current_score = best_2_flip_score
                improved = True
                continue # Restart passes
                
        # Local minimum reached
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs.copy()
            
        # Perturbation
        # Flip 3-5 random coefficients
        num_flips = random.randint(3, 5)
        indices = random.sample(range(N), num_flips)
        for idx in indices:
            best_coeffs[idx] *= -1
            
    # Final refinement of the best found solution
    final_score = refine_peak_value(best_coeffs)
    
    return best_coeffs.tolist()
