import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    N = 24
    M = 4096
    
    j = np.arange(M)
    k = np.arange(N)
    Z = np.exp(1j * 2 * np.pi * np.outer(j, k) / M)
    
    # GR sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    gr_coeffs = np.array(p[:N], dtype=np.int8)
    
    best_coeffs = gr_coeffs.copy()
    best_score = np.inf
    
    def get_score_from_values(values):
        return np.max(np.abs(values))
    
    def local_search(coeffs):
        coeffs = coeffs.copy()
        # Initial values
        values = Z.dot(coeffs)
        current_val = get_score_from_values(values)
        
        improved = True
        while improved:
            improved = False
            for k_idx in range(N):
                # Flip k_idx
                # c_k changes sign.
                # P_new = P_old - 2 * c_k * z^k
                # Note: coeffs[k_idx] is the current value (+1 or -1)
                # So we subtract 2 * coeffs[k_idx] * Z[:, k_idx]
                
                # Create a copy of values to avoid modifying the original until we decide
                # Or just compute the max on the fly
                # V_new = values - 2 * coeffs[k_idx] * Z[:, k_idx]
                
                # To avoid allocating a new array of size M for every flip, 
                # we can do this in a vectorized way if we were checking all flips at once, 
                # but we check one by one.
                # Allocation of size 4096 complex is cheap.
                
                # Optimization: We only care if the max decreases.
                # We can compute the new max.
                
                # Let's just do the calculation
                # v_new = values - 2.0 * coeffs[k_idx] * Z[:, k_idx]
                # new_val = np.max(np.abs(v_new))
                
                # Even faster: 
                # We can precompute Z_k = Z[:, k_idx]
                # But Z is already in memory.
                
                # Let's try to avoid the copy of 'values' if possible?
                # No, we need the original 'values' for the next flip if this one is rejected.
                # So we must compute v_new without overwriting 'values'.
                
                v_new = values - 2.0 * coeffs[k_idx] * Z[:, k_idx]
                new_val = np.max(np.abs(v_new))
                
                if new_val < current_val - 1e-12:
                    # Accept flip
                    coeffs[k_idx] = -coeffs[k_idx]
                    values = v_new  # Update values for next steps
                    current_val = new_val
                    improved = True
        
        return coeffs, current_val

    num_restarts = 0
    while time.time() - start_time < time_limit:
        if num_restarts == 0:
            start_coeffs = gr_coeffs.copy()
        else:
            if num_restarts % 2 == 0:
                start_coeffs = np.random.choice([1, -1], size=N).astype(np.int8)
            else:
                start_coeffs = gr_coeffs.copy()
                num_flips = np.random.randint(1, 4)
                flip_indices = np.random.choice(N, size=num_flips, replace=False)
                start_coeffs[flip_indices] *= -1
        
        found_coeffs, found_score = local_search(start_coeffs)
        
        if found_score < best_score:
            best_score = found_score
            best_coeffs = found_coeffs
        
        num_restarts += 1
        
        target_raw = 1.2583758 * np.sqrt(24)
        if best_score <= target_raw * (1 + 1e-9):
            break
            
    return best_coeffs.tolist()
