import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate initial Golay-Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:24], dtype=float)
    
    # 2. Precompute DFT matrix for N=24 on a fine grid
    # We need enough points to resolve the peaks. 
    # For N=24, the spacing in frequency is 2*pi/M.
    # Let's use M = 2^15 = 32768 for high resolution, but that's 32768x24 matrix.
    # Memory: 32768 * 24 * 8 bytes ~ 6 MB. This is fine.
    # Time to compute: 32768 * 24 * 2 flops ~ 1.5M ops. Negligible.
    M = 2**15
    k = np.arange(M)
    # DFT matrix D[j, n] = exp(-2j * pi * j * n / M)
    # We only need n=0..23
    n_idx = np.arange(24)
    # Compute phases: j * n
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    
    def get_supnorm_and_peaks(coeffs):
        # coeffs: shape (24,)
        # F = D @ coeffs
        F = D @ coeffs
        abs_F = np.abs(F)
        max_val = np.max(abs_F)
        # Find indices of top peaks (e.g., top 5)
        # We want to know where the peaks are to guide the search
        top_k = 5
        peak_indices = np.argsort(abs_F)[-top_k:]
        return max_val, peak_indices, abs_F

    def evaluate_flips(coeffs, peak_indices):
        # coeffs: shape (24,)
        # We want to find the flip that reduces the sup norm the most.
        # Instead of computing full sup norm for all 24 flips, 
        # we can compute the new F for each flip and check the max.
        # But even better: we can just compute the new sup norm for all 24 flips 
        # using the precomputed D matrix.
        # F_new_n = F - 2 * c_n * D[:, n]
        # sup_norm_n = max(|F_new_n|)
        
        current_F = D @ coeffs
        current_abs_F = np.abs(current_F)
        current_max = np.max(current_abs_F)
        
        best_flip = -1
        best_max = current_max + 1e-9 # Start high
        
        for n in range(24):
            # Flip c_n: c_n -> -c_n
            # F_new = F - 2 * c_n * D[:, n]
            F_new = current_F - 2.0 * coeffs[n] * D[:, n]
            max_val = np.max(np.abs(F_new))
            if max_val < best_max:
                best_max = max_val
                best_flip = n
                
        return best_flip, best_max

    current_c = c.copy()
    current_val, _, _ = get_supnorm_and_peaks(current_c)
    best_c = current_c.copy()
    best_val = current_val
    
    # Local search loop
    # We will try to improve by flipping bits that reduce the peak.
    # If no single flip improves, we might try 2-flips or restart.
    # Given the small N=24, 2-flips (528 combos) are cheap to evaluate.
    
    def evaluate_2_flips(coeffs):
        current_F = D @ coeffs
        best_flip1 = -1
        best_flip2 = -1
        best_max = np.max(np.abs(current_F)) + 1e-9
        
        for n1 in range(24):
            for n2 in range(n1+1, 24):
                # Flip c_n1 and c_n2
                # F_new = F - 2*c_n1*D[:,n1] - 2*c_n2*D[:,n2]
                F_new = current_F - 2.0 * coeffs[n1] * D[:, n1] - 2.0 * coeffs[n2] * D[:, n2]
                max_val = np.max(np.abs(F_new))
                if max_val < best_max:
                    best_max = max_val
                    best_flip1 = n1
                    best_flip2 = n2
        return best_flip1, best_flip2, best_max

    # Main loop
    no_improvement_count = 0
    while time.time() < deadline:
        # Try 1-flip
        flip_idx, new_val = evaluate_flips(current_c, None)
        if flip_idx != -1 and new_val < current_val:
            current_c[flip_idx] *= -1.0
            current_val = new_val
            no_improvement_count = 0
            if current_val < best_val:
                best_c = current_c.copy()
                best_val = current_val
        else:
            # Try 2-flips
            f1, f2, new_val_2 = evaluate_2_flips(current_c)
            if f1 != -1 and new_val_2 < current_val:
                current_c[f1] *= -1.0
                current_c[f2] *= -1.0
                current_val = new_val_2
                no_improvement_count = 0
                if current_val < best_val:
                    best_c = current_c.copy()
                    best_val = current_val
            else:
                # Stuck in local minimum. 
                # Random restart or perturbation.
                no_improvement_count += 1
                if no_improvement_count > 10:
                    # Perturb: flip 3-5 random bits
                    n_perturb = random.randint(3, 5)
                    for _ in range(n_perturb):
                        idx = random.randint(0, 23)
                        current_c[idx] *= -1.0
                    current_val, _, _ = get_supnorm_and_peaks(current_c)
                    no_improvement_count = 0
                    # If perturbation made it worse, we accept it to escape, 
                    # but we keep best_c.
    
    return [int(x) for x in best_c]
