"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def _autoconv_peak(x):
    """Compute the peak of the autoconvolution of x (length N)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    c = np.fft.irfft(X * X, size)[:2 * n - 1]
    return np.max(c)

def _score(x):
    """The normalized maximum of autoconvolution."""
    h = 1.0 / (2 * N)
    integral = h * np.sum(x)
    if integral <= 0:
        return np.inf
    peak = _autoconv_peak(x) * h
    return peak / (integral * integral)

def _compute_conv(x):
    """Compute the linear autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    c = np.fft.irfft(X * X, size)[:2 * n - 1]
    return c

def _exact_coordinate_descent(x, deadline):
    """
    Exact coordinate descent for minimizing max(autoconv) / (sum(x))^2.
    """
    x = x.copy()
    S = np.sum(x)
    C = _compute_conv(x)
    best_score = _score(x)
    
    # Precompute indices for shifts
    # For coordinate i, the update to C depends on x shifted by i.
    # C'_k = C_k + 2(v - x_i) x_{k-i} for k != 2i
    # C'_{2i} = C_{2i} - x_i^2 + v^2
    
    for iteration in range(500):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = x[i]
            if old_val == 0 and np.max(C) == 0:
                continue
                
            # Determine the range for v
            # v should be non-negative.
            # A reasonable upper bound is 2 * old_val or 2 * max(x)
            v_max = 2.0 * old_val
            if v_max < 1e-6:
                v_max = 1.0
            
            # Function to evaluate score for a given v
            # We need to update C efficiently.
            # C_new = C + delta
            # delta_k = 2(v - old_val) * x_{k-i}
            # except for k=2i, where delta = v^2 - old_val^2
            
            # x_{k-i} is x shifted right by i?
            # (x*x)_k = sum_j x_j x_{k-j}
            # Term with x_i is x_i x_{k-i}.
            # So we need x_{k-i}.
            # Let's construct the vector x_shifted such that x_shifted[k] = x_{k-i}
            # Using np.roll is tricky with boundaries for linear convolution.
            # It's safer to just compute the change directly or use a precomputed mask.
            
            # Actually, for N=600, we can just recompute the peak from scratch for ternary search steps?
            # No, that's O(N) per step. Ternary search is ~40 steps. 40*600 = 24000 ops.
            # 600 coords * 24000 = 14.4M ops per pass.
            # This is fast enough in Python if we use vectorized ops for the max.
            
            # Let's try to vectorize the evaluation of max(C_new)
            
            def eval_score(v):
                # Update C
                # x_{k-i} for k in 0..2N-2
                # We need to access x[j] where j = k-i.
                # If 0 <= k-i < N, term is x[k-i]. Else 0.
                
                # Create vector of x_{k-i}
                # k ranges from 0 to 2N-2
                # k-i ranges from -i to 2N-2-i
                # Valid indices in x are 0 to N-1.
                
                # Efficient way:
                # delta = np.zeros(2*N - 1)
                # mask = (np.arange(2*N - 1) - i) >= 0
                # mask &= (np.arange(2*N - 1) - i) < N
                # indices = np.arange(2*N - 1) - i
                # delta[mask] = 2 * (v - old_val) * x[indices[mask]]
                
                # Handle k=2i specifically
                k_center = 2 * i
                if 0 <= k_center < 2 * N - 1:
                    # The formula delta = 2(v-old)x_{k-i} gives 2(v-old)x_i
                    # But we need v^2 - old^2
                    # So we correct it: delta[k_center] = v^2 - old^2
                    # Note: the general formula added 2(v-old)x_i to C[k_center]
                    # C[k_center] originally had old^2.
                    # New C should have v^2.
                    # So delta should be v^2 - old^2.
                    # The general formula gave 2(v-old)x_i = 2(v-old)old_val.
                    # We want v^2 - old^2.
                    # Difference: (v^2 - old^2) - 2(v-old)old = (v-old)(v+old) - 2(v-old)old = (v-old)(v-old) = (v-old)^2.
                    # So we just add (v-old)^2 to delta[k_center].
                    pass
                
                # This per-step vectorization is overhead.
                # Let's just use the scalar update logic but vectorized for the max.
                
                # Actually, simpler:
                # C_new = C.copy()
                # This is O(N) memory copy.
                
                # Let's precompute the shift indices once outside the ternary search?
                # No, v changes.
                
                # Let's just do:
                # C_new = C + 2*(v-old_val)*x_shifted
                # Then fix center.
                
                # To avoid creating x_shifted every time, we can precompute the mask and indices.
                # But x_shifted depends on i.
                
                # Let's just compute the max directly from the formula?
                # max_k (C_k + 2(v-old)x_{k-i})
                
                # This is still O(N).
                
                # Let's implement a fast eval.
                # Precomputed for this i:
                # x_shifted is a vector of length 2N-1.
                
                pass
            
            # Precompute x_shifted for this i
            k_indices = np.arange(2 * N - 1)
            j_indices = k_indices - i
            mask = (j_indices >= 0) & (j_indices < N)
            x_shifted = np.zeros(2 * N - 1)
            x_shifted[mask] = x[j_indices[mask]]
            
            k_center = 2 * i
            center_valid = 0 <= k_center < 2 * N - 1
            
            def calc_max_conv(v):
                # C_new = C + 2*(v-old_val)*x_shifted
                # Correction at center:
                # The term 2*(v-old_val)*x_shifted[k_center] adds 2*(v-old_val)*old_val
                # We need to add v^2 - old_val^2 instead.
                # So add (v^2 - old_val^2) - 2*(v-old_val)*old_val = (v-old_val)^2
                val = C + 2.0 * (v - old_val) * x_shifted
                if center_valid:
                    val[k_center] += (v - old_val) ** 2
                return np.max(val)
            
            # Ternary search for v in [0, v_max]
            lo = 0.0
            hi = v_max
            
            # Check endpoints and a few points to bound the search
            # The function is unimodal?
            # max(linear functions) / quadratic.
            # It should be unimodal for v > 0.
            
            for _ in range(40):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                
                f1 = calc_max_conv(m1) / (S - old_val + m1) ** 2
                f2 = calc_max_conv(m2) / (S - old_val + m2) ** 2
                
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            
            v_new = (lo + hi) / 2.0
            
            # Verify improvement
            f_new = calc_max_conv(v_new) / (S - old_val + v_new) ** 2
            f_old = _score(x) # This is expensive, use best_score?
            # Actually, we can just compare with the current max conv / S^2
            # current score = max(C) / S^2
            # new score = max(C_new) / (S - old + v_new)^2
            
            if f_new < best_score * 1.000001: # Allow tiny improvements
                # Update x and C
                x[i] = v_new
                S = S - old_val + v_new
                C = C + 2.0 * (v_new - old_val) * x_shifted
                if center_valid:
                    C[k_center] += (v_new - old_val) ** 2
                best_score = f_new
                improved = True
            else:
                # Revert if not improved (shouldn't happen if ternary search is good, but safe)
                pass
        
        if not improved:
            break
            
    return x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize with a Gaussian
    n = np.arange(N)
    center = N / 2.0
    width = N / 4.0
    x = np.exp(-((n - center) ** 2) / (2 * width ** 2))
    
    # Normalize to avoid overflow/underflow issues during optimization
    x = x / np.sum(x)
    
    # Run exact coordinate descent
    x_opt, score_opt = _exact_coordinate_descent(x, deadline)
    
    # Final check
    if time.time() < deadline - 5:
        # Try a few other initializations
        # Raised Cosine
        x2 = 0.5 * (1 + np.cos(2 * np.pi * (n / N)))
        x2 = x2 / np.sum(x2)
        x2_opt, score2 = _exact_coordinate_descent(x2, deadline)
        if score2 < score_opt:
            x_opt = x2_opt
            score_opt = score2
            
    return list(x_opt)
# EVOLVE-BLOCK-END
