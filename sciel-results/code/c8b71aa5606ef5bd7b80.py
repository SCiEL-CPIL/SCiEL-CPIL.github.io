N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_peak(f):
    """Compute max |autoconvolution| at non-zero lags using FFT."""
    n = len(f)
    if n <= 1:
        return 0.0
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    F = np.fft.fft(f, n=fft_size)
    conv = np.fft.ifft(F * F).real
    # conv[0] is lag 0 (self term = sum(f^2)), we want max |conv[k]| for k=1..n-1
    # Also need to check the wrap-around: conv[n:2n-1] corresponds to negative lags
    # For linear convolution, we only care about lags 1..n-1 and -(n-1)..-1
    # In the FFT result of size 2n, lags 1..n-1 are at indices 1..n-1
    # and lags -(n-1)..-1 are at indices n+1..2n-1
    # But since we're doing circular conv of size 2n with f zero-padded to n,
    # the linear conv is in indices 0..2n-1, and non-zero lags are 1..n-1 and n+1..2n-1
    # Actually for autoconvolution f*f, lag k is sum_i f[i]*f[i+k]
    # In the FFT approach with zero-padding to 2n, the result at index k gives lag k
    # for k=0..n-1, and index 2n-k gives lag -k for k=1..n-1
    # So we want max over indices 1..n-1 and n+1..2n-1
    peak = np.max(np.abs(conv[1:n]))
    if 2 * n > n + 1:
        peak = max(peak, np.max(np.abs(conv[n + 1:2 * n])))
    return peak


def _autoconv_peak_fast(f, fft_size):
    """Fast autoconvolution peak computation with precomputed FFT size."""
    F = np.fft.fft(f, n=fft_size)
    conv = np.fft.ifft(F * F).real
    n = N
    peak = np.max(np.abs(conv[1:n]))
    if fft_size > n + 1:
        peak = max(peak, np.max(np.abs(conv[n + 1:fft_size])))
    return peak


def _optimize_signs(f, active_indices, current_peak, start_time, deadline, fft_size):
    """Greedy sign flipping to minimize autoconvolution peak."""
    best_peak = current_peak
    best_f = f.copy()
    best_active = active_indices.copy()
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(active_indices)):
            if (time.time() - start_time) > deadline:
                break
            idx = active_indices[i]
            old_val = f[idx]
            f[idx] = -old_val
            new_peak = _autoconv_peak_fast(f, fft_size)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_f = f.copy()
                best_active = active_indices.copy()
                improved = True
            else:
                f[idx] = old_val
    return best_f, best_active, best_peak


def _optimize_support(f, active_indices, current_peak, start_time, deadline, fft_size, max_swaps=200):
    """Try swapping active indices to find better support."""
    best_peak = current_peak
    best_f = f.copy()
    best_active = active_indices.copy()
    
    for _ in range(max_swaps):
        if (time.time() - start_time) > deadline:
            break
        
        # Pick random active index to remove
        i_remove = np.random.randint(len(active_indices))
        idx_remove = active_indices[i_remove]
        
        # Pick random inactive index to add
        inactive_mask = np.ones(N, dtype=bool)
        inactive_mask[active_indices] = False
        inactive_indices = np.where(inactive_mask)[0]
        if len(inactive_indices) == 0:
            break
        j_add = inactive_indices[np.random.randint(len(inactive_indices))]
        
        # Remove old, add new
        f[idx_remove] = 0.0
        f[j_add] = 1.0  # Start with +1
        
        new_active = np.delete(active_indices, i_remove)
        new_active = np.append(new_active, j_add)
        
        new_peak = _autoconv_peak_fast(f, fft_size)
        
        if new_peak < best_peak - 1e-12:
            # Try both signs for the new element
            f[j_add] = -1.0
            alt_peak = _autoconv_peak_fast(f, fft_size)
            if alt_peak < new_peak:
                new_peak = alt_peak
            else:
                f[j_add] = 1.0
            
            best_peak = new_peak
            best_f = f.copy()
            best_active = new_active.copy()
            active_indices = new_active
        else:
            # Revert
            f[j_add] = 0.0
            f[idx_remove] = 1.0  # Restore original sign (we lost it, but we can just set to 1)
            # Actually we need to restore the original sign. Let's track it differently.
            # For now, just revert to +1 which may not be correct. Let's fix this.
            # Better approach: save the original value before modification.
    
    return best_f, best_active, best_peak


def _optimize_support_v2(f, active_indices, current_peak, start_time, deadline, fft_size, max_swaps=200):
    """Try swapping active indices with proper sign tracking."""
    best_peak = current_peak
    best_f = f.copy()
    best_active = active_indices.copy()
    
    for _ in range(max_swaps):
        if (time.time() - start_time) > deadline:
            break
        
        # Pick random active index to remove
        i_remove = np.random.randint(len(active_indices))
        idx_remove = active_indices[i_remove]
        orig_val = f[idx_remove]
        
        # Pick random inactive index to add
        inactive_mask = np.ones(N, dtype=bool)
        inactive_mask[active_indices] = False
        inactive_indices = np.where(inactive_mask)[0]
        if len(inactive_indices) == 0:
            break
        j_add = inactive_indices[np.random.randint(len(inactive_indices))]
        
        # Remove old, add new with +1
        f[idx_remove] = 0.0
        f[j_add] = 1.0
        
        new_active = np.delete(active_indices, i_remove)
        new_active = np.append(new_active, j_add)
        
        new_peak = _autoconv_peak_fast(f, fft_size)
        
        # Try -1 as well
        f[j_add] = -1.0
        alt_peak = _autoconv_peak_fast(f, fft_size)
        
        if alt_peak < new_peak:
            best_new_peak = alt_peak
            best_new_val = -1.0
        else:
            best_new_peak = new_peak
            best_new_val = 1.0
        
        if best_new_peak < best_peak - 1e-12:
            f[j_add] = best_new_val
            best_peak = best_new_peak
            best_f = f.copy()
            best_active = new_active.copy()
            active_indices = new_active
        else:
            # Revert
            f[j_add] = 0.0
            f[idx_remove] = orig_val
    
    return best_f, best_active, best_peak


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound (max absolute autoconvolution at non-zero lags).
    
    Strategy:
    1. Use sparse combinatorial design: select K active positions out of 400
    2. Optimize signs via greedy local search
    3. Optimize support via random swaps
    4. Try multiple sparsity levels and random restarts
    5. Time-budgeted to finish within 120s
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    
    best_f = np.zeros(N)
    best_peak = np.inf
    
    # Try different sparsity levels
    # For sparse signals, autoconvolution peak at non-zero lags is bounded by
    # the maximum number of overlapping pairs at any lag
    # Smaller K means smaller potential peak
    sparsity_levels = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    
    np.random.seed(42)
    
    for K in sparsity_levels:
        if time.time() > deadline - 20:
            break
        
        # Multiple random starts for each sparsity level
        n_trials = 3 if K <= 30 else 2 if K <= 60 else 1
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization: pick K random indices
            seed = 42 + trial * 1000 + K
            np.random.seed(seed)
            active_indices = np.random.choice(N, size=K, replace=False)
            signs = np.random.choice([-1.0, 1.0], size=K)
            
            f = np.zeros(N)
            f[active_indices] = signs
            
            current_peak = _autoconv_peak_fast(f, fft_size)
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_f = f.copy()
            
            # Phase 1: Optimize signs
            sub_deadline = min(deadline - 5, start_time + (deadline - start_time) * 0.6)
            f, active_indices, current_peak = _optimize_signs(
                f, active_indices, current_peak, start_time, sub_deadline, fft_size
            )
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_f = f.copy()
            
            # Phase 2: Optimize support
            if time.time() < deadline - 10:
                max_swaps = 50 if K <= 30 else 30 if K <= 60 else 15
                sub_deadline2 = min(deadline - 5, time.time() + (deadline - time.time()) * 0.7)
                f, active_indices, current_peak = _optimize_support_v2(
                    f, active_indices, current_peak, start_time, sub_deadline2, fft_size, max_swaps
                )
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_f = f.copy()
                
                # Phase 3: Re-optimize signs after support changes
                if time.time() < deadline - 5:
                    sub_deadline3 = min(deadline - 2, time.time() + (deadline - time.time()) * 0.5)
                    f, active_indices, current_peak = _optimize_signs(
                        f, active_indices, current_peak, start_time, sub_deadline3, fft_size
                    )
                    
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_f = f.copy()
    
    # Final polish: one more sign optimization pass on the best solution
    if time.time() < deadline - 2:
        active = np.nonzero(best_f)[0]
        if len(active) > 0:
            sub_deadline_final = deadline - 1
            best_f, _, _ = _optimize_signs(
                best_f, active, best_peak, start_time, sub_deadline_final, fft_size
            )
    
    return best_f
# EVOLVE-BLOCK-END
