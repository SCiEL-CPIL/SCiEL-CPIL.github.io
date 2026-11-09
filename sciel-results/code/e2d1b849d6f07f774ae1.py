"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

def _autocorr_peak(h):
    """Compute the peak absolute autocorrelation for lags 1..N-1."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    # r[0] is the DC component (sum of squares = N for +/-1)
    # We care about lags 1 to N-1
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_h = np.ones(N)
    best_peak = _autocorr_peak(best_h)
    
    # Strategy: Multi-restart local search with random initializations
    # 1. Start with a random sequence
    # 2. Perform local search (flip bits) to minimize peak autocorrelation
    # 3. If stuck, restart with a new random seed
    
    # We'll use a deterministic sequence of seeds for reproducibility if needed, 
    # but random is fine. Let's use np.random.
    
    # To ensure we don't spend too long on a single bad local minimum,
    # we limit the number of non-improving steps per restart or use a time slice per restart.
    
    # However, simple steepest descent or best-improvement local search is effective.
    # Given N=400, a full scan of all N flips per step is O(N^2) per step.
    # N=400 -> 160,000 ops per step. 
    # 1000 steps -> 160M ops. This might be too slow in pure Python loop if not vectorized well.
    
    # Let's use a vectorized evaluation of all possible flips simultaneously.
    # For a given h, the change in autocorrelation r[k] when flipping h[i] is:
    # delta_r[k] = -2 * h[i] * (h[i+k] + h[i-k])  (with bounds checks)
    # This is hard to vectorize for all i and all k simultaneously easily without large matrices.
    
    # Alternative: Use the FFT to compute the new peak quickly?
    # No, recomputing FFT for each flip is O(N log N). 400 flips * 400 log 400 ~ 400 * 3000 = 1.2M ops per step.
    # 100 steps -> 120M ops. This is feasible in NumPy if we do it in a loop of 400 FFTs?
    # Actually, looping 400 times and doing an FFT each time is slow due to Python overhead.
    
    # Better approach: 
    # 1. Compute current autocorrelation r.
    # 2. For each index i, calculate the new peak if we flip h[i].
    #    This requires updating r. 
    #    r_new = r - 2*h[i]*(h shifted + h reversed shifted)
    #    This is O(N) per flip to update r, then O(N) to find max. Total O(N^2) per step.
    #    400 * 400 = 160,000 operations per step.
    #    500 steps = 80,000,000 operations. This is very fast in NumPy if vectorized per flip?
    #    No, we have to loop over i.
    
    # Let's try a hybrid: 
    # Use a random walk / simulated annealing style approach which is cheaper per step (1 FFT per step).
    # Or, use a small number of random flips per "step" and recompute FFT once.
    
    # Inspiration 1 used: 
    # - 20 random flips per iteration
    # - Recompute FFT for each flip? No, it said "Recomputing FFT is simple and fast... 2000 FFTs. Very fast."
    # Wait, 20 flips * 500 iterations = 10,000 FFTs? 
    # FFT of size 1024 is ~1024*log2(1024) ~ 10k operations. 
    # 10,000 * 10,000 = 100M operations. This is borderline but likely okay in 115s?
    # Actually, np.fft is highly optimized. 10,000 FFTs of size 1024 should take < 1 second?
    # Let's test the speed mentally. 
    # If it's too slow, we can reduce iterations or flips.
    
    # Let's implement a robust local search:
    # 1. Random init
    # 2. Loop:
    #    a. Pick a random index i
    #    b. Flip h[i]
    #    c. Compute new peak
    #    d. If better, keep. Else, revert.
    #    e. If no improvement for X steps, break and restart.
    
    # To speed up, we can try "best improvement" by checking a subset of indices or all indices.
    # Let's try checking ALL indices in a random order, and keep the first improvement found (first improvement heuristic).
    # Or, check all and pick the best (steepest descent).
    # Steepest descent is safer but slower.
    # Let's try: 
    # For each restart:
    #   h = random
    #   peak = calc(h)
    #   for step in range(max_steps):
    #      best_delta = 0
    #      best_idx = -1
    #      # Sample a subset of indices to check? Or all?
    #      # Let's check all 400 indices.
    #      for i in range(N):
    #          h[i] *= -1
    #          new_peak = calc(h)
    #          h[i] *= -1
    #          if new_peak < peak - 1e-9:
    #              if new_peak < best_delta: # Actually we want min peak
    #                  best_delta = new_peak
    #                  best_idx = i
    #      if best_idx != -1:
    #          h[best_idx] *= -1
    #          peak = best_delta
    #      else:
    #          break
    #   if peak < best_peak:
    #       best_peak = peak
    #       best_h = h.copy()
    
    # Cost per step: 400 FFTs.
    # Max steps: maybe 100?
    # Total FFTs: 40,000.
    # 40,000 FFTs of size 1024.
    # This might be too slow. 40,000 * 0.0001s = 4s? Maybe.
    # Let's optimize the peak calculation.
    
    # We can avoid full FFT if we just update the autocorrelation array directly?
    # r[k] = sum h[i]h[i+k]
    # If we flip h[i], r[k] changes by -2*h[i]*(h[i+k] + h[i-k])
    # We can maintain r array.
    # Initial r: FFT.
    # Flip i:
    #   r_new = r - 2*h[i]*(shifted_h + reversed_shifted_h)
    #   This is O(N) vectorized operation.
    #   Then find max of r_new[1:N]. O(N).
    #   Total O(N) per flip check.
    #   400 flips * 400 ops = 160,000 ops per step.
    #   100 steps = 16,000,000 ops. Very fast.
    
    def _update_peak_after_flip(h, r, i):
        """
        Returns the new peak autocorrelation if h[i] is flipped, 
        and the new r array (if we wanted to keep it, but we revert, so just return peak).
        Actually, we need to compute the new peak.
        """
        # h[i] is currently val. New h[i] is -val.
        # Delta r[k] = -2 * h[i] * (h[i+k] + h[i-k])
        # Note: r is symmetric, r[k] = r[-k].
        # We only care about k=1..N-1.
        
        val = h[i]
        # Create zero array for delta
        # We need to handle boundaries.
        # h[i+k] exists if i+k < N
        # h[i-k] exists if i-k >= 0
        
        # Let's construct the vector of changes for all k.
        # This is tricky to vectorize perfectly without masks, but N=400 is small.
        
        # Alternative: Just recompute the affected parts?
        # No, let's just do the full vectorized update on a copy of r? 
        # r is size N. 
        # delta_r = np.zeros(N)
        # For k in 1..N-1:
        #   if i+k < N: delta_r[k] += -2*val*h[i+k]
        #   if i-k >= 0: delta_r[k] += -2*val*h[i-k]
        # This is a loop over k, which is O(N). 
        # Doing this for 400 flips is O(N^2).
        
        # Can we vectorize the update of r for a single flip i?
        # r_new = r.copy()
        # # Add contribution from i+k
        # idx1 = i + np.arange(1, N - i) # k values where i+k < N
        # r_new[idx1] -= 2 * val * h[i + idx1]
        # # Add contribution from i-k
        # idx2 = i - np.arange(1, i + 1) # k values where i-k >= 0 -> k <= i
        # # Wait, k is the lag. 
        # # Term is h[i-k]. 
        # # For k=1..i, h[i-k] is valid.
        # # r_new[k] -= 2 * val * h[i-k]
        # r_new[idx2] -= 2 * val * h[i - idx2]
        
        # This is O(N) with vectorized operations.
        # Let's implement this.
        
        r_new = r.copy()
        
        # Part 1: i+k < N => k < N-i
        # k ranges from 1 to N-i-1
        if N - i > 1:
            k_vals = np.arange(1, N - i)
            # h[i+k]
            r_new[k_vals] -= 2.0 * val * h[i + k_vals]
            
        # Part 2: i-k >= 0 => k <= i
        # k ranges from 1 to i
        if i > 0:
            k_vals = np.arange(1, i + 1)
            # h[i-k]
            r_new[k_vals] -= 2.0 * val * h[i - k_vals]
            
        # Note: r[0] is not updated, but we don't care about r[0].
        # Also, r is symmetric, so updating k and N-k? 
        # The autocorrelation r[k] = sum h[j]h[j+k].
        # r[N-k] = sum h[j]h[j+N-k] = sum h[j-k]h[j] = r[k].
        # So r is symmetric. 
        # Does the update formula respect this?
        # The change in r[k] is derived from the change in the sum.
        # If we update r[k] for k=1..N-1, do we need to update r[N-k]?
        # r[N-k] = r[k]. So they must change by the same amount?
        # Let's check.
        # Delta r[k] = -2 h[i] (h[i+k] + h[i-k])
        # Delta r[N-k] = -2 h[i] (h[i+N-k] + h[i-(N-k)])
        # This is not obviously the same. 
        # Wait, the autocorrelation of a finite sequence is NOT symmetric in the same way as infinite.
        # r[k] = sum_{j=0}^{N-1-k} h[j]h[j+k]
        # r[N-k] = sum_{j=k}^{N-1} h[j]h[j-(N-k)] = sum_{j=0}^{N-1-k} h[j+N-k]h[j] = r[k].
        # Yes, it is symmetric.
        # So Delta r[N-k] must equal Delta r[k].
        # Let's verify the formula.
        # The flip changes h[i] to -h[i].
        # Terms involving h[i] in r[k]:
        # 1. j=i: h[i]h[i+k] (if i+k < N)
        # 2. j=i-k: h[i-k]h[i] (if i-k >= 0)
        # So r[k] contains h[i]h[i+k] + h[i-k]h[i] = h[i](h[i+k]+h[i-k]).
        # When h[i] flips, this term becomes -h[i](h[i+k]+h[i-k]).
        # Change is -2 h[i](h[i+k]+h[i-k]).
        # This formula is correct for r[k].
        # By symmetry, r[N-k] must have the same change.
        # Does the formula for r[N-k] yield the same?
        # r[N-k] terms involving h[i]:
        # 1. j=i: h[i]h[i+N-k] (invalid if i+N-k >= N, i.e., i>=k)
        # 2. j=i-(N-k) = i-N+k: h[i-N+k]h[i] (invalid if i-N+k < 0, i.e., i < N-k)
        
        # Case 1: i >= k. Then i+N-k >= N. Term 1 invalid.
        # Term 2: i-N+k. Since i >= k, i-N+k >= 2k-N. 
        # This is getting complicated. 
        # It is safer to assume the formula holds for k and rely on the symmetry of the final r array?
        # No, the update is local. 
        # Actually, if we just update r[k] for k=1..N-1 using the formula, 
        # and the true r is symmetric, then the updated r_new might NOT be symmetric if we only apply the formula to k.
        # But we calculated the change for r[k] specifically.
        # Since r[k]=r[N-k], the change must be the same.
        # So we can just update k and N-k with the same delta?
        # Or does the formula for r[k] already account for it?
        # The formula is derived from the definition of r[k].
        # So if we apply it to all k=1..N-1, we get the correct r_new[k].
        # Since r_new must be symmetric, the values will be consistent.
        
        return np.max(np.abs(r_new[1:N]))
    
    max_restarts = 20
    max_steps_per_restart = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        np.random.seed(42 + restart)
        h = np.random.choice([-1.0, 1.0], N)
        r = np.fft.irfft(np.fft.rfft(h, n=1 << (2*N).bit_length()) * np.conj(np.fft.rfft(h, n=1 << (2*N).bit_length())), n=1 << (2*N).bit_length())[:N]
        current_peak = _autocorr_peak(h)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_h = h.copy()
            
        no_improve_count = 0
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            best_idx = -1
            best_new_peak = current_peak
            
            # Try all flips
            for i in range(N):
                new_peak = _update_peak_after_flip(h, r, i)
                if new_peak < best_new_peak - 1e-9:
                    best_new_peak = new_peak
                    best_idx = i
                    
            if best_idx != -1:
                # Apply the best flip
                h[best_idx] *= -1
                # Update r
                # We need to update r to reflect the flip for the next iteration
                # We can reuse the logic from _update_peak_after_flip but without the max
                val = -h[best_idx] # Old value
                # Wait, h[best_idx] is now the new value. Old was -new.
                # Let's just recompute r from scratch? 
                # No, that's O(N log N). 
                # Let's update r incrementally.
                old_val = -h[best_idx]
                r_new = r.copy()
                if N - best_idx > 1:
                    k_vals = np.arange(1, N - best_idx)
                    r_new[k_vals] -= 2.0 * old_val * h[best_idx + k_vals]
                if best_idx > 0:
                    k_vals = np.arange(1, best_idx + 1)
                    r_new[k_vals] -= 2.0 * old_val * h[best_idx - k_vals]
                r = r_new
                current_peak = best_new_peak
                no_improve_count = 0
            else:
                break
                
            if current_peak < best_peak:
                best_peak = current_peak
                best_h = h.copy()
                
    return best_h

# EVOLVE-BLOCK-START
# The above code is the evolved block. 
# We need to ensure the interface is preserved.
# The original code had:
# def construct():
#     return np.ones(N)
# My construct() returns best_h, which is a numpy array of floats (-1.0, 1.0).
# The task says "sequence of signed step heights". 
# The seed returned np.ones(N) (floats).
# So returning floats is consistent.
# EVOLVE-BLOCK-END
