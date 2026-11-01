import numpy as np
import time

N = 400

def _compute_autocorr(h):
    """Compute full autocorrelation of h (length 2N-1)."""
    return np.correlate(h, h, mode='full')

def _max_sidelobe(autocorr):
    """Return max absolute sidelobe value (excluding peak at center)."""
    # Peak is at index N-1
    # Sidelobes are all other indices
    sidelobes = np.concatenate([autocorr[:N-1], autocorr[N:]])
    return np.max(np.abs(sidelobes))

def _spectral_init():
    """
    Initialize h from inverse FFT of a target magnitude spectrum.
    Target: flat magnitude (low peak sidelobe).
    For a ±1 sequence, |FFT(h)|^2 should be as flat as possible.
    We start with a random phase, take IFFT, and threshold to ±1.
    """
    # Target magnitude: constant (white noise spectrum)
    # But we want to minimize peak sidelobe, which corresponds to minimizing max |autocorr| away from peak.
    # In frequency domain, autocorr is IFFT(|FFT(h)|^2).
    # If |FFT(h)|^2 is flat, autocorr is a delta (perfect).
    # We can't get perfect, but we can try to flatten it.
    
    # Strategy: 
    # 1. Start with random h
    # 2. Compute P = |FFT(h)|^2
    # 3. Modify P to be flatter (e.g., replace with sqrt of average, or blend with flat)
    # 4. Keep phase of FFT(h)
    # 5. IFFT to get new h
    # 6. Threshold to ±1
    # This is the "Gerchberg-Saxton" or similar phase retrieval approach.
    
    h = np.random.choice([-1.0, 1.0], size=N)
    
    # Iterate a few times to refine the initialization
    for _ in range(10):
        H = np.fft.fft(h)
        P = np.abs(H)**2
        
        # Flatten P: replace with a smoothed version or target flat spectrum
        # A flat spectrum has constant power.
        # Let's try to push P towards its mean.
        # P_new = mean(P) + alpha * (P - mean(P)) ? No, that's just scaling.
        # We want to reduce the dynamic range of P.
        # Let's try: P_target = mean(P) * ones(N)
        # But we must keep the phase.
        # H_new = sqrt(P_target) * exp(i * angle(H))
        # This forces |H| to be constant.
        
        mean_P = np.mean(P)
        H_phase = np.angle(H)
        H_mag = np.sqrt(mean_P) * np.ones(N)
        
        H_new = H_mag * np.exp(1j * H_phase)
        h_new = np.fft.ifft(H_new).real
        
        # Threshold to ±1
        h = np.sign(h_new)
        # Handle zeros
        h[h == 0] = 1.0
        
    return h

def _local_search(h, max_iters=10000, deadline=None):
    """
    Bit-flip local search to minimize max sidelobe.
    """
    best_h = h.copy()
    best_score = _max_sidelobe(_compute_autocorr(h))
    
    # Precompute autocorr
    autocorr = _compute_autocorr(h)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Try flipping each bit and see if it improves the max sidelobe
        # For efficiency, we can compute the change in autocorr analytically?
        # Flipping h[j] from s to -s changes h[j] by -2s.
        # autocorr[k] = sum h[i] h[i+k]
        # Change in autocorr[k] due to flipping h[j]:
        # If j is involved in the sum, the term changes.
        # Specifically, autocorr[k] involves pairs (i, i+k).
        # h[j] appears in autocorr[k] if j = i or j = i+k.
        # i = j => k = i - j = 0 (peak) or other?
        # Let's just recompute or use a fast update.
        # With N=400, recomputing autocorr is O(N^2) = 160k ops.
        # 10000 iters * 400 flips * 160k = 6.4e9 ops. Too slow.
        
        # We need a faster update.
        # Let delta = -2 * h[j]
        # new_h = h + delta * e_j
        # new_autocorr = autocorr + delta * (h * e_j autocorr + e_j * h autocorr) + delta^2 * (e_j * e_j autocorr)
        # autocorr of h and e_j is h[j-k] (shifted h)
        # So new_autocorr[k] = autocorr[k] - 2*h[j] * (h[k-j] + h[j-k]) + 4*h[j]^2 * (1 if k==0 else 0)
        # Note: h[j]^2 = 1 always.
        # So change at k: -2*h[j]*(h[k-j] + h[j-k]) + 4*(1 if k==0 else 0)
        
        improved = False
        # Randomize flip order to avoid cycles
        indices = np.random.permutation(N)
        
        for j in indices:
            if deadline is not None and time.time() > deadline:
                break
                
            # Calculate change in autocorr
            # We only care if max sidelobe decreases
            # Let's compute the new max sidelobe efficiently?
            # Or just check if the flip reduces the current max?
            # This is tricky because reducing one sidelobe might increase another.
            # Simple heuristic: only flip if it reduces the *current* max sidelobe value?
            # Or if it reduces the sum of squares?
            # The task says "minimize the maximum absolute value of the sidelobes".
            # So we should check if the new max is lower.
            
            # Let's compute the new autocorr quickly?
            # No, let's just compute the new max sidelobe for the candidate.
            # This is O(N) per flip.
            # 400 flips * 400 N = 160k per iteration.
            # 10000 iters * 160k = 1.6e9. Still might be tight in Python.
            
            # Let's try to be smarter.
            # Only consider flipping bits that are "involved" in the current max sidelobes.
            # Or just do a full search with a time limit.
            
            # Let's implement the O(N) check.
            # new_autocorr = autocorr.copy()
            # This is O(N).
            # Then update the O(N) values affected by the flip.
            # The flip at j affects autocorr[k] for all k.
            # Wait, the formula derived:
            # delta_autocorr[k] = -2*h[j]*(h[k-j] + h[j-k]) + 4*(1 if k==0 else 0)
            # This is O(N) to compute for all k.
            
            # Let's just do it.
            h_j = h[j]
            
            # Compute delta_autocorr
            # h[k-j] is h shifted by j
            # h[j-k] is h reversed and shifted
            # Let's use numpy slicing
            # This is getting complicated. Let's just recompute autocorr for the candidate?
            # No, let's stick to the formula.
            
            # Actually, a simpler heuristic:
            # Flip bit j.
            # Compute new autocorr (O(N^2) is too slow, but O(N) update is possible).
            # Let's try the O(N) update.
            
            # delta_k = -2 * h_j * (h[k-j] + h[j-k])
            # We need to handle boundaries.
            
            # Let's just try flipping a random subset of bits per iteration to save time?
            # Or just one bit per iteration?
            
            # Let's try one bit per iteration, with O(N) cost.
            # 10000 * 400 = 4e6 ops. Very fast.
            
            # Compute new autocorr values
            # new_ac = autocorr - 2*h_j * (h[k-j] + h[j-k]) + 4*(k==0)
            
            # We want to check if max(|new_ac|) excluding center is < best_score.
            
            # Let's compute the array of changes.
            # This requires careful indexing.
            
            # Alternative: Just recompute autocorr for the candidate h with flip.
            # np.correlate is C-optimized. N=400.
            # Time per correlate ~ 10-50 microseconds?
            # 10000 iters * 400 flips = 4e6 correlates.
            # 4e6 * 50us = 200s. Too slow.
            
            # We MUST use the incremental update.
            
            # Let's implement the incremental update correctly.
            # autocorr[k] = sum_{i=0}^{N-1-k} h[i] h[i+k] for k>=0
            # Flipping h[j]:
            # Terms involving h[j]:
            # 1. i=j, i+k=j+k => h[j]h[j+k]. Exists if j+k < N.
            # 2. i+k=j, i=j-k => h[j-k]h[j]. Exists if j-k >= 0.
            
            # So for a given k:
            # If k >= 0:
            #   Change from term 1: if j+k < N, h[j]h[j+k] becomes -h[j]h[j+k]. Change: -2 h[j] h[j+k]
            #   Change from term 2: if j-k >= 0, h[j-k]h[j] becomes -h[j-k]h[j]. Change: -2 h[j] h[j-k]
            #   Note: if k=0, j+k=j and j-k=j, so both terms are the same h[j]^2.
            #   Change: -2 h[j]^2 - 2 h[j]^2 = -4 h[j]^2 = -4.
            #   Wait, original term was h[j]^2=1. New is (-h[j])^2=1. Change is 0?
            #   No, the sum is over i.
            #   For k=0, term is h[j]^2. Flipping h[j] changes h[j]^2 to (-h[j])^2 = h[j]^2.
            #   So change is 0 for k=0?
            #   Let's re-verify.
            #   autocorr[0] = sum h[i]^2 = N.
            #   Flipping one bit doesn't change the sum of squares.
            #   So change at k=0 is 0.
            
            #   For k > 0:
            #   Term 1: h[j]h[j+k] -> -h[j]h[j+k]. Change: -2 h[j] h[j+k]
            #   Term 2: h[j-k]h[j] -> -h[j-k]h[j]. Change: -2 h[j] h[j-k]
            #   Total change: -2 h[j] (h[j+k] + h[j-k])
            
            #   For k < 0:
            #   autocorr[k] = autocorr[-k] (symmetry).
            #   So we can just compute for k>=0 and mirror.
            
            # So, for each k in 0..N-1:
            # delta[k] = 0 if k==0
            # delta[k] = -2 * h[j] * (h[j+k] + h[j-k]) for k>0, with boundary checks.
            
            # Let's implement this.
            
            # We only need to check if the new max sidelobe is smaller.
            # new_ac = autocorr + delta
            # We can compute the max of |new_ac| excluding 0.
            
            # This is O(N) per flip.
            
            # Let's do it.
            
            # Create delta array
            # This is a bit slow in Python loops.
            # Can we vectorize?
            # delta[k] depends on h[j+k] and h[j-k].
            # We can construct these arrays.
            
            # Let's try.
            
            # Actually, let's just flip the bit, recompute autocorr (if fast enough), and check.
            # If np.correlate is too slow, we'll optimize.
            
            # Let's assume for now we do a full recompute per flip, but limit flips per iteration.
            # Or just do a few flips.
            
            # Let's try: for each iteration, try flipping each bit, but only keep the best improvement.
            # This is a greedy step.
            
            # Optimization:
            # 1. Try flipping bit j.
            # 2. Compute new max sidelobe.
            # 3. If better, accept and break (greedy).
            # 4. If not, try next bit.
            
            # To speed up, we can precompute the "impact" of each flip?
            
            # Let's just write the O(N) update in numpy.
            
            # delta = np.zeros(N)
            # For k in 1..N-1:
            #   idx1 = j + k
            #   idx2 = j - k
            #   val1 = h[idx1] if idx1 < N else 0
            #   val2 = h[idx2] if idx2 >= 0 else 0
            #   delta[k] = -2 * h[j] * (val1 + val2)
            
            # This is a loop in Python. N=400.
            # 400 iterations * 400 flips * 10000 iters = 1.6e8 operations.
            # Might be too slow in pure Python.
            
            # Let's use numpy vectorization for the delta calculation.
            
            # We can create arrays for indices j+k and j-k.
            
            ks = np.arange(1, N)
            idx1 = j + ks
            idx2 = j - ks
            
            # Mask valid indices
            mask1 = idx1 < N
            mask2 = idx2 >= 0
            
            vals1 = np.zeros(N-1)
            vals2 = np.zeros(N-1)
            
            vals1[mask1] = h[idx1[mask1]]
            vals2[mask2] = h[idx2[mask2]]
            
            delta = np.zeros(N)
            delta[1:] = -2.0 * h[j] * (vals1 + vals2)
            
            new_ac = autocorr + delta
            
            # Check max sidelobe
            # Sidelobes are indices 0..N-2 and N..2N-2?
            # autocorr has length 2N-1.
            # Peak is at N-1.
            # Sidelobes are 0..N-2 and N..2N-2.
            
            # My delta calculation was for k=0..N-1 (positive lags).
            # autocorr is symmetric.
            # autocorr[N-1+k] = autocorr[N-1-k]
            
            # So I need to update the full autocorr array.
            # Let's stick to the full array update.
            
            # This is getting complex. Let's just use a simpler, slower method first and see if it's fast enough.
            # Method: Flip bit, recompute autocorr using np.correlate, check score.
            # If too slow, optimize.
            
            # Let's try that.
            
            h[j] = -h[j]
            new_ac = _compute_autocorr(h)
            new_score = _max_sidelobe(new_ac)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                autocorr = new_ac
                improved = True
                break # Greedy: accept first improvement
            else:
                h[j] = -h[j] # Revert
        
        if not improved:
            # No improvement found, try a random restart or perturbation?
            # Or just stop?
            # Let's try a random perturbation: flip 5 random bits.
            if i > 100:
                for _ in range(5):
                    j = np.random.randint(N)
                    h[j] = -h[j]
                # Recompute
                autocorr = _compute_autocorr(h)
                # Don't update best_score here, just continue searching
                # If this perturbation leads to improvement, it will be found in next iterations
                # Or we can check if it improved.
                new_score = _max_sidelobe(autocorr)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()

    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initialize using spectral method
    h = _spectral_init()
    
    # Refine using local search
    h = _local_search(h, max_iters=10000, deadline=deadline)
    
    # Check time
    if time.time() > deadline:
        return h
        
    # Try a few more restarts if time permits
    best_h = h
    best_score = _max_sidelobe(_compute_autocorr(h))
    
    while time.time() < deadline - 5.0:
        # New random init
        h_new = np.random.choice([-1.0, 1.0], size=N)
        # Spectral refinement
        h_new = _spectral_init() # This overwrites h_new? No, _spectral_init returns new h.
        # Let's call it properly
        h_new = _spectral_init()
        
        # Local search
        h_new = _local_search(h_new, max_iters=5000, deadline=deadline)
        
        score_new = _max_sidelobe(_compute_autocorr(h_new))
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new
            
    return best_h
