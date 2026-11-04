"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_upper_bound(signed):
    """
    Compute the upper bound on C3 for a signed step function.
    This is the maximum absolute value of the autoconvolution at non-zero lags.
    """
    n = len(signed)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    # Pad to next power of 2 for efficiency, or just 2*n
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.fft(signed, size)
    acf = np.fft.ifft(f * np.conj(f)).real
    # acf[k] for k=0..size-1
    # We want max |acf[k]| for k != 0
    # Note: acf is symmetric for real signals.
    # acf[0] is the zero-lag term.
    # We need to exclude acf[0].
    # The relevant lags are 1..n-1 and size-(n-1)..size-1
    # But simply taking max over all indices except 0 is safer and slightly more expensive,
    # but with size ~ 1024 it's negligible.
    # However, strictly speaking, lags beyond n-1 are zero (or near zero due to numerical error).
    # Let's just take max over 1..size-1.
    return float(np.max(np.abs(acf[1:])))


def _local_search(signed, deadline):
    """
    Local search by flipping individual bits.
    Uses a deadline to stop.
    """
    n = len(signed)
    best = signed.copy()
    best_score = _autocorr_upper_bound(best)
    
    # We will do a few passes, or until deadline
    # To be efficient, we don't want to copy the whole array for every single flip evaluation
    # if we can avoid it, but numpy is fast enough for N=400.
    
    # Strategy: Randomized hill climbing with restarts
    # Or: Greedy sweep until no improvement, then random restart.
    
    # Let's do a simple randomized local search
    # 1. Pick random index, flip, check.
    # 2. If improved, accept.
    # 3. If not, reject.
    # 4. Occasionally (or if stuck) do a random restart or multi-flip.
    
    # Better: Cyclic scan with randomization.
    
    while time.time() < deadline:
        improved = False
        # Random permutation of indices to avoid bias
        indices = np.random.permutation(n)
        
        for idx in indices:
            if time.time() > deadline:
                return best, best_score
            
            # Flip
            best[idx] *= -1
            new_score = _autocorr_upper_bound(best)
            
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                best[idx] *= -1
        
        if not improved:
            # Stuck in local minimum.
            # Try a random perturbation: flip K random bits
            # K can be small (1-5) or larger.
            # Let's try a few random perturbations.
            perturbed = False
            for _ in range(10):
                if time.time() > deadline:
                    return best, best_score
                candidate = best.copy()
                num_flips = np.random.randint(1, 10)
                flip_idx = np.random.choice(n, size=num_flips, replace=False)
                candidate[flip_idx] *= -1
                
                cand_score = _autocorr_upper_bound(candidate)
                if cand_score < best_score:
                    best = candidate
                    best_score = cand_score
                    perturbed = True
                    break
            
            if not perturbed:
                # If we can't escape even with perturbation, break to save time
                # Or continue with different random seed?
                # Given the deadline, breaking is safer to ensure we return something.
                # But maybe try a few more times?
                # Let's just break.
                break

    return best, best_score


def construct():
    start_time = time.time()
    # Safety margin: 150s limit. Let's use 145s.
    deadline = start_time + 145.0
    
    # Initialize with Thue-Morse or random?
    # Thue-Morse has good properties.
    # Let's generate Thue-Morse of length N.
    # t_k = popcount(k) % 2
    # h_k = 1 if t_k == 0 else -1
    
    # Vectorized Thue-Morse generation
    # For N=400, simple loop is fine, but let's be smart.
    # Actually, N is small, so simple generation is fast.
    
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        x = i
        c = 0
        while x:
            c += x & 1
            x >>= 1
        tm[i] = c & 1
    
    signed = 1 - 2 * tm  # 0 -> 1, 1 -> -1
    
    # Also try a random initialization to see if it's better
    np.random.seed(42)
    random_signed = np.random.choice([-1, 1], size=N)
    
    # Compare initial scores
    score_tm = _autocorr_upper_bound(signed)
    score_rand = _autocorr_upper_bound(random_signed)
    
    if score_rand < score_tm:
        current_best = random_signed
        current_score = score_rand
    else:
        current_best = signed
        current_score = score_tm
    
    # Perform local search
    # We can run multiple local searches from different starting points
    # Or just one long one.
    # Let's run one long one, but with internal restarts.
    
    best, best_score = _local_search(current_best, deadline)
    
    # If we have time, maybe try another start?
    if time.time() < deadline - 5:
        # Try starting from the other one
        if score_tm < score_rand:
            start2 = random_signed
        else:
            start2 = signed
            
        best2, score2 = _local_search(start2, deadline)
        if score2 < best_score:
            best = best2
            best_score = score2
            
    return best

# EVOLVE-BLOCK-END
