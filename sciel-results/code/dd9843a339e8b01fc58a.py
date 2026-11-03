"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(x):
    """Compute max absolute value of autoconvolution of x."""
    n = len(x)
    # Use FFT for autoconvolution
    size = 2 * n
    f = np.fft.rfft(x, size=size)
    ac = np.fft.irfft(f * np.conj(f), size=size)[:n]
    return np.max(np.abs(ac))

def _integral_sq(x):
    """Squared integral (sum) of x."""
    return float(np.sum(x)) ** 2

def _score(x):
    """Compute the score: max |AC| / (sum)^2."""
    s = np.sum(x)
    if s == 0:
        return np.inf
    ac = _autoconv_max_abs(x)
    return ac / (s * s)

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Start with all ones (the seed)
    x = np.ones(N, dtype=np.float64)
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Also try a few random starts
    for _ in range(5):
        x_rand = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(x_rand)
        if abs(s) < 10:
            continue
        sc = _score(x_rand)
        if sc < best_score:
            best_score = sc
            best_x = x_rand.copy()
    
    # Local search: peak-targeting bit flips
    x = best_x.copy()
    current_score = best_score
    
    # Precompute for speed
    def evaluate_flip(x, idx):
        """Evaluate score if we flip bit at idx."""
        x[idx] = -x[idx]
        s = np.sum(x)
        if s == 0:
            x[idx] = -x[idx]
            return np.inf
        ac = _autoconv_max_abs(x)
        sc = ac / (s * s)
        x[idx] = -x[idx]
        return sc
    
    # More efficient: compute AC once, then for each flip, update AC incrementally
    # AC[k] = sum_i x[i] * x[i+k]
    # Flipping x[j] changes AC[k] by -2*x[j]*x[j+k] for all valid k
    
    while time.time() < deadline:
        # Compute current AC
        n = N
        size = 2 * n
        f = np.fft.rfft(x, size=size)
        ac = np.fft.irfft(f * np.conj(f), size=size)[:n]
        
        # Find the peak lag
        peak_lag = int(np.argmax(np.abs(ac)))
        peak_val = np.abs(ac[peak_lag])
        
        # For each possible flip, compute how much it changes the peak
        # Flipping x[j]: AC[k] changes by -2*x[j]*x[(j+k) mod n] for circular... 
        # Actually autoconvolution is linear, not circular.
        # AC[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
        # Flipping x[j]: for lag k, if 0 <= j <= N-1-k, AC[k] -= 2*x[j]*x[j+k]
        #               if k <= j <= N-1, AC[k] -= 2*x[j-k]*x[j] (same thing)
        
        # Let's just try a subset of flips per iteration for speed
        # Try flipping each position and see effect on peak
        best_flip_idx = -1
        best_new_peak = peak_val
        s_current = np.sum(x)
        
        for j in range(N):
            # Flipping x[j]
            new_s = s_current - 2 * x[j]
            if new_s == 0:
                continue
            
            # Update AC for all lags affected by flipping x[j]
            # AC[k] for k where j is in range: 
            # For lag k >= 0: term x[j]*x[j+k] exists if j+k < N, and x[j-k]*x[j] if j-k >= 0
            # So AC[k] -= 2 * x[j] * (x[j+k] if j+k<N else 0) - 2 * x[j] * (x[j-k] if j-k>=0 else 0)
            # Wait, more carefully:
            # AC[k] = sum_i x[i]*x[i+k]
            # After flipping x[j]: new_AC[k] = AC[k] - 2*x[j]*x[j+k] (if j+k<N) - 2*x[j]*x[j-k] (if j-k>=0 and j-k != j, i.e., k!=0)
            # Actually for k=0: AC[0] = sum x[i]^2 = N always for ±1, so flipping doesn't change AC[0]
            
            # For efficiency, let's just compute the new AC at the peak lag
            # AC[peak_lag] after flip: 
            # If peak_lag > 0:
            #   term1: x[j]*x[j+peak_lag] if j+peak_lag < N
            #   term2: x[j-peak_lag]*x[j] if j-peak_lag >= 0
            #   new_AC[peak_lag] = AC[peak_lag] - 2*x[j]*(x[j+peak_lag] if j+peak_lag<N else 0) - 2*x[j]*(x[j-peak_lag] if j-peak_lag>=0 else 0)
            #   Wait, that's not right either. Let me think again.
            
            # AC[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
            # Flipping x[j] to -x[j]:
            #   For each i where i=j or i+k=j:
            #     if i=j: term was x[j]*x[j+k], becomes -x[j]*x[j+k], change = -2*x[j]*x[j+k]
            #     if i+k=j, i.e., i=j-k: term was x[j-k]*x[j], becomes x[j-k]*(-x[j]), change = -2*x[j-k]*x[j]
            #   So delta_AC[k] = -2*x[j]*x[j+k] (if 0<=j<=N-1-k) - 2*x[j-k]*x[j] (if 0<=j-k<=N-1-k, i.e., k<=j<=N-1)
            #   For k=0: both terms are -2*x[j]*x[j] = -2, but they're the same term counted twice? 
            #   No, for k=0: i=j gives x[j]*x[j], and i+k=j means i=j, same term. So delta = -2*x[j]^2 = -2.
            #   But AC[0] = N always, so this is consistent: N - 2 + 2 = N... wait no.
            #   Actually for k=0, the sum is over i=0 to N-1 of x[i]*x[i] = N. Flipping one x[j] doesn't change it since x[j]^2 = 1 always.
            #   So my formula is wrong for k=0. The two terms are the same when k=0.
            
            # Let me just do a direct computation for the peak lag
            if peak_lag == 0:
                # AC[0] = N always, no change
                continue
            else:
                # Compute new AC[peak_lag]
                new_ac_peak = ac[peak_lag]
                # Term where i=j: exists if j + peak_lag < N
                if j + peak_lag < N:
                    new_ac_peak -= 2 * x[j] * x[j + peak_lag]
                # Term where i = j - peak_lag: exists if j - peak_lag >= 0
                if j - peak_lag >= 0:
                    new_ac_peak -= 2 * x[j - peak_lag] * x[j]
                
                new_abs_peak = abs(new_ac_peak)
                if new_abs_peak < best_new_peak:
                    best_new_peak = new_abs_peak
                    best_flip_idx = j
        
        if best_flip_idx == -1:
            # No improvement found, try random perturbations
            for _ in range(20):
                j = rng.integers(0, N)
                x[j] = -x[j]
                s = np.sum(x)
                if s == 0:
                    x[j] = -x[j]
                    continue
                sc = _score(x)
                if sc < current_score:
                    current_score = sc
                else:
                    x[j] = -x[j]
            break
        else:
            x[best_flip_idx] = -x[best_flip_idx]
            new_s = np.sum(x)
            if new_s == 0:
                x[best_flip_idx] = -x[best_flip_idx]
                break
            new_ac = _autoconv_max_abs(x)
            current_score = new_ac / (new_s * new_s)
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
