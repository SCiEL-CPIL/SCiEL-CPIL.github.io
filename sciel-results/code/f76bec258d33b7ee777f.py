"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute the max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 2 * n
    H = np.fft.fft(h, n=fft_size)
    acf = np.fft.ifft(np.abs(H)**2).real
    max_val = 0.0
    for k in range(1, n):
        val = abs(acf[k])
        if val > max_val:
            max_val = val
    return max_val


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n: (-1)^popcount(i)."""
    tm = np.ones(n)
    for i in range(1, n):
        if i % 2 == 0:
            tm[i] = tm[i // 2]
        else:
            tm[i] = -tm[(i - 1) // 2]
    return tm


def _logistic_signal(n, r, x0=0.5):
    """Generate a signal from the logistic map, mapped to ±1."""
    x = x0
    seq = np.zeros(n)
    for i in range(n):
        x = r * x * (1 - x)
        seq[i] = 1.0 if x > 0.5 else -1.0
    return seq


def _spectral_flatness(h):
    """Compute spectral flatness (geometric mean / arithmetic mean of PSD)."""
    n = len(h)
    fft_size = 2 * n
    H = np.fft.fft(h, n=fft_size)
    psd = np.abs(H[:n]) ** 2
    psd = psd[psd > 1e-10]  # avoid log(0)
    if len(psd) == 0:
        return 0.0
    log_mean = np.mean(np.log(psd + 1e-15))
    arith_mean = np.mean(psd)
    return np.exp(log_mean) / arith_mean


def _combined_score(h):
    """Combined score: autocorrelation peak + small spectral flatness penalty."""
    ac_peak = _autocorr_peak(h)
    flatness = _spectral_flatness(h)
    return ac_peak + 0.001 * (1.0 - flatness)


def _local_search(h, deadline, max_iters=2000):
    """Local search with single and pair flips."""
    current_h = h.copy()
    current_score = _combined_score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try single flip
        idx = rng.integers(N)
        old_val = current_h[idx]
        current_h[idx] = -old_val
        new_score = _combined_score(current_h)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            current_h[idx] = old_val
    
    # Try pair flips
    for it in range(max_iters // 2):
        if time.time() > deadline:
            break
        
        idx1 = rng.integers(N)
        idx2 = rng.integers(N)
        if idx1 == idx2:
            continue
        
        v1, v2 = current_h[idx1], current_h[idx2]
        current_h[idx1] = -v1
        current_h[idx2] = -v2
        new_score = _combined_score(current_h)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            current_h[idx1] = v1
            current_h[idx2] = v2
    
    return best_h, best_score


def construct():
    """
    Construct a signed step function with N=400 intervals
    that minimizes the autoconvolution upper bound (C3).
    
    Strategy: Use logistic map chaotic signals as base candidates,
    combined with Thue-Morse, then refine with local search guided
    by combined autocorrelation + spectral flatness objective.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_h = np.zeros(N)
    best_score = np.inf
    
    # Generate diverse candidate bases
    candidates = []
    
    # 1. Thue-Morse and variants
    tm = _thue_morse(N)
    candidates.append(tm.copy())
    candidates.append(-tm.copy())
    for shift in range(0, 8, 2):
        candidates.append(np.roll(tm, shift).copy())
    
    # 2. Logistic map signals with various parameters
    r_values = [3.5, 3.7, 3.8, 3.9, 3.95, 3.99, 3.999]
    x0_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for r in r_values:
        for x0 in x0_values:
            if time.time() > deadline:
                break
            sig = _logistic_signal(N, r, x0)
            candidates.append(sig)
            candidates.append(-sig)
    
    # 3. Random ±1 sequences
    for seed in range(10):
        rng = np.random.default_rng(seed + 1000)
        candidates.append(rng.choice([-1, 1], size=N).astype(float))
    
    # 4. Thue-Morse with random sign flips
    for seed in range(5):
        rng = np.random.default_rng(seed + 2000)
        mask = rng.choice([-1, 1], size=N)
        candidates.append(tm * mask)
    
    # Evaluate all candidates
    scored = []
    for h in candidates:
        if time.time() > deadline:
            break
        s = _combined_score(h)
        scored.append((s, h))
    
    scored.sort(key=lambda x: x[0])
    
    # Refine top candidates with local search
    for s, h in scored[:8]:
        if time.time() > deadline:
            break
        
        refined_h, refined_score = _local_search(h, deadline, max_iters=1500)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h.copy()
    
    # Final refinement on the best result
    if time.time() < deadline:
        final_h, final_score = _local_search(best_h, deadline, max_iters=1000)
        if final_score < best_score:
            best_score = final_score
            best_h = final_h.copy()
    
    # Ensure we have a valid result
    if best_score == np.inf:
        best_h = tm
        best_score = _combined_score(tm)
    
    return best_h
# EVOLVE-BLOCK-END
