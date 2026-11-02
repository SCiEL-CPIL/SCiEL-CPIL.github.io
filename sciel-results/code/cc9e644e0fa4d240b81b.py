import time
import numpy as np

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    M = 1
    while M < 2 * n - 1:
        M *= 2
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    return conv

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0

    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered
    x = np.linspace(0, 1, N)
    for width in [0.1, 0.2, 0.3, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks
    for left_frac in [0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.1, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse
    for _ in range(5):
        hr = np.zeros(N)
        mask = rng.random(N) < 0.5
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.1, 0.2, 0.3, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Multi-Start Simulated Annealing directly on 600D heights
    # We will run several independent SA chains
    
    num_chains = 5
    steps_per_chain = 2000
    
    for chain_idx in range(num_chains):
        if deadline_reached():
            break
        
        # Initialize this chain from a random candidate or best
        if chain_idx == 0:
            current_h = best_h.copy()
        else:
            # Pick a random candidate from the list, or generate new random
            if rng.random() < 0.5:
                current_h = candidates[rng.integers(0, len(candidates))].copy()
            else:
                # Random Gaussian perturbation around uniform
                current_h = np.maximum(np.ones(N) + rng.normal(0, 0.5, N), 0.0)
            
            # Normalize to similar scale? Not strictly necessary if score is scale invariant?
            # Score = max_conv * N / (total^2 * w).
            # If h -> c*h, max_conv -> c^2 max_conv, total -> c total.
            # Score -> c^2 / c^2 = 1. Scale invariant.
            # So we don't need to normalize, but non-negativity is crucial.
            
        current_score = compute_score(current_h)
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
            
        # SA Parameters
        T_start = 0.1
        T_end = 0.001
        # Adaptive step size
        step_size = 0.1
        
        for step in range(steps_per_chain):
            if deadline_reached():
                break
                
            # Annealing schedule
            progress = step / steps_per_chain
            T = T_start * (T_end / T_start) ** progress
            
            # Generate perturbation
            # Add Gaussian noise
            noise = rng.normal(0, step_size, N)
            new_h = current_h + noise
            new_h = np.maximum(new_h, 0.0)
            
            if np.sum(new_h) <= 0:
                continue
                
            new_score = compute_score(new_h)
            
            # Acceptance
            if new_score < current_score:
                accept = True
            else:
                delta = new_score - current_score
                # Only accept small deteriorations
                if delta < 0.05: 
                    accept = rng.random() < np.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                
                # Increase step size if we are making progress? Or keep it adaptive.
                # Let's try a simple adaptive strategy: if accepted, increase slightly, if rejected, decrease.
                # But we have T cooling. Let's stick to fixed step size per chain for simplicity first.
                # Actually, let's make step size decay slightly as well to refine.
                step_size = max(0.01, step_size * 0.999)
            else:
                step_size = max(0.01, step_size * 0.95)
                
    # Final polish: Local random search around best_h
    if time_left() > 10:
        polish_steps = 500
        current_h = best_h.copy()
        current_score = best_score
        step_size = 0.05
        
        for _ in range(polish_steps):
            if deadline_reached():
                break
            noise = rng.normal(0, step_size, N)
            new_h = np.maximum(current_h + noise, 0.0)
            new_score = compute_score(new_h)
            
            if new_score < current_score:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                step_size = min(0.1, step_size * 1.05)
            else:
                step_size = max(0.005, step_size * 0.9)

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
