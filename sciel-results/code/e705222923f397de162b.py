import time
import numpy as np
from scipy.optimize import minimize

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

def _two_block_params_to_heights(params):
    """Convert 6 parameters to a height array.
    params: [left_center, left_width, left_height, right_center, right_width, right_height]
    All centers and widths are in [0,1] normalized coordinates.
    """
    lc, lw, lh, rc, rw, rh = params
    # Clamp to valid ranges
    lc = np.clip(lc, 0, 1)
    lw = np.clip(lw, 0.01, 1)
    lh = np.clip(lh, 0, 10)
    rc = np.clip(rc, 0, 1)
    rw = np.clip(rw, 0.01, 1)
    rh = np.clip(rh, 0, 10)
    
    x = np.linspace(0, 1, N)
    h = np.zeros(N)
    # Smooth block using cosine taper
    left_block = np.zeros(N)
    mask_l = np.abs(x - lc) <= lw / 2
    if np.any(mask_l):
        # Cosine shape within the block
        dist = np.abs(x[mask_l] - lc) / (lw / 2)
        left_block[mask_l] = lh * 0.5 * (1 + np.cos(np.pi * dist))
    
    right_block = np.zeros(N)
    mask_r = np.abs(x - rc) <= rw / 2
    if np.any(mask_r):
        dist = np.abs(x[mask_r] - rc) / (rw / 2)
        right_block[mask_r] = rh * 0.5 * (1 + np.cos(np.pi * dist))
    
    h = np.maximum(left_block, right_block)
    return h

def _parametric_score(params):
    """Score for a parametric two-block function."""
    h = _two_block_params_to_heights(params)
    return compute_score(h)

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
    
    # 2. Gaussian centered with various widths
    x = np.linspace(0, 1, N)
    for width in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks with various configurations
    for left_frac in [0.1, 0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.05, 0.1, 0.15, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse with various densities
    for _ in range(10):
        hr = np.zeros(N)
        density = rng.uniform(0.2, 0.8)
        mask = rng.random(N) < density
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)
        
    # 6. Plateau with tapered edges
    for plateau_frac in [0.3, 0.5, 0.7]:
        hp = np.zeros(N)
        p_start = int((1 - plateau_frac) / 2 * N)
        p_end = int((1 + plateau_frac) / 2 * N)
        hp[p_start:p_end] = 1.0
        taper_len = max(1, int(0.1 * N))
        if p_start > taper_len:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            hp[p_start - taper_len:p_start] = ramp
        if p_end < N - taper_len:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            hp[p_end:p_end + taper_len] = ramp[::-1]
        candidates.append(hp)

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

    # Phase 1: Optimize parametric two-block family using scipy
    if time_left() > 30:
        # Try multiple random initializations for the parametric optimizer
        n_param_starts = 20
        for ps in range(n_param_starts):
            if deadline_reached():
                break
            
            # Random initial parameters for two blocks
            lc = rng.uniform(0.1, 0.4)
            lw = rng.uniform(0.1, 0.3)
            lh = rng.uniform(0.5, 2.0)
            rc = rng.uniform(0.6, 0.9)
            rw = rng.uniform(0.1, 0.3)
            rh = rng.uniform(0.5, 2.0)
            
            x0 = np.array([lc, lw, lh, rc, rw, rh])
            
            try:
                result = minimize(_parametric_score, x0, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            except Exception:
                continue
            
            if result.fun < best_score:
                best_params = result.x
                best_h = _two_block_params_to_heights(best_params)
                best_score = result.fun
        
        # Also try symmetric two-block: center at 0.25 and 0.75
        for width_frac in [0.15, 0.2, 0.25, 0.3]:
            if deadline_reached():
                break
            for height in [0.5, 1.0, 1.5, 2.0]:
                x0 = np.array([0.25, width_frac, height, 0.75, width_frac, height])
                try:
                    result = minimize(_parametric_score, x0, method='Nelder-Mead',
                                      options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6})
                except Exception:
                    continue
                if result.fun < best_score:
                    best_params = result.x
                    best_h = _two_block_params_to_heights(best_params)
                    best_score = result.fun

    # Phase 2: Local refinement with coordinate descent
    if time_left() > 20:
        current_h = best_h.copy()
        current_score = best_score
        
        # Coordinate descent: optimize one coordinate at a time
        coord_passes = 3
        for _ in range(coord_passes):
            if deadline_reached():
                break
            for i in range(N):
                if deadline_reached():
                    break
                old_val = current_h[i]
                # Try a few small changes
                for delta in [-0.05, -0.02, 0.02, 0.05]:
                    new_h = current_h.copy()
                    new_h[i] = max(0.0, old_val + delta)
                    new_score = compute_score(new_h)
                    if new_score < current_score:
                        current_h = new_h
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_h = current_h.copy()
                        break
        
        # Local random search around best
        if time_left() > 10:
            polish_steps = 1000
            current_h = best_h.copy()
            current_score = best_score
            step_size = 0.03
            
            for _ in range(polish_steps):
                if deadline_reached():
                    break
                num_perturb = rng.integers(3, 15)
                indices = rng.choice(N, num_perturb, replace=False)
                noise = rng.normal(0, step_size, num_perturb)
                new_h = current_h.copy()
                new_h[indices] = np.maximum(current_h[indices] + noise, 0.0)
                new_score = compute_score(new_h)
                
                if new_score < current_score:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                    step_size = min(0.08, step_size * 1.03)
                else:
                    step_size = max(0.003, step_size * 0.93)

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
