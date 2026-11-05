"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:n + n - 1]
    return conv

def _compute_value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, tau):
    """Numerically stable soft max."""
    x_shifted = x - np.max(x)
    return tau * np.log(np.sum(np.exp(x_shifted / tau)))

def _smooth_value(h, tau):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _fft_autoconv(h)
    sm = _soft_max(conv, tau)
    return abs(2.0 * N * sm / (s * s))

def _coordinate_descent(h, max_iters=200, tol=1e-10, deadline=None):
    """Coordinate descent on individual elements."""
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-12:
        return h
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            old_val = _compute_value(h)
            
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h_new = h.copy()
                h_new[i] += delta
                s_new = np.sum(h_new)
                if abs(s_new) < 1e-12:
                    continue
                new_val = _compute_value(h_new)
                if new_val < old_val - tol:
                    h = h_new
                    old_val = new_val
                    improved = True
                    break
            
            if not improved and it > 50:
                break
        if not improved:
            break
    
    return h

def _anneal_coordinate_descent(h, start_temp=0.1, end_temp=0.01, n_steps=100, deadline=None):
    """Simulated annealing with coordinate descent."""
    h = h.copy()
    best_h = h.copy()
    best_val = _compute_value(h)
    current_val = best_val
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (step / n_steps)
        
        # Pick random coordinate
        i = np.random.randint(N)
        h_new = h.copy()
        h_new[i] += temp * np.random.randn()
        
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-12:
            continue
        
        new_val = _compute_value(h_new)
        
        # Accept if better or with probability exp(-delta/temp)
        if new_val < current_val or np.random.rand() < np.exp(-(new_val - current_val) / max(temp, 1e-10)):
            h = h_new
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Generate diverse initializations
    x = np.linspace(0, 1, N)
    initializations = []
    
    # 1. Cosine window
    initializations.append(0.5 + 0.5 * np.cos(2 * np.pi * x))
    
    # 2. Raised cosine
    initializations.append(0.5 * (1 + np.cos(2 * np.pi * x)))
    
    # 3. Sine oscillatory
    initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * x))
    initializations.append(1.0 + 0.3 * np.sin(4 * np.pi * x))
    
    # 4. Two-lobe
    initializations.append(1.0 + 0.5 * np.sin(np.pi * x))
    
    # 5. Uniform with noise
    initializations.append(np.ones(N) + 0.01 * np.random.randn(N))
    
    # 6. Triangular
    tri = np.minimum(x, 1 - x) * 2
    initializations.append(0.5 + 0.5 * tri)
    
    # 7. Gaussian
    initializations.append(np.exp(-0.5 * ((x - 0.5) * 4) ** 2))
    
    # 8. Bump functions
    for center in [0.25, 0.5, 0.75]:
        width = 0.15
        initializations.append(np.exp(-0.5 * ((x - center) / width) ** 2))
    
    # 9. Multi-bump
    initializations.append(0.5 * np.exp(-0.5 * ((x - 0.3) / 0.1) ** 2) + 0.5 * np.exp(-0.5 * ((x - 0.7) / 0.1) ** 2))
    
    # 10. Cosine with phase shift
    for phase in [0.25, 0.5, 0.75]:
        initializations.append(0.5 + 0.5 * np.cos(2 * np.pi * (x - phase)))
    
    best_overall = best_h.copy()
    best_overall_val = best_val
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        # Ensure sum is not zero
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 1.0
        
        h0 = h0 / s0 * N  # normalize so sum = N
        
        # Try coordinate descent
        h_cd = _coordinate_descent(h0, max_iters=100, deadline=deadline)
        val_cd = _compute_value(h_cd)
        
        if val_cd < best_overall_val:
            best_overall_val = val_cd
            best_overall = h_cd.copy()
        
        # Try annealing
        h_anneal = _anneal_coordinate_descent(h0, n_steps=50, deadline=deadline)
        val_anneal = _compute_value(h_anneal)
        
        if val_anneal < best_overall_val:
            best_overall_val = val_anneal
            best_overall = h_anneal.copy()
        
        # Also try direct evaluation
        val0 = _compute_value(h0)
        if val0 < best_overall_val:
            best_overall_val = val0
            best_overall = h0.copy()
    
    # Final: try more aggressive coordinate descent around best
    if time.time() < deadline - 5:
        h_final = _coordinate_descent(best_overall, max_iters=300, tol=1e-12, deadline=deadline)
        val_final = _compute_value(h_final)
        if val_final < best_overall_val:
            best_overall_val = val_final
            best_overall = h_final.copy()
    
    # Safety: ensure valid output
    s_final = np.sum(best_overall)
    if abs(s_final) < 1e-10:
        return np.ones(N)
    
    return best_overall
# EVOLVE-BLOCK-END
