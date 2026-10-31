import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s limit
    
    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        # Autoconvolution via FFT
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)
    
    def spectral_optimize(h_init, max_iters=5000, flatten_power=0.99):
        """Iterative projection: flatten power spectrum + enforce nonnegativity."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        
        size = 2 * N
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # FFT
            H = np.fft.rfft(h, n=size)
            
            # Power spectrum
            power = np.abs(H) ** 2
            
            # Flatten: reduce peaks in power spectrum
            # Target: make power spectrum more uniform
            # Use a soft clipping: if power > threshold, reduce it
            mean_power = np.mean(power)
            # Only flatten the peaks (above some multiple of mean)
            threshold = mean_power * 1.5
            excess = np.maximum(power - threshold, 0)
            # Reduce the excess proportionally
            power_new = power - flatten_power * excess
            power_new = np.maximum(power_new, 0)
            
            # Reconstruct H with flattened power (keep phase)
            phase = np.angle(H)
            H_new = np.sqrt(power_new) * np.exp(1j * phase)
            
            # IFFT back
            h_new = np.fft.irfft(H_new, n=size)[:N]
            
            # Project onto nonnegative
            h_new = np.maximum(h_new, 0.0)
            
            # Ensure not all zero
            if np.all(h_new == 0):
                h_new = h.copy()
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            h = h_new
        
        return best_h, best_m
    
    def local_refine(h_init, max_iters=50000):
        """Random perturbation local search to refine the result."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        
        np.random.seed(42)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick two random indices and swap some mass
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            # Move small fraction of mass
            frac = np.random.uniform(0.01, 0.1)
            delta = h[j] * frac
            if delta < 1e-15:
                continue
            
            new_h = h.copy()
            new_h[j] -= delta
            new_h[i] += delta
            
            if new_h[j] < -1e-15:
                continue
            
            m_new = compute_metric(new_h)
            
            if m_new < best_m:
                best_m = m_new
                best_h = new_h.copy()
                h = new_h
        
        return best_h, best_m
    
    # Multiple initializations for spectral method
    i_arr = np.arange(N)
    mid = N / 2.0
    
    initializations = [
        np.ones(N),
        0.5 * (1.0 + np.cos(np.pi * (i_arr - mid) / (N / 2.0))),
        np.exp(-2.0 * ((i_arr - mid) / (N / 4.0)) ** 2),
        1.0 - 0.3 * np.cos(2 * np.pi * i_arr / N),
        np.cos(np.pi * (i_arr - mid) / N),
    ]
    
    best_h = None
    best_m = float('inf')
    
    # Phase 1: Spectral optimization from multiple starts
    for init in initializations:
        if time.time() > deadline:
            break
        init = np.maximum(init, 0.0)
        h_opt, m_opt = spectral_optimize(init, max_iters=3000)
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    # Phase 2: Local refinement
    if best_h is not None and time.time() < deadline - 10:
        h_ref, m_ref = local_refine(best_h, max_iters=30000)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref
    
    # Phase 3: If still have time, try more spectral iterations from best
    if best_h is not None and time.time() < deadline - 5:
        h_more, m_more = spectral_optimize(best_h, max_iters=5000, flatten_power=0.95)
        if m_more < best_m:
            best_m = m_more
            best_h = h_more
    
    return list(best_h)

# EVOLVE-BLOCK-END
