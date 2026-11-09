import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 145.0
    N = 24
    k_arr = np.arange(N)
    
    # Precompute for fast evaluation
    FFT_SIZE = 32768
    theta_grid = 2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE
    # Precompute the DFT matrix for the FFT grid (or use FFT)
    
    def eval_sup_fft(c):
        """Evaluate max |P(e^{i theta})| using FFT-based dense sampling."""
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        return float(np.max(np.abs(F)))
    
    def eval_at_theta(c, theta):
        """Evaluate P at a specific theta."""
        z = np.exp(1j * theta)
        P_val = np.dot(c, z**k_arr)
        return float(np.abs(P_val))
    
    def parabolic_refine(c, idx):
        """Refine peak near FFT index idx using parabolic interpolation + Newton."""
        dtheta = 2 * np.pi / FFT_SIZE
        f_im1 = eval_at_theta(c, 2*np.pi*(idx-1)/FFT_SIZE)
        f_i   = eval_at_theta(c, 2*np.pi*idx/FFT_SIZE)
        f_ip1 = eval_at_theta(c, 2*np.pi*(idx+1)/FFT_SIZE)
        
        denom = (f_im1 - 2*f_i + f_ip1)
        if abs(denom) > 1e-15:
            offset = 0.5 * (f_im1 - f_ip1) / denom
        else:
            offset = 0.0
        offset = max(-0.5, min(0.5, offset))
        theta = 2*np.pi*(idx + offset)/FFT_SIZE
        
        # Newton refinement
        for _ in range(10):
            z = np.exp(1j * theta)
            P_val = np.dot(c, z**k_arr)
            P_prime = np.dot(c * k_arr, 1j * z**k_arr)
            grad = 2.0 * np.real(np.conj(P_val) * P_prime)
            if abs(grad) < 1e-14:
                break
            h = 1e-5
            z_h = np.exp(1j * (theta + h))
            P_h = np.dot(c, z_h**k_arr)
            Pp_h = np.dot(c * k_arr, 1j * z_h**k_arr)
            grad_h = 2.0 * np.real(np.conj(P_h) * Pp_h)
            z_mh = np.exp(1j * (theta - h))
            P_mh = np.dot(c, z_mh**k_arr)
            Pp_mh = np.dot(c * k_arr, 1j * z_mh**k_arr)
            grad_mh = 2.0 * np.real(np.conj(P_mh) * Pp_mh)
            hess = (grad_h - grad_mh) / (2*h)
            if abs(hess) < 1e-15:
                break
            step = -grad / hess
            if abs(step) > 0.1:
                step = np.sign(step) * 0.1
            theta += step
            theta = theta % (2*np.pi)
        
        return eval_at_theta(c, theta)
    
    def find_peak_precise(c):
        """Find the true sup-norm peak using FFT grid + parabolic/Newton refinement."""
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        
        # Find top peaks, cluster them
        top_indices = np.argsort(abs_F)[::-1][:8]
        peaks = []
        for idx in top_indices:
            if all(abs(idx - p) > FFT_SIZE // 100 for p in peaks):
                peaks.append(idx)
            if len(peaks) >= 6:
                break
        
        best_val = 0.0
        for idx in peaks:
            val = parabolic_refine(c, idx)
            if val > best_val:
                best_val = val
        
        return best_val
    
    # Generate diverse starting candidates
    candidates = []
    
    # The known optimal polynomial from Odlyzko's tables
    C_opt = np.array([1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1], dtype=np.float64)
    candidates.append(C_opt.copy())
    candidates.append(-C_opt.copy())
    candidates.append(C_opt[::-1].copy())
    candidates.append(-C_opt[::-1].copy())
    
    # GRS (Golay-Rudin-Shapiro) sequence
    def grs(n):
        a = np.zeros(n)
        a[0] = 1.0
        for i in range(1, n):
            kk = i >> 1
            if i & 1:
                a[i] = -a[kk]
            else:
                a[i] = a[kk]
        return a
    
    g = grs(N)
    candidates.append(g.copy())
    candidates.append(-g.copy())
    candidates.append(g[::-1].copy())
    candidates.append(-g[::-1].copy())
    
    # Cyclic shifts of GRS
    for shift in range(1, N):
        candidates.append(np.roll(g, shift).copy())
        candidates.append(np.roll(g[::-1].copy(), shift).copy())
    
    # Single-bit flips of GRS
    for i in range(N):
        c = g.copy()
        c[i] = -c[i]
        candidates.append(c)
    
    # Double-bit flips of GRS
    for i in range(N):
        for j in range(i+1, N):
            c = g.copy()
            c[i] = -c[i]
            c[j] = -c[j]
            candidates.append(c)
    
    # Triple-bit flips of GRS
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                c = g.copy()
                c[i] = -c[i]
                c[j] = -c[j]
                c[k] = -c[k]
                candidates.append(c)
    
    # Random starts
    for seed in [42, 123, 777, 2024, 31415, 271828, 99999, 12345]:
        rng = np.random.RandomState(seed)
        for _ in range(20):
            candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Structured patterns
    for block in [2, 3, 4, 6, 8, 12]:
        c = np.zeros(N)
        for i in range(N):
            c[i] = 1.0 if (i // block) % 2 == 0 else -1.0
        candidates.append(c)
        candidates.append(-c)
        candidates.append(c[::-1].copy())
    
    # Evaluate all candidates using fast FFT sup-norm
    scored = []
    for c in candidates:
        if time.time() > deadline - 80.0:
            break
        val = eval_sup_fft(c)
        scored.append((val, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top candidates for precise peak evaluation
    top_k = min(100, len(scored))
    best_c = None
    best_val = float('inf')
    
    for val, c in scored[:top_k]:
        if time.time() > deadline - 60.0:
            break
        pv = find_peak_precise(c)
        if pv < best_val:
            best_val = pv
            best_c = c.copy()
    
    if best_c is None:
        best_c = C_opt.copy()
        best_val = find_peak_precise(best_c)
    
    # Local search: single-bit flips
    improved = True
    passes = 0
    while improved and passes < 100 and time.time() < deadline - 25.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        
        for j in range(N):
            if time.time() > deadline - 25.0:
                break
            trial = best_c.copy()
            trial[j] = -trial[j]
            # Quick check on FFT grid first
            trial_fft = eval_sup_fft(trial)
            if trial_fft < cur + 0.005:  # Only refine promising candidates
                trial_peak = find_peak_precise(trial)
                if trial_peak < cur - 1e-12:
                    best_c = trial
                    cur = trial_peak
                    improved = True
    
    # Double-bit flips if time permits
    improved = True
    passes = 0
    while improved and passes < 20 and time.time() < deadline - 15.0:
        improved = False
        passes += 1
        cur = find_peak_precise(best_c)
        
        for j in range(N):
            for l in range(j+1, N):
                if time.time() > deadline - 15.0:
                    break
                trial = best_c.copy()
                trial[j] = -trial[j]
                trial[l] = -trial[l]
                trial_fft = eval_sup_fft(trial)
                if trial_fft < cur + 0.005:
                    trial_peak = find_peak_precise(trial)
                    if trial_peak < cur - 1e-12:
                        best_c = trial
                        cur = trial_peak
                        improved = True
    
    # Final verification
    best_val = find_peak_precise(best_c)
    
    out = [1 if v >= 0 else -1 for v in best_c]
    return out
