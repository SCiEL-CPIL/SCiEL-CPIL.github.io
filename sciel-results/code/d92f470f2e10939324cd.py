import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def eval_poly(c, theta):
        z = np.exp(1j * theta)
        vals = np.zeros_like(theta, dtype=np.complex128)
        for k in range(24):
            vals += c[k] * (z ** k)
        return np.abs(vals)
    
    def sup_norm(c):
        n_samples = 4096
        theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        mags = eval_poly(c, theta)
        
        peak_indices = []
        for i in range(n_samples):
            prev_i = (i - 1) % n_samples
            next_i = (i + 1) % n_samples
            if mags[i] >= mags[prev_i] and mags[i] >= mags[next_i]:
                peak_indices.append(i)
        
        if not peak_indices:
            peak_indices = [np.argmax(mags)]
        
        peak_indices.sort(key=lambda i: mags[i], reverse=True)
        top_peaks = peak_indices[:20]
        
        best = 0.0
        gr = (np.sqrt(5) - 1) / 2
        for idx in top_peaks:
            t_left = theta[(idx - 1) % n_samples]
            t_right = theta[(idx + 1) % n_samples]
            if t_left > t_right:
                t_right += 2 * np.pi
            
            a, b = t_left, t_right
            for _ in range(50):
                c1 = b - gr * (b - a)
                c2 = a + gr * (b - a)
                v1 = eval_poly(c, np.array([c1]))[0]
                v2 = eval_poly(c, np.array([c2]))[0]
                if v1 < v2:
                    a = c1
                else:
                    b = c2
            mid = (a + b) / 2
            vmid = eval_poly(c, np.array([mid]))[0]
            best = max(best, vmid)
        
        return best
    
    # GR initial
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=np.float64)
    best_val = sup_norm(best_coeffs)
    
    def local_search(c, current_val):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(24):
                if time.time() > deadline:
                    break
                trial = c.copy()
                trial[i] = -trial[i]
                val = sup_norm(trial)
                if val < current_val - 1e-12:
                    c = trial
                    current_val = val
                    improved = True
        return c, current_val
    
    best_coeffs, best_val = local_search(best_coeffs, best_val)
    
    # Try a few random starts if time permits
    rng = np.random.RandomState(42)
    for trial_start in range(20):
        if time.time() > deadline - 5.0:
            break
        c = rng.choice([1, -1], size=24).astype(np.float64)
        val = sup_norm(c)
        if val < best_val:
            c, val = local_search(c, val)
            if val < best_val:
                best_coeffs = c
                best_val = val
    
    result = np.where(best_coeffs > 0, 1, -1).astype(int)
    return result.tolist()
