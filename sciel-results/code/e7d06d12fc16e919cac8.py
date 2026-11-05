import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    # Dense grid for fast evaluation
    M = 8192
    thetas = 2 * np.pi * np.arange(M) / M
    ks = np.arange(N)
    angles = np.outer(thetas, ks)
    C = np.cos(angles)
    S = np.sin(angles)
    
    def eval_score(c):
        pr = C @ c
        pi = S @ c
        mag2 = pr**2 + pi**2
        return np.max(mag2)
    
    def refine_peak(c, theta_center, width=0.01):
        # Golden section refinement around theta_center
        a = theta_center - width
        b = theta_center + width
        gr = (np.sqrt(5) - 1) / 2
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        
        def f(t):
            z = np.exp(1j * t)
            # Evaluate P(z) = sum c_k z^k
            # Using Horner's method or direct sum
            # Direct sum is fine for N=24
            vals = np.array([c[k] * z**k for k in range(N)])
            return np.abs(np.sum(vals))
        
        f1 = f(c1)
        f2 = f(c2)
        for _ in range(50):
            if f1 < f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = f(c1)
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = f(c2)
        
        return max(f1, f2)
    
    def precise_score(c):
        # Fast grid eval to find top peaks, then refine
        pr = C @ c
        pi = S @ c
        mag = np.sqrt(pr**2 + pi**2)
        
        # Find top 5 peaks
        idx = np.argsort(mag)[::-1][:5]
        best_val = mag[idx[0]]
        
        for i in idx:
            theta = thetas[i]
            refined = refine_peak(c, theta, width=2*np.pi/M)
            if refined > best_val:
                best_val = refined
        
        return best_val
    
    # Generate initial candidates
    candidates = []
    
    # 1. Golay-Rudin-Shapiro
    def gr_sequence(n):
        p = [1]
        q = [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=np.float64)
    candidates.append(gr_sequence(N))
    
    # 2. Spectral synthesis: inverse FFT of flattened spectrum
    # Target: constant magnitude spectrum -> flat time domain? No, we want flat frequency domain.
    # We want P(z) to be flat on |z|=1.
    # Let's try to synthesize a sequence whose DFT is as flat as possible.
    # This is equivalent to finding a ±1 sequence with low PAPR.
    # Start with a random sequence, take its DFT, flatten it (take phase of DFT, magnitude 1),
    # inverse DFT, then threshold to ±1.
    
    def spectral_init(seed):
        np.random.seed(seed)
        c = np.random.choice([-1, 1], size=N).astype(np.float64)
        # Iterate: DFT -> flatten phase -> IFFT -> threshold
        for _ in range(100):
            F = np.fft.fft(c)
            # Keep phase, set magnitude to 1
            F_flat = F / np.abs(F)
            F_flat[np.abs(F) < 1e-10] = 1.0
            c_new = np.real(np.fft.ifft(F_flat))
            # Threshold to ±1
            c = np.sign(c_new)
            c[c == 0] = 1.0
        return c
    
    for s in range(20):
        candidates.append(spectral_init(s))
    
    # 3. Random
    np.random.seed(42)
    for _ in range(20):
        candidates.append(np.random.choice([-1, 1], size=N).astype(np.float64))
    
    best_c = None
    best_score = np.inf
    
    # Local search: 1-flip and 2-flip
    def local_search(c, time_limit):
        c = c.copy()
        current_score = eval_score(c)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            pr = C @ c
            pi = S @ c
            current_mag2 = pr**2 + pi**2
            current_max = np.max(current_mag2)
            
            # 1-flip
            for k in range(N):
                new_pr = pr - 2 * c[k] * C[:, k]
                new_pi = pi - 2 * c[k] * S[:, k]
                new_mag2 = new_pr**2 + new_pi**2
                new_max = np.max(new_mag2)
                if new_max < current_max:
                    c[k] = -c[k]
                    pr = new_pr
                    pi = new_pi
                    current_max = new_max
                    current_score = new_max
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2-flip if 1-flip didn't help
            for k1 in range(N):
                for k2 in range(k1 + 1, N):
                    new_pr = pr - 2 * c[k1] * C[:, k1] - 2 * c[k2] * C[:, k2]
                    new_pi = pi - 2 * c[k1] * S[:, k1] - 2 * c[k2] * S[:, k2]
                    new_mag2 = new_pr**2 + new_pi**2
                    new_max = np.max(new_mag2)
                    if new_max < current_max:
                        c[k1] = -c[k1]
                        c[k2] = -c[k2]
                        pr = new_pr
                        pi = new_pi
                        current_max = new_max
                        current_score = new_max
                        improved = True
                        break
                if improved:
                    break
        
        return c, current_score
    
    # Evaluate and refine candidates
    for c in candidates:
        if (time.time() - start_time) > deadline:
            break
        
        # Quick eval first
        quick_score = eval_score(c)
        if quick_score < best_score:
            # Do local search
            c_ls, score_ls = local_search(c, deadline - 5)
            if score_ls < best_score:
                best_score = score_ls
                best_c = c_ls.copy()
        
        # Also try refining the best candidate with more aggressive search
        # If we have time, do a few random restarts from best_c
        if (time.time() - start_time) > deadline - 10:
            break
        
        if best_c is not None:
            # Perturb best_c
            c_pert = best_c.copy()
            for _ in range(3):
                idx = np.random.randint(N)
                c_pert[idx] = -c_pert[idx]
            
            c_ls, score_ls = local_search(c_pert, deadline - 5)
            if score_ls < best_score:
                best_score = score_ls
                best_c = c_ls.copy()
    
    # Final precise evaluation of best_c
    if best_c is not None:
        # Use precise score for final check, but return best_c
        # The evaluator will do its own precise scoring
        pass
    
    return best_c.astype(int).tolist()
