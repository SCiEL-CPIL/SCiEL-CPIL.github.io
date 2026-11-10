import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n = 24
    sqrt_n = np.sqrt(n)
    
    # Use a large FFT size for high-resolution fast scoring
    M = 1 << 16  # 65536 points
    
    def fast_score(c):
        """
        Compute max |P(z)| / sqrt(24) using FFT on a dense grid.
        c: list or array of 24 coefficients in {+1, -1}
        """
        c_arr = np.array(c, dtype=np.complex128)
        # Zero-pad to M
        c_pad = np.zeros(M, dtype=np.complex128)
        c_pad[:n] = c_arr
        # FFT
        P_vals = np.fft.fft(c_pad)
        max_val = np.max(np.abs(P_vals))
        return max_val / sqrt_n

    def precise_score(c):
        """
        Compute sup norm with dense sampling and golden-section refinement of top peaks.
        Returns the max |P(z)| / sqrt(24).
        """
        c_arr = np.array(c, dtype=np.complex128)
        
        # Get peaks from FFT
        c_pad = np.zeros(M, dtype=np.complex128)
        c_pad[:n] = c_arr
        P_vals = np.fft.fft(c_pad)
        abs_P = np.abs(P_vals)
        
        # Find top 10 peaks
        top_indices = np.argsort(abs_P)[-10:][::-1]
        
        best_val = float(np.max(abs_P))
        dtheta = 2 * np.pi / M

        # Precompute coefficients for single-point eval
        # P(t) = sum_{k=0}^{23} c_k e^{ikt}
        ks = np.arange(n)
        
        def f_single(t):
            # Vectorized evaluation for a single t
            # e^{ikt} = cos(kt) + i sin(kt)
            # P(t) = sum c_k cos(kt) + i sum c_k sin(kt)
            # |P(t)|^2 = (sum c_k cos(kt))^2 + (sum c_k sin(kt))^2
            # This might be slower than complex exp for small n, let's use complex exp
            exp_ik_t = np.exp(1j * ks * t)
            P_val = np.dot(c_arr, exp_ik_t)
            return np.abs(P_val)

        for idx in top_indices:
            center = idx * dtheta
            a = center - dtheta
            b = center + dtheta
            
            # Handle wrap-around
            if a < 0:
                a += 2 * np.pi
            if b > 2 * np.pi:
                b -= 2 * np.pi
            
            a_c = max(0, a)
            b_c = min(2 * np.pi, b)
            
            if a_c < b_c:
                def obj(t):
                    return -f_single(t)
                
                res = minimize_scalar(obj, bounds=(a_c, b_c), method='bounded', options={'xatol': 1e-12})
                if res.success:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        
        return best_val / sqrt_n

    # Local Search using fast_score
    def local_search(c, time_limit=0.5):
        current = list(c)
        current_score = fast_score(current)
        improved = True
        start = time.time()
        
        while improved:
            improved = False
            if time.time() - start > time_limit:
                break
            
            # Single flips
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                old = current[i]
                current[i] = -old
                new_score = fast_score(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old
            
            # Double flips
            if time.time() - start > time_limit:
                break
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() - start > time_limit:
                        break
                    vi, vj = current[i], current[j]
                    current[i] = -vi
                    current[j] = -vj
                    new_score = fast_score(current)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = vi
                        current[j] = vj
                        
        return current, current_score

    best_c = None
    best_score = float('inf')
    
    # 1. Start with Rudin-Shapiro and its variants
    def rudin_shapiro(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    rs = rudin_shapiro(n)
    candidates = [rs]
    
    # Add symmetries
    for base in candidates:
        candidates.append(list(reversed(base)))
        candidates.append([-c for c in base])
        candidates.append([-c for c in reversed(base)])
        for shift in range(n):
            rotated = base[shift:] + base[:shift]
            candidates.append(rotated)
            candidates.append(list(reversed(rotated)))
            candidates.append([-c for c in rotated])
            candidates.append([-c for c in reversed(rotated)])
            
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # 2. Random restarts
    random.seed(42)
    num_restarts = 500
    
    # Process initial candidates
    for c in unique_candidates:
        if time.time() > deadline - 10:
            break
        c_ls, s_ls = local_search(c, time_limit=0.2)
        if s_ls < best_score:
            best_score = s_ls
            best_c = c_ls

    # Process random restarts
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
        c = [random.choice([-1, 1]) for _ in range(n)]
        c_ls, s_ls = local_search(c, time_limit=0.1)
        if s_ls < best_score:
            best_score = s_ls
            best_c = c_ls

    # 3. Polish the best candidate with precise_score
    if best_c is not None and time.time() < deadline - 5:
        # Final local search using precise score
        current = list(best_c)
        current_precise = precise_score(current)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old = current[i]
                current[i] = -old
                new_precise = precise_score(current)
                if new_precise < current_precise - 1e-9:
                    current_precise = new_precise
                    improved = True
                else:
                    current[i] = old
        best_c = current
        best_score = current_precise

    return best_c
