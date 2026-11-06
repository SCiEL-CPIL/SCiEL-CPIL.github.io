import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 145.0  # Safety margin

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly(n):
    """Generate Rudin-Shapiro coefficients of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    
    # Recursive generation
    # R_{k+1} = R_k S_k
    # S_{k+1} = R_k (-S_k)
    # Start with R_1=[1], S_1=[1]
    R = [1]
    S = [1]
    while len(R) < n:
        new_R = R + S
        new_S = R + [-s for s in S]
        R = new_R
        S = new_S
    return R[:n]

def _calc_peak_fast(coeffs, M=512):
    """
    Fast peak calculation using dense sampling without refinement.
    Used for local search steps.
    """
    c = np.array(coeffs, dtype=complex)
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    return math.sqrt(np.max(mag2))

def _calc_peak(coeffs, M=2048, refine_top=5, refine_iters=15):
    """
    Accurate peak calculation with golden-section refinement.
    Used for final validation and initial seed evaluation.
    """
    c = np.array(coeffs, dtype=complex)
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        for idx in top_indices:
            lo = theta[idx] - 2.0 * math.pi / M
            hi = theta[idx] + 2.0 * math.pi / M
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

def _local_search(seed, M_fast=512, max_time=120.0):
    """
    Steepest descent local search.
    """
    cur = list(seed)
    cur_peak = _calc_peak_fast(cur, M_fast)
    best = list(cur)
    best_peak_acc = _calc_peak(cur) # Get accurate peak for the seed
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Evaluate all single flips
        best_flip_idx = -1
        best_flip_peak = cur_peak
        
        for j in range(_N):
            cur[j] = -cur[j]
            p = _calc_peak_fast(cur, M_fast)
            if p < best_flip_peak:
                best_flip_peak = p
                best_flip_idx = j
            cur[j] = -cur[j] # Revert
        
        if best_flip_idx != -1:
            # Apply the best flip
            cur[best_flip_idx] = -cur[best_flip_idx]
            cur_peak = best_flip_peak
            
            # Check if this is a new global best (using fast peak as proxy, 
            # but we should verify periodically or at the end)
            # To save time, we only calculate accurate peak every few steps or at the end?
            # Or we can check if fast peak is significantly better.
            # Let's just track the best fast peak and do accurate check at the end of the loop or periodically.
            # For now, let's say if fast peak is better than best_peak_acc, we update best.
            # But fast peak is an underestimate of the true peak? No, it's a sample.
            # The true peak is >= max sample.
            # So if fast peak is low, true peak is likely low.
            # We should update best if cur_peak < best_peak_acc.
            # But best_peak_acc is accurate. cur_peak is fast.
            # Let's do a cheap accurate check if cur_peak is significantly lower.
            if cur_peak < best_peak_acc - 0.01:
                acc_peak = _calc_peak(cur)
                if acc_peak < best_peak_acc:
                    best = list(cur)
                    best_peak_acc = acc_peak
        else:
            # No single flip improves. Try double flips?
            # Or break?
            # Let's try double flips for a few iterations if stuck.
            double_improved = False
            # Limit double flip search to save time
            for j in range(_N):
                for k in range(j + 1, _N):
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    p = _calc_peak_fast(cur, M_fast)
                    if p < cur_peak:
                        cur_peak = p
                        double_improved = True
                        # Apply
                        # Note: cur[j] and cur[k] are currently flipped in cur
                        # We need to keep them flipped
                        # But we need to update best if better
                        if cur_peak < best_peak_acc - 0.01:
                            acc_peak = _calc_peak(cur)
                            if acc_peak < best_peak_acc:
                                best = list(cur)
                                best_peak_acc = acc_peak
                        break
                    else:
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
                if double_improved:
                    break
            
            if not double_improved:
                break

    # Final accurate check on the best found
    final_peak = _calc_peak(best)
    return best, final_peak

def construct():
    global _T0
    _T0 = time.time()
    
    # Primary Seed: Rudin-Shapiro
    seed_rs = _rs_poly(_N)
    
    best_coeffs = seed_rs
    best_peak = _calc_peak(best_coeffs)
    
    # Run local search on RS seed
    # Allocate most of the time to this
    res_coeffs, res_peak = _local_search(seed_rs, M_fast=512, max_time=100.0)
    if res_peak < best_peak:
        best_coeffs = res_coeffs
        best_peak = res_peak
    
    # If time remains, try a few random restarts or perturbations
    if _time_left() > 20.0:
        rng = np.random.default_rng(42)
        # Try perturbing the RS seed
        for _ in range(5):
            if _time_left() < 10.0:
                break
            seed_alt = list(seed_rs)
            # Flip 2 random coefficients
            idx1 = int(rng.integers(0, _N))
            idx2 = int(rng.integers(0, _N))
            if idx1 == idx2:
                idx2 = (idx2 + 1) % _N
            seed_alt[idx1] = -seed_alt[idx1]
            seed_alt[idx2] = -seed_alt[idx2]
            
            r_coeffs, r_peak = _local_search(seed_alt, M_fast=256, max_time=min(15.0, _time_left() - 5.0))
            if r_peak < best_peak:
                best_coeffs = r_coeffs
                best_peak = r_peak
        
        # Try a purely random balanced seed
        if _time_left() > 10.0:
            # Generate random +/- 1 with sum close to 0
            # 12 ones, 12 minus ones
            arr = [1]*12 + [-1]*12
            rng.shuffle(arr)
            r_coeffs, r_peak = _local_search(arr, M_fast=256, max_time=min(10.0, _time_left() - 5.0))
            if r_peak < best_peak:
                best_coeffs = r_coeffs
                best_peak = r_peak

    # Final validation
    best_coeffs = [1 if c >= 0 else -1 for c in best_coeffs]
    
    return best_coeffs

if __name__ == "__main__":
    # Self-test
    coeffs = construct()
    print(coeffs)
    print(len(coeffs))
    print(_calc_peak(coeffs, M=4096))
