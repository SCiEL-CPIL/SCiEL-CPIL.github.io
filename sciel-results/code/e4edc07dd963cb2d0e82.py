# EVOLVE-BLOCK-START
import math
import time
from math import isqrt
import numpy as np

def _compute_k_sigma_sieve(N, deadline):
    """
    Compute k*sigma(k) for k=1..N using a sieve.
    Returns array of length N+1 where result[k] = k*sigma(k) (capped at 10^18+1).
    Uses add.at for accumulation to avoid intermediate overflow issues.
    """
    CAP = 10**18 + 1
    # We accumulate sigma(k) first, then multiply by k
    # sigma(k) = sum of d|k of d
    # k*sigma(k) = sum of d|k of k*d
    # So we can directly accumulate k*d for each divisor d of k
    # For each d from 1 to N, for each multiple k of d, add k*d = d*(d,2d,3d,...)
    # But k*d can overflow int64. Let's use a different approach:
    # Accumulate sigma in int64 (sigma(k) <= k*H_k which for k=15M is ~15M*17 ≈ 2.5e8, fits easily)
    # Then multiply by k at the end.
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for d in range(1, N + 1):
        if time.time() > deadline:
            break
        # Add d to sigma[d], sigma[2d], sigma[3d], ...
        sigma[d::d] += d
    
    # Now compute k * sigma(k)
    k_vals = np.arange(1, N + 1, dtype=np.int64)
    # k*sigma(k) for k up to 15M: sigma(k) is at most ~k*ln(k)*ln(ln(k)) ≈ 15M * 17 * 2.8 ≈ 7e8
    # So k*sigma(k) ≈ 15M * 7e8 = 1.05e16, well within int64
    # But to be safe, check for values that would exceed 10^18
    ks = k_vals * sigma[1:N+1]
    # Cap values exceeding 10^18
    ks[ks > 10**18] = 10**18 + 1
    
    return ks

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s buffer before 150s limit
    
    # Try larger sieve first, fall back to smaller if needed
    L = 15_000_000
    
    try:
        ks = _compute_k_sigma_sieve(L, deadline)
    except Exception:
        L = 8_000_000
        try:
            ks = _compute_k_sigma_sieve(L, deadline)
        except Exception:
            L = 5_000_000
            try:
                ks = _compute_k_sigma_sieve(L, deadline)
            except Exception:
                return {'M': 333312, 'k': [336, 372, 434]}
    
    if time.time() > deadline:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Filter out values > 10^18
    mask = ks[1:L+1] <= 10**18
    M_vals = ks[1:L+1][mask]
    k_vals = np.arange(1, L + 1, dtype=np.int64)[mask]
    
    if len(M_vals) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Sort by M value to group collisions
    sorted_indices = np.argsort(M_vals, kind='stable')
    sorted_M = M_vals[sorted_indices]
    sorted_k = k_vals[sorted_indices]
    
    best_count = 0
    best_M = 0
    best_ks = []
    
    n = len(sorted_M)
    i = 0
    while i < n:
        if time.time() > deadline:
            break
        j = i + 1
        while j < n and sorted_M[j] == sorted_M[i]:
            j += 1
            
        group_size = j - i
        if group_size > best_count:
            M_val = int(sorted_M[i])
            group_k = sorted_k[i:j]
            
            sqrt_M = isqrt(M_val)
            valid_ks = group_k[group_k <= sqrt_M]
            
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M_val
                best_ks = valid_ks.tolist()
                
        i = j
    
    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    best_ks = [int(k) for k in best_ks]
    
    return {'M': int(best_M), 'k': best_ks}
# EVOLVE-BLOCK-END
