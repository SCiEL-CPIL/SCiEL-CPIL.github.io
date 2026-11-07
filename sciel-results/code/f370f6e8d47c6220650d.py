import time
import math
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    def compute_sigma_numpy(L):
        """Compute sigma(n) for all n from 1 to L using NumPy vectorized sieve."""
        sigma = np.zeros(L + 1, dtype=np.int64)
        for i in range(1, L + 1):
            sigma[i::i] += i
        return sigma

    L = 10_000_000
    if time.time() - start_time > deadline - 30:
        # Fallback for very small limits
        best_a, best_b = 122, 80
        return [best_a, best_b]

    sigma = compute_sigma_numpy(L)

    # Start with known good pair: sigma(122)=222, sigma(80)=186... wait let me verify
    # Actually from cell findings: (122, 80) has error 1/40
    # |122/80 - 3/2| = |122/80 - 120/80| = 2/80 = 1/40
    # So sigma(122) == sigma(80)
    best_error = Fraction(1, 40)
    best_a, best_b = 122, 80

    # For each b, check a near 3b/2
    # We want |a/b - 3/2| = |2a - 3b| / (2b)
    # So we want 2a ≈ 3b, i.e., a ≈ 3b/2
    
    # Process in chunks to keep memory reasonable
    chunk_size = 1_000_000
    b_start = 1
    best_num = best_error.numerator
    best_den = best_error.denominator

    # Phase 1: narrow search (offsets -2 to +2)
    offsets_phase1 = [-2, -1, 0, 1, 2]
    
    while b_start <= L:
        b_end = min(b_start + chunk_size - 1, L)
        if time.time() - start_time > deadline - 15:
            break
        
        bs = np.arange(b_start, b_end + 1, dtype=np.int64)
        three_b = 3 * bs
        base_a = three_b // 2
        
        for offset in offsets_phase1:
            if time.time() - start_time > deadline - 10:
                break
            as_ = base_a + offset
            valid = (as_ >= 1) & (as_ <= L) & (as_ != bs)
            if not np.any(valid):
                continue
            as_valid = as_[valid]
            bs_valid = bs[valid]
            sig_a = sigma[as_valid]
            sig_b = sigma[bs_valid]
            match_mask = sig_a == sig_b
            if not np.any(match_mask):
                continue
            as_match = as_valid[match_mask]
            bs_match = bs_valid[match_mask]
            
            # Vectorized error computation
            num = np.abs(2 * as_match - 3 * bs_match)
            den = 2 * bs_match
            # Exclude exact matches (num == 0)
            nonzero = num > 0
            if not np.any(nonzero):
                continue
            num = num[nonzero]
            den = den[nonzero]
            
            # Compare with best: num/den < best_num/best_den => num*best_den < best_num*den
            # Use 64-bit: num, den, best_num, best_den are all <= 10^7 scale
            # num <= 2*10^7, best_den <= 2*10^7, product <= 4*10^14 < 2^63
            comparison = num * best_den
            threshold = best_num * den
            improved = comparison < threshold
            if np.any(improved):
                # Find the best among improved
                # We want to minimize num/den, which is equivalent to minimizing comparison/den
                # Just find index with minimum ratio
                ratios = comparison[improved].astype(np.float64) / den[improved].astype(np.float64)
                idx_in_improved = np.argmin(ratios)
                idxs_improved = np.where(improved)[0]
                best_idx = idxs_improved[idx_in_improved]
                
                new_num = int(num[best_idx])
                new_den = int(den[best_idx])
                a_val = int(as_match[best_idx])
                b_val = int(bs_match[best_idx])
                
                best_error = Fraction(new_num, new_den)
                best_a, best_b = a_val, b_val
                best_num = best_error.numerator
                best_den = best_error.denominator
        
        b_start = b_end + 1

    # Phase 2: if time remains, do a wider search with larger offsets
    remaining = deadline - 10 - (time.time() - start_time)
    if remaining > 20:
        offsets_phase2 = [-5, -4, -3, 3, 4, 5]
        b_start = 1
        while b_start <= L:
            b_end = min(b_start + chunk_size - 1, L)
            if time.time() - start_time > deadline - 8:
                break
            
            bs = np.arange(b_start, b_end + 1, dtype=np.int64)
            three_b = 3 * bs
            base_a = three_b // 2
            
            for offset in offsets_phase2:
                if time.time() - start_time > deadline - 5:
                    break
                as_ = base_a + offset
                valid = (as_ >= 1) & (as_ <= L) & (as_ != bs)
                if not np.any(valid):
                    continue
                as_valid = as_[valid]
                bs_valid = bs[valid]
                sig_a = sigma[as_valid]
                sig_b = sigma[bs_valid]
                match_mask = sig_a == sig_b
                if not np.any(match_mask):
                    continue
                as_match = as_valid[match_mask]
                bs_match = bs_valid[match_mask]
                
                num = np.abs(2 * as_match - 3 * bs_match)
                den = 2 * bs_match
                nonzero = num > 0
                if not np.any(nonzero):
                    continue
                num = num[nonzero]
                den = den[nonzero]
                
                comparison = num * best_den
                threshold = best_num * den
                improved = comparison < threshold
                if np.any(improved):
                    ratios = comparison[improved].astype(np.float64) / den[improved].astype(np.float64)
                    idx_in_improved = np.argmin(ratios)
                    idxs_improved = np.where(improved)[0]
                    best_idx = idxs_improved[idx_in_improved]
                    
                    new_num = int(num[best_idx])
                    new_den = int(den[best_idx])
                    a_val = int(as_match[best_idx])
                    b_val = int(bs_match[best_idx])
                    
                    best_error = Fraction(new_num, new_den)
                    best_a, best_b = a_val, b_val
                    best_num = best_error.numerator
                    best_den = best_error.denominator
            
            b_start = b_end + 1

    # Phase 3: final polish - very tight search around the best ratio found
    remaining = deadline - 5 - (time.time() - start_time)
    if remaining > 5:
        # Do a focused search near the current best a/b ratio
        # b in a range, a = round(b * best_a / best_b) with small offsets
        ratio_num = best_a
        ratio_den = best_b
        # Search b in [best_b - 1000, best_b + 1000] and nearby
        b_lo = max(1, best_b - 2000)
        b_hi = min(L, best_b + 2000)
        
        for b_val in range(b_lo, b_hi + 1):
            if time.time() - start_time > deadline - 3:
                break
            # a should be close to b * ratio_num / ratio_den
            target_a = (b_val * ratio_num) // ratio_den
            for offset in range(-3, 4):
                a_val = target_a + offset
                if a_val < 1 or a_val > L or a_val == b_val:
                    continue
                if sigma[a_val] != sigma[b_val]:
                    continue
                num = abs(2 * a_val - 3 * b_val)
                if num == 0:
                    continue
                den = 2 * b_val
                if num * best_den < best_num * den:
                    best_error = Fraction(num, den)
                    best_a, best_b = a_val, b_val
                    best_num = best_error.numerator
                    best_den = best_error.denominator

    return [int(best_a), int(best_b)]
