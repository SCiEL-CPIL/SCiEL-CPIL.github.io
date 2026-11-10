"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 12
MAX_M = 1000000
MAX_L = 144

def construct():
    deadline = time.time() + 80  # 80s safety margin for 90s limit
    best_m = 12
    best_f = 0
    
    # Precompute: for offset d in 1..MAX_L, which indices i divide (m+d)?
    # This depends on m, so we compute on the fly but efficiently.
    
    # For Hall's theorem: for subset S (bitmask), count numbers in (m, m+L] 
    # divisible by at least one i in S.
    
    def compute_f(m, max_l=MAX_L):
        """Compute f(N, m): least L such that (m, m+L] has distinct b_i with i|b_i."""
        # For each L from 1 to max_l, check if perfect matching exists via Hall
        for L in range(1, max_l + 1):
            # For each number x = m+1, ..., m+L, compute bitmask of i that divide x
            # mask_x: bit i-1 set if i divides x
            masks = []
            for d in range(1, L + 1):
                x = m + d
                mask = 0
                for i in range(1, N + 1):
                    if x % i == 0:
                        mask |= (1 << (i - 1))
                masks.append(mask)
            
            # Hall's condition: for every non-empty subset S of {1..N},
            # |{x in (m,m+L] : exists i in S with i|x}| >= |S|
            # Only need to check subsets up to size N (all 4096)
            # But we can short-circuit: if a subset of size k violates, 
            # we fail. Check all subsets.
            
            # For efficiency, for each subset mask, count how many x have (masks[x] & mask) != 0
            # Precompute for each possible mask value (0 to 2^12-1) how many x have that exact mask
            from collections import Counter
            mask_counts = Counter(masks)
            
            # For a given subset S (bitmask s), the numbers covered are those x 
            # where masks[x] & s != 0
            # Count = L - (number of x where masks[x] & s == 0)
            # = L - sum of mask_counts[m] for m where m & s == 0
            
            # Precompute for each s: count of x with (masks[x] & s) == 0
            # This is: for each x, it contributes to all s where s is a subset of ~masks[x] (within N bits)
            # Alternative: just iterate over all 4096 subsets and count
            
            all_bits = (1 << N) - 1
            failed = False
            for s in range(1, 1 << N):
                # Count x where (masks[x] & s) == 0
                cnt_zero = 0
                for mx in masks:
                    if (mx & s) == 0:
                        cnt_zero += 1
                covered = L - cnt_zero
                # |S| = popcount(s)
                pc = bin(s).count('1')
                if covered < pc:
                    failed = True
                    break
            if not failed:
                return L
        return max_l  # should not reach here normally
    
    # Search strategy: sample m values, prioritize those likely to have large f
    # m=0 gives small f. We want m where multiples are sparse.
    # Try a range of m values with increasing step, then refine.
    
    # First pass: coarse search
    step = 10000
    m = 0
    while m <= MAX_M:
        if time.time() > deadline:
            break
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        m += step
    
    # Second pass: finer search around best region
    if time.time() < deadline:
        step2 = 1000
        m = max(0, best_m - 100000)
        end = min(MAX_M, best_m + 100000)
        while m <= end:
            if time.time() > deadline:
                break
            f_val = compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
            m += step2
    
    # Third pass: even finer
    if time.time() < deadline:
        step3 = 100
        m = max(0, best_m - 10000)
        end = min(MAX_M, best_m + 10000)
        while m <= end:
            if time.time() > deadline:
                break
            f_val = compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
            m += step3
    
    # Final polish: fine search around best
    if time.time() < deadline:
        m = max(0, best_m - 1000)
        end = min(MAX_M, best_m + 1000)
        while m <= end:
            if time.time() > deadline:
                break
            f_val = compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
            m += 1
    
    return int(best_m)

# EVOLVE-BLOCK-END
