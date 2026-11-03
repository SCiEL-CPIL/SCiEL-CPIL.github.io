import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def score_set(U):
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        n = len(U_set)
        if n == 0:
            return 0.0

        # Compute sumset and difference set using NumPy for speed
        U_arr = np.array(list(U_set), dtype=np.int64)
        
        # Sumset: U + U
        # Difference set: U - U
        # For small n (< 50), O(n^2) is fine. 
        # Using broadcasting creates n x n arrays.
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0

        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def build_geo_set(base, scale, m):
        """
        Builds U = { floor(scale * base^k) for k=0..m }
        Ensures 0 is in the set.
        """
        if scale == 0:
            return [0]
        
        # Use logs to avoid overflow if base is large and m is large
        # base^k = exp(k * log(base))
        try:
            log_base = math.log(base)
            # k = 0 to m
            ks = np.arange(m + 1)
            # values = scale * exp(ks * log_base)
            # To avoid precision issues with very large numbers, 
            # we can compute in log space then exp, or just use power if small.
            # Given m is small (< 50) and base is not huge, direct power is safer for integer floor.
            # But numpy power on floats is fast.
            
            # We want distinct integers.
            # If base is close to 1, distinctness might fail.
            # We enforce base > 1.
            
            vals = scale * (base ** ks)
            # Floor to integers
            int_vals = np.floor(vals).astype(np.int64)
            
            # Ensure 0 is included
            # The set must contain 0.
            # If scale * base^0 = scale. If scale >= 1, 0 is not naturally in {scale, ...} unless scale=0.
            # The problem requires U to contain 0.
            # So we explicitly add 0.
            # U = {0} union {floor(scale * base^k) ...}
            
            unique_vals = np.unique(int_vals)
            # Add 0
            if 0 not in unique_vals:
                unique_vals = np.unique(np.concatenate(([0], unique_vals)))
            
            # Filter out negative values (though scale > 0, base > 1 ensures positive)
            unique_vals = unique_vals[unique_vals >= 0]
            
            return list(unique_vals)
        except Exception:
            return [0]

    best_set = [0]
    best_score = 0.0
    
    # 1. Try a range of bases and scales
    # The score depends on the ratio |U-U|/|U+U| and max U.
    # For a geometric progression, |U-U| and |U+U| grow roughly linearly with n if the gaps are large enough.
    # max U is roughly scale * base^m.
    
    # We want to maximize 1 + log(ratio) / log(2*max_U + 1)
    # If ratio is constant (e.g., ~3 for large n), we want to minimize max_U.
    # Minimizing max_U means small scale and small base.
    # But we need enough elements to get a good ratio?
    # Actually, for any set, |U-U| >= |U+U| is not always true, but for APs it is equal.
    # For random sets, |U-U| ~ n^2, |U+U| ~ n^2. Ratio ~ 1.
    # For structured sets like {0, 1, 3, 9, ...} (base 3), the sums are unique?
    # If base > 2, the sumset is the set of all distinct sums, which has size n(n+1)/2.
    # The diffset has size n(n+1)/2 - 1 (excluding 0? No, 0 is in diffset).
    # Actually, if base > 2, all sums a_i + a_j with i<=j are distinct?
    # Let's check: 1*3^0 + 1*3^1 = 1+3=4. 1*3^2 = 9. No collision.
    # In general, if base > 2, the representation in base 'base' with digits 0 or 1 (for sum of two distinct elements) is unique.
    # So |U+U| = n(n+1)/2.
    # |U-U|: a_i - a_j. Also unique representations?
    # Yes, if base > 2, the difference is also unique for each pair (i,j) with i!=j.
    # So |U-U| = n(n-1) + 1 (including 0).
    # Ratio = (n^2 - n + 1) / (n^2/2 + n/2) ~ 2 for large n.
    # Score = 1 + log(2) / log(2 * scale * base^m + 1).
    # To maximize this, we need to minimize the denominator log(2 * scale * base^m + 1).
    # This implies we want small scale and small base^m.
    # But we need n to be large enough for the ratio to approach 2.
    # If n is small, the ratio might be different.
    # Let's just search.
    
    # Search space:
    # m: number of elements (0 to 50)
    # base: 1.1 to 5.0
    # scale: 1 to 100
    
    # We will do a coarse grid search first, then refine the best region.
    
    # Phase 1: Coarse Search
    m_values = list(range(2, 21)) # 2 to 20 elements
    base_values = np.linspace(1.2, 4.0, 20)
    scale_values = [1, 2, 5, 10, 20, 50]
    
    best_combo = None
    
    for m in m_values:
        if time.time() > deadline - 10:
            break
        for b in base_values:
            if time.time() > deadline - 10:
                break
            for s in scale_values:
                U = build_geo_set(b, s, m)
                # Check if distinctness is maintained (base > 2 usually ensures this for sums/diffs)
                # If base is small, collisions might happen, reducing the ratio.
                sc = score_set(U)
                if sc > best_score:
                    best_score = sc
                    best_set = U
                    best_combo = (b, s, m)

    # Phase 2: Local Refinement (Ternary Search on base and scale)
    if best_combo:
        b0, s0, m0 = best_combo
        
        # Refine base around b0
        # We can't do ternary search on scale easily if it's integer, but we can try nearby integers.
        # Base is continuous.
        
        # Try to optimize base for the best m and s found
        # We will iterate over m and s near the best found, and do ternary search on base.
        
        m_range = range(max(2, m0 - 5), min(50, m0 + 10))
        s_range = range(max(1, s0 - 2), s0 + 5)
        
        for m in m_range:
            if time.time() > deadline - 5:
                break
            for s in s_range:
                if time.time() > deadline - 5:
                    break
                
                # Ternary search for base in [1.01, 10.0]
                # Actually, base > 2 is ideal for unique sums.
                # If base < 2, sums might collide.
                # Let's search [1.01, 5.0]
                lo = 1.01
                hi = 5.0
                
                # Ternary search takes O(log((hi-lo)/epsilon)) iterations
                # Each eval is O(n^2). n is small.
                # We can do ~50 iterations of ternary search quickly.
                
                for _ in range(60):
                    if time.time() > deadline - 2:
                        break
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0
                    
                    U1 = build_geo_set(m1, s, m)
                    U2 = build_geo_set(m2, s, m)
                    
                    sc1 = score_set(U1)
                    sc2 = score_set(U2)
                    
                    if sc1 < sc2:
                        lo = m1
                    else:
                        hi = m2
                    
                    if hi - lo < 1e-4:
                        break
                
                mid = (lo + hi) / 2.0
                U_mid = build_geo_set(mid, s, m)
                sc_mid = score_set(U_mid)
                
                if sc_mid > best_score:
                    best_score = sc_mid
                    best_set = U_mid

    # Phase 3: Randomized Local Search on the best set found
    # Try to tweak the set elements slightly
    # Since the set is derived from a formula, we can try to "fix" collisions or improve ratio by manual tweaks
    # But the structural change was the geometric model. Let's stick to that.
    
    # One more check: ensure 0 is in the set and it's a list of ints
    if 0 not in best_set:
        best_set.append(0)
        
    # Final safety check
    if not best_set:
        best_set = [0]
        
    return sorted(best_set)

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Score: {score_set(res):.4f}, Size: {len(res)}")
    print(res[:10])
