import time
import sys
from math import gcd
from fractions import Fraction

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin within 90s

    N = 256
    vertices = list(range(2, N + 1))
    
    # The key insight: for large numbers (say > 64), most pairs have lcm > 256
    # So we can be greedy about large numbers
    # For small numbers, we need careful selection
    
    # Precompute conflict matrix: conflict[i][j] = True if lcm(vertices[i], vertices[j]) <= N
    # We'll use a different approach: work with the full set but use branch and bound
    
    # Strategy: 
    # 1. Identify "easy" large numbers that rarely conflict
    # 2. Enumerate subsets of "hard" small numbers (those with many conflicts)
    # 3. For each small subset, greedily add large numbers
    
    # Let's split: small = 2..48, large = 49..256
    # Small numbers have more conflicts, so enumerate their subsets
    # Large numbers are mostly compatible with each other
    
    SPLIT = 48
    small = list(range(2, SPLIT + 1))
    large = list(range(SPLIT + 1, N + 1))
    n_small = len(small)
    n_large = len(large)
    
    # Precompute conflicts
    # For small-small
    conflict_ss = [[False] * n_small for _ in range(n_small)]
    for i in range(n_small):
        a = small[i]
        for j in range(i + 1, n_small):
            b = small[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict_ss[i][j] = True
                conflict_ss[j][i] = True
    
    # For small-large
    conflict_sl = [[False] * n_large for _ in range(n_small)]
    for i in range(n_small):
        a = small[i]
        for j in range(n_large):
            b = large[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict_sl[i][j] = True
    
    # For large-large
    conflict_ll = [[False] * n_large for _ in range(n_large)]
    for i in range(n_large):
        a = large[i]
        for j in range(i + 1, n_large):
            b = large[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict_ll[i][j] = True
                conflict_ll[j][i] = True
    
    # Count conflicts for large numbers
    ll_conflict_count = [sum(conflict_ll[j][k] for k in range(n_large) if k != j) for j in range(n_large)]
    
    # Sort large by: fewer conflicts first, then by higher reciprocal weight
    large_order = sorted(range(n_large), key=lambda j: (ll_conflict_count[j], -1.0/large[j]))
    
    # Precompute weights as floats for speed, use Fraction only for final comparison
    weights = [1.0 / v for v in vertices]
    # Small indices 0..n_small-1, large indices n_small..n_small+n_large-1
    weight_small = [weights[i] for i in range(n_small)]
    weight_large = [weights[n_small + i] for i in range(n_large)]
    
    best_val = 0.0
    best_result = [2]
    
    # Enumerate valid small subsets
    # n_small = 47, so 2^47 is too large
    # We need a smarter approach
    
    # Alternative: use greedy with backtracking on small numbers
    # Or: recognize that for small numbers, the constraint graph is dense
    # so independent sets are small
    
    # Let's try: greedy on all numbers, ordered by reciprocal weight descending
    # Then local search
    
    # Actually, let me think about this differently.
    # The reciprocal sum is maximized by including small numbers.
    # But small numbers conflict with each other.
    # 
    # Key observation: if a | b (a divides b), then lcm(a,b) = b.
    # So if b <= 256, they conflict.
    # This means we can't include both a and b if a|b and b<=256.
    #
    # For numbers > 128, they can't divide each other (except trivially),
    # and lcm of two numbers > 128 is at least min(a,b) * max(a,b) / gcd(a,b)
    # If gcd is 1, lcm = a*b > 128*129 > 256
    # If gcd > 1, we need to check
    #
    # Let me try a different strategy:
    # 1. Start with all numbers 2..256
    # 2. Greedily remove conflicts, prioritizing keeping smaller numbers (higher weight)
    # 3. Local search to improve
    
    # Actually, let me use the split approach but with a better enumeration
    # For small numbers 2..24 (23 numbers), enumerate all independent sets
    # For each, greedily fill large numbers
    
    SPLIT2 = 24
    small2 = list(range(2, SPLIT2 + 1))
    large2 = list(range(SPLIT2 + 1, N + 1))
    n_s2 = len(small2)  # 23
    n_l2 = len(large2)
    
    # Precompute conflicts for small2
    conflict_s2 = [0] * n_s2
    for i in range(n_s2):
        a = small2[i]
        for j in range(i + 1, n_s2):
            b = small2[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict_s2[i] |= (1 << j)
                conflict_s2[j] |= (1 << i)
    
    # Conflicts between small2 and large2
    conflict_s2_l2 = [0] * n_s2
    for i in range(n_s2):
        a = small2[i]
        mask = 0
        for j in range(n_l2):
            b = large2[j]
            g = gcd(a, b)
            if a * b // g <= N:
                mask |= (1 << j)
        conflict_s2_l2[i] = mask
    
    # Conflicts within large2
    conflict_l2 = [0] * n_l2
    for i in range(n_l2):
        a = large2[i]
        for j in range(i + 1, n_l2):
            b = large2[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict_l2[i] |= (1 << j)
                conflict_l2[j] |= (1 << i)
    
    # Weight for small2
    w_s2 = [1.0 / v for v in small2]
    # Weight for large2
    w_l2 = [1.0 / v for v in large2]
    
    # Sort large2 by weight descending (smaller value = higher weight)
    large2_order = list(range(n_l2))  # already sorted by value
    
    # Enumerate all valid small2 subsets
    for sm in range(1 << n_s2):
        if time.time() > deadline - 10:
            break
        
        # Check validity of small2 subset
        valid = True
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if conflict_s2[i] & sm:
                valid = False
                break
            m ^= lsb
        
        if not valid:
            continue
        
        # Compute small2 weight
        val_s2 = 0.0
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            val_s2 += w_s2[i]
            m ^= lsb
        
        # Forbidden large2 vertices
        forbidden = 0
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            forbidden |= conflict_s2_l2[i]
            m ^= lsb
        
        # Greedy select large2
        selected = 0
        val_l2 = 0.0
        for j in large2_order:
            if forbidden & (1 << j):
                continue
            if conflict_l2[j] & selected:
                continue
            selected |= (1 << j)
            val_l2 += w_l2[j]
        
        # Local search on large2
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try adding
            for j in range(n_l2):
                if selected & (1 << j):
                    continue
                if forbidden & (1 << j):
                    continue
                if conflict_l2[j] & selected:
                    continue
                selected |= (1 << j)
                val_l2 += w_l2[j]
                improved = True
                break
            
            # Try 1-for-1 swap
            if not improved:
                for j_out in range(n_l2):
                    if not (selected & (1 << j_out)):
                        continue
                    new_sel = selected & ~(1 << j_out)
                    for j_in in range(n_l2):
                        if new_sel & (1 << j_in):
                            continue
                        if forbidden & (1 << j_in):
                            continue
                        if conflict_l2[j_in] & new_sel:
                            continue
                        if w_l2[j_in] > w_l2[j_out]:
                            selected = new_sel | (1 << j_in)
                            val_l2 += w_l2[j_in] - w_l2[j_out]
                            improved = True
                            break
                    if improved:
                        break
            
            # Try 1-for-2 swap
            if not improved:
                for j_out in range(n_l2):
                    if not (selected & (1 << j_out)):
                        continue
                    new_sel = selected & ~(1 << j_out)
                    # Try to add two vertices
                    added = 0
                    for j_in1 in range(n_l2):
                        if new_sel & (1 << j_in1):
                            continue
                        if forbidden & (1 << j_in1):
                            continue
                        if conflict_l2[j_in1] & new_sel:
                            continue
                        trial_sel = new_sel | (1 << j_in1)
                        for j_in2 in range(n_l2):
                            if trial_sel & (1 << j_in2):
                                continue
                            if forbidden & (1 << j_in2):
                                continue
                            if conflict_l2[j_in2] & trial_sel:
                                continue
                            delta = w_l2[j_in1] + w_l2[j_in2] - w_l2[j_out]
                            if delta > 0:
                                selected = trial_sel | (1 << j_in2)
                                val_l2 += delta
                                added = 2
                                break
                        if added == 2:
                            break
                    if added == 2:
                        improved = True
                        break
        
        total_val = val_s2 + val_l2
        if total_val > best_val:
            best_val = total_val
            # Build result
            result = []
            m = sm
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                result.append(small2[i])
                m ^= lsb
            m = selected
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                result.append(large2[j])
                m ^= lsb
            best_result = result
    
    # Final polish: try to add any remaining numbers that fit
    result_set = set(best_result)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for v in range(2, N + 1):
            if v in result_set:
                continue
            # Check if v can be added
            can_add = True
            for u in result_set:
                g = gcd(v, u)
                if v * u // g <= N:
                    can_add = False
                    break
            if can_add:
                result_set.add(v)
                best_val += 1.0 / v
                improved = True
    
    return list(result_set)
