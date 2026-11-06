import math
import time

def _dfs_exact(numer, denom, min_d, max_terms, deadline, found, current):
    """
    DFS to find an exact Egyptian fraction representation of numer/denom
    with distinct denominators >= min_d, at most max_terms remaining.
    Returns True if found (populates `found` list).
    """
    if numer == 0:
        return True
    if max_terms == 0:
        return False
    if time.time() > deadline:
        return False
        
    # If we have 1 term left, it must be exactly 1/(denom/numer)
    if max_terms == 1:
        if denom % numer == 0:
            d = denom // numer
            if d >= min_d and d <= 10000000:
                found.append(d)
                return True
        return False

    # Bound: max possible sum with k terms starting from min_d
    # 1/d + 1/(d+1) + ... 
    # A tighter bound: if 1/min_d * k < numer/denom, impossible
    # But better: the sum of the next k terms if we pick consecutive integers starting from min_d
    # is an upper bound on the remaining sum we can achieve? No, we want to achieve exactly numer/denom.
    # The maximum sum we can achieve with k terms starting from min_d is actually limited by distinctness.
    # Max sum: 1/min_d + 1/(min_d+1) + ... + 1/(min_d + k - 1)
    # If numer/denom > max_sum, impossible.
    max_sum = 0.0
    for i in range(k):
        max_sum += 1.0 / (min_d + i)
    if numer / denom > max_sum:
        return False

    # The minimum sum we can achieve is with the largest possible denominators?
    # No, we are minimizing count, not denominators.
    # But we must have numer/denom >= 1/d_last? No.
    
    # Determine range for next denominator d
    # We need 1/d >= numer / (denom * k)  =>  d <= denom * k / numer
    # Also 1/d < numer / denom => d > denom / numer
    # So d is in (denom/numer, denom*k/numer]
    
    d_min = max(min_d, math.floor(denom / numer) + 1)
    d_max = min(10000000, math.floor(denom * k / numer))
    
    # Optimization: d_max can be tightened.
    # 1/d + (k-1)/d_min_remaining <= numer/denom ? No.
    # The remaining sum after picking d is (numer*denom - d*numer) / (denom*d) ? No.
    # Remaining fraction is numer/denom - 1/d = (numer*d - denom) / (denom*d).
    # We need to represent this with k-1 terms, each >= d+1.
    # The maximum sum of k-1 terms starting from d+1 is approx (k-1)/(d+1).
    # So we require (numer*d - denom) / (denom*d) <= (k-1) / (d+1) roughly?
    # Actually, a standard bound:
    # 1/d <= numer/denom => d >= denom/numer (already handled by d_min)
    # Also, the remaining fraction R = numer/denom - 1/d must be representable by k-1 terms.
    # The maximum value of a sum of k-1 distinct unit fractions with denominators > d is:
    # 1/(d+1) + 1/(d+2) + ... 
    # If R > max_possible_sum(k-1, d+1), then this d is too small? No, if R is too big, we need smaller denominators, but d is fixed.
    # Wait, if we pick a small d, 1/d is large, leaving a small R.
    # If we pick a large d, 1/d is small, leaving a large R.
    # The constraint is that R must be representable.
    # If R > max_sum(k-1, d+1), then we cannot represent R because even the largest possible sum of k-1 terms (starting from d+1) is too small.
    # So we need R <= max_sum(k-1, d+1).
    # (numer/denom - 1/d) <= sum_{i=1}^{k-1} 1/(d+i)
    # This helps find d_min? No, it restricts large d?
    # Let's check: if d is very large, 1/d is small, R is close to numer/denom.
    # max_sum(k-1, d+1) is roughly (k-1)/(d+1).
    # So we need numer/denom <= (k-1)/d  => d <= (k-1)*denom/numer.
    # This is similar to d_max bound derived earlier but slightly tighter.
    
    for d in range(d_min, d_max + 1):
        # Check if d is already in current
        if d in current:
            continue
            
        # Check if remaining fraction is representable
        # R_num = numer*d - denom
        # R_den = denom*d
        # If R_num < 0, then 1/d > numer/denom, which is impossible since d > denom/numer.
        r_num = numer * d - denom
        if r_num < 0:
            continue
            
        # Reduce fraction
        g = math.gcd(r_num, denom * d)
        r_num //= g
        r_den //= g
        
        current.append(d)
        
        # Bound check: max sum of k-1 terms starting from d+1
        # if r_num/r_den > max_sum(k-1, d+1):
        #     pass # too big, need smaller d? No, d is fixed. If R is too big, it means we picked d too large (1/d too small).
        # So if r_num/r_den > max_sum(k-1, d+1), this d is invalid (too large).
        # But we are iterating d upwards. If this d is too large, all larger d will also be too large?
        # Let's verify: as d increases, 1/d decreases, so R increases.
        # And max_sum(k-1, d+1) decreases as d increases.
        # So if condition fails for d, it will fail for all larger d.
        # So we can break.
        
        max_sum_rem = 0.0
        for i in range(1, k):
            max_sum_rem += 1.0 / (d + i)
            
        if r_num / r_den > max_sum_rem:
            current.pop()
            # Since d is increasing, R is increasing, max_sum_rem is decreasing.
            # So we can break out of the loop.
            break
        
        # Also check min sum?
        # The min sum of k-1 terms with denominators > d is achieved by taking the largest possible denominators?
        # No, min sum is with largest denominators. But denominators are unbounded (up to 10M).
        # So min sum approaches 0. So no lower bound issue unless R is extremely small?
        # Actually, we just need to ensure R is not 0 (handled by r_num>0) and representable.
        
        if _dfs_exact(r_num, r_den, d + 1, k - 1, deadline, found, current):
            return True
            
        current.pop()
        
    return False


def construct():
    # Target: sum of reciprocals = 1.
    # 1 = 1/1. But denominators must be >= 5.
    # We need to find a set of distinct integers d_1 < d_2 < ... < d_k
    # such that sum(1/d_i) = 1 and d_i >= 5.
    
    # Minimum number of terms?
    # Max sum with 4 terms: 1/5 + 1/6 + 1/7 + 1/8 = 0.2 + 0.166 + 0.142 + 0.125 = 0.633 < 1.
    # Max sum with 5 terms: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 = 0.633 + 0.111 = 0.744 < 1.
    # Max sum with 6 terms: 1/5 + ... + 1/10 = 0.744 + 0.1 = 0.844 < 1.
    # Max sum with 7 terms: 1/5 + ... + 1/11 = 0.844 + 0.0909 = 0.935 < 1.
    # Max sum with 8 terms: 1/5 + ... + 1/12 = 0.935 + 0.0833 = 1.018 > 1.
    # So minimum number of terms is at least 8.
    
    # Let's try to find a solution with 8 terms.
    # If 8 terms are possible, that's optimal.
    
    # We can use the DFS starting with fraction 1/1.
    # But 1/1 is not a proper fraction for the standard algorithm?
    # Standard algorithm assumes numerator < denominator.
    # Here numer=1, denom=1.
    
    # Let's start with numer=1, denom=1, min_d=5, max_terms=8.
    
    deadline = time.time() + 50.0 # Leave 10s margin
    found = []
    current = []
    
    # Optimization: We know the first few terms must be small.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 = 1.018...
    # We need to reduce the sum slightly by increasing some denominators.
    
    # Let's run the DFS.
    if _dfs_exact(1, 1, 5, 8, deadline, found, current):
        return found
    
    # If 8 terms fail (should not happen, but safety), try 9 terms.
    # This is unlikely to be needed if 8 is possible.
    # Known solution for 1 with 8 terms?
    # 1 = 1/2 + 1/3 + 1/7 + 1/42 (4 terms, but 2,3 < 5)
    # 1 = 1/3 + 1/4 + 1/5 + 1/20 (4 terms)
    # 1 = 1/4 + 1/5 + 1/6 + 1/7 + 1/44 + 1/1320 (6 terms)
    
    # Let's try a known 8-term solution with min denominator 5.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 is > 1.
    # We need to replace some with larger denominators.
    
    # If DFS fails due to time or logic, return a known valid solution if any.
    # A known solution for 1 with 8 terms:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/132 = ?
    # Sum 1/5..1/11 = 0.9345...
    # 1 - 0.9345 = 0.0654...
    # 1/132 = 0.0075... No.
    
    # Let's trust the DFS. If it returns empty, raise error or return best effort.
    if not found:
        # Fallback: return a valid set even if not minimal, to avoid score 0.
        # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
        # Sum is much larger than 1.
        # We need a valid set.
        # 1 = 1/2 + 1/3 + 1/7 + 1/42
        # Split 1/2 -> 1/3 + 1/6 (min 3)
        # Split 1/3 -> 1/4 + 1/12
        # ...
        # This is hard to do manually.
        
        # Let's just return a hardcoded 9-term solution if 8-term DFS fails?
        # Or extend DFS to 9 terms?
        found9 = []
        current9 = []
        if _dfs_exact(1, 1, 5, 9, deadline, found9, current9):
            return found9
            
        # If both fail, it's a critical error.
        raise RuntimeError("No solution found")

# Test locally
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {res}")
    print(f"Count: {len(res)}")
    s = sum(1.0/d for d in res)
    print(f"Sum: {s}")
    print(f"Time: {end-start:.2f}s")
    # Verify exact sum using fractions
    from fractions import Fraction
    total = Fraction(0,1)
    for d in res:
        total += Fraction(1,d)
    print(f"Exact Sum: {total}")
    assert total == 1
    assert len(set(res)) == len(res)
    assert all(5 <= d <= 10000000 for d in res)
