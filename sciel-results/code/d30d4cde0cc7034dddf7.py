import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    target_num = 5
    target_den = 31
    
    # Initial known solution
    best_solution = [8, 31, 248]
    best_max_den = 248
    
    # Helper to check if a list of denominators sums to 5/31
    # We assume the search generates valid candidates, but a final check is good practice
    # However, the search logic guarantees correctness if implemented with integer arithmetic.
    
    def search(rem_num, rem_den, min_d, max_d_limit, k, path):
        nonlocal best_solution, best_max_den
        
        if time.time() > deadline:
            return
        
        # If we need 1 more term
        if k == 1:
            # rem_num / rem_den must be 1/d
            # So rem_den must be divisible by rem_num
            if rem_num == 0:
                return
            if rem_den % rem_num != 0:
                return
            d = rem_den // rem_num
            if d > min_d and d < max_d_limit:
                # Valid solution
                if d < best_max_den:
                    best_max_den = d
                    best_solution = path + [d]
            return
        
        # Lower bound for d:
        # 1/d <= rem_num/rem_den  => d >= rem_den/rem_num
        # Also d > min_d
        d_min = max(min_d + 1, (rem_den + rem_num - 1) // rem_num)
        
        # Upper bound for d:
        # We need k terms to sum to rem_num/rem_den
        # Max sum of k terms starting with d is k/d
        # So k/d >= rem_num/rem_den => d <= k * rem_den / rem_num
        d_max = (k * rem_den) // rem_num
        
        # Additional pruning:
        # The largest denominator in the final solution must be < best_max_den
        # So the last term d_k < best_max_den
        # This implies that even if we pick the smallest possible d's for the first k-1 terms,
        # the remaining fraction must be representable by 1 term < best_max_den.
        # A simple bound: if d_min >= best_max_den, we can't beat the current best (since d increases)
        if d_min >= best_max_den:
            return
            
        # Also, if the theoretical max d exceeds best_max_den, we cap it
        # But we must ensure we don't miss solutions where d is small but later terms are large?
        # No, we want to minimize the LARGEST denominator.
        # If we pick d, the remaining terms must be > d.
        # So all subsequent terms will be > d.
        # Thus, if d >= best_max_den, the max denominator will be >= d >= best_max_den.
        # So we only care about d < best_max_den.
        d_max = min(d_max, best_max_den - 1)
        
        if d_min > d_max:
            return
            
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return
            
            # Calculate new remainder: rem - 1/d
            # rem = rem_num / rem_den
            # new_rem = (rem_num * d - rem_den) / (rem_den * d)
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num <= 0:
                # Since d is increasing, new_num will become even more negative
                # Wait, if new_num <= 0, it means 1/d >= rem.
                # But we ensured d >= rem_den/rem_num, so 1/d <= rem.
                # So new_num should be >= 0.
                # If new_num == 0, rem is exactly 1/d. But k > 1, so we need more terms.
                # If rem == 1/d, we can't add more positive terms.
                # So new_num must be > 0.
                if new_num == 0:
                    return # Can't add more terms
                else:
                    continue
            
            # Simplify fraction
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Recurse
            search(new_num, new_den, d, best_max_den, k - 1, path + [d])
            
            # Optional: If we found a better solution, best_max_den is updated.
            # We can tighten d_max in the loop?
            # The loop bound is fixed at start.
            # We could break if d >= best_max_den (already handled by d_max calculation at start,
            # but best_max_den might decrease inside the loop).
            if d >= best_max_den:
                break

    # Try different numbers of terms
    # 2 terms: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 31a+31b=5ab => 31a(5-b) = 31b => a(5-b)=b => a = b/(5-b).
    # Since a>0, b<5. But b>a>=7. Impossible.
    
    # 3 terms
    search(5, 31, 6, best_max_den, 3, [])
    
    # 4 terms
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 4, [])
        
    # 5 terms
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 5, [])
        
    # 6 terms
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 6, [])
        
    # 7 terms
    if time.time() < deadline:
        search(5, 31, 6, best_max_den, 7, [])

    return best_solution
