import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 256
    S = []
    
    # Precompute nothing, just iterate
    # We iterate from n down to 1
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if x can be added to S
        # New conflicts must involve x as the smallest element 'a'
        # So we check if there exist b, c in S such that x divides b+c
        # Since S contains elements > x, b > x and c > x
        # So b+c > 2x.
        
        valid = True
        # Optimization: only check multiples of x that can be formed by sums in S
        # Max sum in S is 2 * max(S) <= 512
        # Min sum in S is 2 * min(S) + 1 ? No, min(S) > x.
        
        # Iterate through multiples of x
        # First possible multiple is 3*x (since b,c > x => b+c >= (x+1)+(x+1) = 2x+2)
        # Actually if S has elements close to x, sum is close to 2x.
        # But S only has elements > x.
        
        max_sum = 2 * S[0] if S else 0 # S[0] is the largest element added so far (since we add descending)
        # Wait, S is a list. We can keep a set for faster sum checking?
        # Or just check sums directly.
        
        # If S is empty, x is always valid.
        if not S:
            S.append(x)
            continue
            
        # To check if any b+c is divisible by x:
        # We can iterate through b in S, and for each b, check if (k*x - b) is in S.
        # This is O(|S| * (max_sum/x)).
        
        # Better: compute set of sums?
        # |S| is at most ~100.
        # Sums set size ~ |S|^2 / 2.
        # Constructing sums set every step is O(|S|^2).
        # 256 steps * 100^2 = 2.5M ops. Fine.
        
        # However, we can be smarter.
        # Just check multiples.
        # Multiples of x in range [2*x+2, 512]
        
        found_conflict = False
        # We need to check if any sum in S+S is a multiple of x.
        # Let's just compute the set of sums for the current S.
        # This is repeated work, but S changes slowly.
        # Actually, S grows.
        
        # Alternative:
        # For each b in S:
        #   For each c in S where c >= b:
        #     if (b+c) % x == 0:
        #        found_conflict = True
        #        break
        #   if found_conflict: break
        
        # This is O(|S|^2) per x.
        # Total O(n * |S|^2).
        # Max |S| ~ 100. n=256.
        # 256 * 10000 = 2.5M.
        # Python might be slow with nested loops.
        # Let's use a set for S to allow O(1) lookup?
        # For each b in S:
        #   We want to know if there is c in S such that b+c = k*x.
        #   i.e., c = k*x - b.
        #   Check if c in S_set and c >= b (to avoid double counting, though not strictly needed for existence).
        
        S_set = set(S)
        for b in S:
            # c must be in S
            # b+c must be multiple of x
            # Let b+c = m*x
            # c = m*x - b
            # We need c in S_set.
            # Also c >= b is not required for validity check, just existence.
            # But we must ensure c is a valid element.
            
            # Range of m:
            # Min sum: 2 * min(S)
            # Max sum: 2 * max(S)
            
            # We can just iterate m?
            # Or iterate c?
            # Iterating m is better if x is large.
            # If x is small, m is large.
            
            # Let's just iterate through possible sums? No.
            # Let's iterate m from 3 to 512//x + 1
            # And check if (m*x - b) is in S_set.
            
            # Optimization:
            # We only need to check m such that m*x is in the range of sums [2*min(S), 2*max(S)].
            
            min_S = min(S)
            max_S = max(S)
            
            # Actually, we can just check all pairs?
            # If |S| is small, pairs are fine.
            # If |S| is large, it's still fine.
            
            # Let's stick to the pair check but optimized with break.
            pass
            
        # Re-implementation of the inner loop for clarity and speed
        # We will check if any pair (b,c) in S sums to a multiple of x.
        
        # To avoid O(|S|^2) if possible:
        # For each b in S:
        #   We need c in S such that (b+c) % x == 0.
        #   This implies c % x == (-b) % x.
        #   Group S by remainder mod x.
        #   If any remainder group has size >= 2, or if b's remainder group has another element?
        #   Wait. If b and c are in the same remainder group r, then b+c is 2r mod x.
        #   We need b+c divisible by x.
        #   So 2r == 0 mod x.
        #   This is getting complicated.
        
        # Given the small n=256, O(|S|^2) is likely fast enough in Python if implemented carefully.
        # 256 * (100*100) = 2.5 million iterations.
        # Each iteration is a modulo and comparison.
        # Should run in < 1 second.
        
        conflict = False
        for b in S:
            for c in S:
                if b < c: # ensure b < c? No, condition is a < b <= c.
                    # Here a=x. b,c are from S.
                    # We just need b <= c?
                    # The condition is for ALL a<b<=c.
                    # If we find ANY b,c (regardless of order) such that x | b+c,
                    # we can swap b,c to satisfy b<=c.
                    # So checking unordered pairs is sufficient.
                    # But checking all pairs (b,c) including b=c is fine.
                    pass
                    
            # Let's just do:
            # for b in S:
            #   for c in S:
            #     if (b+c) % x == 0:
            #        conflict = True
            #        break
            #   if conflict: break
            
        # This is O(|S|^2).
        
        # Let's implement this directly.
        
        conflict = False
        # Optimization: if x > max(S) * 2, no sum can be multiple of x (since max sum < 2*max(S) < x? No, max sum ~ 500, x can be 100. 500 > 100).
        # If x > max_sum, no conflict.
        # max_sum = 2 * S[0] (since S[0] is max)
        if x > 2 * S[0]:
            conflict = False
        else:
            for b in S:
                # We can break early if b + min(S) > 2 * max(S)? No.
                for c in S:
                    if (b + c) % x == 0:
                        conflict = True
                        break
                if conflict:
                    break
        
        if not conflict:
            S.append(x)
            
    return S
