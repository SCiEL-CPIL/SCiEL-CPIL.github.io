import time
import random

def construct():
    start_time = time.time()
    deadline = 88.0  # 2s margin
    
    N = 256
    # Start with the upper half
    S = set(range(129, 257))
    
    def get_violations(S):
        viol = []
        S_list = sorted(S)
        S_set = S
        # Only check a in S such that 3*a <= 512 and 3*a >= 2*(min(S)+1)?
        # Actually b,c >= 129 => b+c >= 258.
        # So 3a >= 258 => a >= 86.
        # And 3a <= 512 => a <= 170.
        # So a must be in [86, 170] intersect S.
        # But we also need b > a.
        # If a < 129, b,c are in [129, 256].
        # If a >= 129, b,c are in [a+1, 256].
        
        for a in S_list:
            if a > 170: # 3a > 510, max sum 512. 3*170=510. 3*171=513.
                break
            target = 3 * a
            # b + c = target
            # b > a, b <= c
            # c = target - b
            # b <= target - b => 2b <= target => b <= target // 2
            # c >= 129 (since S subset of [129, 256] initially, but let's keep it general)
            # b in S, c in S
            
            # b range:
            # b > a
            # b <= target // 2
            # c = target - b must be in S
            # c >= b => target - b >= b => b <= target // 2
            # c <= 256 => target - b <= 256 => b >= target - 256
            # b >= 129 (if we assume S subset [129, 256])
            
            b_lo = max(a + 1, target - 256, 1)
            b_hi = min(target // 2, 256)
            
            for b in range(b_lo, b_hi + 1):
                c = target - b
                if b <= c and b in S_set and c in S_set:
                    viol.append((a, b, c))
        return viol
    
    # Phase 1: Clean up S_upper
    # We will remove elements to break violations.
    # To be efficient, we can just iterate and remove one element at a time.
    
    # Optimization: Precompute potential conflicts for a in [86, 170]
    # But S changes.
    
    # Simple Greedy:
    # While there are violations, remove the element that appears in the most violations.
    
    # To avoid O(N^3) repeatedly, we can just pick a random violation and remove a random element from it,
    # or better, remove 'a' as it's the "root" cause for that specific sum multiple.
    # However, removing 'a' might be suboptimal if 'a' is only in one violation but 'b' is in 10.
    
    # Let's try a few strategies and keep the best S.
    
    best_S = set(S)
    best_size = len(best_S)
    
    for strategy in range(10): # Try a few random restarts or variations
        if time.time() > deadline - 5:
            break
            
        S_curr = set(range(129, 257))
        
        # Randomize order of removal slightly or use different heuristic
        # Heuristic: Remove element with highest degree in violation graph
        
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            viols = get_violations(S_curr)
            if not viols:
                break
                
            # Count degrees
            counts = {}
            for a, b, c in viols:
                counts[a] = counts.get(a, 0) + 1
                counts[b] = counts.get(b, 0) + 1
                counts[c] = counts.get(c, 0) + 1
            
            # Find max count
            max_count = 0
            to_remove = None
            # Tie-break randomly
            for v, cnt in counts.items():
                if cnt > max_count:
                    max_count = cnt
                    to_remove = v
                elif cnt == max_count and to_remove is not None:
                    if random.random() < 0.5:
                        to_remove = v
            
            if to_remove is not None:
                S_curr.remove(to_remove)
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_S = set(S_curr)
            
    # Phase 2: Try to add elements from [1, 128] to best_S
    # Sort candidates in descending order (larger numbers are less likely to cause issues)
    candidates = list(range(128, 0, -1))
    
    def is_valid_add(x, S):
        S_set = S
        # x as a
        # a=x. b,c in S, x < b <= c. x | (b+c)
        # b+c multiples of x: 2x, 3x, ...
        # b,c >= x+1 => b+c >= 2x+2.
        # So multiples 3x, 4x, ...
        for k in range(3, 513 // x + 1):
            m = k * x
            if m < 2 * (x + 1):
                continue
            if m > 512:
                break
            # b + c = m, b > x, b <= c, b,c in S
            b_lo = max(x + 1, m - 256)
            b_hi = min(256, m // 2)
            for b in range(b_lo, b_hi + 1):
                c = m - b
                if b <= c and b in S_set and c in S_set:
                    return False
        
        # x as b
        # a < x <= c. a,c in S. a | (x+c)
        # x+c multiples of a.
        # c in S. c >= x.
        # a in S, a < x.
        for a in S:
            if a >= x:
                continue
            # a | (x+c) => x+c = k*a
            # c = k*a - x
            # c >= x => k*a >= 2x
            # c <= 256 => k*a <= 256+x
            k_lo = (2 * x + a - 1) // a
            k_hi = (256 + x) // a
            for k in range(k_lo, k_hi + 1):
                c = k * a - x
                if c >= x and c in S_set:
                    return False
        
        # x as c
        # a < b <= x. a,b in S. a | (b+x)
        # b+x = k*a
        # b = k*a - x
        # a < b <= x
        # b > a => k*a - x > a => k*a > a+x => k > 1 + x/a
        # b <= x => k*a - x <= x => k*a <= 2x => k <= 2x/a
        for a in S:
            if a >= x:
                continue
            k_lo = (a + x) // a + 1
            k_hi = (2 * x) // a
            for k in range(k_lo, k_hi + 1):
                b = k * a - x
                if a < b <= x and b in S_set:
                    return False
        
        return True

    S_final = set(best_S)
    
    for x in candidates:
        if time.time() > deadline - 2:
            break
        if is_valid_add(x, S_final):
            S_final.add(x)
            
    return sorted(S_final)
