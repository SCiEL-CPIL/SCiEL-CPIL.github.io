import time
import random

def _is_valid(S):
    """Check if set S satisfies the condition."""
    S_list = sorted(S)
    n = len(S_list)
    # For each a, check pairs b,c > a
    # To speed up, we can break early if a is small? No, a is small means more pairs.
    # Actually, if a is large, there are fewer b,c.
    # Let's just iterate.
    for i in range(n):
        a = S_list[i]
        # b, c must be in S_list[i+1:]
        # We need to check if any b+c is divisible by a.
        # b < c or b <= c.
        # b+c ranges from min_pair to max_pair.
        # If min_pair > k*a for all k, then no conflict.
        # Smallest b+c is S_list[i+1] + S_list[i+2] (if exists)
        # Actually just check all pairs.
        sub = S_list[i+1:]
        m = len(sub)
        if m < 2:
            continue
        
        # Optimization: if a > max(sub) + max(sub), i.e., a > 2*max(sub), 
        # then b+c < 2a, so only multiple is 2a? No, b+c can be > a.
        # If a > max(sub), then b < a, c < a. b+c < 2a.
        # So we only need to check if b+c == 2a (or k*a where k*a < 2a, i.e. k=1? No b+c>a).
        # If a > max(sub), then b+c < 2a. Since b,c > a is false, b,c < a.
        # Wait, the condition is a < b <= c.
        # So b, c are ALWAYS > a.
        # So b+c > 2a.
        # So the smallest multiple to check is 3a.
        # If 3a > max(sub) + max(sub), then no multiple fits.
        
        max_sum = sub[-1] + sub[-2] # Max sum of two distinct? No, b<=c, so max is last+last? No, b<c usually, but b=c allowed? "a<b<=c". So b can equal c.
        # If b=c, sum = 2b.
        # Max sum is 2 * max(sub).
        
        if 3 * a > 2 * sub[-1]:
            continue
            
        # Check all pairs in sub
        # We can use a set for O(1) lookup
        # For each b in sub, for each k, check if k*a - b in sub and >= b
        # k starts from 3.
        max_k = (2 * sub[-1]) // a
        for k in range(3, max_k + 1):
            target = k * a
            # Find b, c in sub such that b+c = target, b <= c
            # b must be >= sub[0]
            # c = target - b
            # c must be in sub and c >= b
            # So b <= target/2
            # b >= target - sub[-1]
            
            # Iterate b in sub
            # To be fast, we can iterate b and check if target-b is in sub.
            # But we must ensure b <= target-b => b <= target/2.
            # And target-b must be in sub.
            
            # We can break if b > target/2
            half_target = target // 2
            for b in sub:
                if b > half_target:
                    break
                c = target - b
                # c must be in sub
                if c in S: # S contains all, but we need c in sub (c > a)
                    # Since c = k*a - b and k>=3, b>=sub[0]>a, 
                    # c = k*a - b. 
                    # Is c > a?
                    # k*a - b > a <=> (k-1)a > b.
                    # Since k>=3, (k-1)a >= 2a.
                    # b < a is false, b > a.
                    # If b is close to a, 2a > b is likely true.
                    # But we need c in sub, i.e., c > a.
                    # And c must be in S.
                    if c > a and c in S:
                        # Found conflict
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Initial set: (2n/3, n]
    n = 256
    S = set(range(171, n + 1))
    best_S = set(S)
    best_size = len(S)
    
    # Precompute list of candidates to add
    candidates = list(range(1, 171))
    random.shuffle(candidates)
    
    # Local search loop
    while time.time() < deadline:
        # Try to add a random candidate
        # Pick a random index to rotate or just pick random
        # To avoid bias, we can shuffle candidates occasionally
        
        if random.random() < 0.1:
            random.shuffle(candidates)
        
        added = False
        # Try a few candidates in this iteration? Or just one?
        # One is safer for time.
        x = candidates[0]
        candidates.pop(0)
        candidates.append(x) # Rotate
        
        if x in S:
            continue
            
        # Check if adding x makes S valid
        # New S is S union {x}
        # We only need to check conflicts involving x.
        # Conflicts can be:
        # 1. x is 'a': x must not divide b+c for any b,c in S, b>c > x? No, b,c > x.
        #    So check if any pair in S (elements > x) sums to multiple of x.
        # 2. x is 'b' or 'c': 
        #    If x is b, then for any a in S, a < x, we need a not divide x+c for any c in S, c >= x.
        #    If x is c, then for any a in S, a < x, and any b in S, a < b <= x, 
        #    we need a not divide b+x.
        
        # This incremental check is faster than full re-validation.
        
        S_list = sorted(S)
        valid = True
        
        # Case 1: x is the smallest 'a' in a potential triple? No, x is added.
        # x can be 'a', 'b', or 'c'.
        
        # 1. x as 'a':
        #    Need to check if x divides b+c for any b,c in S such that x < b <= c.
        #    Let S_x = [y for y in S if y > x]
        #    Check if any pair in S_x sums to k*x.
        S_x = [y for y in S if y > x]
        if len(S_x) >= 2:
            max_sum_x = S_x[-1] + S_x[-2] # Approx
            if 3 * x <= 2 * max(S_x):
                S_x_set = set(S_x)
                max_k = (2 * max(S_x)) // x
                for k in range(3, max_k + 1):
                    target = k * x
                    half = target // 2
                    for b in S_x:
                        if b > half: break
                        c = target - b
                        if c in S_x_set:
                            valid = False
                            break
                    if not valid: break
        
        # 2. x as 'b' or 'c':
        #    Here x is one of the larger summands.
        #    Let S_a = [a for a in S if a < x]
        #    For each a in S_a:
        #       We must ensure a does not divide x + c for any c in S, c >= x (if x is b, c>=x; if x is c, b<=x).
        #       Actually, the triple is (a, b, c) with a < b <= c.
        #       If x is b: b=x. c can be any element in S such that c >= x.
        #           Check if a divides x + c.
        #       If x is c: c=x. b can be any element in S such that a < b <= x.
        #           Check if a divides b + x.
        
        #    Combined: For each a in S_a:
        #       For each y in S such that y >= x:
        #           Check if a divides x + y. (Covers x as b, y as c, or x as c, y as b if y=x)
        #           Wait, if x is c, b is the other element.
        #           The pair is {x, y}. One is b, one is c.
        #           Since b <= c, the smaller is b, larger is c.
        #           So we just need to check if a divides x + y.
        #           Constraint: a < min(x,y).
        
        if valid:
            S_a = [a for a in S if a < x]
            # Elements in S that can pair with x: y in S, y >= x?
            # If x is b, y=c >= x.
            # If x is c, y=b <= x. But we also have a < y.
            # So we need to check pairs (x, y) where y in S.
            # Condition: a < min(x, y).
            
            # To be safe and simple, iterate all y in S.
            # For each y in S:
            #   Let b, c be x, y sorted.
            #   Let m = min(x, y).
            #   We need to check all a in S such that a < m.
            #   If any such a divides x+y, conflict.
            
            # Optimization: 
            # For a fixed x, and y in S:
            # Sum = x + y.
            # We need to check if any a in S, a < min(x,y), divides Sum.
            
            # Let's iterate y in S.
            # If x+y is large, few divisors.
            
            for y in S:
                if y == x: continue # x is not in S yet
                m = min(x, y)
                total = x + y
                # Check divisors of total that are in S and < m
                # We can just iterate a in S_a (which is S elements < x, but we need < m)
                # Actually S_a is elements < x.
                # If y < x, then m = y. We need a < y.
                # If y > x, then m = x. We need a < x.
                
                # Let's just check all a in S that are < m.
                # To speed up, precompute sorted S?
                # S is a set.
                
                # Simple check:
                # for a in S:
                #    if a < m and total % a == 0: conflict
                
                # This is O(|S|) per y. O(|S|^2) per x.
                # |S| ~ 100. 10,000 ops. Fast.
                
                for a in S:
                    if a < m and total % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            S.add(x)
            added = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                
    return list(best_S)

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    valid = True
    S_list = sorted(S)
    for i in range(len(S_list)):
        a = S_list[i]
        sub = S_list[i+1:]
        for j in range(len(sub)):
            b = sub[j]
            for k in range(j, len(sub)):
                c = sub[k]
                if b + c % a == 0:
                    print(f"Conflict: {a}, {b}, {c}")
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print("Valid:", valid)
