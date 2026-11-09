"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    n = 256
    deadline = time.time() + 75  # safety margin before 90s limit
    
    # Precompute divisibility: divs[a] = set of numbers in [1,n] divisible by a
    # Actually we need: for each a, which pairs (b,c) with b<=c have a | (b+c)
    # Let's precompute for each a, the set of sums s in [2, 2*n] such that a|s
    # Then a conflicts with (b,c) if b+c is in that set and b>=a+1, c>=b
    
    # Better approach: for a candidate set S, check if any a in S divides b+c for a<b<=c in S
    # For efficiency, precompute for each a: which values of (b+c) are divisible by a
    # Since b,c are in [1,n], b+c is in [2, 2n]
    
    # Precompute: for each a in [1,n], the set of sums s in [2, 2n] where a|s
    # And for each sum s, the pairs (b,c) with b<=c, b+c=s, b,c in [1,n]
    
    # Key insight: the constraint is that for all a < b <= c in S, a does not divide b+c.
    # Equivalently: for all a in S, and all b,c in S with a < b <= c, a does not divide b+c.
    
    # Strategy: Start with a good seed (upper interval), then try to add elements from below.
    
    # First, find a strong seed using the interval [k, n] where k is chosen so that
    # for all a in [k,n], a < b <= c implies a does not divide b+c.
    # If a >= k and b,c >= k with a < b <= c, then b+c >= 2k+1 (since b >= a+1 >= k+1, c >= b >= k+1)
    # Wait, more carefully: a >= k, b >= a+1 >= k+1, c >= b >= k+1
    # So b+c >= 2(k+1) = 2k+2
    # For a to divide b+c, we need b+c >= a (which is true) and b+c = m*a for some m >= 1
    # The minimum b+c is 2(k+1) and maximum is 2n
    # For a in [k, n], the multiples of a in [2(k+1), 2n] are:
    #   ceil(2(k+1)/a)*a, ..., floor(2n/a)*a
    # If 2(k+1) > 2n/a * a... this is getting complex.
    
    # Simpler: for the interval [k, n], the condition is:
    # For all a in [k,n] with a < b <= c in [k,n], a does not divide b+c.
    # b+c ranges from (k+1)+(k+1) = 2k+2 to n+n = 2n.
    # a divides b+c means b+c = m*a for integer m.
    # Since b+c >= 2k+2 and a <= n, m <= 2n/a <= 2n/k.
    # Since b+c <= 2n and a >= k, m >= (2k+2)/a >= (2k+2)/n.
    
    # For the interval to be valid, we need that no a in [k,n] has a multiple m*a in [2k+2, 2n]
    # that can be written as b+c with b,c in [a+1, n] and b <= c.
    # Actually b,c just need to be in [k,n] with b > a.
    
    # Let's just use a verified interval. From the observations, k=171 gives size 86.
    # Let me verify: for a in [171, 256], b in [a+1, 256], c in [b, 256]:
    # b+c is in [2*172, 512] = [344, 512]
    # a is in [171, 256]
    # m = (b+c)/a must be integer. m >= 344/256 ≈ 1.34, m <= 512/171 ≈ 3.0
    # So m can be 2 or 3.
    # m=2: b+c = 2a. Since b > a and c >= b, b+c >= (a+1)+(a+1) = 2a+2 > 2a. So m=2 impossible.
    # m=3: b+c = 3a. b > a, c >= b. b+c = 3a. Since b <= c, b <= 3a/2. And b > a.
    #   Also b,c <= n=256, so 3a <= 512, a <= 170.66. But a >= 171, so 3a >= 513 > 512 = 2n. Impossible.
    # So [171, 256] is valid. Size = 256-171+1 = 86.
    
    # Can we do better? Try smaller k.
    # For k=170: a=170, b=171, c=171: b+c=342=2*171. Does 170 divide 342? 342/170=2.01... No.
    # But we need to check all. Let me just try to find the smallest k where [k,n] is valid.
    
    def is_interval_valid(k, n):
        for a in range(k, n+1):
            for b in range(a+1, n+1):
                for c in range(b, n+1):
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Find the best (smallest) k such that [k, n] is valid
    best_k = n
    for k in range(n, 0, -1):
        if is_interval_valid(k, n):
            best_k = k
        else:
            break  # Once invalid, larger k-1 will also be invalid? Not necessarily monotonic... 
            # Actually it IS monotonic: if [k,n] is valid, then [k+1,n] is valid (subset).
            # So we want the smallest k. Let's search from small to large.
    
    # Actually let me search properly
    best_k = n
    for k in range(1, n+1):
        if is_interval_valid(k, n):
            best_k = k
            break
    
    seed = list(range(best_k, n+1))
    
    # Now try to add elements from [1, best_k-1] to the seed
    # Greedy: try each candidate, add if feasible
    selected = set(seed)
    
    def can_add(candidate, sel):
        """Check if adding candidate to sel maintains the property."""
        # For all a < b <= c in sel ∪ {candidate}, a does not divide b+c
        # New violations can only involve candidate
        # Case 1: candidate is a (smallest): candidate < b <= c in sel, candidate | (b+c)
        # Case 2: candidate is b: a in sel < candidate, candidate <= c in sel, a | (candidate+c)
        # Case 3: candidate is c: a in sel < b in sel <= candidate, a | (b+candidate)
        
        for a in sel:
            if a >= candidate:
                continue
            # candidate is b or c
            for b in sel:
                if b < candidate:
                    # a < b < candidate: check a | (b + candidate)
                    if (b + candidate) % a == 0:
                        return False
                elif b == candidate:
                    # a < candidate = b: check a | (candidate + c) for c in sel, c >= candidate
                    for c in sel:
                        if c >= candidate:
                            if (candidate + c) % a == 0:
                                return False
                else:
                    # a < candidate < b: candidate is b, b is c... wait
                    # a < candidate < b <= c? No, b > candidate here means candidate < b
                    # So a < candidate < b <= c: check a | (candidate + b)
                    if (candidate + b) % a == 0:
                        return False
        # Case: candidate is a (smallest element)
        # candidate < b <= c in sel
        for b in sel:
            if b <= candidate:
                continue
            for c in sel:
                if c < b:
                    continue
                if (b + c) % candidate == 0:
                    return False
        return True
    
    # Greedy add from small to large
    for num in range(1, best_k):
        if time.time() > deadline:
            break
        if can_add(num, selected):
            selected.add(num)
    
    result = sorted(selected)
    
    # Verify the result
    def verify(S):
        S = sorted(S)
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                for k in range(j, len(S)):
                    c = S[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    if not verify(result):
        # Fallback to seed
        result = seed
    
    return result
# EVOLVE-BLOCK-END
