import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    n = 128
    O = [i for i in range(1, n+1) if i % 2 != 0]
    E = set()
    
    # Precompute sums of pairs in O
    # sum_val -> list of (a,b)
    # We only care if sum_val is even (always true for O+O)
    
    # To speed up checks, we can precompute for each even c, the set of pairs (a,b) in O
    # such that a+c <= n and b+c <= n.
    # Let's just iterate.
    
    # Sort evens descending
    evens = [i for i in range(2, n+1, 2)]
    evens.sort(reverse=True)
    
    for c in evens:
        if time.time() > deadline:
            break
            
        valid = True
        
        # Check 1: Conflict with existing E via pair (a,b) in O
        # If a+b in E and a+c <= n and b+c <= n -> invalid
        # Note: a,b distinct in O.
        
        # Optimization:
        # a+c <= n => a <= n-c
        # b+c <= n => b <= n-c
        # So a,b must be in O_small = {x in O | x <= n-c}
        
        max_val = n - c
        if max_val < 3: # Smallest distinct pair sum is 1+3=4
             # If max_val < 3, no pair a,b in O with a,b <= max_val exists?
             # 1, 3. If max_val=1, only 1. No pair.
             # If max_val=2, only 1. No pair.
             # If max_val=3, 1,3. Pair (1,3). Sum 4.
             pass
             
        # Iterate pairs in O_small
        O_small = [x for x in O if x <= max_val]
        # If len < 2, no pair
        if len(O_small) >= 2:
            # Check all pairs
            for i in range(len(O_small)):
                for j in range(i+1, len(O_small)):
                    a = O_small[i]
                    b = O_small[j]
                    s = a + b
                    if s in E:
                        valid = False
                        break
                    # Also check if s == c?
                    # If s == c, then a+b is in S (as c).
                    # So if s == c, valid = False?
                    # Yes, because a+b, a+c, b+c must be in S.
                    # a+b=c is in S.
                    # a+c = a+c. Is it in S? Yes if <= n.
                    # b+c = b+c. Is it in S? Yes if <= n.
                    # So if a+b == c, and a+c<=n, b+c<=n, then invalid.
                    if s == c:
                        valid = False
                        break
                if not valid:
                    break
        if not valid:
            continue
            
        # Check 2: Conflict with triples in O
        # If a+b, a+d, b+d all in E U {c}
        # This is harder.
        # We need to check if adding c completes a triple of sums for some a,b,d in O.
        # i.e., exists a,b,d distinct in O such that:
        # {a+b, a+d, b+d} \subseteq E \cup {c}
        # This means:
        # 1. a+b, a+d, b+d all in E (already checked? No, this is static)
        # 2. Two in E, one is c.
        # 3. One in E, two are c (impossible, sums distinct? a+b != a+d if b!=d)
        
        # So we check:
        # Does there exist a,b,d distinct in O such that
        # exactly one of {a+b, a+d, b+d} is c, and the other two are in E?
        
        # If c is large, a+b=c implies a,b small.
        # Let's find all pairs (a,b) in O such that a+b == c.
        # If no such pair, then c cannot be one of the sums.
        # So Check 2 is only needed if there are pairs summing to c.
        
        pairs_summing_to_c = []
        for i in range(len(O)):
            for j in range(i+1, len(O)):
                if O[i] + O[j] == c:
                    pairs_summing_to_c.append((O[i], O[j]))
                    
        if pairs_summing_to_c:
            for a, b in pairs_summing_to_c:
                # We have a+b = c.
                # We need a+d in E and b+d in E for some d in O, d != a, d != b.
                # Also d must be such that a+d, b+d are in range?
                # Sums are just values. If they are in E, they are in S.
                # E contains values in 1..128.
                # So we just check if there exists d in O \ {a,b} such that
                # (a+d) in E and (b+d) in E.
                
                found_d = False
                for d in O:
                    if d == a or d == b:
                        continue
                    if (a + d) in E and (b + d) in E:
                        found_d = True
                        break
                if found_d:
                    valid = False
                    break
        if not valid:
            continue
            
        E.add(c)
        
    result = list(O) + list(E)
    return result
