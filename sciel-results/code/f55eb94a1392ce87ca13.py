import time

def construct():
    start_time = time.time()
    deadline = 45.0  # 45s to be safe
    
    n = 128
    S = set(range(1, n + 1))
    
    # Precompute triples
    triples = []
    triples_by_elem = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 <= n and s2 <= n and s3 <= n:
                    t = (a, b, c, s1, s2, s3)
                    triples.append(t)
                    # Add to buckets
                    for v in set(t): # Unique elements in this tuple
                        triples_by_elem[v].append(t)
                        
    # Active triples: initially all are candidates, but we check against S
    # Since S is full, all triples are initially bad.
    active_triples = list(triples)
    
    # Peeling
    while active_triples:
        if time.time() > deadline:
            break
            
        freq = {}
        for t in active_triples:
            for x in set(t):
                freq[x] = freq.get(x, 0) + 1
        
        # Pick max freq
        max_v = max(freq, key=lambda k: (freq[k], -k))
        S.remove(max_v)
        
        # Filter active_triples
        # Optimization: instead of filtering the whole list, 
        # we can just check validity later? 
        # No, we need to remove to keep freq correct.
        # But filtering 350k list 128 times is okay.
        active_triples = [t for t in active_triples if max_v not in t]
        
    # Add-back
    candidates = [x for x in range(1, n + 1) if x not in S]
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        safe = True
        for t in triples_by_elem[x]:
            # Check if t is subset of S U {x}
            # Since x is not in S, we check if all other elements are in S
            # Note: t might contain x multiple times? 
            # a,b,c distinct. Sums distinct from each other? 
            # a+b = a+c => b=c (no). 
            # a+b = c? Possible.
            # So t as a tuple might have duplicates.
            # But "all values in t are in S U {x}" is equivalent to 
            # "set(t) is subset of S U {x}".
            if all(v in S or v == x for v in t):
                safe = False
                break
                
        if safe:
            S.add(x)
            
    return list(S)
