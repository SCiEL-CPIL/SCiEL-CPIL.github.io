import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    N = 180
    
    # Precompute all valid distinct triples (a, b, c) with a < b < c? 
    # No, just distinct. But a is always smallest.
    # Store for each a, the list of (b, c) pairs.
    triples_by_a = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if c > a and c != b and 1 <= c <= N:
                    triples_by_a[a].append((b, c))
    
    # Start with the known safe large range
    S = set(range(81, N + 1))
    
    # Try to add smaller numbers
    for a in range(1, 81):
        if time.time() > deadline:
            break
        
        # Check if adding a is safe
        safe = True
        for b, c in triples_by_a[a]:
            if b in S and c in S:
                safe = False
                break
        
        if safe:
            S.add(a)
            
    return list(S)
