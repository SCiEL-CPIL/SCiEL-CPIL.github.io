import time

def construct():
    start_time = time.time()
    deadline = 50.0
    
    N = 180
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a < b and a >= 1:
                    triples.append((a, b, c))
    
    in_triple = set()
    for t in triples:
        in_triple.update(t)
    
    V = set(range(1, N + 1)) - in_triple
    U = in_triple
    
    S_U = set(U)
    removed = []
    
    node_to_triples = {x: [] for x in U}
    for i, t in enumerate(triples):
        for x in t:
            node_to_triples[x].append(i)
            
    def is_conflict(x, S):
        # Check if adding x to S creates a triple
        for idx in node_to_triples[x]:
            t = triples[idx]
            # Check if other two elements are in S
            others = [y for y in t if y != x]
            if others[0] in S and others[1] in S:
                return True
        return False

    # Greedy removal
    while time.time() < deadline - 10:
        conflict = None
        for t in triples:
            if t[0] in S_U and t[1] in S_U and t[2] in S_U:
                conflict = t
                break
        
        if conflict is None:
            break
        
        a, b, c = conflict
        
        # Calculate current degrees
        def deg(x):
            count = 0
            for idx in node_to_triples[x]:
                t = triples[idx]
                if t[0] in S_U and t[1] in S_U and t[2] in S_U:
                    count += 1
            return count
            
        da, db, dc = deg(a), deg(b), deg(c)
        
        if da >= db and da >= dc:
            S_U.remove(a)
            removed.append(a)
        elif db >= da and db >= dc:
            S_U.remove(b)
            removed.append(b)
        else:
            S_U.remove(c)
            removed.append(c)
            
    # Add-back phase
    # Try to add back removed elements in reverse order
    for x in reversed(removed):
        if time.time() > deadline - 2:
            break
        if not is_conflict(x, S_U):
            S_U.add(x)
            
    result = list(V.union(S_U))
    return result
