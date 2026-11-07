import itertools

def construct():
    n = 128
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # such that a+b, a+c, b+c are in 1..n
    # We need to check if a set S contains any such triple.
    
    forbidden_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 <= n and s2 <= n and s3 <= n:
                    forbidden_triples.append((a, b, c))
    
    # Greedy Construction in Descending Order
    S = set()
    for x in range(n, 0, -1):
        # Check if adding x creates a forbidden triple
        # A forbidden triple in S U {x} must contain x.
        # So we only need to check triples containing x.
        valid = True
        for (a, b, c) in forbidden_triples:
            if a == x or b == x or c == x:
                # Check if the other two are in S
                # And the sums are in S U {x}
                # Wait, the definition of forbidden triple is on the BASE numbers.
                # The condition is: a,b,c in S AND a+b, a+c, b+c in S.
                # So we need:
                # 1. {a,b,c} subset of S_new
                # 2. {a+b, a+c, b+c} subset of S_new
                
                if a == x:
                    others = [b, c]
                elif b == x:
                    others = [a, c]
                else:
                    others = [a, b]
                
                if all(o in S for o in others):
                    # Base numbers are in S_new
                    # Check sums
                    sums = [a+b, a+c, b+c]
                    if all(s in S or s == x for s in sums):
                        valid = False
                        break
        if valid:
            S.add(x)
            
    # Local Search to improve
    # Try to swap elements
    improved = True
    while improved:
        improved = False
        for x in range(1, n + 1):
            if x in S:
                continue
            # Try to add x by removing 1 or 2 elements from S
            # First try removing 0 (already done in greedy, but maybe order matters?)
            # Greedy was descending. Maybe ascending allows different set?
            # Let's try swapping: remove y, add x.
            for y in list(S):
                S_temp = S - {y}
                S_temp.add(x)
                # Check validity of S_temp
                # Efficient check:
                # Only need to check triples involving x or y?
                # No, removing y might break a triple, adding x might create one.
                # Just check all triples in S_temp?
                # Too slow.
                
                # Heuristic: Only check if S_temp is valid.
                # We can reuse the logic:
                # Check if any forbidden triple is fully contained in S_temp.
                # This is O(|forbidden_triples|) per swap.
                # 8000 swaps * 350k triples = too slow.
                
                # Optimization:
                # A set is invalid if it contains a forbidden triple.
                # We can precompute for each element z, the list of forbidden triples it participates in.
                
                pass # Placeholder for efficient check
                
    return sorted(S)
