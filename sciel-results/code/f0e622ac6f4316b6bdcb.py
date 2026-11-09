def construct():
    n = 256
    candidates = list(range(2, n + 1))
    S = set(candidates)
    
    # Helper to check if a triple is valid
    def is_bad(a, b, c):
        return (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0

    # We will iteratively remove elements that are part of bad triples.
    # To be efficient, we'll just scan for a bad triple, remove one, and repeat.
    # Since N is small, this is feasible.
    
    # To avoid O(N^3) scan every time, we can just do it.
    # 255^3 is ~16M. 
    # If we do this 200 times, it's 3.2B operations. Too slow in Python.
    
    # Optimization:
    # Precompute for each a, the list of (b, c) such that a | bc.
    # Store as a list of pairs.
    # When checking S, for each a in S, check if any (b, c) from its list are in S.
    
    bad_pairs = {a: [] for a in candidates}
    for a in candidates:
        for b in candidates:
            if b == a: continue
            for c in candidates:
                if c == a or c == b: continue
                if (b * c) % a == 0:
                    bad_pairs[a].append((b, c))
                    
    # Now, to find a violation in S:
    # Iterate a in S.
    # For each (b, c) in bad_pairs[a]:
    #   if b in S and c in S: violation found (a, b, c)
    
    while True:
        violation = None
        for a in S:
            for b, c in bad_pairs[a]:
                if b in S and c in S:
                    violation = (a, b, c)
                    break
            if violation:
                break
        if not violation:
            break
        
        # Remove one element from the violation
        # Heuristic: Remove the one with the most "bad pairs" in S
        a, b, c = violation
        # Count how many bad pairs are active for each
        count_a = sum(1 for x, y in bad_pairs[a] if x in S and y in S)
        count_b = sum(1 for x, y in bad_pairs[b] if x in S and y in S)
        count_c = sum(1 for x, y in bad_pairs[c] if x in S and y in S)
        
        # Remove the one with highest count
        max_count = max(count_a, count_b, count_c)
        if count_a == max_count:
            S.remove(a)
        elif count_b == max_count:
            S.remove(b)
        else:
            S.remove(c)
            
    return list(S)
