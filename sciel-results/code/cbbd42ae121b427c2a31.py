"""Rewritten program: greedy construction avoiding forbidden triples."""

def construct():
    n = 128
    # Greedy: try to add each number if it doesn't create a forbidden triple
    # A triple {a,b,c} is forbidden if a+b, a+c, b+c are all in the set.
    # When adding x, check pairs (a,b) in current set: if a+b in set and x+a in set and x+b in set, then {x,a,b} would be forbidden.
    
    def is_safe(x, S, Sset):
        # Check if adding x creates a forbidden triple
        # For any pair a,b in S: if a+b in Sset and x+a in Sset and x+b in Sset, then forbidden
        # Note: a+b, x+a, x+b must all be in the set AFTER adding x.
        # Since x is being added, x+a and x+b are not yet in Sset (they'd be > x potentially, or could be in Sset if already present)
        # Actually x+a and x+b could already be in Sset if they were added earlier.
        # a+b is in Sset if already present.
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                # Need a+b, x+a, x+b all in set after adding x
                if a + b in Sset and (x + a) in Sset and (x + b) in Sset:
                    return False
        return True
    
    S = []
    Sset = set()
    
    for x in range(1, n+1):
        if is_safe(x, S, Sset):
            S.append(x)
            Sset.add(x)
    
    # Ensure all are Python ints
    return [int(v) for v in S]
