import time

def construct():
    start = time.time()
    deadline = start + 80.0  # 80s safety margin within 90s limit
    N = 180
    
    # Greedy: try to include each number from N down to 1
    # A set S is valid if no three DISTINCT elements a,b,c in S satisfy 1/a = 1/b + 1/c
    # i.e., a*(b+c) = b*c, equivalently 1/a = 1/b + 1/c
    
    # We build S greedily from largest to smallest.
    # When considering adding x to current set S:
    #   Check if there exist distinct a,b in S (a != b, a != x, b != x) such that
    #   1/x = 1/a + 1/b  =>  x*(a+b) = a*b
    #   OR 1/a = 1/x + 1/b for some distinct b in S (b != a, b != x)
    #       => a*(x+b) = x*b
    #   OR 1/b = 1/x + 1/a for some distinct a in S (a != b, a != x)
    #       => b*(x+a) = x*a
    
    # Actually, the condition is: no three DISTINCT elements a,b,c in S satisfy 1/a = 1/b + 1/c.
    # WLOG let a < b < c (since 1/a > 1/b > 1/c, we need a to be the smallest for 1/a to be largest).
    # Wait: 1/a = 1/b + 1/c means 1/a > 1/b and 1/a > 1/c, so a < b and a < c.
    # So a is the smallest of the three. b and c are both larger than a.
    # Without loss of generality, assume a < b < c. Then we need 1/a = 1/b + 1/c.
    # This means: a*c = b*(a+c) => a*c = a*b + b*c => a*c - a*b = b*c => a*(c-b) = b*c
    # So for given a and b, c = a*b/(a-b+... wait let me redo.
    # 1/a = 1/b + 1/c => c = a*b/(a+b) * ... no.
    # 1/a - 1/b = 1/c => (b-a)/(a*b) = 1/c => c = a*b/(b-a)
    # So c = a*b/(b-a), which requires b > a and (b-a) divides a*b.
    
    # So the constraint is: for any a < b in S, if c = a*b/(b-a) is an integer and c > b and c in S, that's invalid.
    # Also need c distinct from a and b (c > b > a ensures distinctness).
    
    S = []
    for x in range(N, 0, -1):
        if time.time() > deadline:
            break
        # Check if adding x to S creates a violation
        # x could be:
        # 1) The smallest element a in a triple (a, b, c) with a < b < c in S ∪ {x}
        #    Then we need b, c in S, b > x, c > b, and c = x*b/(b-x)
        # 2) The middle element b in a triple (a, b, c) with a < b < c in S ∪ {x}
        #    Then a < x, c > x in S, and c = a*x/(x-a)
        # 3) The largest element c in a triple (a, b, c) with a < b < c in S ∪ {x}
        #    Then a < b < x in S, and x = a*b/(b-a)
        
        valid = True
        
        # Case 3: x is the largest (c). Check if there exist a < b in S with a < b < x
        # such that x = a*b/(b-a), i.e., a*b = x*(b-a) => a*b - x*b = -x*a => b*(a-x) = -x*a => b = x*a/(x-a)
        # For each a in S with a < x: b = x*a/(x-a), check if b is integer, a < b < x, b in S
        S_set = set(S)
        for a in S:
            if a >= x:
                break  # S is sorted descending, but we need a < x
            # Actually S is built from large to small, so S is in descending order
            # Let me just iterate all
        # Let me restructure: S is a list, we can check membership via set
        S_set = set(S)
        
        # Case 3: x is c (largest). Need a < b < x, all in S, and x = a*b/(b-a)
        # => b = x*a/(x-a)
        for a in S:
            if a >= x:
                continue
            denom = x - a
            if denom <= 0:
                continue
            num = x * a
            if num % denom != 0:
                continue
            b = num // denom
            if b <= a or b >= x:
                continue
            if b in S_set:
                valid = False
                break
        if not valid:
            continue
            
        # Case 2: x is b (middle). Need a < x < c, a,c in S, and c = a*x/(x-a)
        # For each a in S with a < x: c = a*x/(x-a), check c > x, c in S
        for a in S:
            if a >= x:
                continue
            denom = x - a
            if denom <= 0:
                continue
            num = a * x
            if num % denom != 0:
                continue
            c = num // denom
            if c <= x:
                continue
            if c in S_set:
                valid = False
                break
        if not valid:
            continue
            
        # Case 1: x is a (smallest). Need x < b < c, b,c in S, and c = x*b/(b-x)
        # For each b in S with b > x: c = x*b/(b-x), check c > b, c in S
        for b in S:
            if b <= x:
                continue
            denom = b - x
            if denom <= 0:
                continue
            num = x * b
            if num % denom != 0:
                continue
            c = num // denom
            if c <= b:
                continue
            if c in S_set:
                valid = False
                break
        if not valid:
            continue
            
        # All checks passed, add x
        S.append(x)
        
    return S
