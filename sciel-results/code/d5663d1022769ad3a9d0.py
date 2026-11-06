"""Precomputed lookup table for minimal antichain smooth representations."""

def _precompute():
    import time
    start = time.time()
    
    # Generate all 2^a * 3^b * 5^c smooth numbers <= 128
    smooth = []
    for a in range(0, 8):  # 2^7 = 128
        for b in range(0, 5):  # 3^4 = 81
            for c in range(0, 4):  # 5^3 = 125
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= val <= 128:
                    smooth.append(val)
    smooth = sorted(set(smooth))
    
    # For each target, find minimal-cardinality antichain representation
    answers = [None] * 129  # index 0 unused
    
    for n in range(1, 129):
        # Candidate smooth numbers that could appear in a sum to n
        candidates = [x for x in smooth if x <= n]
        
        found = None
        for count in range(1, 11):  # try 1, 2, 3, ... terms
            if found is not None:
                break
            # Enumerate combinations of 'count' distinct candidates
            # that sum to n and form an antichain
            from itertools import combinations
            for combo in combinations(candidates, count):
                if sum(combo) != n:
                    continue
                # Check antichain: no term divides another
                valid = True
                for i in range(count):
                    for j in range(i + 1, count):
                        a, b = combo[i], combo[j]
                        if a % b == 0 or b % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    found = list(combo)
                    break
            # Time guard
            if time.time() - start > 75:
                break
        
        if found is None:
            # Fallback: use up to 30 terms via greedy/brute force
            # This should not happen for n<=128, but safety net
            from itertools import combinations
            for count in range(1, 31):
                for combo in combinations(candidates, count):
                    if sum(combo) != n:
                        continue
                    valid = True
                    for i in range(count):
                        for j in range(i + 1, count):
                            a, b = combo[i], combo[j]
                            if a % b == 0 or b % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        found = list(combo)
                        break
                if found is not None:
                    break
        
        answers[n] = found
    
    return answers

# Precompute at import time
_TABLE = _precompute()

def construct():
    """Return 128 rows corresponding to targets 1..128."""
    return [_TABLE[n] for n in range(1, 129)]
