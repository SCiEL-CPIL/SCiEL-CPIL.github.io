<<<<<<< SEARCH
def score_of(U):
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U.sort()
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    diffs = set()
    sums = set()
    for a in U:
        for b in U:
            diffs.add(a - b)
            sums.add(a + b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0 # Or slightly above if we want to encourage growth, but mathematically if ratio<1 score<1? 
                   # Actually log(ratio) is negative. 
                   # We want to maximize. 
                   # If ratio < 1, score < 1. 
                   # We should avoid this.
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val
=======
def score_of(U):
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U.sort()
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    n = len(U)
    diffs = set()
    sums = set()
    for i, a in enumerate(U):
        for b in U[i:]:
            diffs.add(a - b)
            diffs.add(b - a)
            sums.add(a + b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val
>>>>>>> REPLACE

