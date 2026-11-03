import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0, 1, 2, 4]
    best_score = 0.0
    
    def score_set(U):
        if not U or 0 not in U:
            return 0.0
        U = list(set(U))
        if max(U) == 0:
            return 1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        size_sums = len(sums)
        size_diffs = len(diffs)
        if size_sums == 0:
            return 0.0
        ratio = size_diffs / size_sums
        if ratio <= 1:
            return 1.0
        max_u = max(U)
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom

    # Calculate score for initial guess
    best_score = score_set(best_set)
    
    # Search small sets
    # N from 2 to 6
    # Max value up to 20
    # This is a small search space
    max_val = 20
    for n in range(2, 7):
        if time.time() > deadline:
            break
        # Generate combinations of size n-1 from 1..max_val
        # 0 is always included
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    return best_set
