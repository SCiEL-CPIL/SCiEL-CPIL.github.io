import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    TARGET = 48
    
    def is_mean_free(S):
        """Check if no element in S is the mean of any subset of >=2 others."""
        for a in S:
            others = [x for x in S if x != a]
            n_others = len(others)
            if n_others < 2:
                continue
            dp = [0] * (n_others + 1)
            dp[0] = 1
            for x in others:
                for k in range(n_others - 1, -1, -1):
                    dp[k+1] |= (dp[k] << x)
            for k in range(2, n_others + 1):
                if (dp[k] >> (k * a)) & 1:
                    return False
        return True
    
    def greedy_mean_free(lo, hi, order, deadline):
        """Greedy construction following given order."""
        S = []
        for x in order:
            if time.time() > deadline:
                break
            if lo <= x <= hi and is_mean_free(S + [x]):
                S.append(x)
        return S
    
    best = []
    rng = random.Random(42)
    
    # Strategy 1: Two-group separation
    # A in [1, 200], B in [401, 512], min(B)=401 > 2*200=400
    A1 = greedy_mean_free(1, 200, list(range(1, 201)), deadline)
    B1 = greedy_mean_free(401, 512, list(range(401, 513)), deadline)
    r1 = A1 + B1
    if len(r1) > len(best):
        best = r1
    
    # Strategy 2: A in [1, 250], B in [501, 512]
    A2 = greedy_mean_free(1, 250, list(range(1, 251)), deadline)
    B2 = greedy_mean_free(501, 512, list(range(501, 513)), deadline)
    r2 = A2 + B2
    if len(r2) > len(best):
        best = r2
    
    # Strategy 3: Single range, ascending
    r3 = greedy_mean_free(1, N, list(range(1, N+1)), deadline)
    if len(r3) > len(best):
        best = r3
    
    # Strategy 4: Single range, descending
    r4 = greedy_mean_free(1, N, list(range(N, 0, -1)), deadline)
    if len(r4) > len(best):
        best = r4
    
    # Strategy 5: Random orderings
    for trial in range(10):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        r5 = greedy_mean_free(1, N, order, deadline)
        if len(r5) > len(best):
            best = r5
    
    # Try to extend best
    if len(best) < TARGET:
        best_set = set(best)
        for x in range(1, N+1):
            if time.time() > deadline:
                break
            if x not in best_set and is_mean_free(best + [x]):
                best.append(x)
                best_set.add(x)
                if len(best) == TARGET:
                    break
    
    return best[:TARGET]

if __name__ == "__main__":
    s = construct()
    print(len(s))
