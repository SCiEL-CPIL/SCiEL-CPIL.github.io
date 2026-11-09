import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80

    def check_validity(S):
        n = len(S)
        for i, a in enumerate(S):
            others = [S[j] for j in range(n) if j != i]
            m = len(others)
            if m < 2:
                continue
            d = [o - a for o in others]
            # Check if any subset of d with size >= 2 sums to 0
            min_possible = sum(min(0, x) for x in d)
            max_possible = sum(max(0, x) for x in d)
            offset = -min_possible
            size = max_possible - min_possible + 1
            if size > 300000:
                # Fallback: check small subset sizes
                from itertools import combinations
                for k in range(2, min(4, m + 1)):
                    for comb in combinations(d, k):
                        if sum(comb) == 0:
                            return False
                return True
            dp = [0] * (m + 1)
            dp[0] = 1 << offset
            for di in d:
                for j in range(m - 1, -1, -1):
                    if dp[j]:
                        if di >= 0:
                            dp[j + 1] |= (dp[j] << di)
                        else:
                            dp[j + 1] |= (dp[j] >> (-di))
            for j in range(2, m + 1):
                if dp[j] & (1 << offset):
                    return False
        return True

    def greedy_random(seed):
        random.seed(seed)
        S = []
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            trial = S + [x]
            if check_validity(trial):
                S = trial
        return S

    best = []
    seed = 0
    while time.time() < deadline:
        result = greedy_random(seed)
        if len(result) > len(best):
            best = result
            if len(best) >= 48:
                break
        seed += 1

    return best
