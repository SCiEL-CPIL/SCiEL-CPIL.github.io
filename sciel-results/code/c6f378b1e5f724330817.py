import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 512

    def is_compatible(y, S):
        for x in S:
            if (y * x) // math.gcd(y, x) > N:
                return False
        return True

    best_set = []

    # 1. Try divisor sets of highly composite numbers
    # These are strong candidates
    highly_composite = [360, 420, 480, 240, 336, 180, 120, 72, 60, 30, 12, 6, 4, 2, 1]
    for m in highly_composite:
        S = [i for i in range(1, N + 1) if m % i == 0]
        # Greedy expand
        candidates = [i for i in range(1, N + 1) if i not in S]
        candidates.sort()
        
        for y in candidates:
            if time.time() > deadline:
                break
            if is_compatible(y, S):
                S.append(y)
                
        if len(S) > len(best_set):
            best_set = S

    # 2. Randomized restarts
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline:
            break
        # Start with a random subset of small numbers
        # Small numbers are more compatible
        max_start = random.randint(10, 30)
        seed = random.sample(range(1, max_start + 1), random.randint(5, 10))
        S = list(seed)
        
        candidates = [i for i in range(1, N + 1) if i not in S]
        # Sort by value, smaller values are more likely to fit
        candidates.sort()
        
        for y in candidates:
            if time.time() > deadline:
                break
            if is_compatible(y, S):
                S.append(y)
                
        if len(S) > len(best_set):
            best_set = S

    return best_set
