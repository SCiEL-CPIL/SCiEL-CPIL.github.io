import time
import random

def construct():
    deadline = time.time() + 85.0
    n = 512
    
    def can_add_increasing(S, x):
        # x is the last term
        for d in range(1, (x - 1) // 3 + 1):
            if (x - d) in S and (x - 2*d) in S and (x - 3*d) in S:
                return False
        return True

    def can_add_random(S, x):
        # x is 1st term
        for d in range(1, (n - x) // 3 + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return False
        # x is 2nd term
        for d in range(1, x + 1):
            if (x - d) >= 1 and (x + 2*d) <= n:
                if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                    return False
        # x is 3rd term
        for d in range(1, x // 2 + 1):
            if (x - 2*d) >= 1 and (x + d) <= n:
                if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                    return False
        # x is 4th term
        for d in range(1, x // 3 + 1):
            if (x - 3*d) >= 1:
                if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                    return False
        return True

    def greedy_increasing():
        S = set()
        for x in range(1, n + 1):
            if can_add_increasing(S, x):
                S.add(x)
        return S

    def greedy_random():
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        S = set()
        for x in candidates:
            if can_add_random(S, x):
                S.add(x)
        return S

    # Start with deterministic greedy
    best_set = greedy_increasing()
    best_size = len(best_set)
    
    # Randomized restarts
    # Use a fixed seed for reproducibility in testing, but random for production?
    # The task says "Maximize cardinality". Randomness helps.
    # We will just use the system random.
    
    attempts = 0
    # Run as many as possible
    while time.time() < deadline - 5.0:
        S = greedy_random()
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
        attempts += 1
        
    return sorted(best_set)
