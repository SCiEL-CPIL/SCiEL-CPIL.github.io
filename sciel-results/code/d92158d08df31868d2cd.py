import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 70
    
    n = 256
    
    def has_conflict(S, a):
        for b in S:
            if b <= a:
                continue
            remainder = (-b) % a
            c = b + remainder
            if c < b:
                c += a
            while c <= n:
                if c in S:
                    return True
                c += a
        return False
    
    def greedy(order):
        S = set()
        for a in order:
            if not has_conflict(S, a):
                S.add(a)
        return S
    
    # Descending greedy
    best = greedy(range(n, 0, -1))
    
    # Try random orderings
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        S = greedy(order)
        if len(S) > len(best):
            best = S
    
    # Local search: add any non-conflicting missing element
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for a in range(1, n+1):
            if time.time() > deadline - 5:
                break
            if a in best:
                continue
            if not has_conflict(best, a):
                best.add(a)
                improved = True
    
    return sorted(best)
