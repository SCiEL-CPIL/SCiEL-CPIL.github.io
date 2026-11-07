import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78.0
    
    def is_valid(S):
        n = len(S)
        if n == 0:
            return True
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for j in range(n):
                if mask & (1 << j):
                    s += S[j]
            if s in sums:
                return False
            sums.add(s)
        sl = sorted(sums)
        for i in range(len(sl)):
            a = sl[i]
            for j in range(i + 1, len(sl)):
                if sl[j] % a == 0:
                    return False
        return True
    
    best = []
    pool = list(range(100, 257))
    random.seed(42)
    
    # Multiple greedy trials
    for trial in range(200):
        if time.time() > deadline:
            break
        order = pool[:]
        random.shuffle(order)
        S = []
        for x in order:
            if is_valid(S + [x]):
                S.append(x)
        if len(S) > len(best):
            best = S[:]
    
    # Try to extend
    for x in pool:
        if time.time() > deadline:
            break
        if x not in best and is_valid(best + [x]):
            best.append(x)
    
    return [int(x) for x in best]
