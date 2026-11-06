import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 180
    
    # Precompute conflict pairs
    conflict_pairs = {}
    for a in range(1, N+1):
        a2 = a * a
        pairs = set()
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                d1 = d
                d2 = a2 // d
                b1 = a + d1
                c1 = a + d2
                if b1 != c1 and 1 <= b1 <= N and 1 <= c1 <= N:
                    pairs.add((min(b1,c1), max(b1,c1)))
            d += 1
        conflict_pairs[a] = list(pairs)
    
    def is_valid(S):
        S_set = set(S)
        for a in S:
            for (b, c) in conflict_pairs[a]:
                if b in S_set and c in S_set:
                    return False
        return True
    
    def add_element(S, a):
        pairs = conflict_pairs[a]
        conflicting = [(b, c) for (b, c) in pairs if b in S and c in S]
        if not conflicting:
            S.add(a)
            return S
        while True:
            if time.time() > deadline:
                break
            conflicting = [(b, c) for (b, c) in pairs if b in S and c in S]
            if not conflicting:
                break
            elem_count = Counter()
            for (b, c) in conflicting:
                elem_count[b] += 1
                elem_count[c] += 1
            to_remove = max(elem_count, key=lambda x: elem_count[x])
            S.discard(to_remove)
        S.add(a)
        return S
    
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Phase 1: Try different random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        S = set(range(91, 181))
        candidates = list(range(1, 91))
        random.shuffle(candidates)
        
        for a in candidates:
            if time.time() > deadline:
                break
            add_element(S, a)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Phase 2: Local search on the best set
    S = set(best_set)
    for _ in range(2000):
        if time.time() > deadline:
            break
        if not S:
            break
        to_remove = random.choice(list(S))
        S.discard(to_remove)
        for _ in range(30):
            a = random.randint(1, N)
            if a not in S:
                add_element(S, a)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    if not is_valid(best_set):
        return sorted(range(91, 181))
    
    return sorted(best_set) if best_set else [91]

