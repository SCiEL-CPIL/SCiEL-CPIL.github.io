import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    def get_subset_sums(S):
        sums = set()
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            sums |= new_sums
        return sums
    
    def is_valid(S):
        if not S:
            return True
        sums = get_subset_sums(S)
        sorted_sums = sorted(sums)
        n = len(sorted_sums)
        for i in range(n):
            a = sorted_sums[i]
            for j in range(i+1, n):
                if sorted_sums[j] % a == 0:
                    return False
        return True
    
    best_set = set()
    
    random.seed(42)
    
    # Phase 1: Greedy with random restarts
    for attempt in range(300):
        if time.time() > deadline - 15.0:
            break
        
        S = set()
        candidates = list(range(256, 0, -1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 10.0:
                break
            if len(S) >= 30:
                break
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
        
        if len(S) > len(best_set):
            best_set = S.copy()
    
    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            S = best_set.copy()
            if not S:
                break
            rem = random.choice(list(S))
            S.remove(rem)
            
            for x in random.sample(range(1, 257), 256):
                if x in S:
                    continue
                new_S = S | {x}
                if is_valid(new_S):
                    S = new_S
                    if len(S) > len(best_set):
                        break
            
            if len(S) > len(best_set):
                best_set = S.copy()
    
    return sorted(best_set)
