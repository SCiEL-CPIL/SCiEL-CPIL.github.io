import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    
    def run_greedy(order):
        """
        Greedily build a set from 'order' such that for any two subsets of different
        cardinalities, their sums are distinct.
        """
        S = [set() for _ in range(61)]
        S[0] = {0}
        A = []
        current_size = 0
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            
            valid = True
            for j in range(1, current_size + 2):
                if not S[j-1]:
                    continue
                for s in S[j-1]:
                    candidate = s + x
                    for i in range(current_size + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                A.append(x)
                current_size += 1
                for k in range(current_size, 0, -1):
                    S[k].update(s + x for s in S[k-1])
        
        return A

    # Deterministic runs with various orderings
    best_set = run_greedy(list(range(1, 257)))
    
    # Try reverse order
    candidate = run_greedy(list(range(256, 0, -1)))
    if len(candidate) > len(best_set):
        best_set = candidate
    
    # Try alternating patterns
    candidate = run_greedy(list(range(128, 0, -1)) + list(range(129, 257)))
    if len(candidate) > len(best_set):
        best_set = candidate
    
    candidate = run_greedy(list(range(129, 257)) + list(range(128, 0, -1)))
    if len(candidate) > len(best_set):
        best_set = candidate
    
    # Randomized runs
    rng = random.Random(42)
    while time.time() - start_time < deadline:
        order = list(range(1, 257))
        rng.shuffle(order)
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Try to extend the best set found by trying to add more elements
    if time.time() - start_time < deadline:
        current_set = set(best_set)
        S = [set() for _ in range(61)]
        S[0] = {0}
        current_size = len(best_set)
        
        # Rebuild S from current_set
        for k in range(1, current_size + 1):
            for combo in combinations(best_set, k):
                S[k].add(sum(combo))
        
        # Try adding each element from 1-256 not in current_set
        for x in range(1, 257):
            if x in current_set:
                continue
            if time.time() - start_time > deadline:
                break
            
            valid = True
            for j in range(1, current_size + 2):
                if not S[j-1]:
                    continue
                for s in S[j-1]:
                    candidate = s + x
                    for i in range(current_size + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_set.append(x)
                current_set.add(x)
                current_size += 1
                for k in range(current_size, 0, -1):
                    S[k].update(s + x for s in S[k-1])
    
    return [int(x) for x in best_set]
