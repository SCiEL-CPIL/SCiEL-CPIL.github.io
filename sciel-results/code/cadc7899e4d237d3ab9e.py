import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    best_set = []
    best_score = 0.0
    
    # Helper to check if adding b to S is valid
    def can_add(b, S):
        # S is a set
        # Check if b creates a 3-AP with any two elements in S
        # 1. b is middle: u, b, w => u + w = 2b => w = 2b - u
        for u in S:
            w = 2 * b - u
            if w in S and w != u:
                return False
        # 2. b is end: u, v, b => u + b = 2v => u = 2v - b
        #    OR b, v, w => b + w = 2v => w = 2v - b
        #    In both cases, for some v in S, the other end is 2v - b.
        for v in S:
            other = 2 * v - b
            if other in S and other != v and other != b:
                return False
        return True

    # Helper to calculate score
    def score(S):
        return sum(1.0 / x for x in S)

    # Greedy initialization
    def greedy_init(randomize=False):
        S = set()
        candidates = list(range(1, n + 1))
        if randomize:
            # Add noise to priority
            priorities = {x: 1.0/x + random.random()*0.01 for x in candidates}
            candidates.sort(key=lambda x: -priorities[x])
        else:
            candidates.sort(key=lambda x: x) # Smallest first for max reciprocal
        
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S

    # Local search
    def local_search(S):
        S = set(S)
        current_score = score(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Sort elements to try removing
            # We want to replace large x (small weight) with small y (large weight)
            # So iterate a in S sorted by 1/a ascending (largest a first)
            # And b not in S sorted by 1/b descending (smallest b first)
            
            # To avoid O(N^2) full scan every time, we can just scan.
            # N=256 is small.
            list_S = list(S)
            list_S.sort(key=lambda x: x) # Largest a at end? No, we want to try removing largest a first?
            # Actually, if we remove a large a, we can add a small b.
            # Let's just iterate all a in S.
            
            # Precompute list of not in S
            not_in_S = [x for x in range(1, n+1) if x not in S]
            not_in_S.sort(key=lambda x: x) # Smallest b first
            
            for a in list(S):
                if time.time() > deadline:
                    return S, current_score
                # Try to find a better b to swap with a
                # We only care about b such that 1/b > 1/a => b < a
                # So we only need to check b in not_in_S where b < a
                for b in not_in_S:
                    if b >= a:
                        break # Since not_in_S is sorted ascending, all subsequent b are >= a, so 1/b <= 1/a
                    # Check if S \ {a} U {b} is valid
                    S_temp = S.copy()
                    S_temp.remove(a)
                    if can_add(b, S_temp):
                        # Valid swap
                        S_temp.add(b)
                        new_score = current_score - 1.0/a + 1.0/b
                        if new_score > current_score:
                            S = S_temp
                            current_score = new_score
                            improved = True
                            # Restart search with new S
                            # Update not_in_S? It's easier to just break and let outer loop restart
                            # But we are in nested loops. 
                            # We can just return from local_search and let the main loop call it again?
                            # Or break out.
                            # Let's break out of all loops by returning
                            return local_search(S)
            # If we finished loops without improvement, break
            if not improved:
                break
        
        return S, current_score

    # Run multiple restarts
    # 1. Deterministic greedy
    S0 = greedy_init(randomize=False)
    S_best, score_best = local_search(S0)
    best_set = S_best
    best_score = score_best
    
    # 2. Randomized restarts
    while time.time() < deadline:
        S_rand = greedy_init(randomize=True)
        S_cand, score_cand = local_search(S_rand)
        if score_cand > best_score:
            best_set = S_cand
            best_score = score_cand

    return list(best_set)

# EVOLVE-BLOCK-END
