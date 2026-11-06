import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial candidate: Union of interval and odds
    # S = {1..22} U {23, 25, ..., 41}
    # Size: 22 + 10 = 32
    # Covers d=1..20
    
    base_set = list(range(1, 23)) + [x for x in range(23, 42, 2)]
    current_score = _score_set(base_set)
    best_set = list(base_set)
    best_score = current_score
    
    # Local Search
    # We try to swap elements to increase score
    # Or add/remove if size allows? No, size must be 32.
    
    # Precompute all possible d up to 256
    max_d = 256
    
    def get_covered_diffs(S):
        S_set = set(S)
        covered = set()
        # To speed up, we can iterate d and check
        # But checking all d is slow.
        # Better: iterate pairs?
        # For each a, b in S, d = b-a.
        # Then check if a+2d in S.
        # This is O(N^2) per score calc. N=32. 1024 ops. Fast.
        for i in range(len(S)):
            for j in range(len(S)):
                if i == j: continue
                d = S[j] - S[i]
                if d > 0:
                    a = S[i]
                    if a + 2*d in S_set:
                        covered.add(d)
        return len(covered), covered

    # Optimize score calculation for speed
    def fast_score(S):
        S_set = set(S)
        covered = set()
        for a in S:
            for b in S:
                if b > a:
                    d = b - a
                    if a + 2*d in S_set:
                        covered.add(d)
        return len(covered)

    # Use fast_score for iterations
    best_score = fast_score(best_set)
    
    current_set = list(best_set)
    current_score = best_score
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all swaps
        # To avoid O(N^2 * ScoreCost) being too slow, we can randomize or limit
        # N=32, swaps ~ 32*500 = 16000. Score ~ 1000 ops. 16M ops. OK in Python?
        # Might be slow. 120s limit.
        # Let's limit swaps per iteration or use random sampling.
        
        # Random sampling of swaps
        for _ in range(2000):
            if time.time() > deadline:
                break
            
            # Pick two random indices
            i = random.randint(0, 31)
            j = random.randint(0, 31)
            if i == j:
                continue
            
            x = current_set[i]
            y = current_set[j]
            
            # Pick a new element for x
            # Sample from 1..512
            new_x = random.randint(1, 512)
            if new_x in current_set:
                continue
            
            # Create new set
            new_set = list(current_set)
            new_set[i] = new_x
            
            # Check validity: distinct
            if len(set(new_set)) != 32:
                continue
            
            new_sc = fast_score(new_set)
            if new_sc > current_score:
                current_set = new_set
                current_score = new_sc
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            elif new_sc == current_score:
                # Accept with some probability (Simulated Annealing style)
                # Or just accept to explore
                if random.random() < 0.1:
                    current_set = new_set
                    improved = True
                    
    return best_set

def _score_set(S):
    S_set = set(S)
    covered = set()
    for a in S:
        for b in S:
            if b > a:
                d = b - a
                if a + 2*d in S_set:
                    covered.add(d)
    return len(covered)
