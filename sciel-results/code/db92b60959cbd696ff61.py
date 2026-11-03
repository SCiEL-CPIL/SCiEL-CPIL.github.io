import itertools
import math
import time

def _score_of(U):
    """Calculate the score for a given set U."""
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    max_u = max(U)
    if max_u == 0:
        return 1.0 # Limit case
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    size_sum = len(sums)
    size_diff = len(diffs)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_set = [0, 1, 4, 9, 15]
    best_score = _score_of(best_set)
    
    # Exhaustive search for small N and M
    # N is size of set, M is max value
    for N in range(2, 8):
        for M in range(1, 25):
            if time.time() > deadline:
                return best_set
            
            # Choose N-1 numbers from 1..M
            # Include 0 always
            candidates = list(range(1, M + 1))
            
            # Optimization: if N is large and M is small, skip
            if M < N - 1:
                continue
                
            # To reduce search space, we can fix 0 and search for others
            # But itertools.combinations is fast for small M
            # However, for N=7, M=24, C(24,6) ~ 134k. 7*24*134k ~ 22M ops.
            # This might be slow in Python.
            
            # Let's limit M based on N
            if N > 5:
                M_limit = 20
            else:
                M_limit = 30
                
            if M > M_limit:
                continue

            for combo in itertools.combinations(candidates, N - 1):
                if time.time() > deadline:
                    return best_set
                    
                U = [0] + list(combo)
                # Quick check: if max U is same as before but ratio worse, skip?
                # Just compute score
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    # Local search / Random restart for larger sets if time permits
    if time.time() < deadline - 20:
        import random
        random.seed(42)
        
        for _ in range(50): # 50 restarts
            if time.time() > deadline:
                break
                
            # Start with a random set of size 5 or 6
            N = random.choice([5, 6, 7])
            M_max = random.randint(20, 50)
            U = [0] + sorted(random.sample(range(1, M_max + 1), N - 1))
            
            current_score = _score_of(U)
            if current_score > best_score:
                best_score = current_score
                best_set = U[:]
                
            # Local improvement
            for _ in range(100): # 100 steps
                if time.time() > deadline:
                    break
                    
                # Pick an index to change
                idx = random.randint(1, len(U) - 1) # Don't change 0
                old_val = U[idx]
                
                # Try changing to a random value
                new_val = random.randint(1, 60)
                
                # Ensure uniqueness
                if new_val in U:
                    continue
                    
                U[idx] = new_val
                U.sort()
                
                new_score = _score_of(U)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_set = U[:]
                else:
                    # Revert
                    U[idx] = old_val
                    U.sort()
                    
    return best_set
