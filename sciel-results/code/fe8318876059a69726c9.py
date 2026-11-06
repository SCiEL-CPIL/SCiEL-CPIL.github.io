import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    def check_valid_addition(S, x, sums_set):
        """Check if adding x to S preserves unique triple sums. Returns (valid, new_sums) or (False, None)."""
        n = len(S)
        new_sums = []
        # (x,x,x)
        new_sums.append(3*x)
        # (x,x,s) for each s in S
        for s in S:
            new_sums.append(2*x + s)
        # (x,s,t) for s <= t in S
        for i in range(n):
            for j in range(i, n):
                new_sums.append(x + S[i] + S[j])
        
        # Check all new sums are not in existing sums_set
        for s in new_sums:
            if s in sums_set:
                return False, None
        
        # Check new sums are unique among themselves
        if len(new_sums) != len(set(new_sums)):
            return False, None
        
        return True, new_sums
    
    def greedy_forward(start_val=1, end_val=4096, step=1):
        S = []
        sums_set = set()
        for x in range(start_val, min(end_val+1, 4097), step):
            valid, new_sums = check_valid_addition(S, x, sums_set)
            if valid:
                S.append(x)
                for s in new_sums:
                    sums_set.add(s)
        return S, sums_set
    
    def local_search(S, sums_set, max_iterations=100):
        """Try to improve S by swapping elements or adding new ones."""
        S = list(S)
        sums_set = set(sums_set)
        n = len(S)
        
        # First try to add elements
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try adding a random element not in S
            added = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                x = random.randint(1, 4096)
                if x in S:
                    continue
                valid, new_sums = check_valid_addition(S, x, sums_set)
                if valid:
                    S.append(x)
                    for s in new_sums:
                        sums_set.add(s)
                    added = True
                    break
            
            if not added:
                break
        
        return S, sums_set
    
    best = []
    best_sums = set()
    
    # First greedy pass
    S, sums_set = greedy_forward(1, 4096, 1)
    if len(S) > len(best):
        best = S[:]
        best_sums = sums_set.copy()
    
    # Try different starting points and steps
    for start in range(1, 20):
        if time.time() > deadline:
            break
        S, sums_set = greedy_forward(start, 4096, 1)
        if len(S) > len(best):
            best = S[:]
            best_sums = sums_set.copy()
    
    # Randomized greedy: shuffle candidate order
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        # Create a shuffled list of candidates
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        S = []
        sums_set = set()
        for x in candidates:
            if x > 4096:
                continue
            valid, new_sums = check_valid_addition(S, x, sums_set)
            if valid:
                S.append(x)
                for s in new_sums:
                    sums_set.add(s)
        
        # Apply local search to improve
        S, sums_set = local_search(S, sums_set)
        
        if len(S) > len(best):
            best = S[:]
            best_sums = sums_set.copy()
    
    return best

# EVOLVE-BLOCK-END
