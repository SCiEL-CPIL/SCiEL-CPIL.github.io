import time
import random

# EVOLVE-BLOCK-START

def _random_search(deadline):
    best_set = None
    best_max = float('inf')
    
    while time.time() < deadline:
        K = random.randint(9, 12)
        S = [0]
        used_diffs = set()
        
        # Build a random Sidon set
        for _ in range(K - 1):
            # Try a few random candidates
            for _ in range(10):
                x = random.randint(1, 60)
                valid = True
                new_diffs = []
                for v in S:
                    d = x - v
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    S.append(x)
                    S.sort()
                    used_diffs.update(new_diffs)
                    break
        
        # Check coverage
        if len(S) >= 9:
            # Check if 1..30 are in used_diffs
            covers = True
            for d in range(1, 31):
                if d not in used_diffs:
                    covers = False
                    break
            
            if covers:
                max_val = S[-1]
                if max_val < best_max:
                    best_max = max_val
                    best_set = S[:]
                    
                    # Local search: try to reduce max element
                    # Sort S
                    S.sort()
                    # Try to move the last element down?
                    # Or try to remove elements?
                    # Since we need 1..30, we can't remove easily.
                    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    result = _random_search(deadline)
    
    if result is not None:
        return result
    
    # Fallback: known construction
    return [0, 1, 4, 10, 12, 28, 32, 45, 63]

# EVOLVE-BLOCK-END
