"""Find a 12-element set in [0,128] minimizing the largest Sidon subset (with diagonals)."""
import random
import time
from itertools import combinations

def _is_sidon_with_diag(subset):
    """Check if a sorted list has all pair sums (i<=j) unique."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size_exhaustive(arr):
    """Exact: largest subset of arr with all pair sums (i<=j) unique."""
    n = len(arr)
    best = 0
    # Try all subsets from largest size down
    # For n=12, C(12,6)=924, C(12,7)=792, etc.
    # Start from size 6 downward; if found, return
    for k in range(n, 0, -1):
        for combo in combinations(arr, k):
            if _is_sidon_with_diag(list(combo)):
                return k
    return 0

def _max_sidon_size_greedy(arr):
    """Greedy upper bound on max Sidon subset size."""
    # Greedily add elements that maintain Sidon property
    chosen = []
    for x in arr:
        test = chosen + [x]
        if _is_sidon_with_diag(test):
            chosen.append(x)
    return len(chosen)

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    best_set = list(range(12))
    best_size = _max_sidon_size_exhaustive(best_set)
    
    random.seed(42)
    
    def evaluate(s):
        """Return the exact max Sidon subset size for set s."""
        return _max_sidon_size_exhaustive(s)
    
    # Local search: swap two elements to reduce max Sidon size
    current = list(range(12))
    current_size = best_size
    
    # Try random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current = list(range(12))
        else:
            current = random.sample(range(129), 12)
        
        current_size = evaluate(current)
        if current_size < best_size:
            best_size = current_size
            best_set = list(current)
            if best_size <= 4:
                break
        
        # Local search: try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                for j in range(12):
                    if i == j:
                        continue
                    # Try replacing current[i] with a random new value
                    # Or swap with another position
                    # Let's try: replace current[i] with a random value in [0,128]
                    new_val = random.randint(0, 128)
                    if new_val in current:
                        continue
                    old_val = current[i]
                    current[i] = new_val
                    new_size = evaluate(current)
                    if new_size < current_size:
                        current_size = new_size
                        improved = True
                        if current_size < best_size:
                            best_size = current_size
                            best_set = list(current)
                            if best_size <= 4:
                                return best_set
                    else:
                        current[i] = old_val
        
        # Also try full swaps between two positions
        if time.time() > deadline:
            break
        for i in range(12):
            for j in range(i+1, 12):
                if time.time() > deadline:
                    break
                # Swap values
                current[i], current[j] = current[j], current[i]
                new_size = evaluate(current)
                if new_size < current_size:
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best_set = list(current)
                        if best_size <= 4:
                            return best_set
                else:
                    current[i], current[j] = current[j], current[i]
    
    return best_set
