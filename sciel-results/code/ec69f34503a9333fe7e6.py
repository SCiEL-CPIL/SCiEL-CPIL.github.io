import time
import random
from itertools import combinations

def _is_sidon_4(subset):
    """Check if a 4-element subset is Sidon (including diagonals)."""
    a, b, c, d = sorted(subset)
    sums = [
        2*a,
        a+b, a+c, a+d,
        2*b, b+c, b+d,
        2*c, c+d,
        2*d
    ]
    return len(set(sums)) == 10

def _is_sidon(subset):
    """Check if a subset is Sidon (including diagonals)."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _has_sidon_4(A):
    """Check if A contains any 4-element Sidon subset. Returns (bool, subset)."""
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _is_sidon_4((A[i], A[j], A[k], A[l])):
                        return True, (A[i], A[j], A[k], A[l])
    return False, None

def _max_sidon_size(A):
    """Calculate exact max Sidon size for a set of 12."""
    n = len(A)
    # Check from largest down
    for k in range(n, 3, -1):
        for comb in combinations(A, k):
            if _is_sidon(comb):
                return k
    # If no size > 3, check size 4
    has4, _ = _has_sidon_4(A)
    if has4:
        return 4
    # Check size 3 (always exists if n>=3? Not necessarily, but likely)
    # Actually, any 3 elements might not be Sidon. 
    # But the task asks for the MAX. If max is <=3, we just need to know if it's 3 or less.
    # For the purpose of the evaluator, if we return a set, it computes the exact max.
    # Our local search only needs to know if it's 3 or 4+.
    # So we can return 3 if no 4 is found.
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: evaluator kills at 90s
    
    # Strategy:
    # 1. Quick random search for a set with NO 4-Sidon subset (max size 3).
    # 2. If not found, use local search to minimize the number of 4-Sidon subsets,
    #    hoping to eliminate them all.
    # 3. Fallback: return the best set found (lowest max Sidon size).
    
    best_set = list(range(12))
    best_val = _max_sidon_size(best_set)
    
    rng = random.Random(42)
    
    # Phase 1: Random Search
    # We want to find a set where NO 4-element subset is Sidon.
    # This is the primary goal.
    for seed in range(5000):
        if time.time() > deadline:
            break
        
        current_set = sorted(rng.sample(range(0, 129), 12))
        
        has4, _ = _has_sidon_4(current_set)
        
        if not has4:
            # Found a set with max Sidon size <= 3
            # Verify exact max size just in case (should be 3 or less)
            val = _max_sidon_size(current_set)
            if val < best_val:
                best_val = val
                best_set = current_set
            return current_set
    
    # Phase 2: Local Search
    # If random search failed, try to improve a random set by removing 4-Sidon subsets.
    # Objective: Minimize the number of 4-Sidon subsets. If we reach 0, we win.
    
    for start_seed in range(50):
        if time.time() > deadline:
            break
        
        rng2 = random.Random(start_seed)
        current_set = sorted(rng2.sample(range(0, 129), 12))
        
        # Count 4-Sidon subsets for current set
        def _count_sidon_4(A):
            count = 0
            n = len(A)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            if _is_sidon_4((A[i], A[j], A[k], A[l])):
                                count += 1
            return count

        current_count = _count_sidon_4(current_set)
        best_local_set = current_set[:]
        best_local_count = current_count
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            if current_count == 0:
                # Found a set with no 4-Sidon subsets
                val = _max_sidon_size(current_set)
                if val < best_val:
                    best_val = val
                    best_set = current_set
                return current_set
            
            # Pick a random 4-Sidon subset to break
            # We need to find one.
            has4, sidon_sub = _has_sidon_4(current_set)
            if not has4:
                # Should have been caught above, but just in case
                val = _max_sidon_size(current_set)
                if val < best_val:
                    best_val = val
                    best_set = current_set
                return current_set
            
            # Pick one element from the Sidon subset to perturb
            # Try replacing it with a random value
            improved = False
            # Try multiple times to find a good replacement
            for _ in range(10):
                idx_to_change = rng2.choice([0, 1, 2, 3])
                val_to_change = sidon_sub[idx_to_change]
                
                # Find index in current_set
                pos = current_set.index(val_to_change)
                
                # Try several random new values
                for _ in range(5):
                    new_val = rng2.randint(0, 128)
                    if new_val == val_to_change:
                        continue
                    if new_val in current_set:
                        continue
                    
                    new_set = current_set[:]
                    new_set[pos] = new_val
                    new_set.sort()
                    
                    new_count = _count_sidon_4(new_set)
                    
                    if new_count < current_count:
                        current_set = new_set
                        current_count = new_count
                        improved = True
                        
                        if current_count < best_local_count:
                            best_local_count = current_count
                            best_local_set = current_set[:]
                        
                        if current_count == 0:
                            val = _max_sidon_size(current_set)
                            if val < best_val:
                                best_val = val
                                best_set = current_set
                            return current_set
                        break
                    elif new_count == current_count and rng2.random() < 0.3:
                        # Accept with some probability to escape local minima
                        current_set = new_set
                        improved = True
                        break
            
            if not improved:
                # Random restart within this local search
                current_set = sorted(rng2.sample(range(0, 129), 12))
                current_count = _count_sidon_4(current_set)
                if current_count < best_local_count:
                    best_local_count = current_count
                    best_local_set = current_set[:]
    
    # Return the best set found
    return best_set
