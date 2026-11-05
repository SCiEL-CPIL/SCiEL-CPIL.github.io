"""Combined approach: structural insight from inspiration 2 (dense contiguous sets minimize Sidon size)
with refinement machinery from inspirations 1 and 3 (local search with biased perturbation and threshold checking)."""
import time
import random
import itertools

def _is_sidon(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _has_sidon_subset(A, k):
    """
    Check if A has ANY Sidon subset of size k.
    Returns True if found, False otherwise.
    Fast threshold check: if True, max Sidon size >= k.
    """
    n = len(A)
    if k > n:
        return False
    for combo in itertools.combinations(A, k):
        if _is_sidon(combo):
            return True
    return False

def _max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    for k in range(n, 0, -1):
        if _has_sidon_subset(A, k):
            return k
    return 0

def _count_duplicate_sums(A):
    """
    Count the number of duplicate pair sums in A.
    Higher count means more collisions, which is good for minimizing Sidon size.
    """
    sums = {}
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            sums[s] = sums.get(s, 0) + 1
    dup_count = sum(v - 1 for v in sums.values() if v > 1)
    return dup_count

def construct():
    start_time = time.time()
    # Safety margin: stop 5 seconds before 60s limit
    deadline = start_time + 55.0
    
    # Initial candidate: Contiguous block is the densest possible set of 12 distinct integers.
    # Density maximizes sum collisions, which minimizes the size of the largest Sidon subset.
    current_A = list(range(12))
    best_A = list(current_A)
    best_val = _max_sidon_size(best_A)  # Expected 4 or 5
    
    # We want to minimize best_val.
    # If best_val is already 4, we try to find 3.
    # If best_val is 5, we try to find 4.
    
    iterations = 0
    no_improve_count = 0
    best_dup_count = _count_duplicate_sums(best_A)
    
    while time.time() < deadline:
        iterations += 1
        
        # Create a perturbed copy
        A_new = list(current_A)
        idx = random.randint(0, 11)
        
        # Biased perturbation: try to keep the set dense
        # Strategy 1: Small step (70% probability)
        # Strategy 2: Random value in a small window around current set's min/max (30% probability)
        
        if random.random() < 0.7:
            step = random.randint(-4, 4)
            new_val = A_new[idx] + step
        else:
            min_val = min(current_A)
            max_val = max(current_A)
            # Keep in a tight window
            new_val = random.randint(max(0, min_val - 5), min(128, max_val + 5))
        
        # Ensure validity: [0, 128] and distinct
        if new_val < 0 or new_val > 128:
            new_val = random.randint(0, 128)
            
        if new_val in A_new:
            # Try to resolve conflict by picking another random value in the dense region
            attempts = 0
            while new_val in A_new and attempts < 10:
                min_val = min(current_A)
                max_val = max(current_A)
                new_val = random.randint(max(0, min_val - 3), min(128, max_val + 3))
                attempts += 1
            if new_val in A_new:
                continue
        
        A_new[idx] = new_val
        A_new.sort()  # Keep sorted for consistency
        
        # Fast threshold check: if best_val is 4, check if A_new has Sidon subset of size 4
        # If not, then max Sidon size <= 3, which is an improvement
        # If yes, check if it has size 5 (if best_val is 5, we care about 4)
        
        if best_val == 5:
            # Check if A_new has Sidon subset of size 5
            if _has_sidon_subset(A_new, 5):
                # Max Sidon size >= 5, no improvement possible with this move
                # But check if it's exactly 5 (no size 6)
                new_val_score = 5  # At least 5, but we know it's at most 12
                # For efficiency, assume it's 5 if it has size 5 but not 6
                # We can refine later if needed
                if new_val_score < best_val:
                    best_val = new_val_score
                    best_A = A_new
                    current_A = A_new
                    no_improve_count = 0
                elif new_val_score == best_val:
                    # Accept with some probability to allow movement in the plateau
                    if random.random() < 0.3:
                        current_A = A_new
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # No Sidon subset of size 5, so max Sidon size <= 4
                # Calculate exact max Sidon size
                new_val_score = _max_sidon_size(A_new)
                if new_val_score < best_val:
                    best_val = new_val_score
                    best_A = A_new
                    current_A = A_new
                    no_improve_count = 0
                    if best_val <= 3:
                        break
                elif new_val_score == best_val:
                    if random.random() < 0.3:
                        current_A = A_new
                    no_improve_count += 1
                else:
                    no_improve_count += 1
        elif best_val == 4:
            # Check if A_new has Sidon subset of size 4
            if _has_sidon_subset(A_new, 4):
                # Max Sidon size >= 4, no improvement possible
                new_val_score = 4  # At least 4
                if new_val_score == best_val:
                    # Accept with some probability
                    if random.random() < 0.3:
                        current_A = A_new
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # No Sidon subset of size 4, so max Sidon size <= 3
                new_val_score = _max_sidon_size(A_new)
                if new_val_score < best_val:
                    best_val = new_val_score
                    best_A = A_new
                    current_A = A_new
                    no_improve_count = 0
                    if best_val <= 2:
                        break
                else:
                    no_improve_count += 1
        else:
            # best_val <= 3, try to find even smaller
            new_val_score = _max_sidon_size(A_new)
            if new_val_score < best_val:
                best_val = new_val_score
                best_A = A_new
                current_A = A_new
                no_improve_count = 0
            else:
                if random.random() < 0.2:
                    current_A = A_new
                no_improve_count += 1
        
        # Periodic reset if stuck
        if no_improve_count > 5000:
            # Reset current_A to best_A with aggressive perturbation
            current_A = list(best_A)
            for _ in range(3):
                idx = random.randint(0, 11)
                A_new = list(current_A)
                A_new[idx] = random.randint(0, 128)
                while A_new.count(A_new[idx]) > 1:
                    A_new[idx] = random.randint(0, 128)
                current_A = A_new
            no_improve_count = 0
    
    return best_A
