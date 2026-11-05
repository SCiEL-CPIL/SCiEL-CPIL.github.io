import time
import random
import itertools

def is_sidon(subset):
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

def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    # Check from largest possible size downwards
    for k in range(n, 0, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial best candidate
    # Start with a dense set, which tends to have smaller max Sidon subsets
    best_candidate = list(range(12))
    best_score = max_sidon_size(best_candidate) # Should be 4
    
    # If we already found 3, we can't do better
    if best_score <= 3:
        return best_candidate

    random.seed(42)
    
    # Local search
    # We will try to find a set with max Sidon size 3.
    # Strategy: Start with a dense set, and randomly swap elements.
    # Bias towards keeping the set dense to avoid creating large Sidon subsets.
    
    current_candidate = best_candidate[:]
    current_score = best_score
    
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() < deadline:
        # Pick a random index to change
        idx = random.randint(0, 11)
        old_val = current_candidate[idx]
        
        # Generate a new value
        # Bias towards values close to the current set to maintain density
        # Pick a random center from the current set
        center = random.choice(current_candidate)
        # Offset in a small range
        offset = random.randint(-15, 15)
        new_val = center + offset
        
        # Ensure valid range and distinctness
        if new_val < 0 or new_val > 128:
            new_val = random.randint(0, 128)
            
        if new_val in current_candidate and new_val != old_val:
            continue
            
        # Apply change
        current_candidate[idx] = new_val
        
        # Evaluate
        new_score = max_sidon_size(current_candidate)
        
        if new_score < best_score:
            best_score = new_score
            best_candidate = current_candidate[:]
            no_improvement_count = 0
            if best_score <= 3:
                break
        elif new_score == best_score:
            # Accept equal score to explore plateau
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Restart with a new random dense set
                base = random.randint(0, 116)
                current_candidate = list(range(base, base + 12))
                current_score = max_sidon_size(current_candidate)
                no_improvement_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_candidate = current_candidate[:]
                    if best_score <= 3:
                        break
        else:
            # Revert
            current_candidate[idx] = old_val
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Restart
                base = random.randint(0, 116)
                current_candidate = list(range(base, base + 12))
                current_score = max_sidon_size(current_candidate)
                no_improvement_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_candidate = current_candidate[:]
                    if best_score <= 3:
                        break

    return best_candidate
