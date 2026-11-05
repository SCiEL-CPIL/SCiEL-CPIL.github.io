import time
import random
import itertools

def is_sidon_fast(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def has_sidon_subset(A, k):
    """Check if A has a Sidon subset of size k. Returns True if found."""
    n = len(A)
    for combo in itertools.combinations(A, k):
        if is_sidon_fast(combo):
            return True
    return False

def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    for k in range(n, 0, -1):
        if has_sidon_subset(A, k):
            return k
    return 0

def count_pair_collisions(A):
    """Count the number of duplicate pair sums in A (higher = more collisions)."""
    n = len(A)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(A[i] + A[j])
    return len(sums) - len(set(sums))

def construct():
    start_time = time.time()
    deadline = 58.0
    
    # Initial candidate: Contiguous block
    current_A = list(range(12))
    best_A = list(current_A)
    best_val = max_sidon_size(best_A)
    
    iterations = 0
    no_improve = 0
    
    while time.time() < deadline:
        iterations += 1
        no_improve += 1
        
        # Create a perturbed copy
        A_new = list(current_A)
        idx = random.randint(0, 11)
        
        # Smart perturbation: try to create sum collisions
        # Pick a value that would create a collision with existing pairs
        strategy = random.random()
        
        if strategy < 0.4:
            # Small step in dense region
            step = random.randint(-4, 4)
            new_val = A_new[idx] + step
        elif strategy < 0.7:
            # Try to match a difference: if a_i - a_j = d, then setting some a_k = a_i - d creates collision
            # Look for existing differences and try to exploit them
            diffs = set()
            for i in range(12):
                for j in range(i+1, 12):
                    diffs.add(A_new[i] - A_new[j])
            if diffs:
                d = random.choice(list(diffs))
                # Try values that would create a_i + a_j = a_k + a_l
                # If we set A_new[idx] = x, we want x + A_new[a] = A_new[b] + A_new[c] for some a,b,c
                # So x = A_new[b] + A_new[c] - A_new[a]
                candidates = []
                for a in range(12):
                    if a == idx:
                        continue
                    for b in range(12):
                        if b == idx or b == a:
                            continue
                        for c in range(12):
                            if c == idx or c == a or c == b:
                                continue
                            x = A_new[b] + A_new[c] - A_new[a]
                            if 0 <= x <= 128 and x not in A_new:
                                candidates.append(x)
                if candidates:
                    new_val = random.choice(candidates)
                else:
                    new_val = A_new[idx] + random.randint(-5, 5)
            else:
                new_val = A_new[idx] + random.randint(-5, 5)
        else:
            # Random in local window
            min_val = min(current_A)
            max_val = max(current_A)
            new_val = random.randint(max(0, min_val - 8), min(128, max_val + 8))
        
        # Ensure validity
        if new_val < 0 or new_val > 128 or new_val in A_new:
            # Try a few more times
            for _ in range(5):
                new_val = random.randint(0, 128)
                if new_val not in A_new:
                    break
            else:
                continue
        
        A_new[idx] = new_val
        
        # Quick check: does A_new still have a Sidon subset of size best_val?
        # If yes, its max_sidon_size >= best_val
        # If no, its max_sidon_size < best_val, which is an improvement
        
        if best_val >= 2:
            # Check if there's a Sidon subset of size best_val
            if not has_sidon_subset(A_new, best_val):
                # Great! Max Sidon size is at most best_val - 1
                new_val_score = max_sidon_size(A_new)
                if new_val_score < best_val:
                    best_val = new_val_score
                    best_A = A_new
                    current_A = A_new
                    no_improve = 0
                    if best_val <= 3:
                        break
                else:
                    # Shouldn't happen, but accept if equal
                    if random.random() < 0.3:
                        current_A = A_new
            else:
                # Still has Sidon subset of size best_val, so max >= best_val
                # Check if it might be better by computing exact value only if promising
                # Use collision count as a heuristic: more collisions = likely smaller max Sidon
                if random.random() < 0.1:
                    new_val_score = max_sidon_size(A_new)
                    if new_val_score < best_val:
                        best_val = new_val_score
                        best_A = A_new
                        current_A = A_new
                        no_improve = 0
                        if best_val <= 3:
                            break
                    elif new_val_score == best_val:
                        if random.random() < 0.3:
                            current_A = A_new
        else:
            new_val_score = max_sidon_size(A_new)
            if new_val_score < best_val:
                best_val = new_val_score
                best_A = A_new
                current_A = A_new
                no_improve = 0
            elif new_val_score == best_val:
                if random.random() < 0.3:
                    current_A = A_new
        
        # Periodic restart if stuck
        if no_improve > 2000:
            current_A = list(best_A)
            # Aggressive perturbation
            for _ in range(random.randint(1, 3)):
                idx2 = random.randint(0, 11)
                A_temp = list(current_A)
                A_temp[idx2] = random.randint(0, 128)
                while A_temp.count(A_temp[idx2]) > 1:
                    A_temp[idx2] = random.randint(0, 128)
                current_A = A_temp
            no_improve = 0

    return best_A
