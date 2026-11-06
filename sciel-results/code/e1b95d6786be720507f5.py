import random
import time
import itertools

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    n = len(S)
    # Sort S to help with pruning? Not strictly necessary for set sums.
    # Iterate k from n down to 1
    for k in range(n, 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(comb):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial good candidates
    # Based on the idea of creating many sum collisions.
    # A set with small diameter might have small Sidon subsets?
    # No, small diameter [0, 11] has Sidon subsets of size ~4.
    # What about a set that is "dense" but structured?
    
    best_set = None
    best_score = 13
    
    # Heuristic 1: Random
    # Heuristic 2: Two clusters
    # Heuristic 3: Arithmetic Progressions
    
    initial_candidates = [
        list(range(12)),
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 4, 8, 16, 32, 64, 1, 3, 5, 7] # Mixed
    ]
    
    for _ in range(100):
        initial_candidates.append(random.sample(range(129), 12))
        
    for c in initial_candidates:
        score = max_sidon_size(c)
        if score < best_score:
            best_score = score
            best_set = c
        if time.time() > deadline:
            break
            
    current_set = best_set
    current_score = best_score
    
    # Local Search
    # We will try to swap elements or move elements.
    # To keep it fast, we only try a subset of possible moves.
    
    iterations = 0
    while time.time() < deadline:
        improved = False
        iterations += 1
        
        # Try single element replacement
        # Pick a random index and a random new value
        # Or systematically? Systematic is better for convergence.
        
        # To avoid O(N*V) full scan, we can randomize the order or limit steps.
        # Let's try all indices, but only a few new values?
        # No, full scan is 12*129 = 1548 evals. ~10ms.
        # We can do ~50 iterations per second.
        # 85s -> 4000 iterations.
        
        for i in range(12):
            old_val = current_set[i]
            # Try all possible new values
            # Optimization: skip if new_val is in set
            others = set(current_set)
            others.remove(old_val)
            
            for new_val in range(129):
                if new_val in others:
                    continue
                
                new_set = current_set[:]
                new_set[i] = new_val
                score = max_sidon_size(new_set)
                
                if score < current_score:
                    current_set = new_set
                    current_score = score
                    improved = True
                    # If we found a better score, restart the loop?
                    # Or continue?
                    # Continuing might find even better.
                    # But if we drop to 3, we can stop.
                    if current_score <= 3:
                        return sorted(current_set)
                    
            if time.time() > deadline:
                break
        
        if not improved:
            # Try swaps
            # This is more expensive.
            # Maybe just random perturbations if no improvement?
            if random.random() < 0.5:
                # Random swap
                i, j = random.sample(range(12), 2)
                # Try swapping with random values
                # Or just perturb one
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in current_set:
                    current_set[i] = new_val
                    current_score = max_sidon_size(current_set)
            else:
                # Random restart
                new_set = random.sample(range(129), 12)
                score = max_sidon_size(new_set)
                if score < current_score:
                    current_set = new_set
                    current_score = score
                elif iterations > 50:
                    # Stuck, try harder random restarts
                    for _ in range(10):
                        new_set = random.sample(range(129), 12)
                        score = max_sidon_size(new_set)
                        if score < current_score:
                            current_set = new_set
                            current_score = score
                            break

    return sorted(current_set)
