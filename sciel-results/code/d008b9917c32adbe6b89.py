import time
import random
import itertools

def is_sidon(subset):
    """Check if subset has all unordered pair sums (i<=j) unique."""
    s = sorted(subset)
    n = len(s)
    seen_sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in seen_sums:
                return False
            seen_sums.add(sm)
    return True

def max_sidon_size(S):
    """Find largest subset of S that is Sidon (all pair sums i<=j unique)."""
    S = sorted(S)
    n = len(S)
    # Check from largest possible size down
    for k in range(n, 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def count_collisions(S):
    """
    Count the number of 'Sidon-violating' sum collisions in the ambient set S.
    A collision occurs if s[i] + s[j] == s[k] + s[l] for distinct pairs.
    We count how many times a sum value appears more than once among all pairs (i<=j).
    Higher collision count generally implies smaller max Sidon subsets.
    """
    S = sorted(S)
    n = len(S)
    from collections import Counter
    sum_counts = Counter()
    for i in range(n):
        for j in range(i, n):
            sm = S[i] + S[j]
            sum_counts[sm] += 1
    
    # Total number of "excess" sums. If a sum appears c times, it contributes c-1 to the collision count.
    # Alternatively, sum of (c-1) for c > 1.
    collisions = 0
    for c in sum_counts.values():
        if c > 1:
            collisions += c - 1
    return collisions

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Initial best set: contiguous integers
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)
    best_collisions = count_collisions(best_set)
    
    # Strategy 1: Systematic search for dense sets with high collision counts
    # We know that dense sets (small range) have many collisions.
    # Let's explore sets within a small window, e.g., [0, 20] or [0, 30].
    
    # First, evaluate a few specific "structured" candidates that are known to be collision-heavy
    candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16],
        [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17],
        [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
    ]
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        v = max_sidon_size(cand)
        c = count_collisions(cand)
        if v < best_val or (v == best_val and c > best_collisions):
            best_val = v
            best_collisions = c
            best_set = sorted(cand)
            if best_val == 4:
                return best_set

    # Strategy 2: Stochastic Local Search (Simulated Annealing style)
    # Objective: Minimize max_sidon_size. 
    # Proxy: Maximize count_collisions.
    # We use a hybrid: primarily maximize collisions, but verify with max_sidon_size occasionally.
    
    current_set = best_set[:]
    current_val = best_val
    current_coll = best_collisions
    
    # Temperature for simulated annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    # We will perform a series of moves.
    # Move: Pick a random index i, and a random new value v in [0, 128].
    # If v is already in the set, skip or try again.
    
    # To speed up evaluation, we can pre-compute the set for fast membership checks.
    # But max_sidon_size is the expensive part.
    # Let's limit the frequency of max_sidon_size calls.
    # We will track the best (val, coll) pair.
    
    # We'll do a loop until time is up.
    while time.time() < deadline - 5:
        # Generate a neighbor
        neighbor = current_set[:]
        idx = random.randint(0, 11)
        old_val = neighbor[idx]
        
        # Choose a new value. Bias towards dense regions?
        # Let's try a mix: 50% small range [0, 30], 50% full range [0, 128]
        if random.random() < 0.5:
            new_val = random.randint(0, 30)
        else:
            new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val == old_val:
            continue
        if new_val in neighbor:
            # Try a few times to find a distinct value
            found = False
            for _ in range(10):
                if random.random() < 0.5:
                    new_val = random.randint(0, 30)
                else:
                    new_val = random.randint(0, 128)
                if new_val != old_val and new_val not in neighbor:
                    found = True
                    break
            if not found:
                continue
        
        neighbor[idx] = new_val
        neighbor = sorted(neighbor)
        
        # Evaluate neighbor
        # First, check collisions (fast)
        new_coll = count_collisions(neighbor)
        
        # Acceptance criteria:
        # 1. If new_coll > current_coll, accept.
        # 2. If new_coll == current_coll, maybe accept with some probability.
        # 3. If new_coll < current_coll, accept with probability exp((new_coll - current_coll) / T).
        
        delta_coll = new_coll - current_coll
        if delta_coll >= 0:
            accept = True
        else:
            # Boltzmann factor
            accept_prob = pow(2.71828, delta_coll / T)
            accept = random.random() < accept_prob
        
        if accept:
            current_set = neighbor
            current_coll = new_coll
            
            # Periodically check the true objective (max_sidon_size)
            # Do this every 50 moves or if collisions improved significantly
            if delta_coll > 5 or random.random() < 0.05:
                new_val = max_sidon_size(current_set)
                if new_val < best_val:
                    best_val = new_val
                    best_set = current_set[:]
                    best_collisions = current_coll
                    if best_val == 4:
                        return best_set
                # Update current_val for reference, though we primarily optimize collisions
                current_val = new_val
        
        # Cool down
        T = max(T_min, T * cooling_rate)

    # Final polish: Try to improve best_set with targeted local search
    if time.time() < deadline - 2:
        # Try swapping pairs in best_set
        for i in range(12):
            for j in range(i+1, 12):
                if time.time() > deadline - 2:
                    break
                orig_i = best_set[i]
                orig_j = best_set[j]
                
                # Try replacing i with a nearby value
                for delta in [-2, -1, 1, 2, 3, 5]:
                    new_i = orig_i + delta
                    if 0 <= new_i <= 128 and new_i != orig_j and new_i not in best_set:
                        cand = best_set[:]
                        cand[i] = new_i
                        cand = sorted(cand)
                        if len(set(cand)) == 12:
                            v = max_sidon_size(cand)
                            if v < best_val:
                                best_val = v
                                best_set = cand
                                if best_val == 4:
                                    return best_set
                                # Restart local search from new best
                                break
    
    return best_set
