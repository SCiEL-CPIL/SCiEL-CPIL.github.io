import time
import random

def construct():
    n = 12
    # Seed solution: Powers of 2
    seed_seq = [1 << i for i in range(n)]
    max_seed = seed_seq[-1]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    # Helper to check if a sequence has distinct subset sums
    def has_distinct_sums(seq):
        sums = set([0])
        for x in seq:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return len(sums) == (1 << len(seq))

    # Try to improve on the seed by searching for a sequence with max < max_seed
    # We'll try a random local search or greedy reduction
    
    best_seq = seed_seq
    best_max = max_seed
    
    # Heuristic: Try to reduce the largest element by adjusting the previous ones
    # This is a simplified search. A full search is too expensive.
    # We'll try a few random perturbations of the seed sequence.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        # Create a copy of the seed sequence
        seq = seed_seq[:]
        
        # Randomly perturb the last few elements
        # Ensure they are positive integers
        for i in range(8, 12):
            # Small random change
            if random.random() < 0.5:
                seq[i] += random.randint(-10, 10)
            else:
                seq[i] = random.randint(1, best_max)
        
        # Ensure distinctness and positive
        if len(set(seq)) != 12:
            continue
        if any(x <= 0 for x in seq):
            continue
            
        # Sort to maintain a canonical form? The problem doesn't require sorted, 
        # but distinct subset sums is invariant to order.
        # However, for the "max element" metric, order doesn't matter.
        
        if max(seq) < best_max and has_distinct_sums(seq):
            best_seq = seq
            best_max = max(seq)
            # If we find a significantly better one, we could stop, 
            # but let's keep searching a bit to see if we can go lower.
            if best_max < 2040:
                break

    return best_seq
