import random
import math
import time
from itertools import combinations

def construct():
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples
    all_possible_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_possible_triples.append((i, j, k))
                
    # Helper to check if a triple is valid (linear) with respect to a set of triples
    def is_valid(triple, existing):
        t_set = set(triple)
        for e in existing:
            e_set = set(e)
            if len(t_set & e_set) > 1:
                return False
        return True

    # Helper to calculate independence number
    # Optimized: Iterate by size descending
    def get_independence_number(triples):
        if not triples:
            return n
            
        # Precompute triple masks
        t_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            t_masks.append(mask)
        
        # Iterate sizes from n down to 1
        for size in range(n, 0, -1):
            # Generate all masks of this size?
            # It's faster to iterate all masks and filter by popcount?
            # Or use combinations to generate indices and then masks?
            # combinations(range(n), size) generates tuples of indices.
            # We can convert to mask.
            # Number of combinations is C(16, size).
            for comb in combinations(range(n), size):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                
                is_ind = True
                for tm in t_masks:
                    if (mask & tm) == tm:
                        is_ind = False
                        break
                if is_ind:
                    return size
        return 0

    # Initial state: Greedy construction
    initial_triples = []
    shuffled_triples = all_possible_triples[:]
    random.shuffle(shuffled_triples)
    for t in shuffled_triples:
        if is_valid(t, initial_triples):
            initial_triples.append(t)
            if len(initial_triples) >= 37:
                break

    # Simulated Annealing
    best_triples = initial_triples[:]
    best_alpha = get_independence_number(best_triples)
    
    current_triples = initial_triples[:]
    current_alpha = best_alpha
    
    start_time = time.time()
    timeout = 110 # Leave 10s buffer
    
    T = 1.0
    cooling = 0.999
    
    # Precompute a list of triples for random selection
    # We might want to bias towards triples that are "hard" to add or something?
    # For now, random from all_possible_triples.
    
    while time.time() - start_time < timeout:
        # Generate a move
        # Option 1: Remove a random triple
        # Option 2: Add a random valid triple
        # Option 3: Swap
        
        move_type = random.choice(['remove', 'add', 'swap'])
        
        new_triples = current_triples[:]
        
        if move_type == 'remove' and len(new_triples) > 0:
            idx = random.randint(0, len(new_triples) - 1)
            new_triples.pop(idx)
        elif move_type == 'add':
            added = False
            for _ in range(20): # Increased attempts
                cand = random.choice(all_possible_triples)
                # Check if already in new_triples (using tuple for speed)
                if tuple(cand) not in [tuple(t) for t in new_triples]: # This is slow
                    # Better: maintain a set of current triples?
                    pass
            # Optimization: Maintain a set of current triples for O(1) lookup
            # But we are modifying the list.
            # Let's just do a linear check but with tuples.
            # Or better, just check validity which implies uniqueness if we check against existing?
            # No, is_valid doesn't check for exact duplicates, just intersection > 1.
            # A duplicate triple has intersection 3 > 1, so it would be rejected by is_valid?
            # Wait, if t is in existing, t_set & e_set = t_set, len 3 > 1. So is_valid returns False.
            # So we don't need to explicitly check for duplicates if we use is_valid.
            
            added = False
            for _ in range(20):
                cand = random.choice(all_possible_triples)
                if is_valid(cand, new_triples):
                    new_triples.append(cand)
                    added = True
                    break
            if not added:
                continue
        elif move_type == 'swap':
            if len(new_triples) == 0:
                continue
            idx = random.randint(0, len(new_triples) - 1)
            removed_triple = new_triples.pop(idx)
            added = False
            for _ in range(20):
                cand = random.choice(all_possible_triples)
                if cand != removed_triple:
                    if is_valid(cand, new_triples):
                        new_triples.append(cand)
                        added = True
                        break
            if not added:
                new_triples.insert(idx, removed_triple)
                continue

        new_alpha = get_independence_number(new_triples)
        
        if new_alpha < current_alpha:
            current_triples = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = new_triples[:]
        else:
            delta = new_alpha - current_alpha
            if delta == 0:
                if random.random() < 0.5:
                    current_triples = new_triples
            else:
                probability = math.exp(-delta / T)
                if random.random() < probability:
                    current_triples = new_triples
                    current_alpha = new_alpha
        
        T *= cooling

    return best_triples
