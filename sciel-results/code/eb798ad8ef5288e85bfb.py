import random
import math
import time

def construct():
    n = 16
    points = list(range(n))
    
    # Helper to check if a triple is valid (linear) with respect to a set of triples
    def is_valid(triple, existing):
        t_set = set(triple)
        for e in existing:
            e_set = set(e)
            if len(t_set & e_set) > 1:
                return False
        return True

    # Helper to calculate independence number
    # A subset S is independent if it contains no triple from 'triples'
    # We can iterate over all masks 0 to 2^n - 1
    # But we only care about the max size.
    # We can sort masks by popcount descending and return the first valid one.
    # Or just iterate all and keep max.
    # 2^16 = 65536. This is fast.
    def get_independence_number(triples):
        max_size = 0
        # Precompute triple masks for faster check?
        # triple_mask = (1<<a) | (1<<b) | (1<<c)
        # S is independent if (S & triple_mask) != triple_mask for all triple_mask
        t_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            t_masks.append(mask)
        
        # Iterate all subsets
        for mask in range(1 << n):
            # Check if mask is independent
            is_ind = True
            for tm in t_masks:
                if (mask & tm) == tm:
                    is_ind = False
                    break
            if is_ind:
                size = bin(mask).count("1")
                if size > max_size:
                    max_size = size
        return max_size

    # Initial state: Greedy construction
    # Try to add as many triples as possible randomly
    initial_triples = []
    all_possible_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_possible_triples.append([i, j, k])
                
    random.shuffle(all_possible_triples)
    for t in all_possible_triples:
        if is_valid(t, initial_triples):
            initial_triples.append(t)
            # Stop if we hit the theoretical max or just keep going?
            # Theoretical max is 37.
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
            # Pick a random triple from all_possible that is not already in new_triples
            # To avoid duplicates, we can just pick a random candidate and check validity
            # But we also need to ensure it's not already there.
            # Simple approach: try a few random triples from the full list
            added = False
            for _ in range(10): # Try 10 random candidates
                cand = random.choice(all_possible_triples)
                if cand not in new_triples: # O(n) check, n~37, ok
                    if is_valid(cand, new_triples):
                        new_triples.append(cand)
                        added = True
                        break
            if not added:
                continue # Skip this step if we couldn't add
        elif move_type == 'swap':
            if len(new_triples) == 0:
                continue
            idx = random.randint(0, len(new_triples) - 1)
            removed_triple = new_triples.pop(idx)
            # Try to add a different valid triple
            added = False
            for _ in range(10):
                cand = random.choice(all_possible_triples)
                if cand != removed_triple and cand not in new_triples:
                    if is_valid(cand, new_triples):
                        new_triples.append(cand)
                        added = True
                        break
            if not added:
                # Revert
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
                # Accept neutral moves to explore
                if random.random() < 0.5:
                    current_triples = new_triples
            else:
                probability = math.exp(-delta / T)
                if random.random() < probability:
                    current_triples = new_triples
                    current_alpha = new_alpha
        
        T *= cooling

    return best_triples
