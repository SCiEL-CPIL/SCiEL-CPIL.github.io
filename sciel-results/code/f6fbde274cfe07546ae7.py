import random
import math
import time

def construct():
    n = 16
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((1 << i) | (1 << j) | (1 << k))
    
    def is_valid(t_mask, existing_masks):
        for e_mask in existing_masks:
            if (t_mask & e_mask).bit_count() > 1:
                return False
        return True

    def get_independence_number(triple_masks_list):
        """
        Fast independence number calculation.
        Uses the 'count' array optimization.
        """
        if not triple_masks_list:
            return n
        
        # count[m] = number of triples contained in mask m
        count = [0] * (1 << n)
        
        # For each triple, increment count for all supersets
        for t in triple_masks_list:
            # Supersets of t are masks m such that (m & t) == t
            # We can iterate through all masks and check?
            # 65536 * 40 = 2.6M ops.
            # Better: Iterate through all masks, and for each mask, check if it contains t?
            # No, that's the same.
            
            # Efficient way to iterate supersets:
            # Let free_bits = (~t) & ((1 << n) - 1)
            # Iterate submasks of free_bits
            free_bits = (~t) & ((1 << n) - 1)
            sub = free_bits
            while True:
                m = t | sub
                count[m] += 1
                if sub == 0:
                    break
                sub = (sub - 1) & free_bits
        
        # Find max popcount where count[m] == 0
        max_alpha = 0
        for m in range(1 << n):
            if count[m] == 0:
                pc = m.bit_count()
                if pc > max_alpha:
                    max_alpha = pc
                    if max_alpha == n:
                        break
        return max_alpha

    # Initial state: Greedy construction
    initial_masks = []
    shuffled_triples = all_triples[:]
    random.shuffle(shuffled_triples)
    
    for t in shuffled_triples:
        if is_valid(t, initial_masks):
            initial_masks.append(t)
            if len(initial_masks) >= 35:
                break

    best_masks = initial_masks[:]
    best_alpha = get_independence_number(best_masks)
    
    current_masks = initial_masks[:]
    current_alpha = get_independence_number(current_masks)
    
    start_time = time.time()
    timeout = 115
    
    T = 1.0
    cooling = 0.998
    
    # Precompute a list of masks for fast random access
    all_triple_list = all_triples
    
    while time.time() - start_time < timeout:
        new_masks = current_masks[:]
        
        # Move generation
        r = random.random()
        if r < 0.5:
            # Add a triple
            # Prefer triples that "attack" the current independent set?
            # We don't track the set, just the number.
            # Let's just pick a random valid triple.
            added = False
            for _ in range(50):
                t = random.choice(all_triple_list)
                if t not in new_masks and is_valid(t, new_masks):
                    new_masks.append(t)
                    added = True
                    break
            if not added:
                continue
        elif r < 0.8:
            # Remove a triple
            if new_masks:
                idx = random.randint(0, len(new_masks) - 1)
                new_masks.pop(idx)
            else:
                continue
        else:
            # Swap
            if new_masks:
                idx = random.randint(0, len(new_masks) - 1)
                old_t = new_masks.pop(idx)
                added = False
                for _ in range(50):
                    t = random.choice(all_triple_list)
                    if t != old_t and t not in new_masks and is_valid(t, new_masks):
                        new_masks.append(t)
                        added = True
                        break
                if not added:
                    new_masks.insert(idx, old_t)
                    continue
        
        # Evaluate
        new_alpha = get_independence_number(new_masks)
        
        # Acceptance
        if new_alpha < current_alpha:
            current_masks = new_masks
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_masks = new_masks[:]
        else:
            delta = new_alpha - current_alpha
            if delta == 0:
                if random.random() < 0.5:
                    current_masks = new_masks
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_masks = new_masks
                    current_alpha = new_alpha
        
        T *= cooling
        
        # Occasionally restart or kick if stuck?
        # For now, just keep going.

    # Convert masks back to lists of ints
    result = []
    for m in best_masks:
        points = []
        for i in range(n):
            if m & (1 << i):
                points.append(i)
        result.append(points)
        
    return result
