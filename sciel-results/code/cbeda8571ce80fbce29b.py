import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute masks for all triples
    triple_masks_dict = {}
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks_dict[t] = mask

    # Helper to check linearity
    def is_linear(triples):
        # Check if any two triples share more than 1 point
        # Using pair set is faster
        used_pairs = set()
        for t in triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            for p in pairs:
                if p in used_pairs:
                    return False
                used_pairs.add(p)
        return True

    # Fast alpha calculation using NumPy
    def calc_alpha(triples):
        if not triples:
            return 16
        
        # Create numpy array of masks
        masks = np.array([triple_masks_dict[t] for t in triples], dtype=np.uint32)
        
        max_alpha = 0
        # Iterate all subsets
        # Optimization: Skip masks with bit_count <= max_alpha
        for mask in range(1, 1 << 16):
            if mask.bit_count() <= max_alpha:
                continue
            
            # Check if independent: no triple is subset of mask
            # (mask & m) == m for any m in masks means m is subset of mask
            # If any is true, mask is NOT independent
            # We want to check if NO triple is subset
            # So: np.any((mask & masks) == masks) should be False
            
            if not np.any((mask & masks) == masks):
                max_alpha = mask.bit_count()
                
        return max_alpha

    def greedy_build():
        triples = []
        used_pairs = set()
        random.shuffle(all_triples)
        for t in all_triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        return triples

    def simulated_annealing(initial_triples, initial_alpha, deadline):
        current_triples = list(initial_triples)
        current_alpha = initial_alpha
        best_triples = list(initial_triples)
        best_alpha = initial_alpha
        
        # Temperature schedule
        T = 1.0
        cooling = 0.99
        
        # Precompute addable candidates for fast move generation?
        # It's dynamic, so we check on the fly.
        
        steps = 0
        while time.time() < deadline - 5:
            steps += 1
            if steps % 100 == 0:
                T *= cooling
                
            # Generate a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            new_triples = None
            
            if move_type == 'add':
                # Try to add a random triple
                t = random.choice(all_triples)
                if t not in current_triples:
                    # Check linearity
                    new_triples = current_triples + [t]
                    if is_linear(new_triples):
                        pass
                    else:
                        new_triples = None
            elif move_type == 'remove':
                if current_triples:
                    idx = random.randint(0, len(current_triples) - 1)
                    new_triples = current_triples[:idx] + current_triples[idx+1:]
            elif move_type == 'swap':
                if current_triples:
                    # Remove one
                    idx = random.randint(0, len(current_triples) - 1)
                    removed = current_triples[idx]
                    remaining = current_triples[:idx] + current_triples[idx+1:]
                    
                    # Try to add a different one
                    # Pick a few candidates
                    candidates = random.sample(all_triples, min(20, len(all_triples)))
                    added = False
                    for t in candidates:
                        if t == removed:
                            continue
                        new_triples = remaining + [t]
                        if is_linear(new_triples):
                            added = True
                            break
                    if not added:
                        new_triples = None

            if new_triples is None:
                continue
            
            # Calculate new alpha
            new_alpha = calc_alpha(new_triples)
            
            # Acceptance criterion
            if new_alpha < current_alpha:
                current_triples = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
            elif new_alpha == current_alpha:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current_triples = new_triples
            else:
                # new_alpha > current_alpha
                # Accept with probability exp(-(delta)/T)
                delta = new_alpha - current_alpha
                if random.random() < np.exp(-delta / T):
                    current_triples = new_triples
                    current_alpha = new_alpha
        
        return best_triples, best_alpha

    best_alpha = 16
    best_triples = []
    
    # Multiple restarts
    num_restarts = 0
    while time.time() < deadline - 20:
        num_restarts += 1
        
        # Strategy 1: Greedy Maximal Packing
        init_triples = greedy_build()
        init_alpha = calc_alpha(init_triples)
        
        # Strategy 2: Sparse Random Start (for SA to work better)
        if num_restarts % 2 == 0:
            # Start with fewer triples
            random.shuffle(all_triples)
            sparse_triples = []
            used_pairs = set()
            for t in all_triples[:20]: # Take first 20 random
                pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
                valid = True
                for p in pairs:
                    if p in used_pairs:
                        valid = False
                        break
                if valid:
                    sparse_triples.append(t)
                    for p in pairs:
                        used_pairs.add(p)
            init_triples = sparse_triples
            init_alpha = calc_alpha(init_triples)
        
        # Polish with SA
        final_triples, final_alpha = simulated_annealing(init_triples, init_alpha, deadline - 10)
        
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            best_triples = final_triples
            
            # If we reach a very good value, we can try to break early
            if best_alpha <= 6:
                # Continue searching for a bit more to see if we can get 5 or 4
                if time.time() > deadline - 30:
                    break

    # Final polish with remaining time
    if best_triples and time.time() < deadline - 5:
        final_triples, final_alpha = simulated_annealing(best_triples, best_alpha, deadline)
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            best_triples = final_triples

    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return [list(t) for t in best_triples]

# EVOLVE-BLOCK-END
