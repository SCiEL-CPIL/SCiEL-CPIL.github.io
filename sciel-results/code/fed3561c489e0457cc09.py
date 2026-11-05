import itertools
import random
import time
from typing import List, Tuple

def construct():
    """
    Constructs a linear triple system on 16 points minimizing the independence number.
    Uses Simulated Annealing with exact independence number evaluation.
    """
    n = 16
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # State: list of triples (tuples)
    # Helper to check validity of adding a triple
    def get_used_pairs(triples):
        used = set()
        for t in triples:
            a, b, c = t
            used.add((a, b))
            used.add((b, c))
            used.add((a, c))
        return used

    def can_add(triples, used_pairs, new_triple):
        a, b, c = new_triple
        if (a, b) in used_pairs or (b, a) in used_pairs: return False
        if (b, c) in used_pairs or (c, b) in used_pairs: return False
        if (a, c) in used_pairs or (c, a) in used_pairs: return False
        return True

    def calc_alpha(triples):
        if not triples:
            return n
        triple_masks = []
        for t in triples:
            mask = 0
            for v in t:
                mask |= (1 << v)
            triple_masks.append(mask)
        
        # Check from largest k down
        for k in range(n, 0, -1):
            found = False
            # Optimization: If k is large, fewer combinations
            for subset in itertools.combinations(range(n), k):
                subset_mask = 0
                for v in subset:
                    subset_mask |= (1 << v)
                
                is_independent = True
                for tm in triple_masks:
                    if (tm & subset_mask) == tm:
                        is_independent = False
                        break
                if is_independent:
                    found = True
                    break
            if found:
                return k
        return 0

    # Initialize with a random maximal packing
    def random_maximal_packing():
        triples = []
        used = set()
        available = all_triples[:]
        random.shuffle(available)
        for t in available:
            if can_add(triples, used, t):
                triples.append(t)
                a, b, c = t
                used.add((a, b)); used.add((b, a))
                used.add((b, c)); used.add((c, b))
                used.add((a, c)); used.add((c, a))
        return triples

    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    best_triples = random_maximal_packing()
    best_alpha = calc_alpha(best_triples)
    
    current_triples = best_triples[:]
    current_alpha = best_alpha
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        # Generate move
        move_type = random.random()
        
        if move_type < 0.5:
            # Remove a random triple
            if not current_triples:
                continue
            idx = random.randrange(len(current_triples))
            removed = current_triples.pop(idx)
            new_triples = current_triples
            # We don't strictly need to update used pairs for alpha calc if we recalc, 
            # but for validity of future adds we need it. 
            # For simplicity, we just recalc alpha on the new state.
            # However, we need to ensure the state is valid (it is, as we removed one).
            
        elif move_type < 0.9:
            # Add a random valid triple
            # Find a random valid triple
            # To speed up, we can pick a random triple and check
            # If many are invalid, this is slow. 
            # Better: pick 3 random points, check if valid.
            found_add = False
            for _ in range(50): # Try 50 times to find a valid add
                t = tuple(random.sample(range(n), 3))
                # Check validity against current_triples
                # Re-calc used pairs is expensive. 
                # Let's maintain a used_pairs set in the main loop?
                # For now, simple check:
                valid = True
                a, b, c = t
                for ot in current_triples:
                    o_a, o_b, o_c = ot
                    # Check intersection size
                    inter = set(t) & set(ot)
                    if len(inter) > 1:
                        valid = False
                        break
                if valid:
                    new_triples = current_triples + [t]
                    found_add = True
                    break
            if not found_add:
                continue
        else:
            # Swap: Remove one, Add one
            if not current_triples:
                continue
            idx = random.randrange(len(current_triples))
            removed = current_triples.pop(idx)
            
            # Try to add a new one
            found_add = False
            for _ in range(50):
                t = tuple(random.sample(range(n), 3))
                valid = True
                a, b, c = t
                for ot in current_triples:
                    inter = set(t) & set(ot)
                    if len(inter) > 1:
                        valid = False
                        break
                if valid:
                    new_triples = current_triples + [t]
                    found_add = True
                    break
            if not found_add:
                # Restore
                current_triples.append(removed)
                continue

        new_alpha = calc_alpha(new_triples)
        
        # Acceptance
        if new_alpha < current_alpha:
            current_triples = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples[:]
        else:
            delta = new_alpha - current_alpha
            if delta == 0:
                current_triples = new_triples
            else:
                # Accept with probability exp(-delta / T)
                # delta is positive
                prob = 2.718281828 ** (-delta / T)
                if random.random() < prob:
                    current_triples = new_triples
                    current_alpha = new_alpha
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return [list(t) for t in best_triples]
