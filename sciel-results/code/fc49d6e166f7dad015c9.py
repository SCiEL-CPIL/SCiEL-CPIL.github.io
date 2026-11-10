"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_alpha(triples, n=16):
    """Compute the independence number using bitmask checks."""
    if not triples:
        return n
    
    # Precompute masks for the triples
    triple_masks = []
    for t in triples:
        mask = 0
        for x in t:
            mask |= (1 << x)
        triple_masks.append(mask)
        
    # Check subsets in decreasing size order
    # We can optimize by checking only up to a certain limit if we have a current best
    # But for now, exact calculation.
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            subset_mask = 0
            for x in subset:
                subset_mask |= (1 << x)
            
            # Check if any triple is contained in this subset
            found = False
            for tm in triple_masks:
                if (tm & subset_mask) == tm:
                    found = True
                    break
            
            if not found:
                return size
    return 1

def _can_add(triples, masks, new_triple):
    """Check if new_triple can be added without violating linearity."""
    new_mask = 0
    for x in new_triple:
        new_mask |= (1 << x)
        
    for m in masks:
        overlap = m & new_mask
        # If overlap has more than 1 bit set, it's invalid
        if overlap & (overlap - 1):
            return False
    return True

def _greedy_maximal(seed, n=16):
    """Generate a maximal linear system using greedy random selection."""
    rng = random.Random(seed)
    triples = []
    masks = []
    
    all_triples = list(combinations(range(n), 3))
    rng.shuffle(all_triples)
    
    for t in all_triples:
        if _can_add(triples, masks, t):
            triples.append(t)
            m = 0
            for x in t:
                m |= (1 << x)
            masks.append(m)
            
    return triples, masks

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    best_alpha = n + 1
    best_triples = []
    
    # Try multiple initializations
    for init_seed in range(50):
        if time.time() > deadline - 10.0:
            break
            
        triples, masks = _greedy_maximal(init_seed, n)
        if not triples:
            continue
            
        current_alpha = _compute_alpha(triples, n)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        # Simulated Annealing
        # We want to minimize alpha
        T = 1.0
        T_min = 0.05
        cooling = 0.99
        
        # Limit SA steps per initialization to avoid getting stuck
        max_sa_steps = 200
        
        for step in range(max_sa_steps):
            if time.time() > deadline:
                break
                
            if not triples:
                break
                
            # Generate a move: Swap a random triple
            idx = random.randint(0, len(triples) - 1)
            old_triple = triples[idx]
            old_mask = masks[idx]
            
            # Remove
            triples.pop(idx)
            masks.pop(idx)
            
            # Find valid additions
            valid_adds = []
            for t in combinations(range(n), 3):
                if _can_add(triples, masks, t):
                    valid_adds.append(t)
                    
            if not valid_adds:
                # Restore
                triples.append(old_triple)
                masks.append(old_mask)
                T *= cooling
                continue
                
            # Pick new triple
            # Avoid picking the same triple if possible
            if len(valid_adds) > 1:
                candidates = [t for t in valid_adds if t != old_triple]
                if not candidates:
                    candidates = valid_adds
                new_triple = random.choice(candidates)
            else:
                new_triple = valid_adds[0]
                
            # Add
            triples.append(new_triple)
            new_mask = 0
            for x in new_triple:
                new_mask |= (1 << x)
            masks.append(new_mask)
            
            # Evaluate
            new_alpha = _compute_alpha(triples, n)
            delta = new_alpha - current_alpha
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                # Metropolis criterion
                # Probability = exp(-delta / T)
                # Since delta is small integer, we can use simple ratio
                accept = random.random() < (1.0 / (1.0 + delta * T))
                
            if accept:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = triples[:]
            else:
                # Revert
                triples.pop()
                masks.pop()
                triples.append(old_triple)
                masks.append(old_mask)
                
            T *= cooling
            
    if not best_triples:
        # Fallback
        return [[i, i+1, i+2] for i in range(0, 14, 3)]
        
    return [list(t) for t in best_triples]

# EVOLVE-BLOCK-END
