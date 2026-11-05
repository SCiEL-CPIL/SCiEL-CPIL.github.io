import random
import time

def _independence_number(triples):
    """Compute the largest subset of {0..15} containing no triple."""
    n = 16
    if not triples:
        return n
    
    # Precompute triple masks
    triple_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
    
    best = 0
    # Iterate all masks
    # Optimization: We can iterate masks in descending order of popcount and stop early?
    # No, we need the max. 
    # We can skip masks that are supersets of known independent sets? 
    # Simple loop is okay for 16 points.
    for mask in range(1 << n):
        contains_triple = False
        for tm in triple_masks:
            if (mask & tm) == tm:
                contains_triple = True
                break
        if not contains_triple:
            pop = mask.bit_count() # Python 3.8+ has bit_count, otherwise bin(mask).count('1')
            if pop > best:
                best = pop
                # If best is 16, we can break? No, we want the max.
                # If best reaches n, break.
                if best == n:
                    return n
    return best

def _is_valid_addition(triples, new_triple):
    """Check if new_triple can be added to maintain linearity."""
    a, b, c = new_triple
    # Use bitmask for faster intersection check?
    # Or just set intersection.
    new_set = {a, b, c}
    for t in triples:
        if len(new_set & {t[0], t[1], t[2]}) >= 2:
            return False
    return True

def _generate_all_valid_triples(triples, n=16):
    """Generate all triples that can be added to the current system."""
    valid = []
    # Precompute existing pairs to speed up check?
    # For n=16, total triples is 560. 
    # Checking 560 triples against |T| existing triples is 560 * |T|.
    # If |T| ~ 30, that's 16800 checks. Very fast.
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                t = (a, b, c)
                if _is_valid_addition(triples, t):
                    valid.append(t)
    return valid

def construct():
    random.seed(42)
    n = 16
    start_time = time.time()
    time_limit = 115  # seconds
    
    # 1. Greedy Initialization: Degree Balancing
    triples = []
    degrees = [0] * n
    
    while True:
        valid_triples = _generate_all_valid_triples(triples, n)
        if not valid_triples:
            break
        
        # Score: minimize sum of degrees of points in the triple
        # This tends to spread out the triples
        best_t = None
        best_score = float('inf')
        for t in valid_triples:
            score = degrees[t[0]] + degrees[t[1]] + degrees[t[2]]
            if score < best_score:
                best_score = score
                best_t = t
        
        if best_t is None:
            break
            
        triples.append(best_t)
        for p in best_t:
            degrees[p] += 1
            
    current_alpha = _independence_number(triples)
    best_alpha = current_alpha
    best_triples = triples[:]
    
    # 2. Iterated Local Search (Hill Climbing)
    # We try to find a neighbor with lower alpha.
    # If stuck, we perturb.
    
    current_triples = triples[:]
    
    while time.time() - start_time < time_limit:
        # Calculate current alpha
        current_alpha = _independence_number(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = current_triples[:]
            
        # Generate Moves
        # 1. Additions
        valid_adds = _generate_all_valid_triples(current_triples, n)
        # 2. Removals
        removals = [(i, None) for i in range(len(current_triples))]
        
        # Evaluate Moves
        best_move_alpha = current_alpha
        best_move = None
        best_move_type = None
        best_move_idx = None
        
        # Check Additions
        for t in valid_adds:
            # Tentative add
            current_triples.append(t)
            new_alpha = _independence_number(current_triples)
            current_triples.pop()
            
            if new_alpha < best_move_alpha:
                best_move_alpha = new_alpha
                best_move = t
                best_move_type = 'add'
                best_move_idx = -1
                
        # Check Removals
        for i in range(len(current_triples)):
            removed = current_triples.pop(i)
            new_alpha = _independence_number(current_triples)
            current_triples.insert(i, removed)
            
            if new_alpha < best_move_alpha:
                best_move_alpha = new_alpha
                best_move = removed
                best_move_type = 'remove'
                best_move_idx = i
                
        # If no improvement found
        if best_move is None:
            # Perturbation: Random restart or random swap
            # Let's do a random perturbation: remove 2 random triples, add 2 random valid ones
            if len(current_triples) > 2:
                idxs = random.sample(range(len(current_triples)), 2)
                for idx in sorted(idxs, reverse=True):
                    current_triples.pop(idx)
                    
                # Try to add back some
                valid_adds = _generate_all_valid_triples(current_triples, n)
                if valid_adds:
                    for _ in range(2):
                        if valid_adds:
                            t = random.choice(valid_adds)
                            current_triples.append(t)
                            valid_adds = _generate_all_valid_triples(current_triples, n)
            else:
                # Reset to best
                current_triples = best_triples[:]
        else:
            # Apply best move
            if best_move_type == 'add':
                current_triples.append(best_move)
            elif best_move_type == 'remove':
                current_triples.pop(best_move_idx)
                
    return best_triples
