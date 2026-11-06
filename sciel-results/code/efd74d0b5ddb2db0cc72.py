import random
import time
import itertools

def calc_alpha(triples):
    """
    Calculate the independence number of the 3-uniform hypergraph defined by triples.
    n=16.
    Uses bitmask superset marking for speed.
    """
    n = 16
    size = 1 << n
    is_ind = [True] * size
    
    # For each triple, mark all supersets as not independent
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        rest_mask = (size - 1) ^ mask
        sub = rest_mask
        while True:
            s = mask | sub
            is_ind[s] = False
            if sub == 0:
                break
            sub = (sub - 1) & rest_mask
        
    # Find max popcount of independent sets
    max_alpha = 0
    for s in range(size):
        if is_ind[s]:
            c = bin(s).count('1')
            if c > max_alpha:
                max_alpha = c
    return max_alpha

def is_linear(triples):
    """Check if the set of triples is linear."""
    pairs = set()
    for t in triples:
        p1 = (min(t[0], t[1]), max(t[0], t[1]))
        p2 = (min(t[0], t[2]), max(t[0], t[2]))
        p3 = (min(t[1], t[2]), max(t[1], t[2]))
        if p1 in pairs or p2 in pairs or p3 in pairs:
            return False
        pairs.add(p1)
        pairs.add(p2)
        pairs.add(p3)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_triples = None
    best_alpha = 16
    
    # Pre-generate all possible triples
    all_triples = list(itertools.combinations(range(16), 3))
    
    # 1. Greedy Construction to get a dense starting point
    # We do multiple random restarts of greedy construction
    for restart in range(50):
        if time.time() > deadline:
            break
            
        random.shuffle(all_triples)
        triples = []
        used_pairs = set()
        
        for t in all_triples:
            p1 = (min(t[0], t[1]), max(t[0], t[1]))
            p2 = (min(t[0], t[2]), max(t[0], t[2]))
            p3 = (min(t[1], t[2]), max(t[1], t[2]))
            
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                triples.append(list(t))
                
        # Calculate initial alpha
        alpha = calc_alpha(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
            
        # If we found a very good solution, we can stop or refine
        if best_alpha <= 5:
             # Try to improve further with local search
             pass
        else:
            # Local Search Phase
            # We want to minimize alpha.
            # Simulated Annealing or Hill Climbing
            temp = 1.0
            current_triples = [list(t) for t in triples]
            current_alpha = alpha
            current_used_pairs = set(used_pairs)
            
            # Local search loop
            # Budget for local search per restart: ~1-2 seconds
            local_deadline = start_time + 140
            # Let's use a fixed number of steps or time check
            steps = 0
            max_steps = 5000
            
            while steps < max_steps and time.time() < local_deadline:
                steps += 1
                
                # Pick a random triple to remove
                if not current_triples:
                    break
                idx = random.randint(0, len(current_triples) - 1)
                t_remove = current_triples[idx]
                
                # Remove it
                current_triples.pop(idx)
                # Update used_pairs
                p1 = (min(t_remove[0], t_remove[1]), max(t_remove[0], t_remove[1]))
                p2 = (min(t_remove[0], t_remove[2]), max(t_remove[0], t_remove[2]))
                p3 = (min(t_remove[1], t_remove[2]), max(t_remove[1], t_remove[2]))
                current_used_pairs.discard(p1)
                current_used_pairs.discard(p2)
                current_used_pairs.discard(p3)
                
                # Try to add a new triple
                # Bias: try to add triples involving point 15 if it's under-represented
                # Or just random
                # Let's try a few random candidates
                added = False
                for _ in range(10):
                    t_new = list(random.choice(all_triples))
                    np1 = (min(t_new[0], t_new[1]), max(t_new[0], t_new[1]))
                    np2 = (min(t_new[0], t_new[2]), max(t_new[0], t_new[2]))
                    np3 = (min(t_new[1], t_new[2]), max(t_new[1], t_new[2]))
                    
                    if np1 not in current_used_pairs and np2 not in current_used_pairs and np3 not in current_used_pairs:
                        # Valid addition
                        # Check if it's the same as removed?
                        if sorted(t_new) == sorted(t_remove):
                            # Reverting
                            current_triples.append(t_remove)
                            current_used_pairs.add(p1)
                            current_used_pairs.add(p2)
                            current_used_pairs.add(p3)
                            added = True
                            break
                        
                        # Calculate new alpha
                        current_triples.append(t_new)
                        new_alpha = calc_alpha(current_triples)
                        
                        # Accept if better or with prob (SA)
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            current_used_pairs.add(np1)
                            current_used_pairs.add(np2)
                            current_used_pairs.add(np3)
                            added = True
                            
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_triples = [list(t) for t in current_triples]
                                if best_alpha <= 5:
                                    break
                        else:
                            # Reject, revert
                            current_triples.pop()
                        
                        # If we found a valid triple, we stop trying others for this step
                        # unless we rejected it? No, if rejected, we can try another candidate?
                        # To keep it simple, we just accept the first valid one we find (or revert if worse)
                        # But we should probably try to find a *better* one.
                        # For now, first valid is fine.
                        break
                
                if not added:
                    # If we couldn't add any valid triple, revert the removal
                    current_triples.append(t_remove)
                    current_used_pairs.add(p1)
                    current_used_pairs.add(p2)
                    current_used_pairs.add(p3)
                
                if best_alpha <= 5:
                    break
                    
    return best_triples
