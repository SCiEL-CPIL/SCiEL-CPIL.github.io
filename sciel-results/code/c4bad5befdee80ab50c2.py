import time
import random
import numpy as np
from itertools import combinations

def compute_alpha_fast(triples):
    """
    Computes the independence number of a linear triple system on 16 points.
    Uses NumPy for vectorized subset checking.
    Returns the size of the largest subset containing no triple.
    """
    n = 16
    if not triples:
        return n
        
    # Precompute masks for triples
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    if not triple_masks:
        return n

    triple_masks_np = np.array(triple_masks, dtype=np.int32)
    
    # Generate all subset masks 0..65535
    # We can cache this
    if not hasattr(compute_alpha_fast, "all_masks"):
        compute_alpha_fast.all_masks = np.arange(1 << 16, dtype=np.int32)
        # Precompute popcount
        pc = np.zeros(1 << 16, dtype=np.int8)
        for i in range(1, 1 << 16):
            pc[i] = pc[i >> 1] + (i & 1)
        compute_alpha_fast.popcount = pc
        
    all_masks = compute_alpha_fast.all_masks
    popcount = compute_alpha_fast.popcount
    
    # Check if any triple is contained in each mask
    # Shape: (65536, num_triples)
    # (mask & triple) == triple
    # This might be memory heavy if num_triples is large, but typically < 40 triples.
    # 65536 * 40 * 4 bytes = ~10MB. Acceptable.
    
    # Vectorized operation:
    intersects = (all_masks[:, None] & triple_masks_np[None, :]) == triple_masks_np[None, :]
    has_triple = intersects.any(axis=1)
    independent = ~has_triple
    
    if not independent.any():
        return 0
        
    max_alpha = popcount[independent].max()
    return int(max_alpha)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n = 16
    points = list(range(n))
    
    # Pre-generate all possible triples
    all_triples = list(combinations(points, 3))
    
    # Helper to check linearity and get used pairs
    def get_used_pairs(triples):
        used = set()
        for a, b, c in triples:
            used.add((a, b))
            used.add((a, c))
            used.add((b, c))
        return used

    def is_linear(triples):
        used = get_used_pairs(triples)
        return len(used) == 3 * len(triples)

    def try_greedy(seed):
        """
        Random greedy construction of a linear triple system.
        """
        rng = random.Random(seed)
        candidates = all_triples[:]
        rng.shuffle(candidates)
        
        used_pairs = set()
        triples = []
        
        for t in candidates:
            a, b, c = t
            pairs = [(a, b), (a, c), (b, c)]
            
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
                
            if not conflict:
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        
        return triples

    best_triples = []
    best_alpha = n + 1
    
    # Phase 1: Find a good initial solution using random greedy
    # We run many seeds because greedy is fast
    num_initial_attempts = 2000
    for i in range(num_initial_attempts):
        if time.time() > deadline - 20:
            break
        triples = try_greedy(i)
        if not triples:
            continue
        alpha = compute_alpha_fast(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            # If we hit a very low alpha, we might stop early or continue to polish
            if best_alpha <= 4:
                # Keep searching for 3 or 2, but don't spend too much time
                pass

    if not best_triples:
        # Fallback: a valid linear system (e.g., disjoint triples if possible, or just a few)
        # For n=16, we can have 5 disjoint triples.
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    # Phase 2: Local Search (Swapping)
    # We try to improve the current best_triples by swapping one triple for another.
    # This is effective because greedy gets stuck in local optima.
    
    current_triples = list(best_triples)
    current_alpha = best_alpha
    
    # Precompute masks for current triples for fast checking? 
    # The compute_alpha_fast handles it, but we can cache the used pairs.
    
    # To speed up local search, we can limit the number of swap attempts per step
    # and the number of total steps.
    
    max_swap_steps = 5000
    swap_attempts_per_step = 100
    
    # We will perform a simple hill climbing / simulated annealing style search.
    # Since we want to MINIMIZE alpha, we accept moves that decrease alpha.
    # We can also accept some moves that increase alpha to escape local minima,
    # but given the time constraint, simple hill climbing with random restarts of the search phase
    # or just aggressive hill climbing on the best candidate is good.
    
    # Let's do multiple rounds of local search on the best candidate found so far.
    # We'll also occasionally restart local search from a slightly perturbed version.
    
    for round_idx in range(10):
        if time.time() > deadline - 5:
            break
            
        # If we are stuck at a high alpha, maybe we should try a different starting point?
        # For now, let's just polish the current best.
        
        cur_triples = list(best_triples)
        cur_alpha = best_alpha
        
        # To make local search effective, we need to quickly check if a swap is valid (linear)
        # and then compute the new alpha.
        
        used_pairs = get_used_pairs(cur_triples)
        
        # We want to find a triple t_new that can replace t_old.
        # t_new must not conflict with any triple in cur_triples \ {t_old}.
        
        for step in range(max_swap_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random triple to remove
            idx = random.randint(0, len(cur_triples) - 1)
            t_old = cur_triples[idx]
            
            # Temporarily remove t_old from used_pairs
            a, b, c = t_old
            pairs_to_remove = [(a, b), (a, c), (b, c)]
            for p in pairs_to_remove:
                used_pairs.discard(p)
            
            # Find a candidate t_new
            # We can pick randomly from all_triples, or from those that share points with t_old (to maintain density)
            # Let's pick randomly from all_triples for diversity, but check validity.
            
            found_improvement = False
            # Try a few candidates
            for _ in range(swap_attempts_per_step):
                t_new = random.choice(all_triples)
                
                # Check if t_new conflicts with current used_pairs
                x, y, z = t_new
                new_pairs = [(x, y), (x, z), (y, z)]
                
                valid = True
                for p in new_pairs:
                    if p in used_pairs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # If valid, try the swap
                # Apply swap
                cur_triples[idx] = t_new
                for p in new_pairs:
                    used_pairs.add(p)
                
                # Compute new alpha
                new_alpha = compute_alpha_fast(cur_triples)
                
                if new_alpha < cur_alpha:
                    # Accept
                    cur_alpha = new_alpha
                    found_improvement = True
                    # Update best global
                    if cur_alpha < best_alpha:
                        best_alpha = cur_alpha
                        best_triples = list(cur_triples)
                    break
                else:
                    # Reject: revert
                    cur_triples[idx] = t_old
                    for p in new_pairs:
                        used_pairs.discard(p)
                    # Re-add old pairs
                    for p in pairs_to_remove:
                        used_pairs.add(p)
            
            if found_improvement:
                # Reset step counter or continue?
                # If we found an improvement, we can continue from here.
                # used_pairs is already updated correctly.
                pass
            else:
                # Revert t_old removal if we didn't swap
                # Wait, if we didn't swap, we discarded t_old pairs. We need to re-add them.
                # In the loop above, if valid was False, we didn't modify used_pairs.
                # If valid was True but we rejected (new_alpha >= cur_alpha), we reverted.
                # So if we exit the inner loop without found_improvement, used_pairs should be back to state with t_old.
                # Let's verify:
                # 1. Remove t_old pairs.
                # 2. Inner loop:
                #    - If candidate invalid: do nothing.
                #    - If candidate valid:
                #       - Add t_new pairs.
                #       - If accept: break (state has t_new).
                #       - If reject: remove t_new pairs, restore t_old pairs. (State has t_old).
                # 3. If inner loop finishes without break:
                #    - If no valid candidate was found: state has NO t_old pairs. ERROR.
                #    - If valid candidates found but all rejected: state has t_old pairs. OK.
                
                # So we must check if t_old pairs are in used_pairs.
                # Actually, simpler: just rebuild used_pairs if we didn't improve?
                # No, that's expensive.
                # Let's fix the logic:
                # If we didn't find an improvement, we must ensure t_old is back.
                # If the inner loop broke due to improvement, t_old is gone (replaced).
                # If the inner loop finished normally:
                #   Case A: No valid t_new found. used_pairs is missing t_old pairs.
                #   Case B: Valid t_new found, but rejected. used_pairs has t_old pairs.
                
                # To be safe, if not found_improvement, we check if t_old pairs are in used_pairs.
                # If not, add them.
                for p in pairs_to_remove:
                    if p not in used_pairs:
                        used_pairs.add(p)
        
        # End of local search round
        # If we improved best_alpha, we can continue.
        # If we didn't, maybe try a different starting point?
        # For now, we just stop if time is low.
        
    # Final verification
    if best_triples:
        # Ensure linearity
        if not is_linear(best_triples):
             # Fallback
             return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        return [list(t) for t in best_triples]
    
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
