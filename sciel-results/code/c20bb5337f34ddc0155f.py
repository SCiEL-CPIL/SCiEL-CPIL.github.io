import time
import random
import itertools

def construct():
    """
    Constructs a linear triple system on 16 points minimizing the independence number.
    Uses random greedy search with local refinement to find packings with small independence number.
    """
    n = 16
    points = list(range(n))
    
    # Pre-generate all possible triples
    all_triples = list(itertools.combinations(points, 3))
    triple_masks = []
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Precompute point indices for each triple
    triple_points = [set(t) for t in all_triples]
    
    def is_valid_addition(triple_idx, existing_indices):
        t_set = triple_points[triple_idx]
        for e_idx in existing_indices:
            e_set = triple_points[e_idx]
            if len(t_set.intersection(e_set)) > 1:
                return False
        return True
    
    def compute_independence_number_fast(triples):
        """
        Compute independence number using bitmask DP.
        A subset S is independent iff no triple is entirely contained in S.
        """
        if not triples:
            return n
        
        # Build a "bad" array: bad[mask] = True if mask contains any triple
        # We can compute this via: for each triple mask T, mark all supersets of T as bad
        # But 2^16 = 65536, and we have ~30 triples, so let's just check directly
        
        # Actually, let's use a different approach:
        # For each mask from 0 to 2^16-1, check if it's independent
        # A mask is independent if no triple_mask is a subset of it
        
        triple_masks_list = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks_list.append(mask)
        
        # Use SOS (Sum Over Subsets) style DP
        # independent[mask] = True if no triple is subset of mask
        # Start with all True, then for each triple T, mark all supersets of T as False
        
        size = 1 << n
        independent = [True] * size
        
        for tm in triple_masks_list:
            # Mark all supersets of tm as not independent
            # Supersets of tm: mask where (mask & tm) == tm
            # We can iterate: for mask in range(size): if (mask & tm) == tm: independent[mask] = False
            # But that's O(size * num_triples). Let's do it smarter.
            # Actually for 65536 * 30 = ~2M, it's fine.
            for mask in range(size):
                if (mask & tm) == tm:
                    independent[mask] = False
        
        # Find max popcount of independent masks
        max_alpha = 0
        for mask in range(size):
            if independent[mask]:
                pc = bin(mask).count('1')
                if pc > max_alpha:
                    max_alpha = pc
                if max_alpha == n:
                    break
        
        return max_alpha
    
    def local_search(triples, time_limit=5.0):
        """
        Try to improve the triple system by swapping triples.
        A swap: remove triple T1, add triple T2 (not already in system).
        Accept if independence number decreases or stays same (for diversity).
        """
        if not triples:
            return triples
        
        current_alpha = compute_independence_number_fast(triples)
        best_triples = list(triples)
        best_alpha = current_alpha
        
        start = time.time()
        deadline = start + time_limit
        
        triple_set = set(triples)
        
        while time.time() < deadline:
            # Try random swaps
            made_progress = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                # Pick a random triple from current system
                if not triples:
                    break
                idx = random.randint(0, len(triples) - 1)
                remove_idx = idx
                removed = triples[remove_idx]
                
                # Try to find a replacement triple
                # Sample some random triples
                for _ in range(20):
                    candidate = random.choice(all_triples)
                    if candidate in triple_set:
                        continue
                    
                    # Check if we can add candidate after removing 'removed'
                    new_triples = triples[:remove_idx] + triples[remove_idx+1:]
                    
                    # Check linearity with candidate
                    valid = True
                    cand_set = set(candidate)
                    for t in new_triples:
                        t_set = set(t)
                        if len(cand_set.intersection(t_set)) > 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Try the swap
                    new_triples_with_cand = new_triples + [candidate]
                    new_alpha = compute_independence_number_fast(new_triples_with_cand)
                    
                    if new_alpha < best_alpha:
                        # Accept improvement
                        triples = new_triples_with_cand
                        triple_set = set(triples)
                        best_triples = list(triples)
                        best_alpha = new_alpha
                        made_progress = True
                        break
                    elif new_alpha == best_alpha and random.random() < 0.1:
                        # Accept with small probability for diversity
                        triples = new_triples_with_cand
                        triple_set = set(triples)
                        made_progress = True
                        break
            
            if not made_progress:
                # Try a few more iterations before giving up
                continue
        
        return best_triples, best_alpha
    
    best_alpha = n
    best_triples = []
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Seed random
    random.seed(42)
    
    # Phase 1: Random greedy constructions
    greedy_count = 0
    while time.time() < deadline:
        # Greedy construction with randomization
        triples = []
        triple_set = set()
        
        # Shuffle triple indices
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        
        for idx in indices:
            if len(triples) >= 40:  # Max reasonable number of triples
                break
            if is_valid_addition(idx, triples):
                triples.append(idx)
                triple_set.add(idx)
        
        # Convert to actual triples
        actual_triples = [all_triples[i] for i in triples]
        
        # Calculate independence number
        alpha = compute_independence_number_fast(actual_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = actual_triples
            greedy_count += 1
            
            # If we hit a very low alpha, we might stop early
            if best_alpha <= 4:
                break
        
        # Limit greedy phase to leave time for local search
        if greedy_count >= 50 or time.time() > deadline - 60:
            break
    
    # Phase 2: Local search refinement on best solution
    if best_triples and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 5
        best_triples, best_alpha = local_search(best_triples, time_limit=min(remaining_time, 50.0))
    
    # Phase 3: If time remains, do more greedy + local search cycles
    if time.time() < deadline - 5:
        while time.time() < deadline - 3:
            # Quick greedy
            triples = []
            indices = list(range(len(all_triples)))
            random.shuffle(indices)
            
            for idx in indices:
                if len(triples) >= 40:
                    break
                if is_valid_addition(idx, triples):
                    triples.append(idx)
            
            actual_triples = [all_triples[i] for i in triples]
            alpha = compute_independence_number_fast(actual_triples)
            
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = actual_triples
                if best_alpha <= 4:
                    break
            
            # Quick local search if improved
            if alpha <= best_alpha and time.time() < deadline - 8:
                tl = min(10.0, deadline - time.time() - 5)
                if tl > 2:
                    best_triples, best_alpha = local_search(best_triples, time_limit=tl)
    
    return best_triples

# EVOLVE-BLOCK-END
