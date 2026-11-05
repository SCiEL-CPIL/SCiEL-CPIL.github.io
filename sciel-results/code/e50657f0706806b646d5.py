import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)  # 560
    
    # Precompute pairs for each triple for fast conflict checking
    triple_pairs = []
    for t in all_triples:
        pairs = frozenset([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
        triple_pairs.append(pairs)
    
    # Precompute triple masks for fast independence check
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
        
    # Convert to numpy array for vectorized checks
    triple_masks_np = np.array(triple_masks, dtype=np.uint16)
    
    # Map from triple to its index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute masks for all subsets of points for faster lookup?
    # There are 2^16 = 65536 subsets.
    # We can precompute the mask for each integer representation of a subset.
    # But combinations returns tuples.
    # We can just compute the mask on the fly, it's 16 ops.
    
    def get_alpha_fast(triple_indices):
        """Fast independence number calculation using numpy bitmask checks."""
        if not triple_indices:
            return 16
        
        # Extract masks for current triples
        masks = triple_masks_np[triple_indices]
        
        # Iterate k from 16 down to 1
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                # Compute mask for this subset
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                
                # Check if any triple is fully contained in s_mask
                # A triple mask m is contained in s_mask if (m & s_mask) == m
                # Vectorized: (masks & s_mask) == masks
                # If all are False, then no triple is contained -> Independent
                # If any is True, then a triple is contained -> Not Independent
                
                # Note: s_mask is a python int, masks is numpy array.
                # NumPy will broadcast s_mask.
                contained = (masks & s_mask) == masks
                if not contained.any():
                    return k
        return 0

    def get_alpha_with_set(triple_indices):
        """Returns (alpha, independent_set_indices) where independent_set is a tuple of points."""
        if not triple_indices:
            return 16, tuple(points)
        
        masks = triple_masks_np[triple_indices]
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                
                contained = (masks & s_mask) == masks
                if not contained.any():
                    return k, subset
        return 0, ()

    best_alpha = 16
    best_triples = []
    best_indices = []
    
    # Helper to check if a triple can be added
    def can_add(triple_idx, used_pairs):
        t_pairs = triple_pairs[triple_idx]
        for p in t_pairs:
            if p in used_pairs:
                return False
        return True

    # Strategy: Randomized Greedy Construction with Local Repair
    
    num_restarts = 2000 # Reduced restarts to allow more time for local search per restart?
    # Actually, with faster alpha, we can do more local steps.
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Construct an initial linear system
        indices = list(range(n_triples))
        random.shuffle(indices)
        
        current_indices = []
        used_pairs = set()
        
        for idx in indices:
            if can_add(idx, used_pairs):
                current_indices.append(idx)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
                
        # Initial alpha
        alpha = get_alpha_fast(current_indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [all_triples[i] for i in current_indices]
            best_indices = list(current_indices)
        
        # Try to improve locally
        # We will do a fixed number of local search steps or until no improvement
        improved = True
        local_steps = 0
        max_local_steps = 200 # Cap local steps to prevent getting stuck in one restart
        
        while improved and local_steps < max_local_steps and time.time() < deadline:
            local_steps += 1
            improved = False
            alpha, indep_set = get_alpha_with_set(current_indices)
            
            if alpha <= best_alpha:
                break
                
            # Find all triples contained in indep_set that can be added
            candidates = []
            indep_set_set = set(indep_set)
            for idx in range(n_triples):
                t = all_triples[idx]
                if all(p in indep_set_set for p in t):
                    if can_add(idx, used_pairs):
                        candidates.append(idx)
                        
            if candidates:
                # Pick a random candidate
                idx = random.choice(candidates)
                current_indices.append(idx)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
                improved = True
            else:
                # No triple can be added inside the current independent set.
                # Try a perturbation: remove a random triple and re-add
                if current_indices and random.random() < 0.5:
                    rem_pos = random.randrange(len(current_indices))
                    rem_idx = current_indices.pop(rem_pos)
                    for p in triple_pairs[rem_idx]:
                        used_pairs.discard(p)
                    # Try to add a different triple
                    new_candidates = []
                    for idx2 in range(n_triples):
                        if can_add(idx2, used_pairs):
                            new_candidates.append(idx2)
                    if new_candidates:
                        idx2 = random.choice(new_candidates)
                        current_indices.append(idx2)
                        for p in triple_pairs[idx2]:
                            used_pairs.add(p)
                        improved = True

        # After local search, check if we improved best
        alpha = get_alpha_fast(current_indices)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [all_triples[i] for i in current_indices]
            best_indices = list(current_indices)

    return best_triples
