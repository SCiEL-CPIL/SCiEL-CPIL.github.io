import itertools
import random
import time
import numpy as np

def construct():
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute masks for all triples
    triple_masks = np.array([(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples], dtype=np.int32)
    
    # Precompute pairs for linearity check
    triple_pairs = []
    for t in all_triples:
        a, b, c = t
        triple_pairs.append([(a, b), (a, c), (b, c)])
        
    # Precompute popcount for all masks
    popcount = np.zeros(1 << n, dtype=np.int8)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    masks_arr = np.arange(1 << n, dtype=np.int32)

    def get_alpha_fast(active_indices):
        if not active_indices:
            return n
            
        is_ind = np.ones(1 << n, dtype=bool)
        
        for idx in active_indices:
            m = triple_masks[idx]
            mask_check = (masks_arr & m) == m
            is_ind[mask_check] = False
            
        # Find max popcount where is_ind is True
        # If all are False (impossible as 0 is always independent), return 0
        # 0 mask is always independent.
        return int(np.max(popcount[is_ind]))

    def greedy_construct(seed):
        random.seed(seed)
        indices = list(range(num_triples))
        random.shuffle(indices)
        active = []
        used_pairs = set()
        
        for idx in indices:
            conflict = False
            for p in triple_pairs[idx]:
                key = (min(p[0], p[1]), max(p[0], p[1]))
                if key in used_pairs:
                    conflict = True
                    break
            if not conflict:
                active.append(idx)
                for p in triple_pairs[idx]:
                    key = (min(p[0], p[1]), max(p[0], p[1]))
                    used_pairs.add(key)
        return active

    def local_search(active, max_time=3.0):
        current_alpha = get_alpha_fast(active)
        best_alpha = current_alpha
        best_active = active[:]
        
        start_time = time.time()
        
        # Maintain used_pairs for O(1) conflict check
        used_pairs = set()
        for idx in active:
            for p in triple_pairs[idx]:
                key = (min(p[0], p[1]), max(p[0], p[1]))
                used_pairs.add(key)
                
        # Set of active indices for O(1) check
        active_set = set(active)
        
        while time.time() - start_time < max_time:
            # Try to add a random triple
            candidate_idx = random.randint(0, num_triples - 1)
            if candidate_idx in active_set:
                continue
                
            # Check if it conflicts
            conflict = False
            conflict_pair_key = None
            for p in triple_pairs[candidate_idx]:
                key = (min(p[0], p[1]), max(p[0], p[1]))
                if key in used_pairs:
                    conflict = True
                    conflict_pair_key = key
                    break
            
            if not conflict:
                # Add it
                new_active = active + [candidate_idx]
                new_alpha = get_alpha_fast(new_active)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_active = new_active[:]
                    active = new_active
                    active_set.add(candidate_idx)
                    for p in triple_pairs[candidate_idx]:
                        key = (min(p[0], p[1]), max(p[0], p[1]))
                        used_pairs.add(key)
                elif new_alpha == best_alpha and random.random() < 0.5:
                    # Accept to explore
                    active = new_active
                    active_set.add(candidate_idx)
                    for p in triple_pairs[candidate_idx]:
                        key = (min(p[0], p[1]), max(p[0], p[1]))
                        used_pairs.add(key)
            else:
                # Swap: Remove a triple sharing conflict_pair_key, add candidate
                # Find a triple in active that uses conflict_pair_key
                rem_idx = -1
                for idx in active:
                    for p in triple_pairs[idx]:
                        if (min(p[0], p[1]), max(p[0], p[1])) == conflict_pair_key:
                            rem_idx = idx
                            break
                    if rem_idx != -1:
                        break
                
                if rem_idx != -1:
                    # Perform swap
                    new_active = [x for x in active if x != rem_idx] + [candidate_idx]
                    new_alpha = get_alpha_fast(new_active)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_active = new_active[:]
                        active = new_active
                        active_set.remove(rem_idx)
                        active_set.add(candidate_idx)
                        
                        # Update used_pairs
                        for p in triple_pairs[rem_idx]:
                            pk = (min(p[0], p[1]), max(p[0], p[1]))
                            used_pairs.discard(pk)
                        for p in triple_pairs[candidate_idx]:
                            pk = (min(p[0], p[1]), max(p[0], p[1]))
                            used_pairs.add(pk)
                    elif new_alpha == best_alpha and random.random() < 0.1:
                        # Accept to explore
                        active = new_active
                        active_set.remove(rem_idx)
                        active_set.add(candidate_idx)
                        for p in triple_pairs[rem_idx]:
                            pk = (min(p[0], p[1]), max(p[0], p[1]))
                            used_pairs.discard(pk)
                        for p in triple_pairs[candidate_idx]:
                            pk = (min(p[0], p[1]), max(p[0], p[1]))
                            used_pairs.add(pk)
        
        return best_active, best_alpha

    best_alpha = n
    best_active = []
    
    start_time = time.time()
    seed = 0
    while time.time() - start_time < 115:
        seed += 1
        active = greedy_construct(seed)
        # Run local search on this seed
        searched_active, alpha = local_search(active, max_time=2.0)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_active = searched_active[:]
            
        if best_alpha <= 4:
            break

    result = [list(all_triples[i]) for i in best_active]
    
    # Final verification
    pair_count = {}
    for e in result:
        for i in range(3):
            for j in range(i+1, 3):
                p = (min(e[i], e[j]), max(e[i], e[j]))
                if p in pair_count:
                    return [] # Invalid
                pair_count[p] = True
                
    return result
