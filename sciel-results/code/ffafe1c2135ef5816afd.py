import time
import random
from itertools import combinations

# Precompute all possible triples on 16 points
ALL_TRIPLES = []
ALL_TRIPLE_MASKS = []
TRIPLE_PAIRS = [] # List of (pair1, pair2, pair3) for each triple

for i in range(16):
    for j in range(i + 1, 16):
        for k in range(j + 1, 16):
            ALL_TRIPLES.append([i, j, k])
            mask = (1 << i) | (1 << j) | (1 << k)
            ALL_TRIPLE_MASKS.append(mask)
            TRIPLE_PAIRS.append(((i, j), (i, k), (j, k)))

NUM_TRIPLES = len(ALL_TRIPLES)

def compute_alpha_fast(H_indices):
    """
    Compute the independence number using branch and bound.
    H_indices is a set of indices into ALL_TRIPLES.
    """
    if not H_indices:
        return 16
    
    H_masks = [ALL_TRIPLE_MASKS[i] for i in H_indices]
    n = 16
    all_vertices = (1 << n) - 1
    best = 0
    
    def dfs(available, current_mask, current_size):
        nonlocal best
        # Pruning: Even if we add all available, we can't beat best
        if current_size + available.bit_count() <= best:
            return
        if available == 0:
            if current_size > best:
                best = current_size
            return
        
        # Find a dangerous triple
        # A triple is dangerous if it is not already broken by exclusions.
        # Broken if (t & ~(current_mask | available)) != 0
        # So dangerous if (t & ~(current_mask | available)) == 0
        dangerous_triple = None
        for t in H_masks:
            if (t & ~(current_mask | available)) == 0:
                dangerous_triple = t
                break
                
        if dangerous_triple is None:
            # No dangerous triples, we can add all remaining
            new_size = current_size + available.bit_count()
            if new_size > best:
                best = new_size
            return
        
        # Branch on excluding a vertex from the dangerous triple
        # We must exclude a vertex that is in available (since current_mask is fixed)
        candidates = dangerous_triple & available
        c = candidates
        while c:
            bit = c & -c
            new_available = available & ~bit
            dfs(new_available, current_mask, current_size)
            c -= bit
            
    dfs(all_vertices, 0, 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_alpha = 16
    best_system = []
    
    # Strategy:
    # 1. Generate a random greedy packing.
    # 2. Local search with simulated annealing to minimize alpha.
    
    # Parameters for Simulated Annealing
    initial_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # We will do several restarts
    # Each restart does a fresh greedy init and then SA
    
    # To speed up SA, we can precompute some things or use bitmasks for pairs?
    # There are 120 pairs. We can use a bitmask of 120 bits? No, Python ints are arbitrary precision.
    # Let's map pairs to indices 0..119.
    
    pair_to_idx = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            pair_to_idx[(i, j)] = idx
            idx += 1
    
    TRIPLE_PAIR_MASKS = []
    for pairs in TRIPLE_PAIRS:
        mask = 0
        for p in pairs:
            mask |= (1 << pair_to_idx[p])
        TRIPLE_PAIR_MASKS.append(mask)
    
    # used_pairs_mask is an integer bitmask of used pairs
    
    # Increase restarts since compute_alpha is faster
    for restart in range(500): # Cap on restarts, but time is the real limit
        if time.time() > deadline:
            break
            
        # 1. Random Greedy Packing
        H_indices = set()
        used_pairs_mask = 0
        
        available = list(range(NUM_TRIPLES))
        random.shuffle(available)
        
        for idx in available:
            pmask = TRIPLE_PAIR_MASKS[idx]
            if (used_pairs_mask & pmask) == 0:
                H_indices.add(idx)
                used_pairs_mask |= pmask
        
        current_alpha = compute_alpha_fast(H_indices)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = [ALL_TRIPLES[i] for i in H_indices]
        
        # 2. Simulated Annealing
        temp = initial_temp
        
        # We'll run SA for a fixed number of steps or until time limit
        # Or until convergence
        steps = 0
        last_improvement = time.time()
        
        while time.time() < deadline:
            if time.time() - last_improvement > 1.0:
                # No improvement for 1 second, break to new restart (faster alpha allows quicker feedback)
                break
                
            # Choose a move
            # Option A: Swap (remove one, add one)
            # Option B: Add (if possible)
            # Option C: Remove (if possible, unlikely to help)
            
            # For efficiency, let's focus on Swaps and Adds.
            
            # Try to add a random triple first?
            # If we have space, adding usually helps or is neutral.
            # But we want to minimize alpha.
            
            # Let's try a swap: remove a random edge, add a random edge that fits.
            # To make it faster, we can pick a random edge to remove,
            # then pick a random edge from the ones that are compatible with the rest.
            
            if H_indices:
                rem_idx = random.choice(list(H_indices))
                rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                
                # Temporarily remove
                used_pairs_mask_temp = used_pairs_mask & ~rem_pmask
                
                # Find a candidate to add
                # We can pick a random triple and check if it fits in used_pairs_mask_temp
                # But we want to bias towards triples that share few pairs with the rest?
                # Or just random.
                
                # To speed up finding a valid add, we can maintain a list of available triples?
                # That's expensive to update.
                # Let's just pick random and check.
                
                # Optimization:
                # We can precompute for each triple, which other triples conflict with it?
                # Conflict means they share a pair.
                # There are C(16,3) = 560 triples.
                # We can build a conflict graph.
                # But we just need to find *any* valid add.
                
                # Let's try a few random adds.
                found_add = False
                for _ in range(10):
                    add_idx = random.randint(0, NUM_TRIPLES - 1)
                    if add_idx in H_indices:
                        continue
                    add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                    if (used_pairs_mask_temp & add_pmask) == 0:
                        # Valid move
                        # Apply move
                        H_indices.remove(rem_idx)
                        H_indices.add(add_idx)
                        used_pairs_mask = used_pairs_mask_temp | add_pmask
                        
                        new_alpha = compute_alpha_fast(H_indices)
                        delta = new_alpha - current_alpha
                        
                        # Accept if better or with probability exp(-delta/temp)
                        if delta <= 0 or random.random() < __import__('math').exp(-delta / temp):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [ALL_TRIPLES[i] for i in H_indices]
                                last_improvement = time.time()
                        else:
                            # Revert
                            H_indices.remove(add_idx)
                            H_indices.add(rem_idx)
                            used_pairs_mask = used_pairs_mask | rem_pmask # Wait, used_pairs_mask was updated to temp.
                            # Correct revert:
                            used_pairs_mask = (used_pairs_mask_temp & ~add_pmask) | rem_pmask
                            # Actually, used_pairs_mask is currently used_pairs_mask_temp | add_pmask
                            # We want to go back to used_pairs_mask (which had rem_idx)
                            # So: used_pairs_mask = used_pairs_mask_temp | rem_pmask
                            # Wait, used_pairs_mask_temp = used_pairs_mask & ~rem_pmask
                            # So used_pairs_mask = used_pairs_mask_temp | rem_pmask is correct.
                            pass # The revert logic above is slightly messy, let's just recompute from H_indices if needed or be careful.
                            # Let's stick to simple revert:
                            # used_pairs_mask = (used_pairs_mask & ~add_pmask) | rem_pmask
                            # But used_pairs_mask was changed to temp | add.
                            # So: used_pairs_mask = (used_pairs_mask & ~add_pmask) | rem_pmask
                            # which is (temp | add & ~add) | rem = temp | rem = original.
                            # Yes.
                            used_pairs_mask = (used_pairs_mask & ~add_pmask) | rem_pmask
                        found_add = True
                        break
                        
                if not found_add:
                    # Try to add a new triple without removing?
                    # This only works if there are unused pairs.
                    # Pick random add_idx
                    for _ in range(5):
                        add_idx = random.randint(0, NUM_TRIPLES - 1)
                        if add_idx in H_indices:
                            continue
                        add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                        if (used_pairs_mask & add_pmask) == 0:
                            # Valid move
                            H_indices.add(add_idx)
                            used_pairs_mask |= add_pmask
                            
                            new_alpha = compute_alpha_fast(H_indices)
                            delta = new_alpha - current_alpha
                            
                            if delta <= 0 or random.random() < __import__('math').exp(-delta / temp):
                                current_alpha = new_alpha
                                if current_alpha < best_alpha:
                                    best_alpha = current_alpha
                                    best_system = [ALL_TRIPLES[i] for i in H_indices]
                                    last_improvement = time.time()
                            else:
                                # Revert
                                H_indices.remove(add_idx)
                                used_pairs_mask &= ~add_pmask
                            break
            
            # Cooling
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
                # If temp is low, we might want to restart or break
                # Let's break to new restart to get a fresh init
                break

    return best_system

def _fallback_system():
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

_orig_construct = construct

def construct_safe():
    try:
        res = _orig_construct()
        if not res:
            return _fallback_system()
        # Validate linearity just in case
        used = set()
        for t in res:
            i,j,k = t
            pairs = [tuple(sorted((i,j))), tuple(sorted((i,k))), tuple(sorted((j,k)))]
            for p in pairs:
                if p in used:
                    return _fallback_system()
                used.add(p)
        return res
    except:
        return _fallback_system()

# Replace construct
construct = construct_safe
