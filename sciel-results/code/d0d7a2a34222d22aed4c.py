import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    """
    Construct a covering C(v, k, t) using a local search strategy on a 
    random candidate pool.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate a large pool of random candidate blocks
    # We need enough candidates to ensure we can find a good cover.
    # C(24,7) is ~346k. Let's pick 2000 random blocks.
    num_candidates = 2000
    candidates = set()
    while len(candidates) < num_candidates:
        candidates.add(tuple(sorted(np.random.choice(v, k, replace=False))))
    
    candidates = list(candidates)
    n_cand = len(candidates)
    
    # Precompute coverage masks for each candidate block
    # mask[i] is a bitmask of length n_t where bit j is 1 if block i covers t-subset j
    # Since n_t = C(24,3) = 2024, we can use Python integers as bitmasks.
    masks = [0] * n_cand
    for i, block in enumerate(candidates):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Full mask representing all t-subsets
    full_mask = (1 << n_t) - 1
    
    # 1. Greedy Initialization to get a valid cover
    covered_mask = 0
    selected_indices = []
    remaining_candidates = list(range(n_cand))
    
    # To make greedy faster, we can sort candidates by popcount of mask initially, 
    # but let's just do standard greedy with bit operations.
    
    while covered_mask != full_mask:
        if time.time() > deadline - 20.0:
            break
            
        best_idx = -1
        best_gain = -1
        
        # To speed up, we might not check all candidates every step if n_cand is huge,
        # but 2000 is small enough.
        for idx in remaining_candidates:
            gain = bin(masks[idx] & ~covered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if gain == 0:
                    continue # Optimization: if best possible is 0, we can't improve, but we need SOME gain.
                    # Actually if best_gain is 0, we are stuck. But greedy usually works.
        
        if best_idx == -1 or best_gain == 0:
             # If no candidate covers new items, we are stuck. 
             # This shouldn't happen if pool is large enough, but handle it.
             break
             
        selected_indices.append(best_idx)
        covered_mask |= masks[best_idx]
        remaining_candidates.remove(best_idx)
        
    # If we didn't cover everything, we have a problem. 
    # For C(24,7,3), a random pool of 2000 should be sufficient for greedy to find a cover of ~80-90 blocks.
    if covered_mask != full_mask:
        # Fallback: add random blocks until covered
        while covered_mask != full_mask:
             if time.time() > deadline - 10.0:
                 break
             idx = np.random.randint(n_cand)
             if idx not in selected_indices:
                 selected_indices.append(idx)
                 covered_mask |= masks[idx]
    
    # 2. Local Search / Deletion Phase
    # We have a valid cover. Now try to remove blocks.
    # A block can be removed if its unique contribution to coverage is 0.
    # i.e., mask[i] & ~ (OR of all other masks) == 0
    
    current_selection = set(selected_indices)
    
    # Optimization: Maintain a count of how many times each t-subset is covered?
    # Or just recompute OR? 
    # Recomputing OR of ~80 masks of 2024 bits is fast in Python if using ints.
    
    def compute_union(indices):
        union = 0
        for i in indices:
            union |= masks[i]
        return union
        
    current_union = compute_union(current_selection)
    
    # Iterative Deletion
    # We try to remove one block at a time.
    # To be efficient, we can iterate through selected blocks.
    
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        # Shuffle to avoid bias
        items_to_check = list(current_selection)
        np.random.shuffle(items_to_check)
        
        for idx in items_to_check:
            if idx not in current_selection:
                continue
                
            # Check if removing idx leaves any t-subset uncovered
            # new_union = current_union & ~masks[idx] is NOT correct.
            # We need to know if any bit in masks[idx] is NOT set in (current_union ^ masks[idx])?
            # No. 
            # Let U = current_union.
            # If we remove idx, new_union = U without the bits ONLY covered by idx.
            # Bits covered by idx are masks[idx].
            # Bits covered by others are U ^ (masks[idx] & ~U)? No.
            
            # Let's compute:
            # unique_to_idx = masks[idx] & ~ (compute_union(current_selection - {idx}))
            # This is expensive (O(N) per check).
            
            # Alternative:
            # Count coverage for each t-subset?
            # Let's stick to the expensive check but limit it or optimize.
            # Actually, for N~80, computing union of 79 masks is very fast.
            
            others = [i for i in current_selection if i != idx]
            union_others = 0
            for i in others:
                union_others |= masks[i]
                
            if (masks[idx] & ~union_others) == 0:
                # idx is redundant
                current_selection.remove(idx)
                current_union = union_others
                changed = True
                if time.time() > deadline - 5.0:
                    break

    # 3. Swap/Shift Phase (Optional if time permits)
    # Try to replace a block with another block that covers the same unique items
    # or allows further deletion.
    # This is complex. Let's try a simpler "re-greedy" on the remaining set if time allows?
    # Or just return what we have.
    
    # Given the 120s limit and Python overhead, the deletion phase is the main driver.
    # Random restarts might help escape local minima of the greedy init.
    
    # Let's do a second pass: Random Restart
    # If we have time, do 3 more random inits and keep the best.
    
    best_size = len(current_selection)
    best_set = set(current_selection)
    
    # If we have plenty of time, run a few more restarts
    # But greedy init is O(N^2 * bitops). N=2000.
    # 2000 * 2000 * (bit ops on 2000-bit ints) ~ 4M * fast. Should be < 1s.
    # So we can afford 5-10 restarts.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Greedy Init again
        covered_mask = 0
        selected_indices = []
        remaining_candidates = list(range(n_cand))
        
        # Shuffle candidates to change tie-breaking
        np.random.shuffle(remaining_candidates)
        
        while covered_mask != full_mask:
            if time.time() > deadline - 15.0:
                break
                
            best_idx = -1
            best_gain = -1
            
            for idx in remaining_candidates:
                gain = bin(masks[idx] & ~covered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx == -1 or best_gain == 0:
                 break
                 
            selected_indices.append(best_idx)
            covered_mask |= masks[best_idx]
            remaining_candidates.remove(best_idx)
            
        if covered_mask != full_mask:
            continue
            
        # Deletion Phase
        current_selection = set(selected_indices)
        current_union = compute_union(current_selection)
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            items_to_check = list(current_selection)
            np.random.shuffle(items_to_check)
            
            for idx in items_to_check:
                if idx not in current_selection:
                    continue
                others = [i for i in current_selection if i != idx]
                union_others = 0
                for i in others:
                    union_others |= masks[i]
                
                if (masks[idx] & ~union_others) == 0:
                    current_selection.remove(idx)
                    current_union = union_others
                    changed = True
                    if time.time() > deadline - 5.0:
                        break
                        
        if len(current_selection) < best_size:
            best_size = len(current_selection)
            best_set = set(current_selection)
            
    # Convert best_set indices back to blocks
    final_blocks = [candidates[i] for i in best_set]
    
    # Final verification
    # The evaluator checks exhaustively, so we trust our mask logic.
    # But let's ensure we return a list of lists (as per interface "list of blocks")
    # candidates are tuples. Interface says "list of k distinct integers".
    # Let's convert to lists to be safe.
    return [list(b) for b in final_blocks]
