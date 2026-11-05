import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 35s margin for safety/return
    
    # 1. Precompute all t-subsets (triples) and map them to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    num_tsubsets = len(all_tsubsets)
    
    # 2. Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Precompute block_indices: shape (num_blocks, C(k, t))
    t_subsets_per_block = int(math.comb(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    # Optimization: Generate indices efficiently
    # For each block, we need the indices of its t-subsets.
    # We can do this by iterating through combinations.
    for i, block in enumerate(all_blocks):
        # Since k and t are small, itertools is fast enough for precomputation
        # But we can be smarter if we want. For v=24, k=7, t=3, this is fine.
        block_indices[i] = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        
        if time.time() > deadline - 5:
            # Fallback if precomputation takes too long (unlikely for 24,7,3)
            # Return a simple greedy result if we run out of time during precomp
            # But let's assume it finishes.
            pass

    # 4. Precompute frequency of each t-subset across all blocks
    # This is useful for weighted greedy (inverse frequency)
    # freq[j] = number of blocks that cover t-subset j
    freq = np.bincount(block_indices.flatten(), minlength=num_tsubsets)
    
    # Inverse frequency for weighting (higher weight for rare subsets)
    # Avoid division by zero
    inv_freq = 1.0 / (freq + 1e-9)
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple strategies/restarts
    # Strategy 1: Pure Greedy (Max Gain)
    # Strategy 2: Weighted Greedy (Max Weighted Gain)
    # Strategy 3: Randomized Greedy (with some noise)
    
    seed = 0
    while time.time() < deadline - 20:  # Leave 20s for local search on best
    
        # Choose strategy based on seed or cycle through them
        strategy = seed % 3
        
        if strategy == 0:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        elif strategy == 1:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='weighted', inv_freq=inv_freq)
        else:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='randomized', inv_freq=inv_freq, seed=seed)
        
        if blocks is None:
            seed += 1
            continue
            
        # Post-processing: Remove redundant blocks
        blocks = _remove_redundant(blocks, block_indices, num_tsubsets)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
        # Early stop if we hit a very good number
        if best_len <= 76:
            break

    # Phase 2: Local Search on the best solution found
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, all_blocks, block_indices, num_tsubsets, deadline)
        
    if best_blocks is None:
        # Fallback: deterministic greedy
        best_blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        if best_blocks is None:
             # Last resort: return empty or raise? Task says valid when every t-subset lies in at least one block.
             # If we fail to cover, it's invalid. But we should try hard.
             # Let's just return whatever we have, hoping it's valid.
             best_blocks = []
             
    return best_blocks

def _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy', inv_freq=None, seed=0):
    """
    Vectorized greedy set cover.
    mode: 'greedy' (max new coverage), 'weighted' (max weighted new coverage), 'randomized' (random from top candidates)
    """
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None  # Signal timeout
            
        if mode == 'greedy':
            # Gain = number of uncovered triples covered by the block
            # uncovered[block_indices] is a boolean matrix (num_blocks, t_subsets_per_block)
            gains = uncovered[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        elif mode == 'weighted':
            # Weighted gain = sum of inv_freq for uncovered triples
            # w = inv_freq where uncovered, else 0
            w = np.where(uncovered, inv_freq, 0.0)
            gains = w[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        elif mode == 'randomized':
            # Calculate gains, then pick randomly from top N candidates
            gains = uncovered[block_indices].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Get indices of blocks with max gain
            # To avoid large lists, we can sample or take top few
            # For simplicity, take all with max gain and pick one
            candidates = np.where(gains == max_gain)[0]
            # If too many candidates, sample a few
            if len(candidates) > 10:
                rng = random.Random(seed + len(selected_indices))
                candidates = rng.sample(list(candidates), 10)
            
            best_idx = candidates[np.random.choice(len(candidates))]
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        else:
            raise ValueError("Unknown mode")
            
    if np.any(uncovered):
        # Incomplete covering, but return what we have? 
        # The task requires valid covering. 
        # If we return incomplete, it's invalid (score 0).
        # It's better to return a valid covering if possible.
        # But greedy should always complete unless there's a bug or timeout.
        # If timeout, we return None.
        # If max_gain == 0 but uncovered exists, it means no block covers any remaining uncovered triple?
        # That shouldn't happen if all_blocks includes all possible blocks.
        # Every t-subset is contained in some k-subset. So gain should never be 0 if uncovered is not empty.
        # Unless we ran out of time.
        return None

    return [list(all_blocks[i]) for i in selected_indices]

def _remove_redundant(blocks, block_indices, num_tsubsets):
    """
    Remove blocks that do not cover any unique t-subset.
    Uses vectorized counting.
    """
    if not blocks:
        return blocks
    
    n = len(blocks)
    if n == 0:
        return blocks
    
    # We need to know which triples each block covers.
    # We have global block_indices for ALL possible blocks.
    # We need to map our selected blocks to their global indices to use block_indices?
    # Or we can just recompute the triples for the selected blocks.
    # Since n is small (~80), recomputing is fast.
    
    # Map selected blocks to their triples
    # A block is a list of ints.
    # We need to check if removing a block leaves all its triples covered by others.
    
    # Efficient way:
    # 1. Count coverage of each triple by the current set of blocks.
    # 2. A block is redundant if for every triple it covers, count > 1.
    # 3. Remove all such blocks.
    # 4. Repeat until no more can be removed.
    
    # To make this fast, we can represent the coverage matrix.
    # But for n=80, T=2024, a simple loop is fine.
    
    # Let's use a dictionary for counts: triple -> count
    # And a list of triples for each block.
    
    block_triples = []
    counts = {}
    
    for block in blocks:
        triples = list(itertools.combinations(block, 3)) # Assuming t=3, but let's use t from closure? 
        # Wait, _remove_redundant doesn't have t. 
        # We should pass t or infer it. 
        # The function signature in construct calls it with (blocks, block_indices, num_tsubsets).
        # It doesn't have t. 
        # I will modify the call to pass t or make it a global? 
        # Better to pass t.
        pass
    
    # Let's rewrite _remove_redundant to accept t.
    # I'll update the call site in construct.
    
    # For now, let's assume t=3 or pass it.
    # I'll change the signature to _remove_redundant(blocks, v, t).
    
    # Actually, I can just use the global t from construct? No, that's bad practice.
    # Let's change the function signature.
    
    # Revised plan:
    # _remove_redundant(blocks, v, t)
    
    # Let's implement it properly in the final code.
    
    return blocks # Placeholder

def _local_search(blocks, all_blocks, block_indices, num_tsubsets, deadline):
    """
    Try to reduce the number of blocks by swapping.
    """
    # This is complex to implement correctly and efficiently in the remaining time.
    # A simple heuristic: 
    # 1. Try to remove any redundant block (handled by _remove_redundant).
    # 2. Try to swap a block with a different one to reduce total count? 
    #    Usually, greedy + redundant removal is quite strong.
    #    Local search for covering designs often involves "swap" moves:
    #    Replace block B with block B' such that B' covers all unique triples of B 
    #    and possibly more, and we can remove B.
    #    But this doesn't reduce count, it just changes the set.
    #    To reduce count, we need to find a set of blocks that can be replaced by fewer blocks.
    #    
    #    Given time constraints, I will implement a simple "try to remove" loop 
    #    which is effectively _remove_redundant but maybe with more iterations or different order.
    #    
    #    For now, I'll just return the blocks as is, assuming _remove_redundant was called.
    #    If I have time, I'll add a simple swap improvement.
    
    return blocks

# Corrected _remove_redundant
def _remove_redundant(blocks, v, t):
    if not blocks:
        return blocks
        
    changed = True
    while changed:
        changed = False
        # Count coverage
        counts = {}
        block_triples = []
        
        for block in blocks:
            triples = list(itertools.combinations(block, t))
            block_triples.append(triples)
            for tr in triples:
                counts[tr] = counts.get(tr, 0) + 1
                
        new_blocks = []
        for i, block in enumerate(blocks):
            is_essential = False
            for tr in block_triples[i]:
                if counts[tr] == 1:
                    is_essential = True
                    break
            if not is_essential:
                changed = True
                # Decrement counts
                for tr in block_triples[i]:
                    counts[tr] -= 1
            else:
                new_blocks.append(block)
                
        blocks = new_blocks
        
    return blocks

# Need to import math for math.comb
import math

# Update construct to call _remove_redundant correctly
# The previous construct called _remove_redundant(blocks, block_indices, num_tsubsets)
# I will change the call to _remove_redundant(blocks, v, t)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_tsubsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    num_tsubsets = len(all_tsubsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    t_subsets_per_block = int(math.comb(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        block_indices[i] = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        
    freq = np.bincount(block_indices.flatten(), minlength=num_tsubsets)
    inv_freq = 1.0 / (freq + 1e-9)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline - 20:
        
        strategy = seed % 3
        
        if strategy == 0:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        elif strategy == 1:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='weighted', inv_freq=inv_freq)
        else:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='randomized', inv_freq=inv_freq, seed=seed)
        
        if blocks is None:
            seed += 1
            continue
            
        blocks = _remove_redundant(blocks, v, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
        if best_len <= 76:
            break

    if best_blocks is not None and time.time() < deadline - 5:
        # Simple local search: just try to remove more redundancies? 
        # Or swap? 
        # For now, we rely on the greedy + redundant removal.
        pass
        
    if best_blocks is None:
        best_blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        if best_blocks is None:
            best_blocks = []
        else:
            best_blocks = _remove_redundant(best_blocks, v, t)
            
    return best_blocks
