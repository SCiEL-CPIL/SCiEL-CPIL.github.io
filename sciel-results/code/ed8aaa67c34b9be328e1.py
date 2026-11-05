import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Greedy + Local Search.
    """
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute the list of t-subset indices for each k-block
    # This is expensive to do for ALL k-blocks upfront (346k * 35 ops), but allows fast lookups.
    # However, memory might be high. Let's compute on the fly or use a helper.
    # For v=24, k=7, t=3:
    # Number of blocks = 346104.
    # Each block has 35 triples.
    # Total triples to store = ~12M.
    # A list of lists of ints is heavy in Python.
    # Let's compute the triples for a block on the fly using a fast method.
    
    def get_block_triples(block):
        """Returns list of indices of t-subsets contained in block."""
        return [subset_to_idx[tuple(c)] for c in itertools.combinations(block, t)]

    # Greedy Initialization
    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    # To speed up greedy, we can precompute for each point x and each partial block P, the gain.
    # But P changes.
    # Simple greedy:
    while uncovered:
        # Pick the first uncovered subset to start the block
        first_uncovered_idx = next(iter(uncovered))
        first_uncovered_tuple = all_t_subsets[first_uncovered_idx]
        block = list(first_uncovered_tuple)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Calculate gain: how many new triples does adding x cover?
                # Triples involving x and t-1 elements from block
                # Number of such triples is C(len(block), t-1)
                gain = 0
                # Optimization: precompute combinations of block of size t-1?
                # len(block) is small (max 6).
                for rest in itertools.combinations(block, t - 1):
                    triple = tuple(sorted(rest + (x,)))
                    if triple in subset_to_idx:
                        idx = subset_to_idx[triple]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for idx in get_block_triples(block):
            uncovered.discard(idx)
            
    # Local Search
    # counts: numpy array of coverage counts for each t-subset
    counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b in blocks:
        for idx in get_block_triples(b):
            counts[idx] += 1
            
    # Precompute all possible k-blocks and their triple indices for faster swap search?
    # Let's generate them on demand but cache if necessary.
    # Actually, for v=24, k=7, iterating all 346k blocks in Python is slow (~0.1s per iteration if optimized? No, likely 1-2s).
    # We need to be careful.
    
    # Let's precompute the triple indices for all k-blocks once.
    # This takes time and memory, but makes the search loop fast.
    all_blocks = list(itertools.combinations(range(v), k))
    block_triples_cache = {}
    # We only need to cache the ones we look at frequently or all of them?
    # 346k blocks.
    # Let's just compute on the fly in the inner loop but optimize it.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Phase 1: Remove Redundant Blocks
        i = 0
        while i < len(blocks):
            b = blocks[i]
            # Check if any triple in b is uniquely covered by b
            is_redundant = True
            for idx in get_block_triples(b):
                if counts[idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block i
                for idx in get_block_triples(b):
                    counts[idx] -= 1
                blocks.pop(i)
                improved = True
                # Do not increment i, check the same index again (which is now the next block)
            else:
                i += 1
        
        # Phase 2: Swap Blocks
        # Try to replace a block with another that covers its unique triples
        for i in range(len(blocks)):
            if (time.time() - start_time) > time_limit:
                return blocks
            
            b = blocks[i]
            # Find unique triples
            unique_indices = [idx for idx in get_block_triples(b) if counts[idx] == 1]
            
            if not unique_indices:
                continue # Should have been removed, but maybe counts changed?
            
            # Search for a replacement block
            # Optimization: The replacement block must contain the points of the "most constrained" unique triple.
            # Let's pick the first unique triple and iterate blocks containing it.
            first_unique_tuple = all_t_subsets[unique_indices[0]]
            
            # Generate candidate blocks containing first_unique_tuple
            # We can iterate all blocks and check if they contain first_unique_tuple.
            # To make this faster, we can precompute a map: tuple -> list of blocks containing it?
            # Or just iterate. 346k checks is acceptable if done a few times.
            
            found_swap = False
            # To avoid checking the same block multiple times, we can shuffle or just iterate.
            for new_block in all_blocks:
                # Check if new_block is already in blocks (and not the one we are replacing)
                # This check is O(N).
                # Better: maintain a set of block tuples?
                # For now, simple check.
                if new_block == b:
                    continue
                
                # Check if new_block contains all unique triples
                # This is the expensive part.
                # We can prune: if new_block doesn't contain first_unique_tuple, skip.
                if not set(first_unique_tuple).issubset(set(new_block)):
                    continue
                
                # Check other unique triples
                valid = True
                for idx in unique_indices[1:]:
                    u_tuple = all_t_subsets[idx]
                    if not set(u_tuple).issubset(set(new_block)):
                        valid = False
                        break
                
                if valid:
                    # Check if new_block is already in the set of blocks
                    if new_block in blocks:
                        continue
                    
                    # Perform Swap
                    # Remove b
                    for idx in get_block_triples(b):
                        counts[idx] -= 1
                    # Add new_block
                    for idx in get_block_triples(new_block):
                        counts[idx] += 1
                    
                    blocks[i] = new_block
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                break # Restart loop to remove redundancies again
        
        # Optional Phase 3: 2-for-1 Swap
        # This is complex. Let's see if Phase 1+2 is enough.
        # If time allows, we can add it.
        
    return [list(b) for b in blocks]

# To speed up the "new_block in blocks" check and the subset checks, 
# we should use frozensets or sorted tuples and sets.
# The current implementation uses `set(new_block)` which creates a new set every time.
# We can precompute the sets for all_blocks? No, memory.
# We can check membership by `all(x in new_block for x in u_tuple)`.

# Let's refine the inner loop for speed.

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    def get_block_triples(block):
        return [subset_to_idx[tuple(c)] for c in itertools.combinations(block, t)]

    # Greedy
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    while uncovered:
        first_uncovered_idx = next(iter(uncovered))
        first_uncovered_tuple = all_t_subsets[first_uncovered_idx]
        block = list(first_uncovered_tuple)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    triple = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx[triple]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        for idx in get_block_triples(block):
            uncovered.discard(idx)
            
    counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b in blocks:
        for idx in get_block_triples(b):
            counts[idx] += 1
            
    all_blocks = list(itertools.combinations(range(v), k))
    # Precompute sets for all_blocks? 
    # all_block_sets = [set(b) for b in all_blocks] -> 346k sets. Might be slow to create.
    # Let's just use tuple membership.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Phase 1: Remove Redundant
        i = 0
        while i < len(blocks):
            b = blocks[i]
            redundant = True
            for idx in get_block_triples(b):
                if counts[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for idx in get_block_triples(b):
                    counts[idx] -= 1
                blocks.pop(i)
                improved = True
            else:
                i += 1
        
        # Phase 2: Swap
        for i in range(len(blocks)):
            if (time.time() - start_time) > time_limit:
                return [list(b) for b in blocks]
            
            b = blocks[i]
            unique_indices = [idx for idx in get_block_triples(b) if counts[idx] == 1]
            if not unique_indices:
                continue
            
            first_unique_tuple = all_t_subsets[unique_indices[0]]
            
            # Search for replacement
            for new_block in all_blocks:
                if new_block == b:
                    continue
                
                # Quick check: does new_block contain first_unique_tuple?
                # Use a helper for speed
                # Since t=3, we can check:
                if not (first_unique_tuple[0] in new_block and first_unique_tuple[1] in new_block and first_unique_tuple[2] in new_block):
                    continue
                
                # Check other unique triples
                valid = True
                for idx in unique_indices[1:]:
                    u_t = all_t_subsets[idx]
                    if not (u_t[0] in new_block and u_t[1] in new_block and u_t[2] in new_block):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check if new_block is already in blocks
                # This is O(N).
                if new_block in blocks:
                    continue
                
                # Swap
                for idx in get_block_triples(b):
                    counts[idx] -= 1
                for idx in get_block_triples(new_block):
                    counts[idx] += 1
                
                blocks[i] = new_block
                improved = True
                break
            
            if improved:
                break
        
    return [list(b) for b in blocks]
