"""Vectorized greedy covering for C(19,8,5) with time budget."""
import time
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin
    
    # Precompute all t-subsets as a lookup table
    # Map each t-subset to an index 0..n_subsets-1
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # For each possible block (k-subset), precompute which t-subsets it covers
    # This is expensive for v=19, k=8: C(19,8) = 75582 blocks, each covers C(8,5)=56 subsets
    # Total: 75582 * 56 = ~4.2M entries
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build coverage matrix: for each k-block, which subsets it covers
    # Use a sparse-like representation: list of arrays
    block_covers = []
    for block in all_k_blocks:
        covered = []
        for sub in itertools.combinations(block, t):
            covered.append(subset_to_idx[sub])
        block_covers.append(np.array(covered, dtype=np.int32))
    
    # Greedy covering with vectorized gain computation
    uncovered_mask = np.ones(n_subsets, dtype=bool)
    remaining = n_subsets
    chosen_blocks = []
    
    while remaining > 0:
        if time.time() > deadline - 5.0:
            # Not enough time for another block; break early
            break
        
        # Compute gain for each candidate block: number of currently uncovered subsets it covers
        gains = np.zeros(n_k_blocks, dtype=np.int32)
        for i, cover in enumerate(block_covers):
            gains[i] = np.count_nonzero(uncovered_mask[cover])
        
        # Pick the block with maximum gain
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # No block can cover any remaining subset (shouldn't happen if remaining > 0)
            break
        
        chosen_blocks.append(all_k_blocks[best_idx])
        
        # Update uncovered mask
        uncovered_mask[block_covers[best_idx]] = False
        remaining -= best_gain
        
        # Early check
        if time.time() > deadline - 3.0:
            break
    
    # If we ran out of time, try to complete the covering with a simpler method
    if remaining > 0 and time.time() < deadline - 10.0:
        # Fallback: continue with non-vectorized greedy on remaining
        uncovered_set = set()
        for i in range(n_subsets):
            if uncovered_mask[i]:
                uncovered_set.add(all_subsets[i])
        
        while uncovered_set and time.time() < deadline - 2.0:
            block = list(min(uncovered_set))
            while len(block) < k:
                best_x = None
                best_count = -1
                for x in range(v):
                    if x in block:
                        continue
                    count = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered_set:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_x = x
                block.append(best_x)
            block.sort()
            chosen_blocks.append(tuple(block))
            for sub in itertools.combinations(block, t):
                uncovered_set.discard(sub)
    
    # Final validation: ensure all t-subsets are covered
    # If not fully covered, add more blocks via simple greedy
    covered_subsets = set()
    for block in chosen_blocks:
        for sub in itertools.combinations(block, t):
            covered_subsets.add(sub)
    
    if len(covered_subsets) < n_subsets and time.time() < deadline - 2.0:
        uncovered_final = set(all_subsets) - covered_subsets
        while uncovered_final and time.time() < deadline - 1.0:
            block = list(min(uncovered_final))
            while len(block) < k:
                best_x = None
                best_count = -1
                for x in range(v):
                    if x in block:
                        continue
                    count = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered_final:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_x = x
                block.append(best_x)
            block.sort()
            chosen_blocks.append(tuple(block))
            for sub in itertools.combinations(block, t):
                uncovered_final.discard(sub)
    
    # Remove duplicate blocks
    unique_blocks = []
    seen = set()
    for block in chosen_blocks:
        tblock = tuple(sorted(block))
        if tblock not in seen:
            seen.add(tblock)
            unique_blocks.append(list(tblock))
    
    # Sort for determinism
    unique_blocks.sort()
    
    return unique_blocks
