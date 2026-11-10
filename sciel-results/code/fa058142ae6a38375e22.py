"""Direct/structured covering for C(19,8,5) with deadline guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0


def construct(v, k, t):
    start_time = time.time()

    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}

    def block_mask(block):
        mask = np.zeros(n_t, dtype=bool)
        for comb in itertools.combinations(block, t):
            mask[idx_map[tuple(comb)]] = True
        return mask

    def greedy_cover(uncovered_mask, blocks_list, time_budget):
        """Greedy: add blocks to cover all uncovered subsets."""
        while uncovered_mask.any():
            if time.time() - start_time > time_budget:
                break
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            start_subset = all_t_subsets[start_idx]
            block = list(start_subset)

            while len(block) < k:
                if time.time() - start_time > time_budget:
                    break
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        si = idx_map.get(s, -1)
                        if si >= 0 and uncovered_mask[si]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)

            block.sort()
            mask = block_mask(block)
            new_covered = mask & uncovered_mask
            if new_covered.any():
                blocks_list.append(block)
                uncovered_mask = uncovered_mask & ~mask

        return blocks_list, uncovered_mask

    # Phase 1: Initial greedy cover
    uncovered_mask = np.ones(n_t, dtype=bool)
    blocks = []
    blocks, uncovered_mask = greedy_cover(uncovered_mask, blocks, DEADLINE_SECONDS - 60)

    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    blocks = unique_blocks

    # Phase 2: Local search - try to replace pairs of blocks with one
    # For each pair (i,j), check if removing both and adding a new block
    # that covers all subsets they jointly cover is possible.
    # Actually simpler: try removing one block at a time and see if remaining
    # blocks still cover everything. If not, try to find a replacement.
    
    n_blocks = len(blocks)
    # Precompute masks for all blocks
    block_masks = [block_mask(b) for b in blocks]
    
    # Compute coverage count: how many blocks cover each t-subset
    coverage_count = np.zeros(n_t, dtype=np.int32)
    for bm in block_masks:
        coverage_count += bm.astype(np.int32)
    
    # Try to remove redundant blocks (those whose removal doesn't uncover anything)
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE_SECONDS - 30:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() - start_time > DEADLINE_SECONDS - 30:
                break
            # Check if block i is redundant: removing it doesn't leave any subset uncovered
            # i.e., for all subsets covered by block i, at least one other block covers them
            other_coverage = np.zeros(n_t, dtype=np.int32)
            for j in range(len(blocks)):
                if j != i:
                    other_coverage += block_masks[j].astype(np.int32)
            
            # If all subsets covered by block i are also covered by others
            if not (block_masks[i] & (other_coverage == 0)).any():
                # Block i is redundant, remove it
                del blocks[i]
                del block_masks[i]
                improved = True
                break  # Restart loop since indices changed
    
    # Phase 3: Try 2-for-1 exchanges
    # For each pair of blocks, try to find a single block that covers
    # all subsets that the pair uniquely covers (not covered by any other block)
    if time.time() - start_time < DEADLINE_SECONDS - 10:
        for iteration in range(50):  # Limited iterations
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
            
            # Recompute coverage counts
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bm in block_masks:
                coverage_count += bm.astype(np.int32)
            
            # Find pairs where one block's unique contribution can be absorbed
            found_improvement = False
            n_b = len(blocks)
            
            for i in range(n_b):
                if time.time() - start_time > DEADLINE_SECONDS - 10:
                    break
                if found_improvement:
                    break
                    
                # Unique subsets covered only by block i
                unique_to_i = block_masks[i] & (coverage_count == 1)
                if not unique_to_i.any():
                    continue
                    
                # Try to find a block that covers all unique_to_i subsets
                # The new block must be a superset of at least one of the unique subsets
                unique_indices = np.where(unique_to_i)[0]
                
                # Try random candidates: pick a point from a unique subset, 
                # then greedily build a block around it
                best_new_block = None
                best_new_mask = None
                
                for attempt in range(20):
                    if time.time() - start_time > DEADLINE_SECONDS - 10:
                        break
                    
                    # Pick a random unique subset to anchor the new block
                    anchor_idx = unique_indices[random.randint(0, len(unique_indices) - 1)]
                    anchor = all_t_subsets[anchor_idx]
                    new_block = list(anchor)
                    
                    # Greedily add points to cover more unique_to_i subsets
                    for _ in range(k - t):
                        if len(new_block) >= k:
                            break
                        best_x = None
                        best_gain = -1
                        for x in range(v):
                            if x in new_block:
                                continue
                            gain = 0
                            for rest in itertools.combinations(new_block, t - 1):
                                s = tuple(sorted(rest + (x,)))
                                si = idx_map.get(s, -1)
                                if si >= 0 and unique_to_i[si]:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_block.append(best_x)
                    
                    new_block.sort()
                    new_mask = block_mask(new_block)
                    
                    # Check if new block covers all unique_to_i subsets
                    if not (unique_to_i & ~new_mask).any():
                        # Also ensure new block is different from existing
                        if tuple(new_block) not in seen:
                            best_new_block = new_block
                            best_new_mask = new_mask
                            break
                
                if best_new_block is not None:
                    # Replace block i with best_new_block
                    blocks[i] = best_new_block
                    block_masks[i] = best_new_mask
                    seen.add(tuple(best_new_block))
                    found_improvement = True
    
    # Final deduplication
    seen = set()
    final_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            final_blocks.append(b)
    
    return final_blocks


def _generic_greedy(v, k, t, start_time):
    """Fallback for non-(19,8,5) parameters."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            block.append(max(candidates, key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
# EVOLVE-BLOCK-END
