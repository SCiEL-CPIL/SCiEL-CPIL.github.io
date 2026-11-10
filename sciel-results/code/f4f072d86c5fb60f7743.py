import itertools
import time
import random
import sys

_DEADLINE_S = 120.0
_MARGIN_S = 5.0

def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)
    
    def remaining():
        return max(0.0, _DEADLINE_S - _MARGIN_S - (time.time() - start_time))

    # Precompute all t-subsets and map them to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    subset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Helper to get the index of a t-subset
    def get_idx(ts):
        return subset_to_idx[ts]

    # Helper to get all t-subsets of a block
    def block_subsets(block):
        return [get_idx(ts) for ts in itertools.combinations(block, t)]

    # --- Greedy Construction ---
    # We maintain a list of uncovered subsets (indices)
    uncovered = set(range(n_tsubsets))
    blocks = []
    
    # Precompute for each point, which subsets it is part of?
    # Or just calculate gain on the fly.
    # Gain of adding x to current partial block P:
    # Number of subsets in uncovered that can be formed by P + x.
    
    def gain_for_point(partial_block, x, uncovered_set):
        # partial_block is a list of points
        # We need to form t-subsets using x and t-1 points from partial_block
        # But wait, the subset must be a subset of the final block.
        # The final block will be partial_block + x + ...
        # The subsets covered by adding x to partial_block are those
        # that contain x and t-1 points from partial_block.
        # However, these subsets must also be subsets of the *final* block.
        # Since we are building the block, any subset of the current partial + x
        # is a valid candidate.
        
        cnt = 0
        # We need t-1 points from partial_block
        # If len(partial_block) < t-1, gain is 0 (can't form a t-subset yet)
        if len(partial_block) < t - 1:
            return 0
            
        for rest in itertools.combinations(partial_block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            idx = subset_to_idx.get(ts, -1)
            if idx != -1 and idx in uncovered_set:
                cnt += 1
        return cnt

    while uncovered and remaining() > 1.0:
        # Pick a target uncovered subset
        target_idx = min(uncovered)
        target_ts = all_tsubsets[target_idx]
        
        block = list(target_ts)
        block.sort()
        
        while len(block) < k and remaining() > 0.5:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                g = gain_for_point(block, x, uncovered)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None or best_gain == 0:
                # If no gain, pick random to break tie or just first
                best_x = rng.choice(candidates)
            block.append(best_x)
        
        block.sort()
        if len(set(block)) != k:
            continue
            
        # Update uncovered
        covered_idxs = block_subsets(block)
        uncovered -= set(covered_idxs)
        blocks.append(block)

    # If we ran out of time during construction, return what we have (if valid)
    # But we need to ensure validity. The greedy ensures validity by construction.
    
    # --- Local Improvement Phase ---
    if remaining() > 2.0:
        # Initialize coverage counts
        cover_count = [0] * n_tsubsets
        for b in blocks:
            for idx in block_subsets(b):
                cover_count[idx] += 1

        improved = True
        while improved and remaining() > 2.0:
            improved = False
            
            # Step 1: Remove redundant blocks
            new_blocks = []
            for b in blocks:
                redundant = True
                for idx in block_subsets(b):
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(b)
                else:
                    # Update counts
                    for idx in block_subsets(b):
                        cover_count[idx] -= 1
                    improved = True
            
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                continue # Restart to find more redundants

            # Step 2: Try to improve blocks (1-for-1 swap)
            n = len(blocks)
            for i in range(n):
                if remaining() < 2.0:
                    break
                b = blocks[i]
                cov_b = block_subsets(b)
                
                # Find unique subsets
                unique_idxs = [idx for idx in cov_b if cover_count[idx] == 1]
                if not unique_idxs:
                    # Block is redundant? Should have been removed.
                    # Maybe it became redundant due to previous swaps in this pass?
                    # If so, we can just remove it.
                    for idx in cov_b:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break # Restart loop
                    
                # Try to construct a new block that covers unique_idxs
                # Strategy: Start with one unique subset, greedily add points
                # to cover other unique subsets.
                
                target_idx = unique_idxs[0]
                target_ts = all_tsubsets[target_idx]
                new_block = list(target_ts)
                new_block.sort()
                
                # Greedily fill
                while len(new_block) < k and remaining() > 0.5:
                    candidates = [x for x in range(v) if x not in new_block]
                    if not candidates:
                        break
                    best_x = None
                    best_gain = -1
                    for x in candidates:
                        # Gain is number of unique_idxs covered by adding x
                        g = 0
                        for rest in itertools.combinations(new_block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = subset_to_idx.get(ts, -1)
                            if idx in unique_idxs: # This check is slow, use set
                                g += 1
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                    if best_x is None or best_gain == 0:
                        best_x = rng.choice(candidates)
                    new_block.append(best_x)
                
                new_block.sort()
                if len(set(new_block)) != k:
                    continue
                    
                cov_new = block_subsets(new_block)
                cov_new_set = set(cov_new)
                
                # Check if new block covers all unique subsets
                if set(unique_idxs).issubset(cov_new_set) and new_block != b:
                    # Perform swap
                    # 1. Remove old block b from counts
                    for idx in cov_b:
                        cover_count[idx] -= 1
                    # 2. Add new block to counts
                    for idx in cov_new:
                        cover_count[idx] += 1
                    # 3. Replace block
                    blocks[i] = new_block
                    improved = True
                    break # Restart to check for new redundancies

    return blocks
