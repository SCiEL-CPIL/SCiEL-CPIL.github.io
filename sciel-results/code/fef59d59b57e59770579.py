"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    all_tsubsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Track coverage: for each t-subset, how many blocks cover it
    coverage = [0] * n_tsubsets
    blocks = []
    
    # Precompute: for each t-subset, list of t-subsets it's contained in
    # Actually, let's precompute for each point x and each (t-1)-subset, the t-subset
    # For gain computation: given partial block P, gain(x) = #{S in C(P, t-1) : S ∪ {x} uncovered}
    
    # Greedy phase
    uncovered_count = n_tsubsets
    uncovered_set = set(range(n_tsubsets))
    
    while uncovered_set and time.time() < deadline - 5.0:
        # Pick the most "constrained" uncovered t-subset (one with fewest possible blocks)
        # For simplicity, pick the first uncovered one
        target_idx = min(uncovered_set)
        target = all_tsubsets[target_idx]
        
        # Start block with the target subset
        block = list(target)
        
        while len(block) < k:
            block_set = set(block)
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                # Gain: number of (t-1)-subsets of block that with x form an uncovered t-subset
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tset_to_idx[ts]
                    if idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            block.append(best_x)
        
        block.sort()
        
        # Add block and update coverage
        blocks.append(block)
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx[ts]
            if idx in uncovered_set:
                uncovered_set.discard(idx)
                uncovered_count -= 1
        
        # Periodically check time
        if time.time() > deadline - 5.0:
            break
    
    # If we ran out of time before covering everything, we have a problem
    # But let's try to continue if we have time
    while uncovered_set and time.time() < deadline - 2.0:
        target_idx = min(uncovered_set)
        target = all_tsubsets[target_idx]
        
        block = list(target)
        while len(block) < k:
            block_set = set(block)
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tset_to_idx[ts]
                    if idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx[ts]
            uncovered_set.discard(idx)
    
    # Post-processing: try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    if time.time() < deadline - 5.0:
        # Rebuild coverage counts
        coverage = [0] * n_tsubsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                coverage[tset_to_idx[ts]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                # Check if all t-subsets of this block are covered by other blocks
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = tset_to_idx[ts]
                    if coverage[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for ts in itertools.combinations(block, t):
                        coverage[tset_to_idx[ts]] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
    
    # Local search: try to swap points in blocks to reduce count
    if time.time() < deadline - 5.0:
        # Rebuild coverage
        coverage = [0] * n_tsubsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                coverage[tset_to_idx[ts]] += 1
        
        # Try swapping points in each block
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i, block in enumerate(blocks):
                if time.time() > deadline - 5.0:
                    break
                block_set = set(block)
                for j in range(len(block)):
                    for x in range(v):
                        if x in block_set:
                            continue
                        # Try swapping block[j] with x
                        old_point = block[j]
                        # Check if removing old_point still leaves all t-subsets covered
                        # and adding x doesn't cause issues
                        # Simulate: remove block[i], add new block with x instead of old_point
                        # Count how many t-subsets would become uncovered
                        new_block = block[:j] + [x] + block[j+1:]
                        new_block.sort()
                        
                        # Check: after removing block[i], are all t-subsets of block[i] still covered?
                        # And does new_block cover any additional uncovered t-subsets?
                        # This is complex; let's just try it and check validity
                        if time.time() > deadline - 5.0:
                            break
                        
                        # For now, skip complex local search due to time constraints
                        # Just do the redundancy removal which is simpler
                        pass
    
    return blocks

# EVOLVE-BLOCK-END
