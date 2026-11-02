import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave buffer
    
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(range(v), t))
    tsubs_set = set(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute all blocks and their tsub indices
    all_blocks = []
    block_tsub_indices = []
    block_to_idx = {}
    tsub_to_blocks = defaultdict(list)
    
    # Optimization: Generate blocks and their tsub indices
    for block in itertools.combinations(range(v), k):
        b = list(block)
        b.sort()
        tsubs = list(itertools.combinations(b, t))
        tsub_indices = [tsub_to_idx[ts] for ts in tsubs]
        
        idx = len(all_blocks)
        all_blocks.append(b)
        block_tsub_indices.append(tsub_indices)
        block_to_idx[tuple(b)] = idx
        
        for ti in tsub_indices:
            tsub_to_blocks[all_tsubs[ti]].append(idx)
            
    # Greedy Cover
    def greedy_cover(seed):
        r = random.Random(seed)
        # Use a simple greedy: pick random uncovered tsub, pick best block covering it
        # To speed up, we can maintain a count of uncovered tsubs?
        # Or just use a set of uncovered tsub indices.
        uncovered = set(range(num_tsubs))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered tsub
            ts_idx = r.choice(list(uncovered))
            target_ts = all_tsubs[ts_idx]
            
            # Find best block covering target_ts
            candidates = tsub_to_blocks[target_ts]
            best_idx = -1
            best_gain = -1
            
            # Randomize candidates to break ties and add diversity
            r.shuffle(candidates)
            
            for b_idx in candidates:
                tsubs_in_b = block_tsub_indices[b_idx]
                gain = 0
                for ti in tsubs_in_b:
                    if ti in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    # Early stop if we covered all? Unlikely for a single block.
                    
            if best_idx != -1:
                blocks.append(best_idx)
                # Update uncovered
                for ti in block_tsub_indices[best_idx]:
                    uncovered.discard(ti)
            else:
                # Should not happen
                break
                
        return blocks

    def local_search(block_indices, seed):
        r = random.Random(seed)
        block_indices = list(block_indices)
        
        # Initialize coverage counts
        cov_counts = [0] * num_tsubs
        for b_idx in block_indices:
            for ti in block_tsub_indices[b_idx]:
                cov_counts[ti] += 1
                
        improved = True
        no_improve_count = 0
        max_no_improve = 50 # Limit iterations to avoid getting stuck
        
        while improved and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Remove redundant blocks
            removed = False
            for i in range(len(block_indices)):
                b_idx = block_indices[i]
                redundant = True
                for ti in block_tsub_indices[b_idx]:
                    if cov_counts[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    for ti in block_tsub_indices[b_idx]:
                        cov_counts[ti] -= 1
                    del block_indices[i]
                    improved = True
                    removed = True
                    break # Restart loop after removal
            
            if removed:
                no_improve_count = 0
                continue
                
            # 2. Swap to improve slack or just random valid swap
            if block_indices:
                i = r.randrange(len(block_indices))
                b_idx = block_indices[i]
                
                # Find unique tsubs in b_idx
                unique_tis = [ti for ti in block_tsub_indices[b_idx] if cov_counts[ti] == 1]
                
                if not unique_tis:
                    # This block is redundant, but we didn't catch it?
                    # Or it's just safe to swap.
                    # Try to swap with a random block to explore
                    pass
                else:
                    # Must cover all unique_tis
                    # Pick one unique ti to filter candidates
                    ut_idx = r.choice(unique_tis)
                    candidates = tsub_to_blocks[all_tsubs[ut_idx]]
                    
                    found_swap = False
                    # Shuffle candidates for diversity
                    r.shuffle(candidates)
                    
                    for cand_idx in candidates:
                        if cand_idx == b_idx:
                            continue
                        
                        # Check if cand covers all unique_tis
                        cand_tis = set(block_tsub_indices[cand_idx])
                        covers_all = all(ti in cand_tis for ti in unique_tis)
                        
                        if covers_all:
                            # Perform swap
                            # Decrement old
                            for ti in block_tsub_indices[b_idx]:
                                cov_counts[ti] -= 1
                            # Increment new
                            for ti in block_tsub_indices[cand_idx]:
                                cov_counts[ti] += 1
                                
                            block_indices[i] = cand_idx
                            improved = True
                            found_swap = True
                            no_improve_count = 0
                            break
                            
                    if not found_swap:
                        no_improve_count += 1
                if not found_swap:
                    no_improve_count += 1

        return block_indices

    best_blocks_idx = None
    best_len = float('inf')
    
    # ILS Loop
    num_iterations = 0
    while time.time() - start_time < time_limit * 0.9:
        num_iterations += 1
        seed = num_iterations
        
        if num_iterations == 1:
            blocks_idx = greedy_cover(seed)
        else:
            # Perturb best solution
            # Remove random blocks and repair
            blocks_idx = list(best_blocks_idx)
            r = random.Random(seed)
            
            # Remove 1-3 random blocks
            num_remove = min(len(blocks_idx), r.randint(1, 3))
            for _ in range(num_remove):
                if blocks_idx:
                    idx_to_remove = r.randrange(len(blocks_idx))
                    blocks_idx.pop(idx_to_remove)
                    
            # Repair coverage
            cov_counts = [0] * num_tsubs
            for b_idx in blocks_idx:
                for ti in block_tsub_indices[b_idx]:
                    cov_counts[ti] += 1
                    
            uncovered = [ti for ti in range(num_tsubs) if cov_counts[ti] == 0]
            
            while uncovered:
                ts_idx = r.choice(uncovered)
                target_ts = all_tsubs[ts_idx]
                candidates = tsub_to_blocks[target_ts]
                
                best_idx = -1
                best_gain = -1
                r.shuffle(candidates)
                
                for b_idx in candidates:
                    gain = 0
                    for ti in block_tsub_indices[b_idx]:
                        if cov_counts[ti] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = b_idx
                        
                if best_idx != -1:
                    blocks_idx.append(best_idx)
                    for ti in block_tsub_indices[best_idx]:
                        cov_counts[ti] += 1
                    # Update uncovered list
                    # Instead of rebuilding, just remove covered ones
                    # This is O(num_tsubs) each time, which is 2024.
                    # Max additions ~ 76. 76 * 2024 is fine.
                    uncovered = [ti for ti in uncovered if cov_counts[ti] == 0]
                else:
                    break
        
        blocks_idx = local_search(blocks_idx, seed)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks_idx = list(blocks_idx)
            
    # Final local search on best
    if best_blocks_idx is not None:
        best_blocks_idx = local_search(best_blocks_idx, 99999)
        
    # Convert indices to blocks
    best_blocks = [all_blocks[i] for i in best_blocks_idx]
    
    # Verification and Fallback
    cov = defaultdict(int)
    for b in best_blocks:
        for ts in itertools.combinations(b, t):
            cov[ts] += 1
            
    for ts in all_tsubs:
        if cov.get(ts, 0) == 0:
            # Fallback: add a block covering ts
            # Find any block containing ts
            candidates = tsub_to_blocks[ts]
            if candidates:
                best_blocks.append(all_blocks[candidates[0]])
            else:
                # Should not happen
                pass

    return best_blocks
