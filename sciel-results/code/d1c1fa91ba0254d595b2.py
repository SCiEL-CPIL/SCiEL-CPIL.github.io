import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Bitset representation
    # tsub_mask[b] is a Python int where bit i is 1 if tsub i is covered by block b
    tsub_mask = [0] * n_blocks
    # tsub_blocks_mask[i] is a Python int where bit b is 1 if block b covers tsub i
    # n_blocks is ~50k, so this int is ~50k bits. Python handles this.
    tsub_blocks_mask = [0] * n_tsubsets
    
    # List of tsub indices for each block (for cover_count updates)
    block_tsubs = [[] for _ in range(n_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask
        
    full_tsub_mask = (1 << n_tsubsets) - 1
    
    def greedy_cover(seed):
        """
        Standard greedy using bitsets for gain calculation.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        
        # To speed up, we don't re-scan all blocks if we can avoid it, 
        # but 50k blocks * ~150 steps is fine.
        # We can optimize by only considering blocks that intersect uncovered_mask.
        
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break
                
            best_block = -1
            best_gain = 0
            
            # Optimization: If we have many blocks, scanning all is O(N).
            # We can sample or just scan. 50k * 150 = 7.5M iterations. 
            # Each iteration is a bit_count of a big int AND.
            # Python big int AND + bit_count is fast.
            
            for b_idx in range(n_blocks):
                if not (tsub_mask[b_idx] & uncovered_mask):
                    continue
                gain = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    if best_gain == n_tsubsets: # Perfect block? Unlikely
                        break
                
            if best_block == -1:
                break
                
            selected.append(best_block)
            uncovered_mask &= ~tsub_mask[best_block]
            
        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        Uses bitsets to find candidate replacement blocks quickly.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks
            
        block_set = set(blocks)
        
        # Initialize cover counts
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
                    
            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed
            
            # Phase 2: Try swaps
            # We try to swap a block b with a block r that covers all unique subsets of b.
            # This might not reduce size, but it might allow future deletions.
            # Or, we look for 2-for-1 swaps.
            
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for b in current_blocks:
                if time.time() > deadline - 2.0:
                    break
                if b not in block_set:
                    continue
                    
                # Find unique subsets for b
                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # Should have been caught in Phase 1, but safe check
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidate blocks covering all unique_tsubs
                # candidate_mask = AND of tsub_blocks_mask[ts] for ts in unique_tsubs
                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                
                if cand_mask == 0:
                    continue
                    
                # Remove current block b from candidates
                cand_mask &= ~(1 << b)
                
                # If we want to try 2-for-1, we'd need to check pairs. 
                # For now, let's stick to 1-for-1 swaps that enable deletion.
                # We pick a random candidate r.
                # To be efficient, we can just pick the first few set bits.
                
                # Extract set bits from cand_mask
                candidates = []
                m = cand_mask
                while m:
                    # Get lowest set bit
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 20: # Limit candidates
                        break
                        
                if not candidates:
                    continue
                    
                rng.shuffle(candidates)
                
                swapped = False
                for r in candidates:
                    if r in block_set:
                        continue
                        
                    # Perform swap b -> r
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart local search loop to check for new deletions
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    def two_for_one_search(blocks, seed):
        """
        Try to replace two blocks with one.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks
            
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            # Check pairs
            # To limit O(N^2), we can check random pairs or pairs involving blocks with many unique subsets.
            # Let's try random pairs.
            n_checks = 100
            for _ in range(n_checks):
                if time.time() > deadline - 2.0:
                    break
                if len(block_set) < 2:
                    break
                    
                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                if b1 not in block_set or b2 not in block_set:
                    continue
                    
                # Find unique subsets for b1 and b2
                # Note: A subset might be unique to b1, or unique to b2, or covered by both but no one else.
                # For 2-for-1, the replacement block r must cover all subsets that are ONLY covered by b1 or ONLY covered by b2.
                # Subsets covered by both b1 and b2 (and no one else) are also "unique to the pair".
                
                # Let U1 be subsets with count 1 in b1 (relative to current cover_count)
                # Let U2 be subsets with count 1 in b2
                # Actually, if a subset is in b1 and b2, and count is 2, removing both makes count 0.
                # So the set of subsets that MUST be covered by r is:
                # { s | s in b1 and count[s] <= 1 } U { s | s in b2 and count[s] <= 1 }
                # Wait, if s is in b1 and b2, count[s] is at least 2.
                # If we remove b1 and b2, count[s] drops by 2.
                # If count[s] was 2, it becomes 0. So r MUST cover s.
                # If count[s] was 3, it becomes 1. So r does NOT strictly need to cover s, but if it does, it's fine.
                # To be safe and simple, let's require r to cover all s such that (s in b1 and count[s] <= 1) OR (s in b2 and count[s] <= 1).
                # Actually, if s in b1 and b2, and count[s]=2, then s is "unique to the pair".
                
                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                        
                if required_mask == 0:
                    # Both blocks are redundant? Should have been caught.
                    continue
                    
                # Find blocks covering required_mask
                # cand_mask = AND of tsub_blocks_mask[ts] for all ts in required_mask
                # This is expensive if required_mask has many bits.
                # We can iterate over the set bits of required_mask.
                
                cand_mask = (1 << n_blocks) - 1
                m = required_mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    m ^= lsb
                    
                if cand_mask == 0:
                    continue
                    
                # Exclude b1 and b2
                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)
                
                # Get candidates
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 5:
                        break
                        
                if not candidates:
                    continue
                    
                rng.shuffle(candidates)
                r = candidates[0]
                
                # Perform swap: remove b1, b2; add r
                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1
                    
                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1
                    
                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1
                    
                improved = True
                break # Restart while loop to check for new deletions
                
        return list(block_set)

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    # 1. Greedy with different seeds
    # 2. Local search on the result
    # 3. 2-for-1 search on the result
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Greedy
        seed = 1000 + i
        # Add some noise to greedy? 
        # Standard greedy is deterministic. 
        # We can randomize tie-breaking.
        # For now, just run standard greedy.
        # To diversify, we can randomly remove some blocks from the greedy solution before local search?
        # Or just run local search which includes swaps.
        
        greedy_sol = greedy_cover(seed)
        
        if not greedy_sol:
            continue
            
        # Local Search
        ls_sol = local_search(greedy_sol, seed)
        
        # 2-for-1 Search
        t2o_sol = two_for_one_search(ls_sol, seed)
        
        # Final deletion pass
        # (local_search and two_for_one_search already do deletion, but just in case)
        
        if len(t2o_sol) < best_size:
            best_size = len(t2o_sol)
            best_solution = t2o_sol.copy()
            
    # If no solution found (should not happen), fallback to basic greedy
    if best_solution is None:
        best_solution = greedy_cover(0)
        
    return [list(all_blocks[idx]) for idx in best_solution]
