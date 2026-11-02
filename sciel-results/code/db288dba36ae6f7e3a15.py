import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # block_t_indices[i] is an array of indices into all_ts for the i-th block
    # Calculate t_count
    t_count = int(np.prod([k-i for i in range(t)]) / np.prod([i+1 for i in range(t)]))
    
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # triple_to_blocks: list of arrays of block indices
    triple_to_blocks = [[] for _ in range(n_ts)]
    for b_idx in range(n_ks):
        for sub_idx in block_t_indices[b_idx]:
            triple_to_blocks[sub_idx].append(b_idx)
    triple_to_blocks = [np.array(lst, dtype=np.int32) for lst in triple_to_blocks]
    
    def greedy_cover(seed):
        random.seed(seed)
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        
        while not np.all(covered):
            # Calculate gain for each block
            # gain[i] = number of uncovered triples in block i
            # We can vectorize this:
            # uncovered_flags = ~covered
            # gains = np.sum(uncovered_flags[block_t_indices], axis=1)
            
            uncovered_flags = ~covered
            gains = np.sum(uncovered_flags[block_t_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            blocks_indices.append(best_idx)
            covered[block_t_indices[best_idx]] = True
            
        return blocks_indices

    # Phase 1: Multiple Greedy Starts
    best_blocks = None
    best_len = float('inf')
    
    # Run a few greedy starts
    for seed in range(5):
        if time.time() > start_time + time_limit * 0.2:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        best_len = len(best_blocks)

    current_blocks = list(best_blocks)
    
    # Helper to check if a set of blocks covers all triples
    def is_valid(blocks):
        if not blocks:
            return n_ts == 0
        covered = np.zeros(n_ts, dtype=bool)
        for b in blocks:
            covered[block_t_indices[b]] = True
        return np.all(covered)
        
    # Helper to remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[block_t_indices[b]] += 1
            
        new_blocks = []
        for b in blocks:
            subs = block_t_indices[b]
            # If all triples in b have count > 1, b is redundant
            if np.min(coverage_count[subs]) > 1:
                coverage_count[subs] -= 1
            else:
                new_blocks.append(b)
        return new_blocks

    # Helper to try 2-for-1 swap
    def try_swap(blocks):
        if len(blocks) < 2:
            return blocks, False
            
        # Precompute coverage counts
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[block_t_indices[b]] += 1
            
        n = len(blocks)
        improved = False
        
        # Iterate pairs
        # To speed up, we can iterate in random order or specific order
        for i in range(n):
            if time.time() > start_time + time_limit - 5:
                break
            for j in range(i + 1, n):
                if time.time() > start_time + time_limit - 5:
                    break
                    
                b1 = blocks[i]
                b2 = blocks[j]
                
                # Find triples that would be uncovered if b1, b2 removed
                # These are triples in b1 or b2 with coverage_count == 1 (if in one) or 2 (if in both and no other)
                # Actually, simpler:
                # Temporarily decrement coverage for b1 and b2
                subs1 = block_t_indices[b1]
                subs2 = block_t_indices[b2]
                
                # We need to identify U = {t | t in b1 U b2 AND count_{S \ {b1,b2}}(t) == 0}
                # count_{S \ {b1,b2}}(t) = coverage_count[t] - (1 if t in b1 else 0) - (1 if t in b2 else 0)
                
                # Efficient way:
                # Create a mask for b1 and b2
                # But we need to check intersection of candidates.
                
                # Get all unique triples in b1 and b2
                # union_subs = np.unique(np.concatenate([subs1, subs2]))
                
                # For each t in union_subs, check if it's covered by others.
                # is_covered_by_others = coverage_count[t] > (1 if t in b1 else 0) + (1 if t in b2 else 0)
                
                # This is vectorizable if we create a boolean array for b1 and b2.
                # But doing it for every pair is O(N^2 * T).
                
                # Optimization: Only consider pairs that share at least one triple?
                # If they share no triples, U is large, unlikely to be covered by one block.
                
                # Let's just do the check.
                # Create temporary coverage
                # cov_temp = coverage_count.copy() # Too slow
                # cov_temp[subs1] -= 1
                # cov_temp[subs2] -= 1
                # U_indices = np.where(cov_temp[union_subs] == 0)[0]
                # This is still expensive.
                
                # Alternative:
                # U is the set of triples in (b1 U b2) that are NOT in (Union of other blocks).
                # Let's build the set of "critical" triples.
                
                # We can check if there exists a block B_new such that U subset B_new.
                # This is equivalent to: Intersection of Candidates(t) for all t in U is non-empty.
                
                # Let's identify U first.
                # We can use a small local array.
                
                # To avoid creating large arrays, we can iterate.
                # But let's try to be smart.
                
                # If b1 and b2 are disjoint, U is large.
                # If they overlap, U is smaller.
                
                # Let's just compute U explicitly for the pair.
                # Since n_ts is small (2024 for v=24, t=3), we can use a boolean array.
                
                # Actually, let's just use the intersection logic directly without explicit U if possible?
                # No, we need to know which triples are "lost".
                
                # Let's use a numpy array for the specific triples in b1 and b2.
                # unique_subs = np.unique(np.concatenate((subs1, subs2)))
                
                # Check coverage for these unique_subs
                # count_b1 = np.isin(unique_subs, subs1)
                # count_b2 = np.isin(unique_subs, subs2)
                # required_count = count_b1.astype(int) + count_b2.astype(int)
                # available_count = coverage_count[unique_subs] - required_count
                # U_mask = available_count == 0
                # U = unique_subs[U_mask]
                
                unique_subs = np.unique(np.concatenate((subs1, subs2)))
                count_b1 = np.isin(unique_subs, subs1).astype(np.int32)
                count_b2 = np.isin(unique_subs, subs2).astype(np.int32)
                required = count_b1 + count_b2
                available = coverage_count[unique_subs] - required
                U_mask = available == 0
                U = unique_subs[U_mask]
                
                if len(U) == 0:
                    # Both are redundant? Should have been caught by deletion.
                    # But maybe deletion missed it?
                    # If U is empty, we can just remove b1 and b2?
                    # No, if U is empty, it means all triples in b1 U b2 are covered by others.
                    # So b1 and b2 are both redundant.
                    # We can remove both.
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    return new_blocks, True
                
                if len(U) > t_count:
                    # A single block can cover at most t_count triples.
                    # If U has more, impossible.
                    continue
                    
                # Find intersection of candidates for U
                # Sort U by candidate count to prune early
                # candidate_counts = [len(triple_to_blocks[t]) for t in U]
                # sorted_U = sorted(U, key=lambda x: len(triple_to_blocks[x]))
                
                # For efficiency, just take the first one and intersect.
                # If U is small (<= 10), it's fast.
                
                candidates = triple_to_blocks[U[0]]
                for idx in range(1, len(U)):
                    t = U[idx]
                    candidates = np.intersect1d(candidates, triple_to_blocks[t])
                    if len(candidates) == 0:
                        break
                        
                if len(candidates) > 0:
                    # Pick a candidate
                    # Prefer one not in current blocks?
                    # If it is in current blocks (and not b1, b2), it's already there.
                    # But we established B_new cannot be in S' if U is not empty.
                    # So candidates should not be in S'.
                    # Wait, candidates are ALL blocks containing t.
                    # We need to check if any candidate is NOT in S'.
                    # If all candidates are in S', then U is covered by S', which is a contradiction.
                    # So at least one candidate must be outside S'.
                    
                    # Find a candidate not in blocks (excluding b1, b2)
                    # blocks_set = set(blocks)
                    # blocks_set.discard(b1)
                    # blocks_set.discard(b2)
                    
                    # Efficient check:
                    # Create a boolean mask of blocks in current set
                    # in_current = np.zeros(n_ks, dtype=bool)
                    # for b in blocks: in_current[b] = True
                    # in_current[b1] = False
                    # in_current[b2] = False
                    # valid_candidates = candidates[in_current[candidates] == False]
                    
                    # This is O(N) per swap attempt.
                    # We can precompute in_current once per pass.
                    
                    # Let's do this outside the loop.
                    pass # Placeholder, will implement in main loop

        return blocks, improved

    # Main Local Search Loop
    # We need to manage time carefully.
    
    # Precompute in_current mask for swap logic?
    # It changes every time blocks change.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit - 10:
        improved = False
        
        # 1. Deletion
        new_blocks = remove_redundant(current_blocks)
        if len(new_blocks) < len(current_blocks):
            current_blocks = new_blocks
            improved = True
            continue # Restart loop with new set
            
        # 2. Swap
        # Prepare data for swap
        n = len(current_blocks)
        if n < 2:
            break
            
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in current_blocks:
            coverage_count[block_t_indices[b]] += 1
            
        # Mask of blocks in current set
        in_current = np.zeros(n_ks, dtype=bool)
        for b in current_blocks:
            in_current[b] = True
            
        # Try swaps
        swapped = False
        for i in range(n):
            if time.time() > start_time + time_limit - 5:
                break
            for j in range(i + 1, n):
                if time.time() > start_time + time_limit - 5:
                    break
                    
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                subs1 = block_t_indices[b1]
                subs2 = block_t_indices[b2]
                
                unique_subs = np.unique(np.concatenate((subs1, subs2)))
                count_b1 = np.isin(unique_subs, subs1).astype(np.int32)
                count_b2 = np.isin(unique_subs, subs2).astype(np.int32)
                required = count_b1 + count_b2
                available = coverage_count[unique_subs] - required
                U_mask = available == 0
                U = unique_subs[U_mask]
                
                if len(U) == 0:
                    # Remove both
                    current_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                    swapped = True
                    improved = True
                    break
                elif len(U) > t_count:
                    continue
                else:
                    # Find candidate
                    # Sort U by rarity
                    U_list = list(U)
                    U_list.sort(key=lambda x: len(triple_to_blocks[x]))
                    
                    candidates = triple_to_blocks[U_list[0]]
                    for t in U_list[1:]:
                        candidates = np.intersect1d(candidates, triple_to_blocks[t])
                        if len(candidates) == 0:
                            break
                            
                    if len(candidates) > 0:
                        # Find a candidate not in current set (except b1, b2 which are being removed)
                        # We want B_new not in S' = S \ {b1, b2}
                        # So B_new must not be in S, OR B_new is b1/b2 (but they are removed, so we can't add them back effectively as a "new" block in the context of replacement? 
                        # Actually, if B_new is b1, we are replacing {b1, b2} with {b1}, which is just removing b2. 
                        # If b2 is redundant, deletion would have caught it. 
                        # So we should look for B_new NOT in S.
                        
                        # Check if any candidate is NOT in in_current
                        # in_current has True for all blocks in S.
                        # We need candidate c such that in_current[c] is False.
                        
                        valid_mask = ~in_current[candidates]
                        if np.any(valid_mask):
                            # Pick one
                            valid_candidates = candidates[valid_mask]
                            b_new = valid_candidates[0]
                            
                            # Replace
                            # Remove b1, b2, add b_new
                            new_blocks = []
                            for idx, b in enumerate(current_blocks):
                                if idx != i and idx != j:
                                    new_blocks.append(b)
                            new_blocks.append(b_new)
                            
                            current_blocks = new_blocks
                            swapped = True
                            improved = True
                            break
                            
            if swapped:
                break # Restart loop
                
    # Final check
    # If time is running out, return current best
    return [list(all_ks[b]) for b in current_blocks]
