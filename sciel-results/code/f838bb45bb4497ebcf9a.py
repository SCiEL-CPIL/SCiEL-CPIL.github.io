import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets containing a specific t-subset? 
    # We will generate candidates on the fly for replacement.
    
    def greedy_cover(use_random=False):
        covered = [0] * num_t
        blocks = []
        
        uncovered_indices = list(range(num_t))
        
        while uncovered_indices:
            if use_random:
                idx = random.choice(uncovered_indices)
            else:
                idx = uncovered_indices[0]
            
            target_t = all_t_subsets[idx]
            
            remaining_points = [x for x in range(v) if x not in target_t]
            candidates = []
            
            for ext in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_t + ext))
                gain = 0
                for ts in itertools.combinations(block, t):
                    t_idx = t_subset_indices[ts]
                    if covered[t_idx] == 0:
                        gain += 1
                candidates.append((gain, block))
            
            if use_random and len(candidates) > 1:
                max_gain = max(c[0] for c in candidates)
                best_cands = [c[1] for c in candidates if c[0] == max_gain]
                best_block = random.choice(best_cands)
            else:
                best_block = max(candidates, key=lambda x: (x[0], -x[1]))[1]
            
            blocks.append(list(best_block))
            
            for ts in itertools.combinations(best_block, t):
                t_idx = t_subset_indices[ts]
                if covered[t_idx] == 0:
                    covered[t_idx] = 1
            
            uncovered_indices = [i for i in uncovered_indices if covered[i] == 0]

        return blocks

    def try_delete(blocks):
        changed = True
        current_blocks = list(blocks)
        
        while changed:
            changed = False
            counts = [0] * num_t
            for block in current_blocks:
                for ts in itertools.combinations(block, t):
                    counts[t_subset_indices[ts]] += 1
            
            new_blocks = []
            for block in current_blocks:
                essential = False
                for ts in itertools.combinations(block, t):
                    if counts[t_subset_indices[ts]] == 1:
                        essential = True
                        break
                
                if essential:
                    new_blocks.append(block)
                else:
                    changed = True
            
            current_blocks = new_blocks
        
        return current_blocks

    def try_replace(blocks):
        """
        Local search step: Try to replace a block with another that covers its essential subsets.
        This can allow further deletions.
        """
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            # Compute counts and essential sets
            counts = [0] * num_t
            for block in current_blocks:
                for ts in itertools.combinations(block, t):
                    counts[t_subset_indices[ts]] += 1
            
            # Identify essential t-subsets for each block
            # essential_sets[i] is a set of t-subset indices that are ONLY covered by block i
            essential_sets = [set() for _ in range(len(current_blocks))]
            
            for i, block in enumerate(current_blocks):
                for ts in itertools.combinations(block, t):
                    t_idx = t_subset_indices[ts]
                    if counts[t_idx] == 1:
                        essential_sets[i].add(t_idx)
            
            # Try to replace blocks that have essential sets
            # We look for a replacement block that covers all essential t-subsets of block i
            # and is not already in the list (or is a different block)
            # To speed up, we only try to replace if the essential set is small (e.g., size 1 or 2)
            # Or we can just try for all, but limit the search space.
            
            for i in range(len(current_blocks)):
                if not essential_sets[i]:
                    continue
                
                # If the essential set is too large, finding a replacement is hard.
                # For t=3, a block has C(7,3)=35 t-subsets.
                # If many are essential, it's hard to replace.
                # We only attempt replacement if len(essential_sets[i]) <= 3 for efficiency, 
                # or we can try to find a block containing the first essential subset.
                
                # Pick a random essential t-subset to anchor the search?
                # Or iterate through essential sets?
                # Let's try to find a block that covers ALL essential subsets.
                # This is a set cover problem itself, but small.
                
                # Optimization: If len(essential_sets[i]) > 5, skip replacement for this block 
                # as it's unlikely to find a single block covering all, or it's costly.
                if len(essential_sets[i]) > 5:
                    continue
                
                # Generate candidate replacement blocks
                # They must contain all t-subsets in essential_sets[i]
                # This means the union of points in these t-subsets must be in the block.
                # Let core = union of points in essential t-subsets
                core_points = set()
                for t_idx in essential_sets[i]:
                    core_points.update(all_t_subsets[t_idx])
                
                if len(core_points) > k:
                    continue # Cannot fit in a block
                
                # We need to choose k - len(core_points) points from the remaining v - len(core_points) points
                remaining_points = [x for x in range(v) if x not in core_points]
                
                found_replacement = False
                # Limit the number of candidates to check to save time
                # If k - len(core_points) is large, C(v-len(core), k-len(core)) is huge.
                # For v=24, k=7, t=3.
                # If core size is 3 (one essential subset), we choose 4 from 21. C(21,4) = 5985. Manageable.
                # If core size is 4 (two overlapping essential subsets), we choose 3 from 20. C(20,3) = 1140.
                # If core size is 6, we choose 1 from 18. C(18,1) = 18.
                
                # We will iterate through a limited number of candidates or all if small.
                cand_count = 0
                max_cands_to_check = 500
                
                for ext in itertools.combinations(remaining_points, k - len(core_points)):
                    if cand_count > max_cands_to_check:
                        break
                    cand_count += 1
                    
                    new_block = tuple(sorted(tuple(core_points) + ext))
                    
                    # Check if this new block is already in the list
                    if new_block in current_blocks:
                        continue
                    
                    # Check if it covers all essential subsets
                    # We know it covers core, so it covers the points, but does it cover the t-subsets?
                    # Yes, if the t-subset is a subset of core_points, it is in the block.
                    # Wait, essential_sets[i] contains t-subsets. If a t-subset is in essential_sets[i],
                    # its points are in core_points. So any block containing core_points contains that t-subset.
                    # So we just need to ensure we don't introduce new problems?
                    # The replacement block might cover other t-subsets, which is good.
                    
                    # Let's verify coverage of essential sets just in case
                    valid = True
                    for t_idx in essential_sets[i]:
                         if t_subset_indices[all_t_subsets[t_idx]] != t_idx: # sanity
                             pass
                         # Check if all_t_subsets[t_idx] is subset of new_block
                         if not set(all_t_subsets[t_idx]).issubset(set(new_block)):
                             valid = False
                             break
                    if not valid:
                        continue
                             
                    # Perform replacement
                    current_blocks[i] = list(new_block)
                    changed = True
                    found_replacement = True
                    break # Try next block or restart loop? 
                    # Restarting is safer to recompute counts.
                    # But we are inside a for loop. Let's break and restart the while loop.
                    break
                
                if found_replacement:
                    break # Restart the while loop to recompute counts and essentials
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < 110:
        # Alternate between random and deterministic?
        blocks = greedy_cover(use_random=True)
        blocks = try_delete(blocks)
        blocks = try_replace(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            if best_len <= 69:
                break
        
        iterations += 1

    if best_blocks is None:
        blocks = greedy_cover(use_random=False)
        best_blocks = try_delete(blocks)
        best_blocks = try_replace(best_blocks)
        
    return best_blocks
