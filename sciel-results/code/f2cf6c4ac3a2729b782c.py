import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Enumerate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Enumerate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # 3. Precompute coverage relationships
    # masks[i] is a bitmask of t-subsets covered by k-subset i
    masks = [0] * num_k_subsets
    # mask_indices[i] is list of t-subset indices covered by k-subset i
    mask_indices = [[] for _ in range(num_k_subsets)]
    
    # blocks_covering_t[j] is list of k-subset indices that cover t-subset j
    blocks_covering_t = [[] for _ in range(num_t_subsets)]

    for i, ks in enumerate(k_subsets):
        mask = 0
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
            blocks_covering_t[idx].append(i)
        masks[i] = mask
        mask_indices[i] = indices

    best_blocks = None
    best_count = float('inf')

    # Run multiple random restarts
    num_restarts = 200 # Limit restarts to ensure time

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Greedy Construction
        uncovered = (1 << num_t_subsets) - 1
        current_blocks_indices = []
        used_blocks = set()
        
        # To speed up greedy, we can just iterate all blocks and pick max gain.
        # 75k blocks * ~300 steps = 22.5M operations per restart.
        # With 200 restarts, that's 4.5B ops. Too slow for Python if done naively.
        # We need a better greedy or fewer restarts.
        
        # Optimized Greedy: 
        # Instead of recalculating gain for all blocks, we can be smarter, 
        # but for now, let's reduce restarts and rely on local search.
        # Or, we can use a randomized greedy.
        
        # Let's try a single good greedy with tie-breaking by randomness.
        
        while uncovered and time.time() < deadline:
            max_gain = 0
            best_candidates = []
            
            # Iterate through all unused blocks
            # This is the bottleneck. 
            # Optimization: Only consider blocks that cover at least one uncovered t-subset.
            # We can maintain a list of active blocks? 
            # Simpler: Just iterate. 75k * 300 = 22.5M bit ops. 
            # Python can do ~10M simple ops/sec. This might take 2-3 seconds per restart.
            # 200 restarts * 3s = 600s. Too slow.
            
            # Strategy: Reduce restarts to 20-50 and make local search very efficient.
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(i)
            
            if max_gain == 0:
                break
            
            # Pick random from best candidates
            chosen_idx = random.choice(best_candidates)
            current_blocks_indices.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered &= ~masks[chosen_idx]
            
            # Check if all covered
            if uncovered == 0:
                break

        if uncovered != 0:
            continue

        # Local Search
        local_indices = list(current_blocks_indices)
        
        # Initialize coverage counts
        counts = [0] * num_t_subsets
        for bi in local_indices:
            for idx in mask_indices[bi]:
                counts[idx] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove redundant blocks
            for pos in range(len(local_indices)):
                if time.time() > deadline:
                    break
                bi = local_indices[pos]
                is_redundant = True
                for idx in mask_indices[bi]:
                    if counts[idx] < 2:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in mask_indices[bi]:
                        counts[idx] -= 1
                    local_indices.pop(pos)
                    improved = True
                    break # Restart loop

            if improved:
                continue

            # Phase 2: 1-for-1 Swaps
            n_blocks = len(local_indices)
            for pos in range(n_blocks):
                if time.time() > deadline:
                    break
                bi = local_indices[pos]
                
                # Find t-subsets covered ONLY by bi
                critical_ts = []
                for idx in mask_indices[bi]:
                    if counts[idx] == 1:
                        critical_ts.append(idx)
                
                if not critical_ts:
                    # Redundant, remove
                    for idx in mask_indices[bi]:
                        counts[idx] -= 1
                    local_indices.pop(pos)
                    improved = True
                    break

                # Find a replacement block that covers all critical_ts
                # Intersect lists of blocks covering each critical t-subset
                # Start with the list for the first critical t-subset
                candidates = set(blocks_covering_t[critical_ts[0]])
                
                for idx in critical_ts[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(blocks_covering_t[idx])
                
                if candidates:
                    # Pick a candidate not in current solution (or even if it is, swap might still be valid if we allow duplicates? No, blocks must be distinct. But if it's already there, it's not a new block. We need a block NOT in local_indices? 
                    # Actually, if the block is already in local_indices, adding it again is not allowed (distinct blocks). 
                    # So we filter out blocks already in local_indices.
                    valid_candidates = [c for c in candidates if c not in local_indices]
                    if valid_candidates:
                        new_bi = random.choice(valid_candidates)
                        # Perform swap
                        # Remove bi
                        for idx in mask_indices[bi]:
                            counts[idx] -= 1
                        # Add new_bi
                        for idx in mask_indices[new_bi]:
                            counts[idx] += 1
                        local_indices[pos] = new_bi
                        improved = True
                        break

        # Update best
        if len(local_indices) < best_count:
            best_count = len(local_indices)
            best_blocks = [list(k_subsets[i]) for i in local_indices]

    # If no valid solution found (should not happen with greedy), return empty or raise
    if best_blocks is None:
        # Fallback: Return a trivially large covering? 
        # Or just return empty list which is invalid? 
        # The problem says "at most C(v,t) blocks". 
        # A valid covering must cover all. 
        # If we failed, we should return the best effort. 
        # But if best_blocks is None, we haven't found any.
        # Let's force a greedy result if we haven't found one.
        # Actually, the loop above updates best_blocks. 
        # If it's still None, it means we never completed a greedy or local search.
        # Let's do one final greedy without local search to ensure we return something.
        if time.time() < deadline - 5:
            uncovered = (1 << num_t_subsets) - 1
            current_blocks_indices = []
            used_blocks = set()
            while uncovered:
                max_gain = 0
                best_candidates = []
                for i in range(num_k_subsets):
                    if i in used_blocks:
                        continue
                    gain = (masks[i] & uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_candidates = [i]
                    elif gain == max_gain and gain > 0:
                        best_candidates.append(i)
                if max_gain == 0:
                    break
                chosen_idx = random.choice(best_candidates)
                current_blocks_indices.append(chosen_idx)
                used_blocks.add(chosen_idx)
                uncovered &= ~masks[chosen_idx]
            if uncovered == 0:
                best_blocks = [list(k_subsets[i]) for i in current_blocks_indices]
    
    return best_blocks
