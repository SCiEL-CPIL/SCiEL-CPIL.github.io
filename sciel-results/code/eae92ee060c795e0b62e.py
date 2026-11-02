import itertools
import random
import time
import sys

def construct(v, k, t):
    # Use a slightly earlier deadline to ensure we return in time
    deadline = time.time() + 115
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For fast gain calculation, we can precompute for each (t-1) subset R and point x,
    # the index of R U {x} if it's a valid t-subset.
    # But since v=19 is small, direct lookup is fine.
    
    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)
    
    # Helper to remove redundant blocks
    def _remove_redundant(blocks, deadline):
        if not blocks:
            return blocks
        
        # Map block index to the set of t-subset indices it covers
        block_tsub_indices = []
        cover_count = [0] * n_tsubs
        
        for block in blocks:
            tsubs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    tsubs.append(idx)
                    cover_count[idx] += 1
            block_tsub_indices.append(tsubs)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                # Check if any t-subset in this block is covered ONLY by this block
                is_redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        return blocks

    # Helper for local search: try to swap points to enable removal
    def _local_search(blocks, deadline):
        if not blocks:
            return blocks
            
        # Precompute tsub indices for each block
        block_tsub_indices = []
        cover_count = [0] * n_tsubs
        for block in blocks:
            tsubs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    tsubs.append(idx)
                    cover_count[idx] += 1
            block_tsub_indices.append(tsubs)
            
        improved = True
        max_iterations = 50
        iter_count = 0
        
        while improved and time.time() < deadline - 10 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Try to remove a block
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                
                # Find critical t-subsets for block i
                criticals = [idx for idx in block_tsub_indices[i] if cover_count[idx] == 1]
                
                if not criticals:
                    # Block i is redundant, remove it
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    improved = True
                    break
                
                # If not redundant, try to find a swap in another block that covers one critical
                # We pick the first critical one to try to cover
                crit_idx = criticals[0]
                crit_tsub = all_tsubs[crit_idx]
                
                # Find blocks that contain t-1 of the points in crit_tsub
                # And do not contain the last point
                # Actually, any block can be swapped.
                # We want to modify block j such that it covers crit_tsub.
                # Let's try swapping points in each other block j.
                
                found_swap = False
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if time.time() > deadline - 10:
                        break
                    
                    block_j = blocks[j]
                    block_j_set = set(block_j)
                    
                    # Check if block_j already covers crit_tsub
                    if crit_idx in block_tsub_indices[j]:
                        # It already covers it, but cover_count says 1?
                        # This implies i is the only one covering it? No, if j covers it, count >= 2.
                        # So this case shouldn't happen if cover_count is correct.
                        pass
                    
                    # Try to modify block_j to cover crit_tsub
                    # We need to replace some points in block_j with points from crit_tsub
                    # such that the new block covers crit_tsub.
                    
                    # Simple strategy: If block_j and crit_tsub share at least t-1 points,
                    # we might be able to swap 1 point.
                    common = block_j_set.intersection(crit_tsub)
                    
                    if len(common) >= t - 1:
                        # We can swap 1 point to include the missing one
                        missing = [p for p in crit_tsub if p not in block_j_set]
                        if len(missing) == 1:
                            # We need to remove one point from block_j that is NOT in common
                            # and add the missing point
                            extra = [p for p in block_j if p not in common]
                            if extra:
                                # Try swapping each extra point with the missing point
                                for rem_p in extra:
                                    new_block_j = [p if p != rem_p else missing[0] for p in block_j]
                                    new_block_j_set = set(new_block_j)
                                    
                                    # Check if new_block_j is valid (distinct, in range) - yes
                                    # Check if new_block_j covers all its t-subsets?
                                    # We need to ensure that removing rem_p doesn't uncover
                                    # any t-subsets that were ONLY covered by block_j.
                                    
                                    # Calculate t-subsets of old block_j and new block_j
                                    old_tsubs = set(block_tsub_indices[j])
                                    new_tsubs = set()
                                    for ts in itertools.combinations(new_block_j, t):
                                        idx = tsub_to_idx.get(tuple(ts))
                                        if idx is not None:
                                            new_tsubs.add(idx)
                                    
                                    # Check if we lose any unique coverage
                                    can_swap = True
                                    for idx in old_tsubs:
                                        if idx not in new_tsubs and cover_count[idx] == 1:
                                            can_swap = False
                                            break
                                    
                                    if can_swap:
                                        # Apply swap
                                        # Update cover counts
                                        for idx in old_tsubs:
                                            cover_count[idx] -= 1
                                        for idx in new_tsubs:
                                            cover_count[idx] += 1
                                        
                                        # Update block
                                        blocks[j] = new_block_j
                                        block_tsub_indices[j] = list(new_tsubs)
                                        
                                        improved = True
                                        found_swap = True
                                        break
                                    
                                    if found_swap:
                                        break
                                if found_swap:
                                    break
                    if found_swap:
                        break
                
                if found_swap:
                    break

        return blocks

    # Main loop: Randomized Greedy
    for restart in range(200):
        if time.time() > deadline - 20:
            break

        uncovered = set(range(n_tsubs))
        blocks = []
        order = list(range(n_tsubs))
        rng.shuffle(order)

        while uncovered:
            if time.time() > deadline - 10:
                break

            seed_idx = next(idx for idx in order if idx in uncovered)
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)

            while len(block) < k:
                best_gain = -1
                best_point = -1
                # Tie-breaking: prefer points that appear less often in existing blocks?
                # Or just random among ties.
                tie_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        tie_points = [x]
                    elif gain == best_gain and gain > -1:
                        tie_points.append(x)
                
                if best_point >= 0:
                    if len(tie_points) > 1:
                        best_point = rng.choice(tie_points)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(rng.choice(remaining))
                        block_set.add(block[-1])
                    else:
                        break

            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    uncovered.discard(idx)

        if time.time() > deadline - 10 or uncovered:
            continue

        # Post-processing
        blocks = _remove_redundant(blocks, deadline)
        
        if time.time() < deadline - 15:
            blocks = _local_search(blocks, deadline)
            blocks = _remove_redundant(blocks, deadline)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 330:
                break

    if best_blocks is None:
        # Fallback: simple greedy without randomization
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            seed_idx = min(uncovered)
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(remaining[0])
                        block_set.add(block[-1])
                    else:
                        break
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    uncovered.discard(idx)
        best_blocks = blocks

    return best_blocks
