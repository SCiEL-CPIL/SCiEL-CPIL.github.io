import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    
    # points_in_subset: list of lists, index by subset id
    points_in_subset = [set(s) for s in all_subsets]
    
    # subsets_containing_point: list of lists, index by point
    subsets_containing_point = [[] for _ in range(v)]
    for s in all_subsets:
        for p in s:
            subsets_containing_point[p].append(subset_to_id[s])
            
    # mask_S_without_x: dict mapping (sid, x) to bitmask of S \ {x}
    # We only need this for x in S.
    # To save memory/time, we can compute on fly or precompute a list of masks for each sid.
    # Let's precompute mask for each subset.
    subset_masks = [0] * num_subsets
    for i, s in enumerate(all_subsets):
        m = 0
        for p in s:
            m |= (1 << p)
        subset_masks[i] = m
        
    # For gain calculation, we need mask of S \ {x}.
    # We can just do subset_masks[sid] & ~(1 << x)
    
    best_blocks = None
    best_size = float('inf')
    
    def greedy_once(seed_val):
        random.seed(seed_val)
        uncovered = set(range(num_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            start_sid = random.choice(list(uncovered))
            start_pts = all_subsets[start_sid]
            
            # Initialize block with points from start subset? 
            # Or just start with one point?
            # Standard greedy often starts with the points of the "hardest" subset.
            # Let's start with the points of start_sid.
            current_block = list(start_pts)
            current_block_set = set(current_block)
            current_mask = 0
            for p in current_block:
                current_mask |= (1 << p)
                
            # Expand block to size k
            while len(current_block) < k:
                best_x = -1
                best_gain = -1
                # Candidates are points not in current_block
                for x in range(v):
                    if x in current_block_set:
                        continue
                    gain = 0
                    # Iterate through all t-subsets containing x
                    for sid in subsets_containing_point[x]:
                        if sid in uncovered:
                            # Check if S \ {x} is subset of current_block
                            # S \ {x} has t-1 points.
                            # mask of S \ {x} is subset_masks[sid] & ~(1 << x)
                            req_mask = subset_masks[sid] & ~(1 << x)
                            if (req_mask & current_mask) == req_mask:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if gain == len(uncovered): # Max possible gain
                            break
                if best_x == -1:
                    # Should not happen if uncovered is not empty
                    break
                current_block.append(best_x)
                current_block_set.add(best_x)
                current_mask |= (1 << best_x)
                
            # Add block
            blocks.append(current_block)
            
            # Update uncovered
            # Find all t-subsets contained in current_block
            for s in itertools.combinations(current_block, t):
                sid = subset_to_id[s]
                uncovered.discard(sid)
                
        return blocks

    def local_search(blocks):
        # Convert blocks to sets for faster subset checking?
        # We need to check if any S in block has cover_count 1.
        # cover_count array
        cover_count = [0] * num_subsets
        
        # Helper to get list of subset IDs in a block
        def get_block_subset_ids(block):
            ids = []
            for s in itertools.combinations(block, t):
                ids.append(subset_to_id[s])
            return ids
            
        block_ids = [get_block_subset_ids(b) for b in blocks]
        
        # Initialize counts
        for ids in block_ids:
            for sid in ids:
                cover_count[sid] += 1
                
        # Removal loop
        changed = True
        while changed:
            changed = False
            new_block_ids = []
            for i, ids in enumerate(block_ids):
                # Check if any sid in ids has count 1
                removable = True
                for sid in ids:
                    if cover_count[sid] == 1:
                        removable = False
                        break
                if removable:
                    # Remove this block
                    for sid in ids:
                        cover_count[sid] -= 1
                    changed = True
                else:
                    new_block_ids.append(ids)
            block_ids = new_block_ids
            
        # Reconstruct blocks from remaining ids? 
        # We lost the original block structure if we only kept ids.
        # We need to map ids back to blocks.
        # Actually, we can just keep the original blocks list and filter it.
        
        # Let's redo local search to keep track of original blocks
        pass

    def local_search_v2(blocks):
        cover_count = [0] * num_subsets
        # Precompute subset IDs for each block
        block_subset_ids = []
        for b in blocks:
            ids = []
            for s in itertools.combinations(b, t):
                ids.append(subset_to_id[s])
            block_subset_ids.append(ids)
            
        for ids in block_subset_ids:
            for sid in ids:
                cover_count[sid] += 1
                
        # Iterative removal
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                ids = block_subset_ids[i]
                removable = True
                for sid in ids:
                    if cover_count[sid] == 1:
                        removable = False
                        break
                if removable:
                    active[i] = False
                    for sid in ids:
                        cover_count[sid] -= 1
                    changed = True
                    
        return [b for i, b in enumerate(blocks) if active[i]]

    # Main loop: multiple restarts
    attempts = 0
    while time.time() < deadline:
        blocks = greedy_once(attempts)
        optimized_blocks = local_search_v2(blocks)
        
        if len(optimized_blocks) < best_size:
            best_size = len(optimized_blocks)
            best_blocks = optimized_blocks
            
        # If we hit the known best (151), we can stop early or try to improve
        if best_size <= 151:
             # Try a few more to see if we can go lower, but 151 is likely optimal or near it
             if best_size == 151:
                 # Give it a few more shots
                 if attempts > 20:
                     break
        attempts += 1
        # Limit attempts to prevent infinite loop if time check is slow
        if attempts > 500:
            break

    if best_blocks is None:
        # Fallback: run one more time if we ran out of time before finishing first
        blocks = greedy_once(0)
        best_blocks = local_search_v2(blocks)
        
    return best_blocks
