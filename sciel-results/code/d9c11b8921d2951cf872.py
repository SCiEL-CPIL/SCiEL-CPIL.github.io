import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets and map to indices
    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t_subs = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute for each point, the list of t-subset indices that contain it
    # This helps in incremental updates
    point_t_subs = [[] for _ in range(v)]
    for idx, sub in enumerate(all_t_subs):
        for p in sub:
            point_t_subs[p].append(idx)
    
    # Track coverage count for each t-subset
    coverage = np.zeros(n_t_subs, dtype=np.int32)
    
    blocks = []
    
    def block_t_indices(block):
        """Get indices of all t-subsets contained in a block."""
        return [t_sub_to_idx[tri] for tri in itertools.combinations(block, t)]
    
    # Greedy construction with randomized target selection
    while True:
        if time.time() > deadline - 10.0:
            break
        
        # Find uncovered t-subsets
        uncovered_mask = coverage == 0
        n_uncovered = np.count_nonzero(uncovered_mask)
        if n_uncovered == 0:
            break
        
        # Pick a random uncovered t-subset as target
        uncovered_indices = np.where(uncovered_mask)[0]
        target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        target = all_t_subs[target_idx]
        
        # Generate candidate blocks containing the target
        remaining_points = [p for p in all_points if p not in target]
        
        best_block = None
        max_new_cover = -1
        
        # For efficiency, we'll evaluate a random sample of candidates if too many
        candidates = list(itertools.combinations(remaining_points, k - t))
        
        # If too many candidates, sample
        if len(candidates) > 5000:
            random.shuffle(candidates)
            candidates = candidates[:5000]
        
        for combo in candidates:
            block = tuple(sorted(target + combo))
            
            # Count how many currently uncovered t-subsets this block covers
            new_cover = 0
            for tri in itertools.combinations(block, t):
                if coverage[t_sub_to_idx[tri]] == 0:
                    new_cover += 1
            
            if new_cover > max_new_cover:
                max_new_cover = new_cover
                best_block = block
        
        if best_block is None:
            break
        
        # Add the block
        blocks.append(list(best_block))
        
        # Update coverage
        for tri in itertools.combinations(best_block, t):
            coverage[t_sub_to_idx[tri]] += 1
    
    # Redundancy removal (optimized with numpy)
    if blocks:
        n_blocks = len(blocks)
        
        # Precompute for each block, the set of t-subset indices it covers
        block_cover = [set(block_t_indices(b)) for b in blocks]
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_blocks = []
            for i in range(len(blocks)):
                # Check if block i is redundant
                redundant = True
                for tri_idx in block_cover[i]:
                    # Check if tri_idx is covered by any other block
                    found = False
                    for j in range(len(blocks)):
                        if i == j:
                            continue
                        if tri_idx in block_cover[j]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                
                if not redundant:
                    new_blocks.append(blocks[i])
            
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                block_cover = [set(block_t_indices(b)) for b in blocks]
                changed = True
    
    # Local search: try to replace blocks to reduce count
    # This is a simple 1-exchange local search
    if time.time() < deadline - 2.0:
        # Try to remove blocks that are "least useful"
        # For each block, compute how many t-subsets it uniquely covers
        if blocks:
            n_blocks = len(blocks)
            block_cover = [set(block_t_indices(b)) for b in blocks]
            
            for _ in range(5):  # Try a few rounds
                if time.time() > deadline - 2.0:
                    break
                
                # Compute unique coverage for each block
                unique_counts = []
                for i in range(n_blocks):
                    unique = 0
                    for tri_idx in block_cover[i]:
                        is_unique = True
                        for j in range(n_blocks):
                            if i == j:
                                continue
                            if tri_idx in block_cover[j]:
                                is_unique = False
                                break
                        if is_unique:
                            unique += 1
                    unique_counts.append(unique)
                
                # Try to remove the block with the fewest unique covers
                if unique_counts:
                    min_unique = min(unique_counts)
                    if min_unique == 0:
                        # Find a block with 0 unique covers and remove it
                        candidates_to_remove = [i for i, u in enumerate(unique_counts) if u == 0]
                        if candidates_to_remove:
                            remove_idx = random.choice(candidates_to_remove)
                            blocks.pop(remove_idx)
                            block_cover.pop(remove_idx)
                            n_blocks -= 1
    
    return blocks
