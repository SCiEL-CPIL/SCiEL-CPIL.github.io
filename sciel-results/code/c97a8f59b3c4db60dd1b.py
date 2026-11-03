import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for faster counting
    t_index = {t_set: i for i, t_set in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks and their t-coverage
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute t-coverage for each block
    block_t_indices = []
    for block in all_blocks:
        t_subs = list(itertools.combinations(block, t))
        indices = [t_index[ts] for ts in t_subs]
        block_t_indices.append(indices)
    
    # Precompute for each t-subset, which blocks contain it
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_to_blocks[idx].append(b_idx)
    
    random.seed(42)
    
    uncovered_set = set(range(n_t))
    selected_blocks = []
    counts = [0] * n_t
    
    # Dual variables: weight for each t-subset
    # Initialize to 1.0, update periodically
    dual_weights = [1.0] * n_t
    
    # Block scores: sum of dual weights of uncovered t-subsets in the block
    # We'll maintain this incrementally
    block_scores = [0.0] * n_blocks
    
    # Initialize block scores
    for b_idx in range(n_blocks):
        score = 0.0
        for idx in block_t_indices[b_idx]:
            score += dual_weights[idx]
        block_scores[b_idx] = score
    
    # Track which blocks have been selected
    block_selected = [False] * n_blocks
    
    # Greedy with dual variable updates
    # Every N iterations, update dual weights based on uncovered status
    # and recompute block scores
    dual_update_interval = 20
    
    iteration_count = 0
    
    while uncovered_set:
        if time.time() > deadline - 30:  # Reserve 30s for local search
            break
        
        iteration_count += 1
        
        # Periodically update dual weights and block scores
        if iteration_count % dual_update_interval == 0:
            # Update dual weights: increase weight for uncovered t-subsets
            for idx in uncovered_set:
                dual_weights[idx] = 1.0 + 0.5 * len(uncovered_set) / n_t
            for idx in range(n_t):
                if idx not in uncovered_set:
                    dual_weights[idx] = 0.0
            
            # Recompute block scores
            for b_idx in range(n_blocks):
                if block_selected[b_idx]:
                    continue
                score = 0.0
                for idx in block_t_indices[b_idx]:
                    if idx in uncovered_set:
                        score += dual_weights[idx]
                block_scores[b_idx] = score
        
        # Find the best unselected block by score
        best_block_idx = -1
        best_score = -1.0
        
        for b_idx in range(n_blocks):
            if not block_selected[b_idx] and block_scores[b_idx] > best_score:
                best_score = block_scores[b_idx]
                best_block_idx = b_idx
        
        if best_block_idx == -1 or best_score <= 0:
            # Fallback: pick a random uncovered t-subset and find best block containing it
            target = random.choice(list(uncovered_set))
            best_block_idx = -1
            best_gain = -1
            for b_idx in t_to_blocks[target]:
                if block_selected[b_idx]:
                    continue
                gain = 0
                for idx in block_t_indices[b_idx]:
                    if idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
        
        selected_blocks.append(best_block_idx)
        block_selected[best_block_idx] = True
        
        # Update uncovered and counts, and adjust block scores
        for idx in block_t_indices[best_block_idx]:
            counts[idx] += 1
            if counts[idx] == 1:
                uncovered_set.discard(idx)
                # This t-subset is now covered; remove its weight from all blocks
                w = dual_weights[idx]
                if w > 0:
                    for b_idx in t_to_blocks[idx]:
                        if not block_selected[b_idx]:
                            block_scores[b_idx] -= w
    
    # Convert to actual blocks
    blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    # Local search: try to remove redundant blocks
    max_removal_rounds = 10
    for _ in range(max_removal_rounds):
        if time.time() > deadline - 5:
            break
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            block = blocks[i]
            # Check if block is redundant
            redundant = True
            for t_sub in itertools.combinations(block, t):
                idx = t_index[t_sub]
                if counts[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                blocks.pop(i)
                for t_sub in itertools.combinations(block, t):
                    idx = t_index[t_sub]
                    counts[idx] -= 1
                changed = True
        if not changed:
            break
    
    # Additional local search: try swapping blocks
    # For each block, try replacing it with a different block that covers the same critical t-subsets
    if time.time() < deadline - 10:
        for _ in range(5):  # 5 rounds of swap attempts
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                block = blocks[i]
                # Find t-subsets that are only covered by this block
                critical = []
                for t_sub in itertools.combinations(block, t):
                    idx = t_index[t_sub]
                    if counts[idx] == 1:
                        critical.append(idx)
                
                if not critical:
                    continue
                
                # Try to find a replacement block that covers all critical t-subsets
                # and doesn't add too many new blocks
                # This is expensive, so let's just try a few random blocks
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    new_block_idx = random.randint(0, n_blocks - 1)
                    new_block = all_blocks[new_block_idx]
                    
                    # Check if new block covers all critical t-subsets
                    new_block_t_set = set(block_t_indices[new_block_idx])
                    if all(c in new_block_t_set for c in critical):
                        # Check if this swap is beneficial
                        # Count how many t-subsets would become uncovered
                        # and how many would be newly covered
                        would_uncover = 0
                        for t_sub in itertools.combinations(block, t):
                            idx = t_index[t_sub]
                            if counts[idx] == 1 and idx not in new_block_t_set:
                                would_uncover += 1
                        
                        if would_uncover == 0:
                            # Swap is safe
                            # Check if it reduces total count
                            blocks[i] = new_block
                            # Update counts
                            for t_sub in itertools.combinations(block, t):
                                idx = t_index[t_sub]
                                counts[idx] -= 1
                            for t_sub in itertools.combinations(new_block, t):
                                idx = t_index[t_sub]
                                counts[idx] += 1
                            improved = True
                            break
            if not improved:
                break
    
    # Final removal pass
    for _ in range(5):
        if time.time() > deadline - 2:
            break
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            block = blocks[i]
            redundant = True
            for t_sub in itertools.combinations(block, t):
                idx = t_index[t_sub]
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                for t_sub in itertools.combinations(block, t):
                    idx = t_index[t_sub]
                    counts[idx] -= 1
                changed = True
        if not changed:
            break
    
    return blocks
