import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Parameters are fixed for this task: v=19, k=7, t=4
    # But we should handle general inputs if possible, though the task says "evaluator passes this task's parameters"
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Precomputation specific to v=19, k=7, t=4
    # If v, k, t differ, we might need to adjust, but the task implies fixed params.
    # We will assume v=19, k=7, t=4 for optimization, but keep it somewhat general if easy.
    
    if v != 19 or k != 7 or t != 4:
        # Fallback to simple greedy for other params
        return _greedy_generic(v, k, t)

    # Precompute 4-subset IDs
    # map4 is a 4D array or flat array of size 19*19*19*19
    map4 = np.full((19, 19, 19, 19), -1, dtype=np.int32)
    id_counter = 0
    all_4s = []
    for comb in itertools.combinations(range(19), 4):
        a, b, c, d = comb
        map4[a, b, c, d] = id_counter
        all_4s.append(comb)
        id_counter += 1
    n_4s = id_counter # 3876
    
    # Precompute block_4s: for each of the 50388 blocks, list of 4-subset IDs
    all_blocks = list(itertools.combinations(range(19), 7))
    block_4s = []
    for block in all_blocks:
        ids = []
        for sub in itertools.combinations(block, 4):
            ids.append(map4[sub[0], sub[1], sub[2], sub[3]])
        block_4s.append(np.array(ids, dtype=np.int32))
        
    # Greedy Initialization
    counts = np.zeros(n_4s, dtype=np.int32)
    blocks = [] # List of block indices
    
    # Helper to add a block
    def add_block(idx):
        blocks.append(idx)
        counts[block_4s[idx]] += 1
        
    # Helper to remove a block
    def remove_block(idx):
        blocks.remove(idx) # O(N) but N is small
        counts[block_4s[idx]] -= 1

    # Greedy: pick block covering most uncovered 4s
    while True:
        if time.time() - start_time > time_limit:
            break
        uncovered_mask = (counts == 0)
        if not np.any(uncovered_mask):
            break
            
        # Find block that maximizes coverage of uncovered 4s
        # For each block, sum(uncovered_mask[block_4s[i]])
        # This is 50388 * 35 operations. ~1.7M ops. Fast.
        scores = np.array([np.sum(uncovered_mask[block_4s[i]]) for i in range(len(all_blocks))])
        # Tie-breaker: maybe cover more total? Or just first max.
        best_idx = np.argmax(scores)
        add_block(best_idx)
        
    # Simulated Annealing
    # State: blocks list, counts array
    # Cost: len(blocks) + 100 * np.sum(counts == 0)
    
    def get_cost():
        return len(blocks) + 100 * np.sum(counts == 0)
    
    current_cost = get_cost()
    T = 1.0
    best_cost = current_cost
    best_blocks = list(blocks)
    
    # SA Loop
    while time.time() - start_time < time_limit - 5: # Leave 5s for final check
        # Try a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Remove a block
            if len(blocks) == 0: continue
            idx_to_remove = random.choice(blocks)
            # Temporarily remove
            counts[block_4s[idx_to_remove]] -= 1
            new_cost = get_cost()
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                blocks.remove(idx_to_remove)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(blocks)
            else:
                # Reject
                counts[block_4s[idx_to_remove]] += 1
                
        elif move_type < 0.8:
            # Add a block
            # Pick a block that covers many uncovered? Or random?
            # Random is good for exploration.
            idx_to_add = random.randint(0, len(all_blocks)-1)
            if idx_to_add in blocks:
                continue
            counts[block_4s[idx_to_add]] += 1
            new_cost = get_cost()
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                blocks.append(idx_to_add)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(blocks)
            else:
                # Reject
                counts[block_4s[idx_to_add]] -= 1
                
        else:
            # Swap
            if len(blocks) == 0: continue
            idx_old = random.choice(blocks)
            idx_new = random.randint(0, len(all_blocks)-1)
            if idx_new in blocks:
                continue
            
            # Apply swap
            counts[block_4s[idx_old]] -= 1
            counts[block_4s[idx_new]] += 1
            new_cost = get_cost()
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                blocks.remove(idx_old)
                blocks.append(idx_new)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(blocks)
            else:
                # Reject
                counts[block_4s[idx_old]] += 1
                counts[block_4s[idx_new]] -= 1
                
        # Cooling
        T *= 0.995
        if T < 0.01:
            T = 0.01

    # Use best found
    blocks = best_blocks
    # Rebuild counts for best blocks to be sure
    counts = np.zeros(n_4s, dtype=np.int32)
    for idx in blocks:
        counts[block_4s[idx]] += 1
        
    # Final Greedy Repair
    while True:
        if time.time() - start_time > time_limit:
            break
        uncovered_mask = (counts == 0)
        if not np.any(uncovered_mask):
            break
        scores = np.array([np.sum(uncovered_mask[block_4s[i]]) for i in range(len(all_blocks))])
        best_idx = np.argmax(scores)
        if best_idx in blocks:
             # If the best block is already there, we have a problem?
             # No, if it's already there, it would have covered those 4s.
             # This implies the greedy step might pick a block that is already in the set?
             # If so, scores would be 0 for it if it covers no NEW uncovered sets?
             # Wait, if a block is in the set, its 4s are covered (count >= 1).
             # So uncovered_mask[block_4s[idx]] would be 0 for those 4s.
             # So a block already in the set will have score 0 for uncovered 4s.
             # So we will pick a new block.
             pass
        blocks.append(best_idx)
        counts[block_4s[best_idx]] += 1
        
    # Convert indices back to lists of points
    result = [list(all_blocks[i]) for i in blocks]
    return result

def _greedy_generic(v, k, t):
    # Fallback for non-19,7,4
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
