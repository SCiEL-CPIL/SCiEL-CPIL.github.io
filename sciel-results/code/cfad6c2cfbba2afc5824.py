import itertools
import random
import time

def construct(v, k, t):
    # The task is specifically for C(24, 7, 3)
    # We optimize for this case.
    if v == 24 and k == 7 and t == 3:
        return _construct_optimized()
    else:
        return _generic_greedy(v, k, t)

def _construct_optimized():
    v, k, t = 24, 7, 3
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Precomputation
    points = range(v)
    triples = list(itertools.combinations(points, t))
    triple_to_id = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    blocks = list(itertools.combinations(points, k))
    num_blocks = len(blocks)
    
    # Precompute masks and triple IDs for each block
    block_masks = [0] * num_blocks
    block_triple_ids = [None] * num_blocks
    
    for i, block in enumerate(blocks):
        mask = 0
        t_ids = []
        for comb in itertools.combinations(block, t):
            tid = triple_to_id[comb]
            mask |= (1 << tid)
            t_ids.append(tid)
        block_masks[i] = mask
        block_triple_ids[i] = t_ids
        
    all_triples_mask = (1 << num_triples) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # 2. Randomized Greedy Restarts
    # We run as many restarts as possible before the deadline
    while time.time() < deadline:
        uncovered = all_triples_mask
        selected_blocks = []
        
        while uncovered:
            best_gain = -1
            candidates = []
            
            # Find all blocks with maximum gain
            for b_idx in range(num_blocks):
                gain = (uncovered & block_masks[b_idx]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [b_idx]
                elif gain == best_gain:
                    candidates.append(b_idx)
            
            if best_gain == 0:
                # No block covers any new triple? This shouldn't happen if uncovered is not empty
                # unless we ran out of valid blocks or bug.
                break
                
            # Pick randomly from candidates to escape local minima
            chosen_idx = random.choice(candidates)
            selected_blocks.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        # 3. Local Search: Remove Redundant Blocks
        # Iteratively remove any block whose triples are all covered by others
        changed = True
        while changed:
            changed = False
            # Calculate cover counts
            cover_counts = [0] * num_triples
            for b_idx in selected_blocks:
                for tid in block_triple_ids[b_idx]:
                    cover_counts[tid] += 1
            
            # Try to remove blocks from the end to keep indices stable-ish
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                redundant = True
                for tid in block_triple_ids[b_idx]:
                    if cover_counts[tid] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    selected_blocks.pop(i)
                    changed = True
                    break  # Restart the while loop to recalculate counts
        
        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = list(selected_blocks)
            
            # Optional: If we found a very good solution, we could break early?
            # No, keep searching for better.
            
    if best_blocks is None:
        # Fallback
        return [list(b) for b in blocks[:num_blocks]]
        
    return [list(blocks[i]) for i in best_blocks]

def _generic_greedy(v, k, t):
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
