import itertools
import time
import random

def construct(v, k, t):
    # Precompute
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    all_k_blocks = list(itertools.combinations(points, k))
    
    # Map block to its t-subsets
    block_subsets = {}
    for block in all_k_blocks:
        block_subsets[block] = set(itertools.combinations(block, t))
        
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            best_gain = -1
            best_blocks = []
            for block in all_k_blocks:
                gain = 0
                # Optimization: break early if gain can't beat best_gain?
                # Max possible gain is len(uncovered)
                # But we need exact gain for tie-breaking.
                for sub in block_subsets[block]:
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [block]
                elif gain == best_gain:
                    best_blocks.append(block)
            
            if not best_blocks:
                break # Should not happen if uncovered is not empty and k>=t
                
            chosen_block = random.choice(best_blocks)
            blocks.append(chosen_block)
            # Remove covered
            for sub in block_subsets[chosen_block]:
                if sub in uncovered:
                    uncovered.remove(sub)
                    
        return blocks

    def refine(blocks):
        # Convert to list of tuples
        blocks = [tuple(b) for b in blocks]
        
        # Count coverage
        cover_count = {}
        for block in blocks:
            for sub in block_subsets[block]:
                cover_count[sub] = cover_count.get(sub, 0) + 1
        
        refined_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(refined_blocks)):
                block = refined_blocks[i]
                # Check if removal is safe
                safe = True
                # Decrement counts
                for sub in block_subsets[block]:
                    cover_count[sub] -= 1
                    if cover_count[sub] == 0:
                        safe = False
                        # Restore
                        for sub2 in block_subsets[block]:
                            cover_count[sub2] += 1
                        break
                
                if safe:
                    refined_blocks.pop(i)
                    changed = True
                    # Note: we don't need to update cover_count for the rest
                    # because we already decremented and it was safe,
                    # so the decrements stay.
                    break # Restart loop to be safe? Or continue?
                    # If we remove a block, the indices shift.
                    # Also, removing a block might make other blocks removable.
                    # So restart is good.
        
        return refined_blocks

    start_time = time.time()
    deadline = start_time + 110.0
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 42
    while time.time() < deadline:
        blocks = greedy_cover(seed)
        refined = refine(blocks)
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined
        
        # If we hit a very good score, we might want to spend more time refining?
        # For now, just keep looking for a better greedy result.
        
        seed += 1

    if best_blocks is None:
        # Fallback to simple greedy if loop didn't run (should not happen)
        return greedy_cover(0)
        
    return [list(b) for b in best_blocks]
