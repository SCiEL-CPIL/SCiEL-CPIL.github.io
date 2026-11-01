import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    all_points = range(v)
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices for faster counting? 
    # Or just use a set of tuples for uncovered.
    # For deletion, we need counts. A dict of tuple->count is fine for 11k items.
    
    def greedy_cover():
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to break symmetry
            target = random.choice(list(uncovered))
            
            # Generate candidates containing target
            remaining = [x for x in all_points if x not in target]
            need = k - t
            
            best_gain = -1
            best_block = None
            
            # Optimization: If we just want a valid cover, we can pick any.
            # But we want min blocks, so we pick max gain.
            for combo in itertools.combinations(remaining, need):
                block = tuple(sorted(target + combo))
                # Calculate gain
                gain = 0
                for sub in itertools.combinations(block, t):
                    if sub in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early termination? No, we need max.
                            pass
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if best_gain == 56: # Max possible gain for k=8, t=5 is C(8,5)=56
                        break
            
            if best_block is None:
                # Should not happen
                break
                
            blocks.append(best_block)
            
            # Update uncovered
            for sub in itertools.combinations(best_block, t):
                if sub in uncovered:
                    uncovered.remove(sub)
                    
            if time.time() > deadline - 10:
                break
                
        return blocks

    def delete_blocks(blocks):
        # Count coverage
        counts = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] = counts.get(sub, 0) + 1
        
        removed_any = True
        while removed_any:
            removed_any = False
            # Iterate in reverse to potentially remove later added (often more redundant) blocks first?
            # Or random order. Let's use random order to avoid bias.
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                b = blocks[i]
                # Check if any subset in b has count 1
                can_remove = True
                for sub in itertools.combinations(b, t):
                    if counts[sub] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for sub in itertools.combinations(b, t):
                        counts[sub] -= 1
                    blocks[i] = None # Mark for removal
                    removed_any = True
                    if time.time() > deadline - 5:
                        return [b for b in blocks if b is not None]
            
            # Compact list
            if removed_any:
                blocks = [b for b in blocks if b is not None]
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy instances
    while time.time() < deadline - 20:
        blocks = greedy_cover()
        if not blocks:
            continue
            
        # Try to delete
        blocks = delete_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we hit a very good score, we can stop or continue trying to improve
        # The lower bound is 264. Best known is 335.
        # If we get < 335, it's a new record (unlikely but possible).
        # If we get close to 335, keep trying.
        
        # Optional: Local search?
        # For now, just restart greedy with different random seeds.
        
    if best_blocks is None:
        # Fallback to a single deterministic run if something went wrong
        random.seed(42)
        blocks = greedy_cover()
        blocks = delete_blocks(blocks)
        return blocks
        
    return best_blocks
