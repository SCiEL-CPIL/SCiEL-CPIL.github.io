import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Map all t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_subsets = len(t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Precompute for each point x, the list of t-subsets containing x?
    # Or just calculate gains on the fly.
    
    # Strategy: Run multiple randomized greedy constructions, then refine.
    # Given time limits, maybe 5-10 runs.
    
    for run in range(20):
        if time.time() > deadline:
            break
            
        # 1. Greedy Construction
        cover_count = [0] * n_subsets
        blocks = []
        
        # We need to find uncovered subsets.
        # Instead of a set of uncovered, we can just check cover_count.
        # But finding the "first" or "random" uncovered one requires scanning.
        # With 11628 subsets, scanning is fast.
        
        while True:
            # Find an uncovered subset
            uncovered_idx = -1
            # Randomize start to diversify
            start = random.randint(0, n_subsets - 1)
            for i in range(n_subsets):
                idx = (start + i) % n_subsets
                if cover_count[idx] == 0:
                    uncovered_idx = idx
                    break
            
            if uncovered_idx == -1:
                break # All covered
            
            # Start a new block
            # The seed uses min(uncovered). We use the found uncovered_idx.
            # To be more effective, we can try to build a block around this subset.
            # The block must contain this subset.
            # We need to add k-t more points.
            
            current_block = list(t_subsets[uncovered_idx])
            
            # Greedily add points
            while len(current_block) < k:
                best_gain = -1
                best_point = -1
                
                candidates = [x for x in range(v) if x not in current_block]
                
                for x in candidates:
                    # Calculate gain: how many new t-subsets does adding x cover?
                    # A t-subset is covered if it contains the current_block's t-subsets?
                    # No, a block B covers a t-subset S if S is a subset of B.
                    # If we add x to B, the new t-subsets covered are those S
                    # such that S is a subset of B ∪ {x} but not a subset of B.
                    # This is equivalent to: S contains x, and S \ {x} is a subset of B.
                    # Since S has size t, S \ {x} has size t-1.
                    # So we need to count how many (t-1)-subsets of B are part of
                    # an uncovered t-subset that also contains x.
                    
                    # Efficient way:
                    # Iterate over all (t-1)-subsets of current_block.
                    # For each (t-1)-subset R, check if R ∪ {x} is an uncovered t-subset.
                    
                    gain = 0
                    # t-1 is 4. k is 8.
                    # Number of (t-1) subsets of current_block is C(len(current_block), t-1).
                    # Max C(7, 4) = 35. This is very fast.
                    
                    for r in itertools.combinations(current_block, t - 1):
                        s = tuple(sorted(r + (x,)))
                        if s in subset_to_idx:
                            idx = subset_to_idx[s]
                            if cover_count[idx] == 0:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain:
                        # Tie-breaking: randomize to explore different paths
                        if random.random() < 0.5:
                            best_point = x
                            # Don't update best_gain, just pick randomly among ties
                            # Actually, just picking one is fine.
                
                if best_point == -1:
                    # Should not happen if v > k
                    break
                
                current_block.append(best_point)
                
            current_block.sort()
            blocks.append(current_block)
            
            # Update cover_count
            for s in itertools.combinations(current_block, t):
                cover_count[subset_to_idx[s]] += 1
        
        # 2. Refinement: Block Deletion
        # Try to remove blocks
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(blocks) - 1, -1, -1):
                # Check if block i is essential
                # Temporarily decrement counts
                block_to_remove = blocks[i]
                essential = False
                for s in itertools.combinations(block_to_remove, t):
                    idx = subset_to_idx[s]
                    if cover_count[idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove it
                    for s in itertools.combinations(block_to_remove, t):
                        cover_count[subset_to_idx[s]] -= 1
                    blocks.pop(i)
                    improved = True
                    # Continue checking other blocks, but indices shifted?
                    # Since we pop from end, or iterate backwards, it's safe.
                    # Actually, popping changes indices.
                    # It's safer to just restart the loop or be careful.
                    # Let's just break and restart the while loop for simplicity and safety.
                    break
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
    # If no blocks found (shouldn't happen), return empty or handle error
    if best_blocks is None:
        return []
        
    return best_blocks
