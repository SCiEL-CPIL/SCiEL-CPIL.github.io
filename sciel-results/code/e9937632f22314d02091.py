import itertools
import time
import random

def construct(v, k, t):
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    total_subsets = len(all_subsets)
    
    # We will track the best number of blocks found
    best_blocks = None
    best_count = total_subsets # Upper bound: each block covers at least 1 subset
    
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin for safety/return
    
    # Helper to check if a set of blocks covers all subsets
    # This is expensive, so we only do it if we think we have a valid cover.
    # Actually, the greedy construction guarantees validity if we stop when uncovered is empty.
    
    def greedy_cover(random_seed):
        random.seed(random_seed)
        uncovered = set(all_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # Or pick the one with the most "pressure"? 
            # Random pick is good for diversity.
            target_subset = random.choice(list(uncovered))
            
            block = list(target_subset)
            
            # Greedily add points
            while len(block) < k:
                # Calculate gain for each candidate point
                # Candidate points are those not in block
                candidates = [x for x in range(v) if x not in block]
                
                best_gain = -1
                best_point = None
                # To add randomness, we can collect top candidates
                gains = []
                for x in candidates:
                    # Count how many new subsets are covered by adding x
                    # New subsets are formed by taking (t-1) elements from current block and adding x
                    # But we only care if they are in `uncovered`
                    count = 0
                    # Optimization: Pre-compute (t-1) subsets of block? 
                    # Block size is small (<= k).
                    for rest in itertools.combinations(block, t-1):
                        # Create the new tuple
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered:
                            count += 1
                    gains.append((count, x))
                
                # Find max gain
                if gains:
                    # Randomize among top gains to escape local optima
                    max_gain = max(g[0] for g in gains)
                    top_candidates = [g[1] for g in gains if g[0] == max_gain]
                    # Pick randomly from top candidates
                    best_point = random.choice(top_candidates)
                    if max_gain == 0:
                        # If no gain, just pick a random one to complete the block
                        # This shouldn't happen if uncovered is not empty and we started from an uncovered subset
                        # But if the target subset is covered, we might be stuck? 
                        # No, we started with target_subset which is in uncovered.
                        # So at least the subsets involving target_subset should be covered?
                        # Wait, we add points to cover OTHER subsets.
                        # The target subset itself is covered once the block contains it.
                        # But we started with the target subset IN the block.
                        # So the target subset is already "covered" in the sense that the block will contain it.
                        # But `uncovered` still contains it until we finalize the block and remove them.
                        # So we should remove the target subset from `uncovered` immediately?
                        # No, because the block isn't finished. But the block WILL contain it.
                        # So we can mark it as covered.
                        # However, other subsets might be covered by the final block.
                        # Let's just proceed. If gain is 0, it means no OTHER uncovered subset is covered by adding x.
                        # We still need to fill the block.
                        pass
                    
                    block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Remove all t-subsets of this block from uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        return blocks

    def local_search_remove(blocks):
        # Try to remove redundant blocks
        # A block is redundant if all its t-subsets are covered by other blocks
        # To check this efficiently, we can maintain a count of how many blocks cover each t-subset.
        # But rebuilding counts is O(N_blocks * C(k,t)).
        # N_blocks ~ 300, C(8,5) = 56. 300*56 = 16800 ops per check.
        # If we do this for each block, it's 300 * 16800 = 5M ops. Feasible.
        
        # Better: 
        # 1. Build a map: subset -> list of block indices covering it.
        # 2. For each block i, check if any subset it covers is ONLY covered by i.
        #    If all subsets covered by i are covered by at least one other block, i is redundant.
        
        subset_to_blocks = {}
        for i, b in enumerate(blocks):
            for sub in itertools.combinations(b, t):
                if sub not in subset_to_blocks:
                    subset_to_blocks[sub] = []
                subset_to_blocks[sub].append(i)
        
        redundant_indices = []
        for i, b in enumerate(blocks):
            is_redundant = True
            for sub in itertools.combinations(b, t):
                covering_blocks = subset_to_blocks[sub]
                if len(covering_blocks) == 1:
                    # This subset is only covered by block i
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.append(i)
        
        # Remove redundant blocks
        # Note: Removing one block might make another redundant. 
        # So we should iterate or do a single pass and repeat?
        # A single pass is fast. Multiple passes might be slow.
        # Let's do a single pass of removal, then repeat if time permits.
        
        if not redundant_indices:
            return blocks
            
        # Create set of indices to remove
        remove_set = set(redundant_indices)
        new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
        
        # Check if we can remove more in the new set?
        # Let's just return the result of one pass. The outer loop will call this again on the best result?
        # No, the outer loop generates new covers.
        # We should apply local search to the *best* result periodically or at the end of each greedy run.
        
        return new_blocks

    # Main loop
    seed = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed)
        # Apply local search
        # We can do multiple rounds of removal
        improved = True
        while improved and time.time() < deadline:
            improved = False
            prev_len = len(blocks)
            # Rebuild subset_to_blocks
            subset_to_blocks = {}
            for i, b in enumerate(blocks):
                for sub in itertools.combinations(b, t):
                    if sub not in subset_to_blocks:
                        subset_to_blocks[sub] = []
                    subset_to_blocks[sub].append(i)
            
            redundant_indices = []
            for i, b in enumerate(blocks):
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    if len(subset_to_blocks[sub]) == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_indices.append(i)
            
            if redundant_indices:
                remove_set = set(redundant_indices)
                blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                improved = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = blocks.copy()
        
        seed += 1
        
    if best_blocks is None:
        # Fallback: return a valid cover (e.g. from first run)
        # This should not happen if deadline > 0
        best_blocks = greedy_cover(0)
        
    return best_blocks
