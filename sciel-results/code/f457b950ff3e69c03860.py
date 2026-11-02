import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy Set Cover followed by 
    Simulated Annealing / Random Walk with Block Replacement.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    # 1. Pre-compute all possible k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_subsets = {}
    for b in all_blocks:
        subs = tuple(sorted(itertools.combinations(b, t)))
        block_to_subsets[b] = subs
        
    all_subsets = list(itertools.combinations(range(v), t))
    
    # Map from t-subset to list of blocks covering it
    subset_to_blocks = {s: [] for s in all_subsets}
    for b, subs in block_to_subsets.items():
        for s in subs:
            subset_to_blocks[s].append(b)
            
    # 2. Greedy Set Cover
    uncovered = set(all_subsets)
    blocks = []
    used_blocks = set()
    
    # Greedy: Pick uncovered subset, find best block covering it
    while uncovered:
        # Pick a random uncovered subset to break ties and add diversity?
        # Or the one with fewest candidates?
        # For speed, let's pick the first one, but randomize the initial set?
        # Actually, picking the "most constrained" is better for greedy.
        # But calculating "most constrained" is expensive.
        # Let's pick a random uncovered subset.
        s = random.choice(list(uncovered))
        
        candidates = subset_to_blocks[s]
        
        best_block = None
        best_gain = -1
        
        # To speed up, we can sample candidates if the list is large?
        # C(21,4) = 5985. Iterating 5985 is fine.
        for b in candidates:
            if b in used_blocks:
                continue
            gain = 0
            for sub in block_to_subsets[b]:
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
            elif gain == best_gain and random.random() < 0.1:
                # Random tie-breaking
                best_block = b
                
        if best_block is None:
            break
            
        blocks.append(best_block)
        used_blocks.add(best_block)
        
        for sub in block_to_subsets[best_block]:
            uncovered.discard(sub)
            
    # 3. Initial Cleanup: Remove redundant blocks
    coverage_counts = {}
    for b in blocks:
        for sub in block_to_subsets[b]:
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
            
    def is_redundant(block, coverage_counts):
        for sub in block_to_subsets[block]:
            if coverage_counts[sub] <= 1:
                return False
        return True
        
    # Remove redundant blocks initially
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if is_redundant(blocks[i], coverage_counts):
                # Remove block
                for sub in block_to_subsets[blocks[i]]:
                    coverage_counts[sub] -= 1
                del blocks[i]
                changed = True

    # 4. Simulated Annealing / Random Walk
    # We will perform moves that replace a block with another valid block.
    # A move is valid if the new block covers all "unique" subsets of the old block.
    # After a move, we remove any redundant blocks.
    
    # Helper to find unique subsets of a block
    def get_unique_subsets(block, coverage_counts):
        unique = []
        for sub in block_to_subsets[block]:
            if coverage_counts[sub] == 1:
                unique.append(sub)
        return unique

    # Helper to find candidate replacements
    def find_candidates(block, coverage_counts, used_blocks_set):
        unique_subs = get_unique_subsets(block, coverage_counts)
        if not unique_subs:
            return [] # Block is redundant, should be removed
        
        # Intersect lists of blocks covering each unique subset
        # Start with the subset having the fewest blocks to minimize intersection size
        unique_subs.sort(key=lambda s: len(subset_to_blocks[s]))
        
        candidates = set(subset_to_blocks[unique_subs[0]])
        for s in unique_subs[1:]:
            candidates.intersection_update(subset_to_blocks[s])
            
        # Remove the block itself and any already used blocks?
        # Wait, we are swapping, so the new block must NOT be in the current set of blocks.
        candidates.discard(block)
        candidates -= used_blocks_set
        
        return list(candidates)

    # Main SA Loop
    # We track the current set of blocks as a set for O(1) lookup
    blocks_set = set(blocks)
    
    # We will run until time limit
    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant blocks
        removed = False
        for i in range(len(blocks) - 1, -1, -1):
            if is_redundant(blocks[i], coverage_counts):
                # Remove
                for sub in block_to_subsets[blocks[i]]:
                    coverage_counts[sub] -= 1
                blocks_set.discard(blocks[i])
                del blocks[i]
                removed = True
                break
        
        if removed:
            continue # Restart loop to check for more redundancies
        
        # 2. Try a swap move
        if not blocks:
            break
            
        # Pick a random block
        idx = random.randint(0, len(blocks) - 1)
        old_block = blocks[idx]
        
        # Find candidates
        candidates = find_candidates(old_block, coverage_counts, blocks_set)
        
        if not candidates:
            # No valid swap found for this block. 
            # We could try another block, but for now, let's just sleep or try next iteration.
            # To avoid getting stuck, we might need a "perturbation" that allows invalid swaps?
            # No, we must maintain validity.
            # If no candidates, this block is "fixed" in the current configuration.
            # We can try to pick a different block.
            # But picking a random block every time might miss opportunities.
            # Let's just continue to the next iteration.
            continue
            
        # Pick a random candidate
        new_block = random.choice(candidates)
        
        # Perform Swap
        # 1. Remove old_block
        for sub in block_to_subsets[old_block]:
            coverage_counts[sub] -= 1
        blocks_set.discard(old_block)
        
        # 2. Add new_block
        for sub in block_to_subsets[new_block]:
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
        blocks_set.add(new_block)
        
        # Update blocks list
        blocks[idx] = new_block
        
        # 3. Try to remove redundant blocks (including new_block or others)
        # We do this greedily
        for i in range(len(blocks) - 1, -1, -1):
            if is_redundant(blocks[i], coverage_counts):
                for sub in block_to_subsets[blocks[i]]:
                    coverage_counts[sub] -= 1
                blocks_set.discard(blocks[i])
                del blocks[i]
                break # Only remove one per swap step to keep it simple? 
                # Or remove all? Removing all is better.
                # Let's remove all redundant blocks in a loop.
        
        # Let's do a full cleanup after each swap to ensure we are in a "minimal" state for this configuration.
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(blocks[i], coverage_counts):
                    for sub in block_to_subsets[blocks[i]]:
                        coverage_counts[sub] -= 1
                    blocks_set.discard(blocks[i])
                    del blocks[i]
                    changed = True

    return [list(b) for b in blocks]
