import itertools
import random
import time

def construct(v, k, t):
    # Deadline check
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    num_restarts = 20 # Adjust based on speed
    
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        blocks = greedy_cover(v, k, t, seed)
        
        # Prune redundant blocks
        blocks = prune_blocks(blocks, v, t)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Optional: if we reach a very low number, we can stop early or continue to polish
            # C(24,7,3) lower bound is 69. Known best is 76.
            # If we get close to 70, it's great.
            
    if best_blocks is None:
        # Fallback to deterministic if random failed or timed out immediately
        best_blocks = greedy_cover(v, k, t, 0)
        best_blocks = prune_blocks(best_blocks, v, t)
        
    return best_blocks

def greedy_cover(v, k, t, seed):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered triple to start the block
        # Or pick the one with lowest "degree" if we tracked it, but random is simpler for diversity
        start_triple = random.choice(list(uncovered))
        block = list(start_triple)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                # Count how many uncovered triples are completed by adding x
                # A triple is completed if it has 2 elements in block and 1 is x
                # Iterate over pairs in block
                for pair in itertools.combinations(block, t-1):
                    triple = tuple(sorted(pair + (x,)))
                    if triple in uncovered:
                        gain += 1
                        # Early break if gain exceeds current best?
                        # Maybe not, we need max.
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is not None:
                block.append(best_x)
            else:
                # Should not happen if uncovered is not empty, but safety
                break
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        # Remove all triples in this block
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return blocks

def prune_blocks(blocks, v, t):
    # Count coverage for each triple
    triple_count = {}
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_count[triple] = triple_count.get(triple, 0) + 1
            
    new_blocks = []
    for block in blocks:
        is_essential = False
        for triple in itertools.combinations(block, t):
            if triple_count[triple] == 1:
                is_essential = True
                break
        if is_essential:
            new_blocks.append(block)
        else:
            # Remove this block, so decrement counts for its triples
            # Note: We are modifying triple_count while iterating blocks?
            # We should be careful.
            # If we remove a block, we must update counts for future blocks' checks.
            for triple in itertools.combinations(block, t):
                triple_count[triple] -= 1
                
    return new_blocks
