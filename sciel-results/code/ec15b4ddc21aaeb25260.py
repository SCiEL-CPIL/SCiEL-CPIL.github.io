import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # 150s limit, leave buffer
    
    best_blocks = None
    best_len = float('inf')
    
    # Precompute all t-subsets for faster access if needed, 
    # but the greedy set difference is efficient enough.
    
    # Run multiple random restarts
    seed = 0
    while time.time() < deadline:
        blocks = _greedy_cover(v, k, t, seed)
        
        # Post-processing: Remove redundant blocks
        blocks = _remove_redundant(blocks, v, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
        # If we hit a very good number, we might stop early? 
        # Or just keep going to try for better.
        # 76 is the known best. If we hit 76, we can stop.
        if best_len <= 76:
            break

    if best_blocks is None:
        # Fallback: deterministic run
        return _greedy_cover(v, k, t, 0)
        
    return best_blocks

def _greedy_cover(v, k, t, seed):
    rng = random.Random(seed)
    all_points = list(range(v))
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered triple to start the block
        # Using a random choice from the set
        # To avoid O(N) conversion to list every time, we can keep a list 
        # but sets are faster for membership. 
        # Converting set to list is O(N). N=2024. 
        # Doing this ~80 times is fine.
        start_triple = rng.choice(list(uncovered))
        
        block = list(start_triple)
        
        while len(block) < k:
            # Find the point that maximizes gain
            max_gain = -1
            candidates = []
            
            # Optimization: Only consider points not in block
            for x in all_points:
                if x in block:
                    continue
                
                # Calculate gain: number of uncovered triples completed by adding x
                # A triple is completed if it contains x and t-1 elements from block
                gain = 0
                # t-1 = 2 for t=3
                # Pairs in block
                for pair in itertools.combinations(block, t - 1):
                    # Create the triple
                    triple = tuple(sorted(pair + (x,)))
                    if triple in uncovered:
                        gain += 1
                        
                if gain > max_gain:
                    max_gain = gain
                    candidates = [x]
                elif gain == max_gain:
                    candidates.append(x)
            
            if max_gain == 0:
                # If no point adds any new coverage, pick a random one
                # This shouldn't happen if uncovered is not empty and k > t,
                # unless the block is already "full" of points that don't help?
                # Actually, if uncovered is not empty, there must be some point 
                # that helps? Not necessarily if the remaining uncovered triples 
                # are disjoint from the current block prefix in a way that 
                # no single point completes one?
                # Wait, if we have a partial block, and there are uncovered triples,
                # it's possible that adding ANY point doesn't complete an uncovered triple?
                # Example: Block={0,1}. Uncovered={2,3,4}. 
                # Adding 5 doesn't help. Adding 2 makes {0,1,2} which is not in uncovered.
                # So gain can be 0.
                # In this case, we just pick a random point to fill the slot.
                available = [x for x in all_points if x not in block]
                if available:
                    block.append(rng.choice(available))
                else:
                    break
            else:
                # Pick randomly from best candidates to introduce diversity
                block.append(rng.choice(candidates))
                
        # Sort block for consistency
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        # Remove all triples contained in this block
        for triple in itertools.combinations(block, t):
            uncovered.discard(tuple(triple))
            
    return blocks

def _remove_redundant(blocks, v, t):
    # Count coverage of each triple
    # triple -> count of blocks covering it
    counts = {}
    block_triples = []
    
    for i, block in enumerate(blocks):
        triples = list(itertools.combinations(block, t))
        block_triples.append(triples)
        for triple in triples:
            counts[triple] = counts.get(triple, 0) + 1
            
    # Mark blocks for removal if they are redundant
    # A block is redundant if all its triples are covered by other blocks.
    # i.e., for every triple in block i, counts[triple] > 1.
    # However, if we remove block i, counts decrease.
    # We can do a simple pass: if a block has NO unique triple (triple with count 1),
    # it is redundant.
    # But removing one might make another redundant.
    # Iterative removal is safer.
    
    changed = True
    while changed:
        changed = False
        # Recalculate counts? Or just check if any triple has count 1?
        # If a block contains a triple with count 1, it's essential.
        # If it doesn't, it's redundant.
        
        # To be efficient, we can check:
        # For each block, does it contain any triple that is ONLY in this block?
        # If no, remove it.
        
        new_blocks = []
        for i, block in enumerate(blocks):
            is_essential = False
            for triple in block_triples[i]:
                if counts[triple] == 1:
                    is_essential = True
                    break
            if not is_essential:
                # Redundant
                changed = True
                # Decrement counts for this block's triples
                for triple in block_triples[i]:
                    counts[triple] -= 1
                # Do not add to new_blocks
            else:
                new_blocks.append(block)
                
        blocks = new_blocks
        # Re-calculate block_triples if we want to be precise, 
        # but since we only removed blocks, the remaining blocks are the same.
        # However, the 'counts' dict is updated.
        # We need to ensure block_triples corresponds to blocks.
        # Since we filtered blocks, we should filter block_triples too.
        # But we didn't keep an index mapping.
        # Let's just recalculate block_triples in the next iteration if needed?
        # Actually, the loop condition `while changed` implies we might iterate multiple times.
        # But a single pass of removing all non-essential blocks is usually sufficient 
        # if we define essential correctly?
        # No, removing one non-essential block might make another one non-essential.
        # So we need to loop until no more can be removed.
        
        # To avoid index mismatch, let's rebuild block_triples if we changed blocks.
        if changed:
             block_triples = [list(itertools.combinations(b, t)) for b in blocks]
             
    return blocks
