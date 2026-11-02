import itertools
import random
import time

def construct(v, k, t):
    # Parameters for the specific task
    # v=24, k=7, t=3
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    all_triplets = list(itertools.combinations(range(v), t))
    initial_uncovered = set(all_triplets)
    
    best_blocks = None
    best_size = float('inf')
    
    # Helper to reduce blocks
    def reduce_blocks(blocks):
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow easier removal? Or just shuffle
            # Shuffling helps break symmetry
            random.shuffle(blocks)
            for i in range(len(blocks)):
                b = blocks[i]
                # Check if b is redundant
                # A block is redundant if every triplet in b is covered by other blocks
                redundant = True
                for triplet in itertools.combinations(b, t):
                    covered_by_other = False
                    for j in range(len(blocks)):
                        if i == j:
                            continue
                        if triplet in blocks[j]: # This is O(1) if blocks[j] is a set, but list is O(k)
                            # Optimization: convert blocks to sets for checking?
                            # Or maintain a global coverage count?
                            # Given small N, O(N * C(k,t)) is fine.
                            # But `triplet in blocks[j]` is slow for lists.
                            # Let's convert to sets for the reduction phase.
                            pass
                    # Better reduction check:
                    # Precompute sets for blocks
                    pass
                
                # Let's rewrite reduce_blocks with sets for speed
                pass
        return blocks

    def greedy_run(seed_offset=0):
        uncovered = set(initial_uncovered)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet
            # Using next(iter(uncovered)) is deterministic (first in set).
            # To randomize, we can pick a random index from a list?
            # Sets don't support random access.
            # We can convert to list every time? Expensive.
            # Or, we can just pick `min(uncovered)` and rely on randomization in block construction?
            # Or, we can maintain a list of uncovered and pick random index.
            # Let's use a list for uncovered for O(1) random access, and a set for O(1) membership.
            # Actually, `random.choice(list(uncovered))` is O(M). M is small (2024).
            # 80 steps * 2024 = 160k ops. Fine.
            
            seed_triplet = random.choice(list(uncovered))
            
            # Build block containing seed_triplet
            block = list(seed_triplet)
            
            # Greedily add k - t points
            for _ in range(k - t):
                best_gain = -1
                best_point = -1
                # Candidate points
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain
                    # Gain is number of triplets in (block + x) that are in uncovered
                    # Triplets involving x:
                    # 1. x + 2 from block
                    # 2. x + 1 from block + 1 from (block+x)?? No, block size is < k.
                    # We are adding x to current block.
                    # New triplets are those containing x and 2 points from current block.
                    # Wait, triplets are size 3.
                    # If we add x to block B, the new triplets covered are:
                    # {x, a, b} where a, b in B.
                    # There are C(len(B), 2) such triplets.
                    gain = 0
                    for a, b in itertools.combinations(block, 2):
                        triplet = tuple(sorted((x, a, b)))
                        if triplet in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    # Tie-breaking: random among top?
                    # For now, take first max.
                
                if best_point != -1:
                    block.append(best_point)
                else:
                    # Should not happen if uncovered is not empty and v > k
                    break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                uncovered.discard(tuple(triplet))
        
        return blocks

    def reduce_blocks_fast(blocks):
        # Convert to sets for faster triplet check
        block_sets = [set(b) for b in blocks]
        # Count coverage of each triplet
        # Or just check redundancy directly
        
        changed = True
        while changed:
            changed = False
            # Shuffle to break symmetry
            random.shuffle(block_sets)
            for i in range(len(block_sets)):
                b = block_sets[i]
                # Check if all triplets in b are covered by other blocks
                is_redundant = True
                for triplet in itertools.combinations(b, t):
                    triplet_set = set(triplet)
                    covered = False
                    for j in range(len(block_sets)):
                        if i == j:
                            continue
                        if triplet_set.issubset(block_sets[j]):
                            covered = True
                            break
                    if not covered:
                        is_redundant = False
                        break
                
                if is_redundant:
                    block_sets.pop(i)
                    changed = True
                    # Restart loop to check again
                    break # Break inner loop, restart outer while
            
        return [list(b) for b in block_sets]

    # Main loop
    run_count = 0
    while time.time() < deadline:
        # Run greedy
        blocks = greedy_run()
        # Reduce
        reduced_blocks = reduce_blocks_fast(blocks)
        
        if len(reduced_blocks) < best_size:
            best_size = len(reduced_blocks)
            best_blocks = reduced_blocks
            # Optional: print progress (not allowed in final output usually, but good for debugging)
            # print(f"New best: {best_size}")
        
        run_count += 1
        # If we have found 76, we might want to try harder or stop?
        # 76 is the best known. If we find 76, we can try to find 75.
        # But 75 is likely impossible (LB 69, but 76 is tight).
        # We just keep running.

    if best_blocks is None:
        # Fallback: return seed-like result
        return greedy_run()
        
    return best_blocks
