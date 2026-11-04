import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_subsets_set = set(all_t_subsets)
    
    # Map t-subset to index for faster array access if needed, 
    # but set of tuples is fine for v=19, t=4 (3876 subsets).
    
    best_blocks = None
    best_count = float('inf')
    
    # Precompute all possible blocks? 
    # C(19, 7) = 50388. 
    # Storing them as tuples is fine.
    # all_blocks = list(itertools.combinations(range(v), k))
    # This might take memory but is fast to iterate.
    # However, we don't need to store them all if we generate candidates on the fly.
    
    for run_id in range(200): # More runs if we can
        if time.time() > deadline - 5.0: # Leave time for final check
            break
            
        rng = random.Random(run_id)
        
        # 1. Greedy Construction
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            seed_subset = rng.choice(list(uncovered))
            block = list(seed_subset)
            block_set = set(block)
            
            # Greedily extend block
            # We want to maximize coverage of 'uncovered'
            # But we also want to form a valid block of size k.
            
            # To speed up, we can precompute the (t-1) subsets of the current block.
            # But the block changes.
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                # Optimization: Only consider x that are in some uncovered subset?
                # No, any x can be added.
                
                # Get all (t-1) subsets of current block
                t_minus_1_subs = list(itertools.combinations(block, t - 1))
                
                # Count gain for each candidate x
                # Candidates: all points not in block
                candidates = [x for x in range(v) if x not in block_set]
                
                # To make it faster, we can sample candidates if v is large, 
                # but v=19 is small.
                
                for x in candidates:
                    gain = 0
                    for rest in t_minus_1_subs:
                        # Form t-subset
                        # rest is sorted. x is int.
                        # T = tuple(sorted(rest + (x,)))
                        # This sorting is the bottleneck.
                        # Since rest is sorted, we can insert x in sorted order.
                        # But for t=4, it's negligible.
                        T = tuple(sorted(rest + (x,)))
                        if T in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and gain > 0:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_x = x
                            
                if best_x == -1:
                    # Fallback: pick random
                    best_x = rng.choice(candidates)
                    
                block.append(best_x)
                block_set.add(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            # Generate all t-subsets of block
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        # 2. Local Search / Optimization
        # We will try to remove redundant blocks and swap blocks to enable more removals.
        
        # Helper to check if a block is redundant given a list of blocks
        # and coverage counts.
        
        # We maintain coverage counts incrementally.
        
        def get_coverage_counts(block_list):
            counts = {}
            for b in block_list:
                for sub in itertools.combinations(b, t):
                    counts[sub] = counts.get(sub, 0) + 1
            return counts
        
        counts = get_coverage_counts(blocks)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
                
            # Pass 1: Remove redundant blocks
            redundant_indices = []
            for i, b in enumerate(blocks):
                is_essential = False
                for sub in itertools.combinations(b, t):
                    if counts[sub] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    redundant_indices.append(i)
                    
            if redundant_indices:
                # Remove one at a time to be safe, or all if they don't interact?
                # Removing all identified redundant blocks in one pass is risky as discussed.
                # But if we remove one, counts change.
                # Let's remove one, update counts, and set improved=True to restart loop.
                idx = redundant_indices[0]
                b_to_remove = blocks[idx]
                blocks.pop(idx)
                # Update counts
                for sub in itertools.combinations(b_to_remove, t):
                    counts[sub] -= 1
                    if counts[sub] == 0:
                        # Should not happen if we checked correctly, but just in case
                        # If it happens, we have a bug.
                        pass
                improved = True
                continue # Restart loop to re-evaluate redundancies
                
            # Pass 2: Try Swaps
            # If no redundant blocks, try to swap a block to create a redundant one.
            # For each block i, remove it, find U (subsets covered only by it).
            # If U is not empty, try to find a replacement block B_new that covers U.
            # If found, swap.
            
            # To limit time, we only try swaps for a subset of blocks or limit attempts.
            # Let's try swaps for all blocks, but break early if time is tight.
            
            swap_made = False
            # Shuffle blocks to randomize order of swaps
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                    
                b_i = blocks[i]
                
                # Temporarily remove b_i
                # Find U: subsets in b_i with count == 1
                U = []
                for sub in itertools.combinations(b_i, t):
                    if counts[sub] == 1:
                        U.append(sub)
                    
                if not U:
                    # This block is redundant? 
                    # This should have been caught in Pass 1.
                    # If it happens, remove it.
                    blocks.pop(i)
                    for sub in itertools.combinations(b_i, t):
                        counts[sub] -= 1
                    improved = True
                    swap_made = True
                    break # Restart loop
                    
                # U is not empty. We need to cover U with a new block.
                # Pick a random subset S from U to seed the new block
                # We want a block that covers as many of U as possible.
                
                # Strategy: Pick a few random S from U, generate candidate blocks containing S, 
                # and pick the one that covers the most of U.
                
                best_new_block = None
                best_covered_count = 0
                
                # Sample up to 5 seeds from U
                seeds = rng.sample(U, min(5, len(U)))
                
                for S in seeds:
                    # Generate candidate blocks containing S
                    # S is a 4-subset. We need to add 3 more points.
                    # Candidates for the 3 points: v - 4 = 15 points.
                    # C(15, 3) = 455 candidates.
                    remaining_points = [x for x in range(v) if x not in S]
                    
                    # Iterate over combinations of 3 points from remaining
                    for add_triplet in itertools.combinations(remaining_points, k - t):
                        new_block = list(S)
                        new_block.extend(add_triplet)
                        new_block.sort()
                        
                        # Check how many subsets in U are covered by new_block
                        covered_in_U = 0
                        for sub in U:
                            # Check if sub is in new_block
                            # Since sub is size 4 and new_block is size 7, 
                            # we can check if all elements of sub are in new_block.
                            # set(sub) <= set(new_block)
                            # To speed up, we can precompute set(new_block)
                            # But doing this for 455 * 5 * 35 checks is ~ 80k checks.
                            # Let's optimize:
                            # If S is in U, and S is in new_block, that's 1.
                            # We need to check other subsets of new_block.
                            
                            # Actually, we just need to check if sub is a subset of new_block.
                            # We can do this by checking if all 4 elements are in new_block.
                            # new_block is small.
                            if all(x in new_block for x in sub):
                                covered_in_U += 1
                                
                        if covered_in_U > best_covered_count:
                            best_covered_count = covered_in_U
                            best_new_block = new_block
                            
                            # If we cover all of U, we can stop early for this seed
                            if best_covered_count == len(U):
                                break
                                
                    if best_covered_count == len(U):
                        break
                        
                if best_new_block is not None and best_covered_count == len(U):
                    # Swap b_i with best_new_block
                    # Update counts: 
                    # Subtract counts for b_i
                    for sub in itertools.combinations(b_i, t):
                        counts[sub] -= 1
                    # Add counts for best_new_block
                    for sub in itertools.combinations(best_new_block, t):
                        counts[sub] = counts.get(sub, 0) + 1
                        
                    blocks[i] = best_new_block
                    swap_made = True
                    improved = True
                    break # Restart loop to re-evaluate
                    
            if swap_made:
                continue # Restart main while loop
                
        # End Local Search
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    if best_blocks is None:
        # Fallback
        best_blocks = []
        
    return best_blocks
