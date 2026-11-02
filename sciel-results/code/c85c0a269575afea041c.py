import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # Pre-compute all t-subsets? No, too many (2024 is fine actually, but let's be dynamic)
    # Actually, 2024 is small. We can keep a set of all t-subsets initially.
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_score = float('inf')
    
    # We will run the greedy algorithm multiple times with different randomizations
    # to escape local minima.
    
    for run in range(100): # Cap runs, break on time
        if time.time() > deadline:
            break
            
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Greedy construction
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered subset to base the block on
            # This helps diversify the search
            target = random.choice(list(uncovered))
            
            # Find all k-blocks containing 'target'
            # target has t elements. We need k-t more from v-t remaining.
            remaining_points = [p for p in range(v) if p not in target]
            
            best_block = None
            best_gain = -1
            
            # Iterate through combinations of remaining points to form blocks
            # Number of candidates: C(v-t, k-t) = C(21, 4) = 5985
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target + combo))
                
                # Calculate gain: how many uncovered subsets are in this block?
                # A k-block has C(k, t) = C(7, 3) = 35 subsets.
                gain = 0
                for sub in itertools.combinations(candidate_block, t):
                    if sub in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early termination not really possible since we need max, 
                            # but we can skip if max possible is reached?
                            # Max possible is len(uncovered) but locally max is 35.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                break # Should not happen
            
            blocks.append(list(best_block))
            
            # Update uncovered set
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(sub)
        
        if not uncovered: # Valid covering found
            # Post-processing: Remove redundant blocks
            # A block is redundant if every t-subset within it is covered by at least one other block.
            
            # Build a count of how many blocks cover each t-subset
            # This is expensive if done from scratch every time, but we can do it once per run.
            
            # Optimize: Only check removal if we might save a block.
            # Let's do a simple pass:
            # 1. Count coverage for all subsets.
            # 2. Try removing blocks one by one if they don't have unique coverage.
            
            # Efficient check:
            # For each block, check if it has any unique subset.
            # If a block has NO unique subsets, it can be removed.
            
            # To do this efficiently:
            # 1. Create a dict: subset -> count of blocks covering it.
            coverage_count = {}
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[sub] = coverage_count.get(sub, 0) + 1
            
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in blocks:
                    # Check if b covers any subset uniquely
                    is_redundant = True
                    for sub in itertools.combinations(b, t):
                        if coverage_count[sub] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        # Remove b
                        changed = True
                        # Update coverage counts for subsets in b
                        for sub in itertools.combinations(b, t):
                            coverage_count[sub] -= 1
                    else:
                        new_blocks.append(b)
                
                blocks = new_blocks
                
                if time.time() > deadline:
                    break
            
            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = blocks.copy()
                
            # If we found a very good solution, we might want to stop early?
            # Theoretical lower bound is 69. If we hit 70, that's great.
            # But let's keep running to see if we can do better or find 69 (unlikely).
            
    if best_blocks is None:
        # Fallback: Return the last valid one or a trivial one?
        # The greedy should always find a valid one.
        # If timeout occurred before finishing a greedy run, we might have partial blocks.
        # We must return a VALID covering.
        # If best_blocks is None, it means we didn't complete a single greedy run within time?
        # Unlikely for v=24. But to be safe:
        # Re-run a single fast greedy without randomization if needed?
        # Or just return an empty list? No, must be valid.
        
        # Let's assume best_blocks is not None because the first run is fast.
        # If it is None, we have a problem. Let's do a quick deterministic greedy.
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            target = min(uncovered)
            remaining_points = [p for p in range(v) if p not in target]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target + combo))
                gain = sum(1 for sub in itertools.combinations(candidate_block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            blocks.append(list(best_block))
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(sub)
        best_blocks = blocks

    return best_blocks
