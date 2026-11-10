"""Strategy: Local Search (Hill Climbing / Simulated Annealing) on a fixed pool of blocks."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s limit, leaving 5s safety margin
    
    # 1. Precomputation
    # Map all t-subsets to indices for fast array access
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    # Precompute for each point x, the list of t-subsets containing x?
    # Or better: for a block, compute its coverage mask.
    # Since k=7, t=3, a block covers C(7,3)=35 t-subsets.
    # Total t-subsets C(24,3) = 2024.
    
    # We will maintain a count array `cover_count` of size num_tsubs.
    # Objective: Maximize number of covered t-subsets (minimize uncovered).
    # Secondary: Minimize number of blocks? No, we fix the number of blocks to a target N.
    # The task is to find *as few as possible*. 
    # Strategy: Start with a reasonably large N (e.g., 80), try to cover all.
    # If successful, try to reduce N by removing blocks or swapping in more efficient ones.
    # Actually, a common approach is to fix N to a value near the known bound (e.g., 76) 
    # and try to find a valid covering. If found, we can try to optimize for fewer blocks 
    # by attempting to merge or remove, but removing is hard if we fixed the pool size.
    
    # Better approach: 
    # 1. Generate a random initial set of N blocks.
    # 2. Use local search to maximize coverage.
    # 3. If coverage == num_tsubs, we have a valid covering with N blocks.
    # 4. Try to reduce N: 
    #    - Try removing a block if it doesn't break coverage.
    #    - If not, try replacing a block with a "better" one that allows removal of another?
    #    - Or simply restart with N-1 blocks if we have time.
    
    # Given the time limit, let's target N=76 (best known) or slightly higher to ensure validity, 
    # then try to shrink.
    # However, the score is the number of blocks. 
    # If we find a valid covering with 76 blocks, score is 1.0 (assuming seed is 76? No, seed score 1.0093 implies ~75.3 blocks? 
    # Wait, score = seed/value. If seed program produces 76 blocks, score=1.0. 
    # Current program score 1.0093 implies it produces ~75.3 blocks? That's impossible if integer.
    # Maybe the seed program produced 75 blocks? 
    # C(24,7,3) lower bound is 69. Best known 76.
    # If current program score is 1.0093, and seed is 1.0, then value_child = value_seed / 1.0093.
    # If value_seed = 76, value_child = 75.3. This suggests the current program might be returning an INVALID covering?
    # Or maybe the "seed" value is different.
    # Let's assume the goal is to get as close to 69-76 as possible.
    # If I produce a valid covering with 76 blocks, and the seed produced 76, score is 1.0.
    # If I produce 75, score is 76/75 = 1.013.
    # If I produce 74, score is 76/74 = 1.027.
    # The prompt says "Score: the number of blocks, minimized... search score is seed/value".
    # So lower blocks = higher score.
    
    # Let's aim for the minimum possible valid covering.
    # We will try N = 70, 71, ..., 76.
    # For each N, we run local search to see if we can cover all t-subsets.
    # If we find a valid covering for N, we return it.
    # We start with a higher N to ensure we find *some* valid covering quickly, 
    # then try lower N if time permits.
    
    # Actually, finding a valid covering with N=76 is the baseline.
    # Finding one with N=70 would be a major breakthrough (beating best known).
    # It is unlikely to beat 76 significantly in 120s.
    # So the goal is likely to find a valid covering with 76 blocks reliably, or maybe 75.
    
    # Let's start with N=76.
    # If we can't cover all, we increase N until we can.
    # Then we try to reduce N.
    
    def random_block():
        return tuple(sorted(random.sample(range(v), k)))
    
    def block_to_mask(block):
        """Returns list of indices of t-subsets covered by this block."""
        indices = []
        for tsub in itertools.combinations(block, t):
            indices.append(tsub_to_idx[tsub])
        return indices
    
    def evaluate(blocks):
        """Returns number of uncovered t-subsets."""
        cover_count = [0] * num_tsubs
        for b in blocks:
            for idx in block_to_mask(b):
                cover_count[idx] += 1
        uncovered = 0
        for c in cover_count:
            if c == 0:
                uncovered += 1
        return uncovered
    
    def get_cover_count(blocks):
        cover_count = [0] * num_tsubs
        for b in blocks:
            for idx in block_to_mask(b):
                cover_count[idx] += 1
        return cover_count

    best_blocks = None
    best_uncovered = num_tsubs
    
    # Try to find a valid covering with N blocks
    # Start with N=76 (known feasible)
    # If time allows, try N=75, 74...
    
    target_Ns = [76, 75, 74, 73, 72, 71, 70]
    
    for N in target_Ns:
        if time.time() > deadline - 10:
            break
            
        # Initialize random blocks
        current_blocks = [random_block() for _ in range(N)]
        # Ensure distinct blocks? Random collision is low but possible.
        # We'll handle duplicates in local search or just allow them (they waste space).
        
        current_uncovered = evaluate(current_blocks)
        
        if current_uncovered == 0:
            best_blocks = current_blocks
            best_uncovered = 0
            break # Found valid covering for this N, try next lower N? 
                  # The loop continues to next N which is lower. Good.
        
        # Local Search / Simulated Annealing
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        steps = 5000 # Max steps per N attempt? Adjust based on time.
        
        # We have ~100s total. 7 N values. ~14s per N.
        # Let's allocate time dynamically.
        
        current_T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Choose a move
            # 1. Swap a point in a random block with a random other point
            # 2. Replace a random block with a new random block
            # 3. Swap two blocks? (No, set is unordered)
            
            move_type = random.random()
            i = random.randint(0, N - 1)
            old_block = current_blocks[i]
            
            if move_type < 0.7:
                # Point swap
                j = random.randint(0, k - 1)
                old_point = old_block[j]
                # Pick a new point not in the block
                candidates = [x for x in range(v) if x not in old_block]
                if not candidates:
                    continue
                new_point = random.choice(candidates)
                
                new_block_list = list(old_block)
                new_block_list[j] = new_point
                new_block = tuple(sorted(new_block_list))
                
                # Check if this block already exists in the set (optional, but good for diversity)
                # If it does, the move is effectively a no-op or duplicate.
                
                # Calculate delta in uncovered count
                # Old block covers set A, New block covers set B.
                # We need to know how many t-subsets were covered ONLY by old_block 
                # and are not covered by new_block (lost coverage)
                # and how many were covered ONLY by new_block 
                # and are not covered by old_block (gained coverage)
                
                # Efficient delta calculation:
                # cover_count is global.
                # Let's maintain cover_count array globally.
                
                # This requires maintaining state. Let's refactor to maintain cover_count.
                
                pass # This branch is too complex to implement inline without state. 
                     # Let's restart the logic with stateful cover_count.
            
            break # Break out of inner loop if we decided to refactor.
        
        # REFACTORING: Stateful Local Search
        # We need to maintain `cover_count` to calculate deltas quickly.
        
        current_blocks = [random_block() for _ in range(N)]
        cover_count = [0] * num_tsubs
        block_masks = [] # List of lists of indices for each block
        
        for b in current_blocks:
            mask = block_to_mask(b)
            block_masks.append(mask)
            for idx in mask:
                cover_count[idx] += 1
        
        current_uncovered = sum(1 for c in cover_count if c == 0)
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_blocks = current_blocks[:]
        
        T = 10.0
        cooling = 0.9995
        
        while T > 0.1 and time.time() < deadline - 5:
            i = random.randint(0, N - 1)
            old_block = current_blocks[i]
            old_mask = block_masks[i]
            
            # Generate new block
            # 50% chance: point substitution
            # 50% chance: new random block
            if random.random() < 0.5:
                j = random.randint(0, k - 1)
                old_point = old_block[j]
                candidates = [x for x in range(v) if x not in old_block]
                if not candidates:
                    continue
                new_point = random.choice(candidates)
                new_block_list = list(old_block)
                new_block_list[j] = new_point
                new_block = tuple(sorted(new_block_list))
            else:
                new_block = random_block()
            
            new_mask = block_to_mask(new_block)
            
            # Calculate delta
            # For each idx in old_mask:
            #   if cover_count[idx] == 1, it becomes uncovered (if not in new_mask)
            # For each idx in new_mask:
            #   if cover_count[idx] == 0 (and not in old_mask? wait), it becomes covered
            
            # Better:
            # Delta = (new_uncovered - old_uncovered)
            
            # Let's compute the change in uncovered count.
            # Uncovered count increases if a t-subset goes from covered (count 1) to uncovered (count 0).
            # Uncovered count decreases if a t-subset goes from uncovered (count 0) to covered (count 1).
            
            delta_uncovered = 0
            
            # Remove old block's coverage
            # We need to know which indices in old_mask are NOT in new_mask.
            # And which indices in new_mask are NOT in old_mask.
            
            old_set = set(old_mask)
            new_set = set(new_mask)
            
            only_old = old_set - new_set
            only_new = new_set - old_set
            
            # For indices in only_old:
            # cover_count[idx] decreases by 1.
            # If cover_count[idx] was 1, it becomes 0 -> Uncovered increases by 1.
            for idx in only_old:
                if cover_count[idx] == 1:
                    delta_uncovered += 1
            
            # For indices in only_new:
            # cover_count[idx] increases by 1.
            # If cover_count[idx] was 0, it becomes 1 -> Uncovered decreases by 1.
            for idx in only_new:
                if cover_count[idx] == 0:
                    delta_uncovered -= 1
            
            # Acceptance criterion (Simulated Annealing)
            if delta_uncovered <= 0 or random.random() < pow(2.718, -delta_uncovered / T):
                # Accept move
                # Update state
                for idx in only_old:
                    cover_count[idx] -= 1
                for idx in only_new:
                    cover_count[idx] += 1
                
                current_blocks[i] = new_block
                block_masks[i] = new_mask
                current_uncovered += delta_uncovered
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = current_blocks[:]
                    if best_uncovered == 0:
                        break # Found valid covering for this N
                
            T *= cooling
        
        if best_uncovered == 0:
            # We found a valid covering for this N.
            # Since we iterate N from high to low, and we break on success,
            # we should actually try to find a valid covering for the *lowest* N possible.
            # The loop `for N in target_Ns` goes 76, 75...
            # If we succeed at 76, we break the inner while, but the outer for loop continues to 75?
            # No, I put a `break` in the previous draft. 
            # Here, if best_uncovered == 0, we want to try the next lower N.
            # So we do NOT break the outer loop. We just let it continue to N-1.
            # But we should save the best result so far.
            pass
        else:
            # If we didn't find a valid covering for this N, 
            # we keep the best partial covering found so far (best_blocks)
            # and try the next N (which is lower, so harder).
            # Wait, if we fail at 76, we should probably increase N? 
            # But 76 is known feasible. 
            # If local search fails at 76, it's likely due to time or bad initialization.
            # We should keep the best result found at 76 and try 75? 
            # No, 75 is harder. 
            # If we can't cover with 76, we shouldn't expect to cover with 75.
            # So we should probably stop trying lower N if we fail at the current N?
            # Or just keep the best valid covering found.
            # If we never found a valid covering (best_uncovered > 0), 
            # we return the best partial covering? 
            # No, the task requires a VALID covering. 
            # "Valid when every t-subset ... lies in at least one block".
            # If we return an invalid covering, score is 0.
            # So we MUST return a valid covering.
            # If we can't find one with N=76, we should try N=77, 78...
            
            # Let's adjust strategy:
            # 1. Try N=76. If success, save it.
            # 2. Try N=75. If success, save it (better).
            # 3. If no success at 76, try N=77, 78... until success.
            # 4. Return the best valid covering found.
            
            # To do this, we need to restructure the loop.
            # We'll try a range of Ns, keep the best valid one.
            
            pass

    # Let's rewrite the main loop logic clearly.
    
    best_valid_blocks = None
    best_valid_N = float('inf')
    
    # Try N from 70 up to 80.
    # If we find a valid covering with N, it's a candidate.
    # We want the smallest N.
    # So we should try 70, 71, ...
    # But 70 is very hard. 
    # It's safer to ensure we have a valid covering first.
    # Let's try 76 first to get a baseline valid covering.
    # Then try 75, 74... to improve.
    # If 76 fails, try 77, 78... to ensure validity.
    
    # Priority: Validity > Minimality.
    
    # 1. Attempt N=76
    # 2. If valid, attempt N=75, 74...
    # 3. If N=76 invalid, attempt N=77, 78...
    
    def try_N(N):
        if time.time() > deadline - 5:
            return None
        
        # Random restarts
        best_local_uncovered = num_tsubs
        best_local_blocks = None
        
        # Number of restarts depends on time
        # Let's do a few quick restarts.
        
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            current_blocks = [random_block() for _ in range(N)]
            cover_count = [0] * num_tsubs
            block_masks = []
            
            for b in current_blocks:
                mask = block_to_mask(b)
                block_masks.append(mask)
                for idx in mask:
                    cover_count[idx] += 1
            
            current_uncovered = sum(1 for c in cover_count if c == 0)
            
            if current_uncovered < best_local_uncovered:
                best_local_uncovered = current_uncovered
                best_local_blocks = current_blocks[:]
            
            if current_uncovered == 0:
                break
            
            T = 10.0
            cooling = 0.9995
            steps = 0
            max_steps = 2000 # Limit steps per restart
            
            while T > 0.1 and steps < max_steps and time.time() < deadline - 5:
                i = random.randint(0, N - 1)
                old_block = current_blocks[i]
                old_mask = block_masks[i]
                
                if random.random() < 0.5:
                    j = random.randint(0, k - 1)
                    old_point = old_block[j]
                    candidates = [x for x in range(v) if x not in old_block]
                    if not candidates:
                        continue
                    new_point = random.choice(candidates)
                    new_block_list = list(old_block)
                    new_block_list[j] = new_point
                    new_block = tuple(sorted(new_block_list))
                else:
                    new_block = random_block()
                
                new_mask = block_to_mask(new_block)
                
                old_set = set(old_mask)
                new_set = set(new_mask)
                
                only_old = old_set - new_set
                only_new = new_set - old_set
                
                delta_uncovered = 0
                for idx in only_old:
                    if cover_count[idx] == 1:
                        delta_uncovered += 1
                for idx in only_new:
                    if cover_count[idx] == 0:
                        delta_uncovered -= 1
                
                if delta_uncovered <= 0 or random.random() < pow(2.718, -delta_uncovered / T):
                    for idx in only_old:
                        cover_count[idx] -= 1
                    for idx in only_new:
                        cover_count[idx] += 1
                    
                    current_blocks[i] = new_block
                    block_masks[i] = new_mask
                    current_uncovered += delta_uncovered
                    
                    if current_uncovered < best_local_uncovered:
                        best_local_uncovered = current_uncovered
                        best_local_blocks = current_blocks[:]
                        if best_local_uncovered == 0:
                            break
                    steps += 1
                else:
                    steps += 1
            
            if best_local_uncovered == 0:
                break
        
        if best_local_uncovered == 0:
            return best_local_blocks
        else:
            return None

    # Try 76 first
    blocks_76 = try_N(76)
    if blocks_76:
        best_valid_blocks = blocks_76
        best_valid_N = 76
        
        # Try to improve (lower N)
        for N in [75, 74, 73]:
            if time.time() > deadline - 10:
                break
            blocks_N = try_N(N)
            if blocks_N:
                best_valid_blocks = blocks_N
                best_valid_N = N
                # Don't break, keep trying lower? 
                # Yes, if 75 works, try 74.
            else:
                # If 75 fails, 74 is likely to fail too. 
                # But maybe 74 works and 75 didn't due to random chance? Unlikely.
                # Let's just stop if we fail at a lower N? 
                # No, let's just try them all if time permits.
                pass
    else:
        # 76 failed. Try higher N to ensure validity.
        for N in [77, 78, 79, 80]:
            if time.time() > deadline - 10:
                break
            blocks_N = try_N(N)
            if blocks_N:
                best_valid_blocks = blocks_N
                best_valid_N = N
                break # Found valid, stop increasing.

    if best_valid_blocks is None:
        # Fallback: Greedy construction (like seed) to ensure we return *something* valid
        # This is a safety net.
        uncovered = set(all_tsubsets)
        blocks = []
        while uncovered and time.time() < deadline - 2:
            target = random.choice(list(uncovered))
            block = list(target)
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_point = None
                block_combos = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_combos:
                        subset = tuple(sorted(rest + (x,)))
                        if subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None or best_gain <= 0:
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = random.choice(available)
                    else:
                        break
                block.append(best_point)
            
            if len(block) == k:
                block.sort()
                blocks.append(tuple(block))
                for tsub in itertools.combinations(block, t):
                    uncovered.discard(tsub)
        
        # Post-processing: Remove redundant blocks
        cover_count = {tsub: 0 for tsub in all_tsubsets}
        for b in blocks:
            for tsub in itertools.combinations(b, t):
                cover_count[tsub] += 1
        
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            can_remove = True
            for tsub in itertools.combinations(b, t):
                if cover_count[tsub] <= 1:
                    can_remove = False
                    break
            if can_remove:
                del blocks[i]
                for tsub in itertools.combinations(b, t):
                    cover_count[tsub] -= 1
        
        return [list(b) for b in blocks]

    return [list(b) for b in best_valid_blocks]
