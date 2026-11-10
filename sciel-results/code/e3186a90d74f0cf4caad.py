"""
Strategy: Simulated Annealing on a fixed pool of blocks.
1. Initialize a pool of blocks using a greedy heuristic.
2. Define the objective function as the number of covered t-subsets.
3. Perform local search by swapping points within blocks.
4. Accept moves that improve coverage, and probabilistically accept worsening moves
   based on a cooling schedule to escape local optima.
"""
import itertools
import time
import random
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s limit, leaving 5s safety margin
    
    # Precompute all t-subsets for efficiency
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    # State representation:
    # blocks: list of lists of points
    # covered: boolean array of length n_tsubsets, True if covered
    
    # 1. Greedy Initialization
    # We want to start with a reasonable covering.
    # Greedily pick blocks that cover the most currently uncovered t-subsets.
    uncovered_indices = set(range(n_tsubsets))
    blocks = []
    
    # Precompute t-subsets for a given block? No, compute on the fly or use combinatorics.
    # For initialization, we can use a simple greedy set cover.
    # To speed up, we might just generate random blocks or use a fast greedy.
    # Let's use a fast greedy initialization.
    
    # Optimization: Precompute for each point, the list of t-subsets containing it?
    # That's v * C(v-1, t-1) which is 24 * C(23,2) = 24 * 253 = 6072. Small.
    # But we need to know which t-subsets are covered by a block.
    
    # Let's just do a simple greedy initialization.
    # To make it faster, we can limit the number of blocks in initialization to a reasonable estimate.
    # Lower bound is 69. Let's aim for ~80 blocks initially.
    
    max_init_blocks = 100
    while uncovered_indices and len(blocks) < max_init_blocks and time.time() < deadline - 10:
        best_block = None
        best_gain = -1
        
        # To make this faster, we can sample candidate blocks or use a heuristic.
        # A full search over C(24,7) is too slow.
        # Heuristic: Pick a random uncovered t-subset, build a block around it greedily.
        if uncovered_indices:
            target_idx = random.choice(list(uncovered_indices))
            target_tsub = all_tsubsets[target_idx]
            
            # Build a block containing this t-subset
            block = list(target_tsub)
            # Add random points to complete the block? Or greedy?
            # Greedy is better.
            block_set = set(block)
            
            # Calculate gain for adding each point
            # We want to maximize the number of NEW t-subsets covered.
            # A t-subset is covered if it is a subset of the block.
            
            # Let's just add random points for init to be fast, then refine with SA.
            # Actually, a greedy init is important for a good starting point.
            # Let's try a simple greedy: add point that maximizes new t-subsets.
            
            for _ in range(k - len(block)):
                best_point = None
                best_gain = -1
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain
                    # New t-subsets are those that include x and (t-1) points from block
                    # and are currently uncovered.
                    gain = 0
                    # C(len(block), t-1) combinations
                    for rest in itertools.combinations(block, t - 1):
                        tsub = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(tsub, -1)
                        if idx != -1 and idx in uncovered_indices:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is not None:
                    block.append(best_point)
                else:
                    # Pick random
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        block.append(random.choice(available))
                    else:
                        break
            
            if len(block) == k:
                block.sort()
                # Check if this block is new
                if block not in blocks:
                    blocks.append(block)
                    # Update uncovered
                    for tsub in itertools.combinations(block, t):
                        idx = tsubset_to_idx[tsub]
                        uncovered_indices.discard(idx)
        
        if time.time() > deadline - 10:
            break
    
    if not blocks:
        # Fallback: random blocks
        for _ in range(80):
            block = random.sample(range(v), k)
            blocks.append(block)
    
    # Initialize covered array
    covered = [False] * n_tsubsets
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            covered[tsubset_to_idx[tsub]] = True
    
    current_score = sum(covered)
    best_score = current_score
    best_blocks = [b[:] for b in blocks]
    
    # 2. Simulated Annealing
    # Parameters
    T = 1.0  # Initial temperature
    alpha = 0.999  # Cooling rate
    T_min = 0.01
    
    # Move: Pick a random block, pick a random point in it, pick a random point outside, swap.
    # This changes the coverage.
    
    while time.time() < deadline - 5 and T > T_min:
        # Pick a random block
        i = random.randint(0, len(blocks) - 1)
        block = blocks[i]
        
        # Pick a point to remove
        j = random.randint(0, k - 1)
        old_point = block[j]
        
        # Pick a point to add
        available = [x for x in range(v) if x not in block]
        if not available:
            continue
        new_point = random.choice(available)
        
        # Perform swap
        block[j] = new_point
        block.sort()
        
        # Calculate delta score
        # We need to update the covered array efficiently?
        # Or just calculate the change.
        # Removing old_point from block:
        # Any t-subset containing old_point and (t-1) points from the rest of the block
        # might become uncovered if it was ONLY covered by this block.
        # Adding new_point:
        # Any t-subset containing new_point and (t-1) points from the rest of the block
        # might become covered if it was previously uncovered.
        
        # This is complex to calculate exactly without a reference count of how many blocks cover each t-subset.
        # Let's maintain a count array: cover_count[idx] = number of blocks covering t-subset idx.
        
        # Wait, I didn't initialize cover_count. Let's do that.
        # This requires re-initializing.
        
        # Let's restart the SA part with proper state management.
        break

    # Re-implementing with proper state for SA
    # The previous init is okay, let's just use it.
    
    # Let's try a different approach for the SA part to be safe and fast.
    # Instead of complex delta calculations, we can just recalculate the score for the affected t-subsets?
    # No, that's still O(C(k,t)).
    
    # Let's use the cover_count array.
    cover_count = [0] * n_tsubsets
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            cover_count[tsubset_to_idx[tsub]] += 1
    
    current_score = sum(1 for c in cover_count if c > 0)
    best_score = current_score
    best_blocks = [b[:] for b in blocks]
    
    T = 1.0
    alpha = 0.9995
    T_min = 0.01
    
    while time.time() < deadline - 5 and T > T_min:
        i = random.randint(0, len(blocks) - 1)
        block = blocks[i]
        
        # Pick point to remove
        j = random.randint(0, k - 1)
        old_point = block[j]
        
        # Pick point to add
        # Optimization: Precompute points not in block?
        # For v=24, k=7, this is small.
        block_set = set(block)
        available = [x for x in range(v) if x not in block_set]
        if not available:
            continue
        new_point = random.choice(available)
        
        # Perform swap in memory
        block[j] = new_point
        
        # Calculate Delta
        # 1. Remove old_point from block
        # t-subsets containing old_point and (t-1) from (block \ {old_point})
        # Note: block already has new_point, not old_point.
        # So we consider the block WITHOUT old_point (which is current block) 
        # and WITH old_point (which is current block with new_point replaced by old_point).
        
        # Let's define:
        # B_old = block with new_point replaced by old_point
        # B_new = current block (with new_point)
        
        # Delta = (New Covered - Old Covered)
        
        # Subsets affected:
        # Those involving old_point and points in B_new \ {new_point}
        # Those involving new_point and points in B_new \ {new_point}
        
        # Actually, simpler:
        # Subsets that change coverage status are those that include either old_point or new_point
        # and (t-1) points from the rest of the block (which is the same in both cases, except for the swapped point).
        
        rest_of_block = [x for x in block if x != new_point]
        
        # Subsets with old_point:
        # These were in B_old. Are they in B_new? No, because old_point is not in B_new.
        # So for each such subset, cover_count decreases by 1.
        # If it becomes 0, score decreases by 1.
        
        # Subsets with new_point:
        # These are in B_new. Are they in B_old? No, because new_point is not in B_old.
        # So for each such subset, cover_count increases by 1.
        # If it was 0 before, score increases by 1.
        
        delta = 0
        
        # Process removals (old_point)
        for rest in itertools.combinations(rest_of_block, t - 1):
            tsub = tuple(sorted(rest + (old_point,)))
            idx = tsubset_to_idx[tsub]
            cover_count[idx] -= 1
            if cover_count[idx] == 0:
                delta -= 1
        
        # Process additions (new_point)
        for rest in itertools.combinations(rest_of_block, t - 1):
            tsub = tuple(sorted(rest + (new_point,)))
            idx = tsubset_to_idx[tsub]
            if cover_count[idx] == 0:
                delta += 1
            cover_count[idx] += 1
        
        # Accept/Reject
        if delta >= 0 or random.random() < math.exp(delta / T):
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_blocks = [b[:] for b in blocks]
        else:
            # Revert
            block[j] = old_point
            # Revert cover_count
            for rest in itertools.combinations(rest_of_block, t - 1):
                tsub = tuple(sorted(rest + (old_point,)))
                idx = tsubset_to_idx[tsub]
                cover_count[idx] += 1
                # Note: if it was 0, it became 1. If it was 1, it became 0.
                # We don't need to track score here, just counts.
                
                tsub = tuple(sorted(rest + (new_point,)))
                idx = tsubset_to_idx[tsub]
                cover_count[idx] -= 1
                # Note: if it was 0, it became 1. If it was 1, it became 0.
            
            # Wait, the revert logic for cover_count is tricky if we don't store the previous state.
            # But we know the changes.
            # Removal of old_point: count[idx] -= 1
            # Addition of new_point: count[idx] += 1
            # To revert:
            # count[idx] += 1 for old_point subsets
            # count[idx] -= 1 for new_point subsets
            
            # Let's re-verify the revert logic.
            # Original state:
            # Count_old_point_subsets = C_old
            # Count_new_point_subsets = C_new
            
            # After move:
            # Count_old_point_subsets = C_old - 1
            # Count_new_point_subsets = C_new + 1
            
            # To revert:
            # Count_old_point_subsets = (C_old - 1) + 1 = C_old
            # Count_new_point_subsets = (C_new + 1) - 1 = C_new
            
            # So yes, just reverse the operations.
            
            # However, we must be careful if the same tsub appears in both sets?
            # Can a t-subset contain both old_point and new_point?
            # No, because old_point and new_point are distinct, and the rest_of_block doesn't contain either.
            # The t-subsets are disjoint sets of indices?
            # Set 1: {old_point} U rest
            # Set 2: {new_point} U rest
            # Since old_point != new_point, these sets are disjoint.
            # So no overlap. The revert is safe.
            
        T *= alpha
    
    # Return best blocks found
    return [list(b) for b in best_blocks]
