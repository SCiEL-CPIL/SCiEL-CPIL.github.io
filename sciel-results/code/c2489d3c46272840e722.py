import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute for each t-subset, the list of points not in it?
    # No, we need to generate k-subsets containing a specific t-subset.
    # We can precompute the "complement" points for each subset to speed up candidate generation?
    # Actually, generating combinations is fast enough.
    
    best_blocks = None
    best_len = float('inf')
    
    # We will run multiple greedy restarts
    # Estimate cost of one restart:
    # ~150 steps.
    # Each step: generate 455 candidates, compute gain for each.
    # Gain computation: 35 lookups in a boolean array.
    # 455 * 35 = 15925 ops.
    # 150 * 15925 = 2.4M ops.
    # In Python, maybe 0.1s - 0.5s per restart?
    # Let's try to run as many as possible.
    
    for run in range(1000): # Max 1000, but break on time
        if time.time() > deadline:
            break
        
        random.seed(run) # Reproducible per run, but different between runs
        
        covered = np.zeros(num_subsets, dtype=bool)
        blocks = []
        
        # List of indices of uncovered subsets
        # We maintain this list and filter it periodically or just scan it.
        # Scanning is O(N). N starts at 3876.
        # Let's just scan the range(num_subsets) to find an uncovered one?
        # Or maintain a list of uncovered indices.
        uncovered_indices = list(range(num_subsets))
        
        while uncovered_indices:
            # Find an uncovered subset
            # Randomly pick from uncovered_indices to diversify
            # But we need to ensure it's actually uncovered.
            # Since we don't remove from the list immediately, we might pick a covered one.
            # Strategy: Pick a random index from uncovered_indices.
            # If it's covered, we need to find an uncovered one.
            # To be efficient, we can just pick the first uncovered one in the list,
            # but shuffle the list occasionally?
            
            # Let's pick a random candidate from the list
            # If the list is large, random pick is O(1).
            # Check if covered.
            
            # Optimization: If the list is small, just scan.
            # If the list is large, random pick might hit covered ones often.
            # Let's maintain a list of *truly* uncovered indices by filtering every few steps?
            # Or just scan from the beginning to find the first uncovered.
            # Scanning is deterministic. To randomize, we can rotate the start index.
            
            # Let's use a simple approach:
            # Pick a random index from the current uncovered_indices list.
            # If it is covered, we ignore it and try again?
            # This could be slow if most are covered.
            
            # Better: Maintain `uncovered_indices` as a list of indices that are *known* to be uncovered.
            # When we cover a subset, we don't remove it from the list immediately.
            # When we need to pick a target, we scan `uncovered_indices` to find one where `covered[idx]` is False.
            # To make it random, we can shuffle `uncovered_indices` occasionally?
            # Or just pick a random offset and scan forward.
            
            # Let's just scan from a random starting point.
            if not uncovered_indices:
                break
                
            # Pick a random start
            start_pos = random.randint(0, len(uncovered_indices) - 1)
            target_idx = -1
            for i in range(len(uncovered_indices)):
                idx = uncovered_indices[(start_pos + i) % len(uncovered_indices)]
                if not covered[idx]:
                    target_idx = idx
                    break
            
            if target_idx == -1:
                # All are covered?
                uncovered_indices = []
                break
            
            target_subset = all_subsets[target_idx]
            
            # Generate candidates: k-subsets containing target_subset
            # target_subset is a tuple of t points.
            # We need to choose k-t points from the remaining v-t points.
            remaining_points = [p for p in range(v) if p not in target_subset]
            candidates = []
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_subset + combo))
                candidates.append(block)
            
            # Calculate gain for each candidate
            # Gain = number of subsets in block that are currently uncovered
            # We need the indices of the subsets in the block.
            # Precomputing subset indices for a block on the fly:
            # A block has C(k,t) subsets.
            # For each subset in block, get its index from subset_to_idx.
            
            best_candidate = None
            best_gain = -1
            
            for block in candidates:
                gain = 0
                # Iterate through all t-subsets of this block
                for sub in itertools.combinations(block, t):
                    sub_idx = subset_to_idx[sub]
                    if not covered[sub_idx]:
                        gain += 1
                        # Early exit? No, we need total gain.
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = block
                    # Tie breaking: if gain == best_gain, we could randomize.
                    # But simple greedy picks the first max.
                    # To add randomness, we can collect all max gain candidates and pick one randomly.
                    # For now, let's stick to first max for speed, or simple random tie-break.
                    # Let's do random tie-break to improve restart diversity.
                    # Actually, just picking the first max is fine if we randomize the target selection.
                    
            # Add best_candidate to blocks
            blocks.append(list(best_candidate))
            
            # Update covered
            for sub in itertools.combinations(best_candidate, t):
                sub_idx = subset_to_idx[sub]
                covered[sub_idx] = True
            
            # Periodically clean up uncovered_indices?
            # If len(uncovered_indices) > 1000 and run % 10 == 0:
            #    uncovered_indices = [i for i in uncovered_indices if not covered[i]]
            # This is O(N).
            # Let's do it every 10 steps to keep the list manageable.
            if len(blocks) % 10 == 0:
                uncovered_indices = [i for i in uncovered_indices if not covered[i]]
                
            # Check if all covered
            # We can check if sum(~covered) == 0
            # But that's O(N).
            # The loop condition `while uncovered_indices` will handle it if we clean up.
            # If we don't clean up, the list might contain covered indices.
            # The scan `if not covered[idx]` handles it.
            # But if the list is full of covered indices, the scan is O(N).
            # So cleaning up is good.

        # Pruning phase
        if blocks:
            # Count coverage for each subset
            counts = np.zeros(num_subsets, dtype=int)
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    counts[subset_to_idx[sub]] += 1
            
            # Remove non-essential blocks
            # A block is essential if it contains a subset with count 1
            essential = [True] * len(blocks)
            for i, block in enumerate(blocks):
                is_essential = False
                for sub in itertools.combinations(block, t):
                    if counts[subset_to_idx[sub]] == 1:
                        is_essential = True
                        break
                essential[i] = is_essential
            
            # Filter blocks
            new_blocks = [blocks[i] for i in range(len(blocks)) if essential[i]]
            
            # If we removed blocks, we could re-run pruning?
            # No, as reasoned before, one pass is enough.
            # But wait, if we remove a block, counts decrease.
            # A subset that had count 2 might become count 1.
            # This might make a *different* block essential?
            # No, making a subset count 1 makes the blocks covering it *more* likely to be essential.
            # It doesn't make a non-essential block essential.
            # A block is non-essential if all its subsets have count > 1.
            # Decreasing counts cannot turn a count > 1 into a count == 1 for a subset *not* in the block?
            # Wait.
            # Block A is non-essential: all its subsets have count >= 2.
            # We remove Block B.
            # Counts of subsets in B decrease.
            # If a subset S is in A, its count decreases only if S is also in B.
            # If S is in A and B, and count was 2, it becomes 1.
            # Then A becomes essential (because S has count 1).
            # So removing blocks can make other blocks essential.
            # It cannot make an essential block non-essential.
            # So we should iterate until no more blocks are removed.
            
            changed = True
            while changed:
                changed = False
                counts = np.zeros(num_subsets, dtype=int)
                for block in new_blocks:
                    for sub in itertools.combinations(block, t):
                        counts[subset_to_idx[sub]] += 1
                
                essential = [True] * len(new_blocks)
                for i, block in enumerate(new_blocks):
                    is_essential = False
                    for sub in itertools.combinations(block, t):
                        if counts[subset_to_idx[sub]] == 1:
                            is_essential = True
                            break
                    essential[i] = is_essential
                
                old_len = len(new_blocks)
                new_blocks = [new_blocks[i] for i in range(len(new_blocks)) if essential[i]]
                if len(new_blocks) < old_len:
                    changed = True
                else:
                    changed = False
                    
            if len(new_blocks) < best_len:
                best_len = len(new_blocks)
                best_blocks = new_blocks
                
    if best_blocks is None:
        # Fallback: simple greedy without restarts if something went wrong
        # Or return empty? No, must be valid.
        # The first run should always produce a valid covering.
        pass
        
    return best_blocks
