"""Strategy: Multi-restart randomized greedy with local search refinement.
Uses bitmasks for fast coverage checks.
1. Precompute t-subset masks and k-subset masks.
2. Run multiple greedy constructions with random tie-breaking.
3. Apply a local search step: try to remove any block if the rest of the cover
   still covers all t-subsets. Try to swap blocks if it reduces size.
4. Return the best valid cover found within the time limit."""

import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precomputation
    # 1. Generate all t-subsets as bitmasks
    tsubs = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsubs.append(m)
    n_tsubs = len(tsubs)
    
    # 2. Generate all k-subsets as bitmasks and their covered t-subsets
    ksubs = []
    ksub_cover = [] # list of sets of tsub indices covered by each ksub
    tsub_index_map = {m: i for i, m in enumerate(tsubs)}
    
    for comb in itertools.combinations(range(v), k):
        m = 0
        for p in comb:
            m |= (1 << p)
        ksubs.append(m)
        
        # Determine which tsubs are covered by this ksub
        # A tsub is covered if all its points are in the ksub
        # i.e., tsub_mask & ksub_mask == tsub_mask
        covered_indices = []
        for i, tm in enumerate(tsubs):
            if (tm & m) == tm:
                covered_indices.append(i)
        ksub_cover.append(covered_indices)
        
    n_ksubs = len(ksubs)
    
    # Precompute a frequency count array for fast "is it covered?" checks?
    # Actually, for local search, we need to check if removing a block breaks coverage.
    # We can maintain a count of how many times each tsub is covered.
    
    best_blocks = None
    best_size = float('inf')
    
    # Helper to check validity and get size
    def evaluate(blocks):
        if not blocks:
            return 0, False
        # Count coverage
        counts = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            # b is index into ksubs
            counts[ksub_cover[b]] += 1
        if np.any(counts == 0):
            return len(blocks), False
        return len(blocks), True

    # Helper for local search improvement
    def local_search(blocks, rng):
        # blocks is a list of indices into ksubs
        # Try to remove any block that is redundant
        # Try to swap one block for another that allows removal of another?
        # Simplest: Randomly try to remove a block. If valid, keep.
        # If not, try to swap it with a random other block? 
        # Let's stick to removal first, then maybe 1-for-1 swap if removal fails?
        # Actually, greedy covers are often "minimal" (no block is redundant) 
        # because we added them to cover new elements. 
        # However, adding a block might make a PREVIOUS block redundant.
        # So we should check for redundancies after each addition or at the end.
        
        current_blocks = list(blocks)
        
        # Pass 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            # Shuffle to randomize removal order
            indices_to_check = list(range(len(current_blocks)))
            rng.shuffle(indices_to_check)
            
            for idx in indices_to_check:
                if idx >= len(current_blocks):
                    continue
                b_idx = current_blocks[idx]
                
                # Temporarily remove
                remaining = [b for i, b in enumerate(current_blocks) if i != idx]
                
                # Check validity
                counts = np.zeros(n_tsubs, dtype=np.int32)
                valid = True
                for b in remaining:
                    # Optimization: early exit if any count drops to 0?
                    # Hard to do without full count.
                    # Just count.
                    counts[ksub_cover[b]] += 1
                    # If we are just checking validity, we can stop if we find a 0?
                    # No, we need to finish counting to know if ANY is 0.
                
                if np.all(counts > 0):
                    # It was redundant
                    current_blocks.pop(idx)
                    changed = True
                    # Continue checking, maybe more are redundant now
                    # But we broke the index alignment, so restart the loop?
                    # Or just continue with the modified list?
                    # The 'indices_to_check' list is stale.
                    # Let's just break and restart the while loop.
                    break
        
        return current_blocks

    # Main Loop: Multiple Restarts
    rng = random.Random(42)
    
    # We can afford roughly 10-20 restarts if each is fast.
    # Greedy construction is O(N * K) where N is num blocks (~150), K is num candidates (50k).
    # 150 * 50k = 7.5M operations. In Python, this might take ~1-2s per run.
    # So ~50 runs in 115s? Maybe.
    # Let's optimize the greedy step.
    
    def greedy_construct(rng):
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        chosen = []
        
        # Precompute for each ksub, the list of tsub indices it covers (already done in ksub_cover)
        # To speed up "gain" calculation:
        # gain[j] = sum(uncovered[ksub_cover[j]])
        
        while True:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_j = -1
            best_candidates = []
            
            # Calculate gain for all ksubs
            # Vectorized approach:
            # We can't easily vectorize the sum over variable length lists in numpy without a matrix.
            # Let's use a loop but keep it tight.
            # Alternative: Maintain an array 'covered_count' of size n_tsubs.
            # When we pick a block, we increment covered_count for its tsubs.
            # Gain for block j = sum(1 for i in ksub_cover[j] if covered_count[i] == 0)
            
            # Let's maintain 'uncovered' boolean array.
            # gain[j] = np.sum(uncovered[ksub_cover[j]])
            
            # To speed this up, we can precompute the cover lists as numpy arrays.
            # cover_arrays[j] = np.array(ksub_cover[j])
            
            # But creating 50k arrays is memory heavy? 50k * 35 ints = 1.75M ints. OK.
            
            # Let's just loop. 50k iterations is fast in Python if simple.
            for j in range(n_ksubs):
                # If j is already chosen, skip? 
                # Greedy usually doesn't pick same block twice.
                if j in chosen_set: 
                    continue
                
                # Count how many new tsubs this block covers
                # Using the uncovered array
                # sum(uncovered[cover_list])
                # Since uncovered is int8, sum is fast.
                gain = np.sum(uncovered[ksub_cover[j]])
                
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    best_candidates = [j]
                elif gain == best_gain:
                    best_candidates.append(j)
            
            if best_gain <= 0:
                break
                
            # Tie-breaking: Random choice among best candidates
            if len(best_candidates) > 1:
                # Add some diversity
                # Maybe prefer blocks with less overlap with existing?
                # For now, random pick is a good differentiator
                best_j = rng.choice(best_candidates)
            
            chosen.append(best_j)
            chosen_set.add(best_j)
            
            # Update uncovered
            uncovered[ksub_cover[best_j]] = 0
            
            # Early termination if all covered
            if np.all(uncovered == 0):
                break
        
        return chosen

    # Pre-convert ksub_cover lists to numpy arrays for faster indexing in the loop
    cover_arrays = [np.array(lst, dtype=np.int32) for lst in ksub_cover]
    
    # Re-define greedy to use cover_arrays
    def greedy_construct_fast(rng):
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        chosen = []
        chosen_set = set()
        
        while True:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_candidates = []
            
            for j in range(n_ksubs):
                if j in chosen_set:
                    continue
                
                gain = np.sum(uncovered[cover_arrays[j]])
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [j]
                elif gain == best_gain:
                    best_candidates.append(j)
            
            if best_gain <= 0:
                break
                
            if len(best_candidates) > 1:
                best_j = rng.choice(best_candidates)
            else:
                best_j = best_candidates[0]
            
            chosen.append(best_j)
            chosen_set.add(best_j)
            
            uncovered[cover_arrays[best_j]] = 0
            
            if np.all(uncovered == 0):
                break
        
        return chosen

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10.0: # Leave 10s for final repair/validation
        rng = random.Random(num_restarts * 123 + 456)
        
        blocks = greedy_construct_fast(rng)
        
        # Local search: Remove redundant blocks
        # We need to check if removing any block keeps the cover valid.
        # This is O(N^2 * K) in worst case if done naively.
        # N ~ 150, K ~ 35. 150*150*35 = 787,500 ops. Very fast.
        
        # Check redundancy
        # Build coverage counts
        counts = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            counts[cover_arrays[b]] += 1
        
        # Try to remove any block where min count in its coverage is > 1?
        # No, if a block covers a tsub that has count 1, it's essential.
        # If a block only covers tsubs that have count > 1, it's redundant.
        
        changed = True
        while changed:
            changed = False
            # Randomize order for diversity
            idxs = list(range(len(blocks)))
            rng.shuffle(idxs)
            
            for idx in idxs:
                if idx >= len(blocks):
                    continue
                b_idx = blocks[idx]
                
                # Check if all tsubs covered by b_idx have count > 1
                # i.e., min(counts[cover_arrays[b_idx]]) > 1
                if np.min(counts[cover_arrays[b_idx]]) > 1:
                    # It is redundant
                    # Remove it
                    blocks.pop(idx)
                    # Update counts
                    counts[cover_arrays[b_idx]] -= 1
                    changed = True
                    # Restart loop as indices changed
                    break
        
        size = len(blocks)
        if size < best_size:
            best_size = size
            best_blocks = list(blocks)
            
        num_restarts += 1
        
        # Optional: If we have time, do a quick swap search?
        # Swap block A with block B if it allows removal of block C?
        # Too complex for the remaining time budget likely.
        
    # Final Repair: Ensure validity
    # The greedy + removal should be valid, but just in case of bugs or edge cases
    if best_blocks is None:
        # Fallback: Simple greedy without local search
        rng = random.Random(0)
        best_blocks = greedy_construct_fast(rng)
        
    # Verify validity one last time
    counts = np.zeros(n_tsubs, dtype=np.int32)
    for b in best_blocks:
        counts[cover_arrays[b]] += 1
        
    if np.any(counts == 0):
        # Invalid! Add missing blocks
        # Find uncovered tsubs
        uncovered_idx = np.where(counts == 0)[0]
        for i in uncovered_idx:
            # Find a ksub that covers this tsub
            # tsub_mask = tsubs[i]
            # We need a ksub that contains all points of tsub
            # Since we have all ksubs, we can just pick the first one that covers it
            # Or better, pick one that covers the most other uncovered tsubs?
            # For simplicity and speed, just pick any valid one.
            # Actually, we can reconstruct the block from the tsub.
            # tsub has t points. We need k-t more points.
            # Let's just find a ksub index that covers this tsub.
            # Precompute: for each tsub, which ksubs cover it?
            # We didn't do that. 
            # Let's just loop through ksubs.
            found = False
            for j in range(n_ksubs):
                if (tsubs[i] & ksubs[j]) == tsubs[i]:
                    # Check if j is already in best_blocks
                    if j not in best_blocks:
                        best_blocks.append(j)
                        found = True
                        break
            if not found:
                # Should not happen since we have all ksubs
                pass
                
    # Convert indices to lists of points
    result = []
    for b_idx in best_blocks:
        mask = ksubs[b_idx]
        pts = [p for p in range(v) if (mask >> p) & 1]
        result.append(pts)
        
    return result
