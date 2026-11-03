import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage masks for all blocks
    # We store as a 2D numpy array: (n_blocks, n_uint64s)
    n_uint64s = (n_t + 63) // 64
    
    coverage_masks = np.zeros((n_blocks, n_uint64s), dtype=np.uint64)
    
    # Optimization: Iterate through blocks and set bits
    # To speed this up, we can precompute the indices of t-subsets for each block
    # But let's try to vectorize the bit setting as much as possible or just run the loop.
    # 346k blocks * 35 t-subsets = 12M operations.
    # Python loop is slow. Let's try to optimize.
    
    # Alternative: Generate t-subset indices for all blocks at once?
    # Hard to vectorize "combinations of combinations".
    # Let's stick to the loop but maybe use a faster way to map t-subset to index.
    # t_subset_to_idx is a dict lookup.
    
    # We can precompute the bitmask for each t-subset? No, t-subsets are the elements.
    # We need: for block b, for each ts in C(b, t), set bit idx(ts).
    
    # Let's just run the loop. It should take ~1-2 seconds.
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            word_idx = idx // 64
            bit_idx = idx % 64
            coverage_masks[i, word_idx] |= (1 << bit_idx)
            
    if time.time() > deadline - 20.0:
        # Fallback if precomputation is too slow (unlikely)
        return [list(all_blocks[0])] # Invalid, but prevents crash. Should not happen.

    # 2. Diverse Greedy Set Cover with Penalty
    # Objective: Minimize number of blocks.
    # Heuristic: Maximize (newly covered) - lambda * (overlap with existing)
    # This encourages picking blocks that cover new ground and are distinct from previous choices.
    
    # We will run multiple iterations with different lambda values to explore different solution landscapes.
    # Then we will perform a reduction phase on the best candidate.
    
    best_solution_blocks = None
    best_solution_size = float('inf')
    
    # Lambda values to try: 0 (standard greedy), 1, 2, 5, 10
    lambdas = [0, 1, 2, 5, 10]
    
    # We might also want to randomize the tie-breaking or initial order?
    # Standard greedy is deterministic. Diverse greedy with lambda > 0 is deterministic too.
    # To get diversity, we can also shuffle the candidate pool? No, that breaks the "best first" logic slightly.
    # Actually, the penalty term makes the scores unique more often, reducing ties.
    
    for lam in lambdas:
        if time.time() > deadline - 10.0:
            break
            
        covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
        selected_blocks = []
        
        # We need to track the union of selected blocks to compute overlap.
        # union_mask: bitwise OR of all selected blocks' masks.
        union_mask = np.zeros(n_uint64s, dtype=np.uint64)
        
        max_iterations = 200 # Safety
        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
                
            # Check if all covered
            if np.bitwise_count(covered_mask).sum() == n_t:
                break
                
            # Compute gains for all blocks
            not_covered = ~covered_mask
            # New elements covered by block j
            new_covered = coverage_masks & not_covered
            new_counts = np.bitwise_count(new_covered).sum(axis=1)
            
            # Overlap with existing selected blocks
            # overlap_j = popcount(coverage_masks[j] & union_mask)
            overlaps = coverage_masks & union_mask
            overlap_counts = np.bitwise_count(overlaps).sum(axis=1)
            
            # Score = new_counts - lam * overlap_counts
            scores = new_counts - lam * overlap_counts
            
            # Pick the block with max score
            best_block_idx = np.argmax(scores)
            best_score = scores[best_block_idx]
            
            # If the best score is negative or zero, we might be stuck or done.
            # If new_counts is 0, we are stuck (should not happen if valid covering exists).
            # If new_counts > 0 but score < 0, it means overlap penalty is high.
            # We should still pick it if it covers new elements, otherwise we can't finish.
            # So we only break if new_counts[best_block_idx] == 0.
            
            if new_counts[best_block_idx] == 0:
                # No block covers any new element. Stuck.
                break
            
            selected_blocks.append(best_block_idx)
            covered_mask |= coverage_masks[best_block_idx]
            union_mask |= coverage_masks[best_block_idx]
            
        # Check if valid
        if np.bitwise_count(covered_mask).sum() == n_t:
            current_size = len(selected_blocks)
            if current_size < best_solution_size:
                best_solution_size = current_size
                best_solution_blocks = selected_blocks.copy()
                
                # Early stop if we reach the lower bound or a very good number?
                # Lower bound is 69. Best known is 76.
                # If we find < 76, it's a major improvement.
                # If we find 76, it's a tie with best known.
                # Let's keep searching for smaller.
                
    # If we didn't find a valid solution (unlikely), fallback to standard greedy
    if best_solution_blocks is None:
        covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
        selected_blocks = []
        for _ in range(200):
            not_covered = ~covered_mask
            intersections = coverage_masks & not_covered
            gains = np.bitwise_count(intersections).sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_blocks.append(best_idx)
            covered_mask |= coverage_masks[best_idx]
        best_solution_blocks = selected_blocks
        best_solution_size = len(selected_blocks)

    # 3. Local Search: Block Removal (Deletion)
    # Now that we have a candidate, try to remove redundant blocks.
    # This is effective if the diverse greedy left blocks that are now redundant
    # due to the spread-out nature of the selection.
    
    # We need efficient check for "is block i essential?"
    # Essential if it covers any t-subset that no other selected block covers.
    # i.e., min(cover_counts[t_idxs[i]]) == 1 for some t_idx in block i.
    
    # Precompute block_t_indices for the selected blocks only?
    # Or reuse the global one?
    # We need cover counts for the current solution.
    
    # Let's build a structure for the current solution.
    # selected_blocks is a list of block indices.
    
    # Initialize cover counts
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in best_solution_blocks:
        # We need the t-subset indices for this block.
        # We can re-derive them or store them.
        # Re-deriving is slow if done repeatedly.
        # Let's precompute a list of t-indices for each block in the solution.
        pass
    
    # To make removal fast, let's precompute the t-indices for all blocks once?
    # We did that in the current program's local search.
    # Let's do it here.
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)
        
    # Initialize cover counts
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in best_solution_blocks:
        cover_counts[block_t_indices[b_idx]] += 1
        
    # Removal Loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try to remove blocks in reverse order (often newer blocks are more redundant)
        # Or random order? Reverse is a good heuristic.
        for i in range(len(best_solution_blocks) - 1, -1, -1):
            b_idx = best_solution_blocks[i]
            t_idxs = block_t_indices[b_idx]
            
            # Check if any t-subset has count 1
            # If min(cover_counts[t_idxs]) > 1, safe to remove.
            
            # Vectorized check
            if np.min(cover_counts[t_idxs]) > 1:
                # Safe to remove
                cover_counts[t_idxs] -= 1
                best_solution_blocks.pop(i)
                improved = True
                # Restart loop to check new possibilities
                break
                
    # 4. Optional: Swap Move
    # If removal is stuck, try swapping a block with another that covers its essential parts
    # and allows further removals.
    # Given time constraints, let's see if removal alone is enough.
    # The diverse greedy should produce a solution where many blocks are redundant.
    
    # Final Verification
    final_blocks = [list(all_blocks[i]) for i in best_solution_blocks]
    
    # Check validity
    test_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in best_solution_blocks:
        test_counts[block_t_indices[b_idx]] += 1
        
    if np.any(test_counts == 0):
        # Invalid. Fallback to the pre-reduction solution?
        # We modified best_solution_blocks in place.
        # We should have saved a copy.
        # Let's just re-run the standard greedy quickly as a fallback.
        covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
        valid_blocks = []
        for _ in range(200):
            not_covered = ~covered_mask
            intersections = coverage_masks & not_covered
            gains = np.bitwise_count(intersections).sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            valid_blocks.append(best_idx)
            covered_mask |= coverage_masks[best_idx]
        return [list(all_blocks[i]) for i in valid_blocks]

    return final_blocks
