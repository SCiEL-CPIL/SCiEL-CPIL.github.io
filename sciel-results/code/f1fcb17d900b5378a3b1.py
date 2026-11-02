"""Improved Greedy-Then-Exact Hybrid for covering design C(24,7,3)."""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    """
    Construct a covering design: find as few k-element blocks of v points as possible
    such that every t-element subset is contained in at least one block.
    
    Strategy:
    1. Precompute all possible blocks and the indices of t-subsets they cover.
    2. Vectorized Greedy Set Cover: iteratively pick the block covering the most 
       currently uncovered t-subsets.
    3. Post-optimization: Remove redundant blocks (those whose coverage is 
       entirely covered by other selected blocks).
    4. ILP Completion: If any t-subsets remain uncovered, solve a small ILP 
       to complete the covering.
    """
    # Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-blocks and the indices of t-subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, store the indices of the t-subsets it covers
    # Shape: (n_blocks, C(k,t))
    block_triple_indices = np.zeros((n_blocks, len(list(itertools.combinations(range(k), t)))), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, ts in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = tsub_to_idx[ts]
            
    # Greedy Set Cover
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected_blocks = []
    
    # We need to be careful with memory if we create large intermediate arrays, 
    # but (346104, 35) is manageable.
    
    for _ in range(n_blocks): # Safety cap
        if not uncovered.any():
            break
            
        # Calculate gain for each block: number of currently uncovered t-subsets it covers
        # uncovered[indices] broadcasts to (n_blocks, C(k,t))
        gains = uncovered[block_triple_indices].sum(axis=1)
        
        # Find the block with maximum gain
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            # No block covers any new uncovered subset (should not happen if uncovered.any())
            break
            
        selected_blocks.append(all_blocks[best_idx])
        
        # Update uncovered
        uncovered[block_triple_indices[best_idx]] = False
        
    # Post-optimization: Remove redundant blocks
    # For each selected block, check if it's needed.
    # A block is redundant if all t-subsets it covers are covered by other selected blocks.
    
    # First, build a coverage map: for each t-subset, which selected blocks cover it?
    # Or simpler: just iterate and check.
    
    # Let's use a set of covered t-subsets by *other* blocks.
    # Since the number of blocks is small (~76), we can do this efficiently.
    
    # Re-verify coverage to ensure we have a valid cover first.
    # If the greedy failed to cover everything (e.g. due to some logic error or tie-breaking),
    # we might have gaps. But greedy set cover on a finite set always covers if a cover exists.
    
    # Remove redundancy
    final_blocks = []
    n_sel = len(selected_blocks)
    
    # To check redundancy, we can maintain a count of how many times each t-subset is covered by the current final_blocks list.
    # But since we are removing, it's easier to check: 
    # For block B in selected_blocks:
    #   Is there a t-subset in B that is ONLY covered by B (among selected_blocks)?
    #   If no, B is redundant.
    
    # Efficient check:
    # 1. Create an array `cover_count` of size n_tsubs, initialized to 0.
    # 2. For each block in selected_blocks, increment cover_count for its t-subsets.
    # 3. For each block in selected_blocks, check if any of its t-subsets has cover_count == 1.
    #    If yes, the block is essential. If no, it's redundant.
    #    Note: If we remove a block, cover_counts change. 
    #    However, a standard "greedy removal" or "local search" approach is:
    #    Try to remove each block one by one. If removing it leaves a valid cover, remove it.
    
    # Let's try the "try to remove" approach. It's O(N_blocks^2 * C(k,t)) which is fine for N~76.
    
    temp_blocks = list(selected_blocks)
    
    # Helper to check if a set of blocks covers all t-subsets
    def is_valid_cover(blocks_list):
        if not blocks_list:
            return n_tsubs == 0
        covered = np.zeros(n_tsubs, dtype=bool)
        for block in blocks_list:
            # Get indices of t-subsets in this block
            # We can use the precomputed block_triple_indices if we had a map from block tuple to index
            # But we don't have a reverse map easily.
            # Let's just compute on the fly for the small number of blocks.
            for ts in itertools.combinations(block, t):
                covered[tsub_to_idx[ts]] = True
        return covered.all()
        
    # Optimization: Instead of checking all blocks from scratch, we can just check if the union of coverage is full.
    # But re-computing coverage for ~76 blocks * 35 triples is ~2600 operations.
    # Doing this for each of 76 blocks is ~200k operations. Very fast.
    
    for i in range(len(temp_blocks)):
        # Try removing temp_blocks[i]
        if len(temp_blocks) <= 1:
            break
        candidate_blocks = temp_blocks[:i] + temp_blocks[i+1:]
        if is_valid_cover(candidate_blocks):
            # Removal is valid
            temp_blocks = candidate_blocks
            # Since we removed an element, the indices shift. 
            # We can just restart the loop or handle index carefully.
            # Restarting is simpler and still fast enough.
            i = 0 
            continue
            
    # If there are still uncovered t-subsets (shouldn't happen if greedy worked), 
    # or if we want to try to improve further with ILP on the residual.
    
    # Check for residual
    covered_final = np.zeros(n_tsubs, dtype=bool)
    for block in temp_blocks:
        for ts in itertools.combinations(block, t):
            covered_final[tsub_to_idx[ts]] = True
            
    uncovered_indices = np.where(~covered_final)[0]
    
    if len(uncovered_indices) > 0:
        # ILP Completion
        uncovered_set = set(all_tsubs[i] for i in uncovered_indices)
        
        # Generate candidate blocks that cover at least one uncovered t-subset
        # We can reuse the precomputed all_blocks and block_triple_indices
        # A block is a candidate if it covers any index in uncovered_indices
        
        # Create a boolean mask for blocks that cover any uncovered t-subset
        # uncovered_indices is an array of t-subset indices
        # block_triple_indices is (n_blocks, 35)
        
        # Check if any element in block_triple_indices[i] is in uncovered_indices
        # We can use np.isin
        
        # To avoid large memory usage, we can check in chunks or use a set for uncovered_indices
        unc_set = set(uncovered_indices.tolist())
        
        candidate_block_indices = []
        for i in range(n_blocks):
            # Check if any of the t-subsets in block i is in unc_set
            # block_triple_indices[i] is an array of 35 ints
            # We can do: if np.any(np.isin(block_triple_indices[i], unc_set)):
            # But np.isin with a set might be slow. 
            # Let's use a boolean array for uncovered t-subsets
            is_unc = np.zeros(n_tsubs, dtype=bool)
            is_unc[uncovered_indices] = True
            
            # This creates a (n_blocks, 35) boolean array again
            # We can do this once outside the loop? No, we need it for filtering.
            # Actually, we can just compute the mask for all blocks at once.
            pass
            
        # Let's do the vectorized mask calculation once
        is_unc = np.zeros(n_tsubs, dtype=bool)
        is_unc[uncovered_indices] = True
        
        # mask: (n_blocks, C(k,t)) -> (n_blocks,)
        # A block is useful if ANY of its t-subsets is uncovered
        useful_mask = is_unc[block_triple_indices].any(axis=1)
        
        candidate_indices = np.where(useful_mask)[0]
        
        if len(candidate_indices) == 0:
            # No block can cover the remaining? Impossible if the design exists.
            # But if it happens, we just return what we have.
            return [list(b) for b in temp_blocks]
            
        # Build ILP
        # Variables: x_j for j in candidate_indices
        # Minimize sum(x_j)
        # Subject to: for each uncovered t-subset i, sum_{j: block j covers i} x_j >= 1
        
        n_candidates = len(candidate_indices)
        
        # A matrix: (len(uncovered_indices), n_candidates)
        A = np.zeros((len(uncovered_indices), n_candidates), dtype=np.int8)
        
        # Map uncovered t-subset index to row in A
        # uncovered_indices are the original indices in 0..n_tsubs-1
        # We need to map them to 0..len(uncovered_indices)-1
        unc_idx_to_row = {unc_index: i for i, unc_index in enumerate(uncovered_indices)}
        
        for col, block_idx in enumerate(candidate_indices):
            # Get t-subsets covered by this block
            ts_indices = block_triple_indices[block_idx]
            for ts_idx in ts_indices:
                if ts_idx in unc_idx_to_row:
                    A[unc_idx_to_row[ts_idx], col] = 1
                    
        c = np.ones(n_candidates, dtype=float)
        
        constraint = LinearConstraint(A, lb=np.ones(len(uncovered_indices)), ub=np.inf)
        bounds = Bounds(lb=np.zeros(n_candidates), ub=np.ones(n_candidates))
        integrality = np.ones(n_candidates, dtype=np.int8)
        
        try:
            result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if result.status == 0 and result.x is not None:
                selected_candidates = np.where(result.x > 0.5)[0]
                for idx in selected_candidates:
                    block_idx = candidate_indices[idx]
                    temp_blocks.append(all_blocks[block_idx])
        except Exception:
            # Fallback if ILP fails
            # Add blocks greedily for remaining uncovered
            for i in uncovered_indices:
                ts = all_tsubs[i]
                # Find a block containing ts
                # We can just pick the first candidate that contains it
                for block_idx in candidate_indices:
                    if all(x in all_blocks[block_idx] for x in ts):
                        temp_blocks.append(all_blocks[block_idx])
                        break
                        # Note: this might add duplicates if not careful, but blocks are distinct in all_blocks
                        # We should check if it's already in temp_blocks? 
                        # The ILP ensures distinctness if we don't allow duplicates, but here we are adding to temp_blocks.
                        # It's safer to just add and let the final check handle it, or check membership.
                        # Since we are in a fallback, performance is less critical.
                        break
                        
    # Final verification and cleanup
    # Ensure no duplicates in temp_blocks
    unique_blocks = []
    seen = set()
    for b in temp_blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(b)
            
    # Final coverage check
    final_covered = np.zeros(n_tsubs, dtype=bool)
    for block in unique_blocks:
        for ts in itertools.combinations(block, t):
            final_covered[tsub_to_idx[ts]] = True
            
    if not final_covered.all():
        # Emergency fallback: add any block that covers an uncovered triple
        for i in range(n_tsubs):
            if not final_covered[i]:
                ts = all_tsubs[i]
                # Find any block containing ts
                for block in all_blocks:
                    if all(x in block for x in ts):
                        unique_blocks.append(block)
                        # Update final_covered
                        for tts in itertools.combinations(block, t):
                            final_covered[tsub_to_idx[tts]] = True
                        break
                        
    return [list(b) for b in unique_blocks]
