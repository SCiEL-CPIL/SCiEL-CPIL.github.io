"""Greedy covering with fast gain computation, local search pruning, and deadline guard."""
import itertools
import time
import numpy as np

def construct(v, k, t):
    deadline = time.monotonic() + 115.0  # safety margin under 120s
    points = list(range(v))
    
    # Generate all t-subsets and assign them indices for fast lookup
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute all possible k-blocks
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # For each k-block, precompute the list of t-subset indices it covers.
    block_tsubs = []
    for b in all_kblocks:
        b_tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(b_tsubs)
    
    # Build a numpy matrix: for each block, which t-subsets it covers (as a set for fast lookup)
    block_tsub_sets = [set(bts) for bts in block_tsubs]
    
    # Build reverse lookup: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for bi, bts in enumerate(block_tsubs):
        for ts_idx in bts:
            tsub_to_blocks[ts_idx].append(bi)
    
    # --- Phase 1: Weighted greedy construction ---
    covered = np.zeros(num_tsubs, dtype=np.uint8)
    block_indices = []  # store indices into all_kblocks
    
    # Weight: initially 1 for all uncovered t-subsets
    weights = np.ones(num_tsubs, dtype=np.float64)
    
    while True:
        if time.monotonic() > deadline - 20:
            break
        
        best_gain = 0.0
        best_block_idx = -1
        
        # Find the block with maximum weighted gain
        for bi in range(num_kblocks):
            bts = block_tsubs[bi]
            gain = 0.0
            for idx in bts:
                if not covered[idx]:
                    gain += weights[idx]
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi
        
        if best_gain == 0:
            break
            
        blocks_list = list(all_kblocks[best_block_idx])
        block_indices.append(best_block_idx)
        
        # Update covered
        for idx in block_tsubs[best_block_idx]:
            covered[idx] = 1
        
        # Update weights: increase weight of uncovered t-subsets
        # Subsets covered by fewer selected blocks are "harder"
        uncovered_mask = (covered == 0)
        if uncovered_mask.any():
            # Count how many selected blocks cover each uncovered t-subset
            # (should be 0 since they're uncovered, but let's boost based on difficulty)
            # Simpler: boost weights of uncovered subsets that appear in fewer candidate blocks
            pass  # Keep simple for now; the weighted gain already helps
        
        if covered.all():
            break
    
    # --- Phase 2: Local search - remove redundant blocks ---
    if time.monotonic() < deadline - 15:
        changed = True
        while changed and time.monotonic() < deadline - 15:
            changed = False
            n_blocks = len(block_indices)
            for i in range(n_blocks - 1, -1, -1):
                if time.monotonic() > deadline - 15:
                    break
                # Check if block i is redundant: all its t-subsets are covered by other selected blocks
                bi = block_indices[i]
                bts = block_tsubs[bi]
                redundant = True
                for ts_idx in bts:
                    # Check if this t-subset is covered by some other selected block
                    found = False
                    for j in range(n_blocks):
                        if j == i:
                            continue
                        bj = block_indices[j]
                        if ts_idx in block_tsub_sets[bj]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    block_indices.pop(i)
                    changed = True
    
    # --- Phase 3: Local search - try to replace a block with a better one ---
    if time.monotonic() < deadline - 5:
        improved = True
        while improved and time.monotonic() < deadline - 5:
            improved = False
            n_blocks = len(block_indices)
            
            for i in range(n_blocks):
                if time.monotonic() > deadline - 5:
                    break
                
                bi = block_indices[i]
                # Find t-subsets covered ONLY by block i (not by any other selected block)
                bts_i = block_tsub_sets[bi]
                unique_to_i = set()
                for ts_idx in bts_i:
                    covered_by_other = False
                    for j in range(n_blocks):
                        if j == i:
                            continue
                        bj = block_indices[j]
                        if ts_idx in block_tsub_sets[bj]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        unique_to_i.add(ts_idx)
                
                if not unique_to_i:
                    # Block i is redundant
                    block_indices.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all unique_to_i
                # and also covers at least as many other t-subsets as block i
                for candidate in range(num_kblocks):
                    if candidate == bi:
                        continue
                    cand_set = block_tsub_sets[candidate]
                    if unique_to_i.issubset(cand_set):
                        # Check if candidate covers at least as many total t-subsets
                        # as block i (within the current solution context)
                        # Actually, just replace if it covers unique_to_i
                        # But we need to ensure no t-subset becomes uncovered
                        # Since we only require covering unique_to_i, and other t-subsets
                        # of block i are already covered by others, this is safe
                        block_indices[i] = candidate
                        improved = True
                        break
                if improved:
                    break
    
    # Convert block indices back to lists
    blocks = [list(all_kblocks[bi]) for bi in block_indices]
    
    return blocks
