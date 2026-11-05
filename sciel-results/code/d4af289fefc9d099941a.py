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
    
    # Build a numpy matrix: for each t-subset, which blocks cover it
    # Use a list of lists of block indices per t-subset for fast reverse lookup
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for bi, b_tsubs in enumerate(block_tsubs):
        for ts_idx in b_tsubs:
            tsub_to_blocks[ts_idx].append(bi)
    
    # --- Phase 1: Greedy construction ---
    covered = np.zeros(num_tsubs, dtype=np.uint8)
    blocks = []
    
    while True:
        if time.monotonic() > deadline:
            break
        
        best_gain = 0
        best_block_idx = -1
        
        # Find the block with maximum gain
        for i, b_tsubs in enumerate(block_tsubs):
            gain = 0
            for idx in b_tsubs:
                if not covered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        
        if best_gain == 0:
            break
            
        best_block = all_kblocks[best_block_idx]
        blocks.append(list(best_block))
        
        for idx in block_tsubs[best_block_idx]:
            covered[idx] = 1
            
        if covered.all():
            break
    
    # --- Phase 2: Local search - remove redundant blocks ---
    if time.monotonic() < deadline - 10:
        # Try to remove blocks whose t-subsets are all covered by other selected blocks
        changed = True
        while changed and time.monotonic() < deadline - 10:
            changed = False
            n_blocks = len(blocks)
            for i in range(n_blocks - 1, -1, -1):
                if time.monotonic() > deadline - 10:
                    break
                # Check if block i is redundant
                b_tsubs = block_tsubs[all_kblocks.index(tuple(blocks[i]))]
                redundant = True
                for ts_idx in b_tsubs:
                    # Check if this t-subset is covered by some other selected block
                    found = False
                    for j in range(n_blocks):
                        if j == i:
                            continue
                        other_block = all_kblocks.index(tuple(blocks[j]))
                        if ts_idx in block_tsubs[other_block]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
    
    # --- Phase 3: Local search - 2-for-1 swaps ---
    if time.monotonic() < deadline - 5:
        # For pairs of blocks, check if a single block can cover all their unique t-subsets
        improved = True
        while improved and time.monotonic() < deadline - 5:
            improved = False
            n_blocks = len(blocks)
            for i in range(n_blocks):
                if time.monotonic() > deadline - 5:
                    break
                for j in range(i + 1, n_blocks):
                    if time.monotonic() > deadline - 5:
                        break
                    # Find t-subsets covered only by block i or block j (not by others)
                    # A t-subset is "unique" to {i,j} if it's in block i or j but not in any other selected block
                    bi = all_kblocks.index(tuple(blocks[i]))
                    bj = all_kblocks.index(tuple(blocks[j]))
                    ti = set(block_tsubs[bi])
                    tj = set(block_tsubs[bj])
                    
                    # Check if there exists a single block that covers all t-subsets in ti ∪ tj
                    # that are not covered by other selected blocks
                    other_blocks = [all_kblocks.index(tuple(blocks[m])) for m in range(n_blocks) if m != i and m != j]
                    
                    # Find t-subsets in ti ∪ tj not covered by other blocks
                    uncovered_by_others = set()
                    for ts_idx in (ti | tj):
                        covered_by_other = False
                        for ob in other_blocks:
                            if ts_idx in block_tsubs[ob]:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            uncovered_by_others.add(ts_idx)
                    
                    if not uncovered_by_others:
                        # Both blocks are redundant together
                        blocks.pop(j)
                        blocks.pop(i)
                        improved = True
                        break
                    
                    # Try to find a single block that covers all uncovered_by_others
                    found_swap = False
                    for candidate in range(num_kblocks):
                        cand_tsubs = set(block_tsubs[candidate])
                        if uncovered_by_others.issubset(cand_tsubs):
                            # Check that adding candidate doesn't break anything
                            # (it only adds coverage, so it's fine)
                            blocks.pop(j)
                            blocks.pop(i)
                            blocks.append(list(all_kblocks[candidate]))
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if improved:
                    break
    
    return blocks
