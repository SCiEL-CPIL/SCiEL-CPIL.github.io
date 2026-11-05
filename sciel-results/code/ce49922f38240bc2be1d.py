import itertools
import time
import sys
import numpy as np
from collections import Counter

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using optimized greedy with
    vectorized gain computation, multiple randomized restarts, and local search.
    """
    deadline = time.monotonic() + 115.0  # Safety margin before 120s kill
    points = list(range(v))

    # Precomputation: all t-subsets
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    # Precompute all k-blocks and their t-subset indices
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Store for each block, the list of t-subset indices it covers
    block_tsubs = []
    for b in all_kblocks:
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(idxs)
    
    # Build block_tsub matrix: (num_kblocks, num_tsubs) boolean numpy array
    block_tsub_matrix = np.zeros((num_kblocks, num_tsubs), dtype=np.int8)
    for bi, idxs in enumerate(block_tsubs):
        for ti in idxs:
            block_tsub_matrix[bi, ti] = 1
    
    # Reverse index: for each t-sub, list of blocks that cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for bi, idxs in enumerate(block_tsubs):
        for ti in idxs:
            tsub_to_blocks[ti].append(bi)
    
    # Precompute sets for blocks to speed up local search coverage check
    block_tsub_sets = [set(lst) for lst in block_tsubs]

    # Helper to check coverage
    def is_complete(block_indices):
        if not block_indices:
            return num_tsubs == 0
        if len(block_indices) == 0:
            return num_tsubs == 0
        # Use numpy: sum over selected rows, check all columns have sum >= 1
        idx_arr = np.array(block_indices, dtype=np.int32)
        covered = block_tsub_matrix[idx_arr].sum(axis=0)
        return bool(np.all(covered >= 1))

    # Helper to prune redundant blocks
    def prune(block_indices):
        if not block_indices:
            return block_indices
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for bi in block_indices:
            counts += block_tsub_matrix[bi]
        
        changed = True
        while changed:
            if time.monotonic() > deadline - 15:
                break
            changed = False
            new_list = []
            for bi in block_indices:
                # Check if all t-subs in this block have count > 1
                if np.all(counts[block_tsub_matrix[bi] == 1] > 1):
                    counts -= block_tsub_matrix[bi]
                    changed = True
                else:
                    new_list.append(bi)
            block_indices = new_list
        return block_indices

    # Fast Greedy with logarithmic potential gain computation
    def greedy(seed=None):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(num_tsubs, dtype=np.int8)
        selected = []
        remaining = np.ones(num_kblocks, dtype=bool)
        
        while time.monotonic() < deadline - 25:
            # Check if all covered
            if uncovered.sum() == 0:
                break
            
            # Vectorized gain computation: for each remaining block, count uncovered t-subs it covers
            # gain[bi] = sum of uncovered[ti] for ti in block_tsubs[bi]
            gains = (block_tsub_matrix * uncovered[None, :]).sum(axis=1)
            gains[~remaining] = -1  # Mask out non-remaining blocks
            
            # Use logarithmic potential: log(1 + gain) to better approximate submodular objective
            # This penalizes blocks with diminishing returns more strongly
            log_gains = np.log1p(gains)
            log_gains[~remaining] = -1
            
            best_block = int(np.argmax(log_gains))
            best_gain = int(gains[best_block])
            
            if best_gain <= 0:
                break
            
            selected.append(best_block)
            remaining[best_block] = False
            uncovered[block_tsub_matrix[best_block] == 1] = 0
            
        return selected

    # Local Search: drop and swap based, using precomputed structures
    def local_search_fast(block_indices, max_iters=300):
        block_indices = list(block_indices)
        n = len(block_indices)
        if n == 0:
            return block_indices

        # Build counts array
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for bi in block_indices:
            counts += block_tsub_matrix[bi]
        
        # Track which blocks are in the solution for O(1) lookup
        in_solution = set(block_indices)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.monotonic() < deadline - 10:
            improved = False
            iters += 1

            # Try dropping: find a block where all its t-subs have count > 1
            drop_idx = -1
            for idx, bi in enumerate(block_indices):
                if np.all(counts[block_tsub_matrix[bi] == 1] > 1):
                    drop_idx = idx
                    break
            
            if drop_idx >= 0:
                bi = block_indices[drop_idx]
                counts -= block_tsub_matrix[bi]
                in_solution.discard(bi)
                block_indices.pop(drop_idx)
                improved = True
                continue

            # Try swapping: for each block, find critical t-subs (count == 1)
            # Then find a replacement block covering all critical t-subs
            for i, bi in enumerate(block_indices):
                critical_mask = (block_tsub_matrix[bi] == 1) & (counts == 1)
                critical_count = int(critical_mask.sum())
                if critical_count == 0:
                    continue
                
                # Find candidate replacement blocks
                # A candidate must cover all critical t-subs
                # Use reverse index: for each critical t-sub, get blocks covering it, intersect
                critical_tsubs = np.where(critical_mask)[0]
                
                # Start with blocks covering the first critical t-sub
                candidates = set(tsub_to_blocks[int(critical_tsubs[0])])
                for ct in critical_tsubs[1:]:
                    candidates &= set(tsub_to_blocks[int(ct)])
                    if not candidates:
                        break
                
                # Remove blocks already in solution
                candidates -= in_solution
                
                if not candidates:
                    continue
                
                # Pick the first candidate (or best by some heuristic)
                j = next(iter(candidates))
                
                # Perform swap
                counts -= block_tsub_matrix[bi]
                counts += block_tsub_matrix[j]
                in_solution.discard(bi)
                in_solution.add(j)
                block_indices[i] = j
                improved = True
                break

        return block_indices

    # ILS (Iterated Local Search) with adaptive perturbation
    best_blocks = None
    best_count = float('inf')
    
    # Initial solution: deterministic greedy
    t0 = time.monotonic()
    sel = greedy(seed=42)
    if sel:
        sel = prune(sel)
        if is_complete(sel):
            sel_ls = local_search_fast(sel, max_iters=500)
            if is_complete(sel_ls):
                best_count = len(sel_ls)
                best_blocks = list(sel_ls)
                sys.stderr.write(f"Initial: {best_count} blocks\n")
            else:
                best_count = len(sel)
                best_blocks = list(sel)
                sys.stderr.write(f"Initial (no LS): {best_count} blocks\n")
    
    # ILS loop with adaptive perturbation
    if best_blocks is not None and time.monotonic() < deadline - 30:
        rng = np.random.default_rng(12345)
        no_improve = 0
        max_no_improve = 50
        perturb_strength = 0.05
        best_local_count = best_count
        
        while time.monotonic() < deadline - 15 and no_improve < max_no_improve:
            # Perturb the current best solution
            current = list(best_blocks)
            n = len(current)
            
            # Adaptive perturbation: remove some blocks and re-greedy
            # Number of blocks to remove scales with perturb_strength
            num_remove = max(1, int(n * perturb_strength))
            
            # Randomly select blocks to remove
            if n > num_remove:
                remove_indices = rng.choice(n, size=num_remove, replace=False)
                remove_set = set(remove_indices)
                perturbed = [current[i] for i in range(n) if i not in remove_set]
            else:
                perturbed = list(current)
            
            # Re-greedy to fill in coverage
            uncovered = np.zeros(num_tsubs, dtype=np.int8)
            for bi in perturbed:
                uncovered[block_tsub_matrix[bi] == 1] = 0
            # Find uncovered t-subs
            uncovered = (1 - uncovered).astype(np.int8)
            
            if uncovered.sum() > 0:
                # Greedy fill
                remaining = np.ones(num_kblocks, dtype=bool)
                for bi in perturbed:
                    remaining[bi] = False
                
                while time.monotonic() < deadline - 10 and uncovered.sum() > 0:
                    gains = (block_tsub_matrix * uncovered[None, :]).sum(axis=1).astype(np.float64)
                    gains[~remaining] = -1
                    
                    best_block = int(np.argmax(gains))
                    best_gain = int(gains[best_block])
                    
                    if best_gain <= 0:
                        break
                    
                    perturbed.append(best_block)
                    remaining[best_block] = False
                    uncovered[block_tsub_matrix[best_block] == 1] = 0
            
            # Local search on perturbed solution
            if perturbed:
                perturbed = prune(perturbed)
                if is_complete(perturbed):
                    perturbed_ls = local_search_fast(perturbed, max_iters=300)
                    if is_complete(perturbed_ls):
                        if len(perturbed_ls) < best_count:
                            best_count = len(perturbed_ls)
                            best_blocks = list(perturbed_ls)
                            no_improve = 0
                            best_local_count = best_count
                            # Increase perturbation strength slightly
                            perturb_strength = min(0.2, perturb_strength * 1.1)
                            sys.stderr.write(f"ILS improved: {best_count} blocks\n")
                        else:
                            no_improve += 1
                    else:
                        if len(perturbed) < best_count:
                            best_count = len(perturbed)
                            best_blocks = list(perturbed)
                            no_improve = 0
                            best_local_count = best_count
                            perturb_strength = min(0.2, perturb_strength * 1.1)
                            sys.stderr.write(f"ILS improved (no LS): {best_count} blocks\n")
                        else:
                            no_improve += 1
                
                # Decrease perturbation strength if no improvement
                if no_improve > 10:
                    perturb_strength = max(0.03, perturb_strength * 0.8)
    
    # Final local search on best solution
    if best_blocks is not None and time.monotonic() < deadline - 5:
        final_ls = local_search_fast(list(best_blocks), max_iters=200)
        if is_complete(final_ls) and len(final_ls) < best_count:
            best_count = len(final_ls)
            best_blocks = list(final_ls)
            sys.stderr.write(f"Final LS: {best_count} blocks\n")
    
    if best_blocks is None:
        # Fallback
        sel = greedy(seed=0)
        sel = prune(sel)
        best_blocks = sel
        best_count = len(sel)

    return [list(all_kblocks[i]) for i in best_blocks]
