import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build incidence data
    # block_tsubs: list of t-subset indices for each block
    # tsub_blocks: list of block indices for each t-subset
    block_tsubs = []
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            tsubs.append(idx)
            tsub_blocks[idx].append(bi)
        block_tsubs.append(tsubs)
        
    # Precompute sets for faster intersection in local search
    tsub_block_sets = [set(lst) for lst in tsub_blocks]

    best_blocks = None
    best_len = float('inf')

    def time_left():
        return deadline - time.time()

    def remove_redundant(blocks):
        """Remove blocks that do not cover any unique t-subset."""
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for b in blocks:
            tsubs = []
            for ts in itertools.combinations(b, t):
                idx = tsub_index[ts]
                tsubs.append(idx)
                cover_count[idx] += 1
            block_tsub_indices.append(tsubs)
        
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    changed = True
        
        return [blocks[i] for i in range(len(blocks)) if kept[i]]

    def greedy_cover(seed, weights=None):
        """
        Greedy set cover using a max-heap.
        weights: optional array of weights for t-subsets (duals). If None, uniform.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        # gain[bi] = sum of weights of currently uncovered t-subsets covered by block bi
        gain = np.zeros(n_blocks, dtype=np.float64)
        
        if weights is not None:
            for bi in range(n_blocks):
                gain[bi] = np.sum(weights[block_tsubs[bi]])
        else:
            for bi in range(n_blocks):
                gain[bi] = len(block_tsubs[bi])
                
        # Max-heap: (-gain, bi)
        heap = [(-gain[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        
        selected = []
        picked = np.zeros(n_blocks, dtype=bool)
        
        while time_left() > 5:
            # Pop until we find a valid (non-stale) entry with positive gain
            best_bi = -1
            max_gain = 0.0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                # Check if this gain is still valid (matches current gain array)
                if g == gain[bi] and g > 1e-9 and not picked[bi]:
                    best_bi = bi
                    max_gain = g
                    break
                # Otherwise, it's a stale entry or picked, discard
            else:
                break # Heap empty or all gains 0
                
            if max_gain <= 1e-9:
                break
                
            selected.append(best_bi)
            picked[best_bi] = True
            
            # Update gains for all blocks that share t-subsets with best_bi
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    w = weights[ts_idx] if weights is not None else 1.0
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= w
                        if gain[bj] > 1e-9 and not picked[bj]:
                            heapq.heappush(heap, (-gain[bj], bj))
                        
            # Mark best_bi as used (gain -1 to prevent re-selection)
            gain[best_bi] = -1.0
            heapq.heappush(heap, (-gain[best_bi], best_bi))
            
            if not np.any(uncovered):
                break
                
        # Fallback for any remaining uncovered
        if not np.all(uncovered == False):
            # Simple greedy for remaining
            rem_uncovered = np.where(uncovered)[0]
            uncovered_set = set(rem_uncovered)
            
            while uncovered_set and time_left() > 5:
                best_block = None
                best_count = -1
                # Scan all blocks
                for j in range(n_blocks):
                    if picked[j]:
                        continue
                    count = 0
                    for s_idx in block_tsubs[j]:
                        if s_idx in uncovered_set:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_block = j
                    # Early break if we found a block covering all? Unlikely.
                
                if best_block is None or best_count == 0:
                    break
                    
                selected.append(best_block)
                picked[best_block] = True
                for s_idx in block_tsubs[best_block]:
                    if s_idx in uncovered_set:
                        uncovered_set.remove(s_idx)
                        
        return selected

    def local_search(blocks, seed):
        """
        Local search: Remove redundant blocks and perform swaps.
        blocks: list of block indices
        """
        rng = random.Random(seed)
        # Work with a set for O(1) lookup
        block_set = set(blocks)
        
        # cover_count[ts_idx] = number of blocks in block_set covering ts_idx
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        
        # Initialize cover_count
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time_left() > 2:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed
            
            # Phase 2: Try swaps
            # If a block has unique t-subsets, try to replace it with another block that covers those unique t-subsets.
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time_left() < 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Find unique t-subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # This block is redundant (should have been caught in Phase 1, but just in case)
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart local search loop
                    
                # Find candidate blocks that cover ALL unique_tsubs
                # Start with the rarest unique tsubset to minimize candidates
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                    
                # Remove current block from candidates
                candidates.discard(bi)
                
                if not candidates:
                    continue
                    
                # Try candidates - limit to a reasonable number to avoid long stalls
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:50]: # Limit candidates
                    if bj in block_set:
                        continue
                        
                    # Perform swap
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart main loop to re-evaluate redundancy
                    
                if swapped:
                    break # Restart main loop

        return list(block_set)

    # Solve LP to get duals
    duals = None
    try:
        # Build sparse matrix A for LP: M.T @ x >= 1 => -M.T @ x <= -1
        rows = []
        cols = []
        data = []
        for j in range(n_blocks):
            for s_idx in block_tsubs[j]:
                rows.append(s_idx)
                cols.append(j)
                data.append(1.0)
        
        M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
        
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # res.ineqlin.marginals are for A_ub @ x <= b_ub
            # The constraint is -M.T @ x <= -1
            # Dual variable y >= 0 for M @ y >= 1 (primal min, dual max)
            # In linprog, marginals are d(Obj)/d(b). 
            # Obj = c^T x. Constraint -M^T x <= -1.
            # If we relax -1 to -1+delta, obj changes by marginal * delta.
            # Standard duality: c^T x = y^T 1.
            # The marginal for constraint i is the shadow price.
            # For min problem with <= constraints, marginals are <= 0.
            # y_i = -marginal_i.
            duals = -res.ineqlin.marginals
            duals = np.maximum(duals, 0)
            
    except Exception as e:
        # Fallback to uniform weights if LP fails
        duals = np.ones(n_tsubsets)

    # Run multiple trials with different seeds and perturbations
    num_trials = 5
    for i in range(num_trials):
        if time_left() < 15:
            break
            
        # Perturb duals slightly to break symmetry
        if i > 0:
            perturbed_duals = duals * (1 + 0.1 * np.random.random(n_tsubsets))
        else:
            perturbed_duals = duals
            
        blocks = greedy_cover(seed=i, weights=perturbed_duals)
        
        if blocks:
            # Local search refinement
            blocks = local_search(blocks, seed=i)
            blocks = remove_redundant(blocks)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                
        # If we have a very good solution, stop early
        if best_len <= 140:
            break

    if best_blocks is None:
        # Final fallback: standard greedy
        duals = np.ones(n_tsubsets)
        best_blocks = greedy_cover(seed=0, weights=duals)
        best_blocks = local_search(best_blocks, seed=0)
        best_blocks = remove_redundant(best_blocks)

    return best_blocks
