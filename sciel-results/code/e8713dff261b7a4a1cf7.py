"""Covering design C(v,k,t) via LP-relaxation column generation + rounding + deletion."""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115
    start_time = time.time()

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # For each point, list of t-subset indices containing it
    point_to_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for p in ts:
            point_to_tsubs[p].append(idx)

    def block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices

    def greedy_cover(uncovered_mask, seed, randomize=True):
        """Build a cover starting from current uncovered_mask state."""
        rng = random.Random(seed)
        blocks = []
        mask = uncovered_mask.copy()
        
        while mask.any():
            # Find the t-subset with lowest "coverage potential" — pick first uncovered
            uncovered_idx = np.where(mask)[0]
            if len(uncovered_idx) == 0:
                break
            target_idx = uncovered_idx[0] if not randomize else rng.choice(uncovered_idx)
            target_ts = all_tsubs[target_idx]
            
            # Build a block around this t-subset
            block = list(target_ts)
            in_block = set(block)
            candidates = [x for x in range(v) if x not in in_block]
            if randomize:
                rng.shuffle(candidates)
            
            while len(block) < k and candidates:
                best_gain = -1
                best_x = candidates[0]
                for x in candidates:
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and mask[idx]:
                            gain += 1
                    if gain > best_gain or (gain == best_gain and rng.random() < 0.3):
                        best_gain = gain
                        best_x = x
                block.append(best_x)
                candidates.remove(best_x)
            
            block.sort()
            blocks.append(block)
            for idx in block_tsub_indices(block):
                mask[idx] = 0
        
        return blocks

    def lp_column_generation(seed, max_rounds=2000):
        """
        Column generation for set cover LP:
        - Start with a small set of columns (blocks)
        - Solve dual (primal = min sum x_j s.t. Ax >= 1, x >= 0)
        - Price out new columns (blocks) with negative reduced cost
        - Iterate until no improving column exists
        - Return LP solution (fractional) and the column pool
        """
        rng = random.Random(seed)
        
        # Initialize with a greedy solution's blocks
        init_blocks = greedy_cover(np.ones(n_tsubs, dtype=np.int32), seed, randomize=True)
        
        # Build the column pool as a set of frozensets
        col_pool = set()
        for b in init_blocks:
            col_pool.add(frozenset(b))
        
        # Also add random columns
        for _ in range(500):
            cb = frozenset(rng.sample(range(v), k))
            col_pool.add(cb)
        
        # We'll maintain the LP solution via a simple iterative method:
        # Since scipy is available, use scipy.optimize.linprog or milp
        # But for column gen, we need to solve the LP repeatedly.
        # Alternative: use a dual-descent heuristic.
        
        # Let's use a simpler approach: iterated greedy with LP-guided weighting
        # The dual variables y_i (for each t-subset i) tell us how "expensive" covering i is.
        # A column (block) has reduced cost = 1 - sum_{i in block} y_i
        # We add columns with negative reduced cost.
        
        # Initialize dual variables: y_i = 1 for all i (uniform)
        y = np.ones(n_tsubs, dtype=np.float64)
        
        # Precompute for each block in pool, which t-subs it covers
        # We'll work with a growing list of blocks
        active_blocks = list(col_pool)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 30:
                break
            
            # For each active block, compute its "score" = sum of y_i for t-subs it covers
            # We want to find the block with maximum score (most negative reduced cost)
            # But we can't check all C(19,7) blocks each round.
            # Instead, solve a sub-problem: given weights y, find the k-subset maximizing sum of y over its t-subsets.
            
            # This is itself a hard problem, but we can approximate:
            # Greedy: start from the t-sub with highest y, extend to k-subset greedily.
            
            # Find the t-sub with highest y
            best_start = np.argmax(y)
            
            # Try multiple starts
            top_starts = np.argsort(y)[-10:]  # top 10 t-subs by dual weight
            
            best_block_score = -1
            best_block = None
            
            for start_idx in top_starts:
                start_ts = all_tsubs[start_idx]
                block = list(start_ts)
                in_block = set(block)
                candidates = [x for x in range(v) if x not in in_block]
                
                while len(block) < k and candidates:
                    best_gain = -1
                    best_x = candidates[0]
                    for x in candidates:
                        gain = 0
                        for combo in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(combo + (x,)))
                            idx = tsub_index.get(ts)
                            if idx is not None:
                                gain += y[idx]
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    block.append(best_x)
                    candidates.remove(best_x)
                
                # Score this block
                score = 0
                for ts in itertools.combinations(block, t):
                    idx = tsub_index.get(tuple(sorted(ts)))
                    if idx is not None:
                        score += y[idx]
                
                if score > best_block_score:
                    best_block_score = score
                    best_block = tuple(sorted(block))
            
            if best_block is None:
                break
            
            # Add this block to pool if new
            if best_block not in col_pool:
                col_pool.add(frozenset(best_block))
                active_blocks.append(best_block)
            
            # Update dual variables using a simple subgradient method:
            # Primal: min 1^T x  s.t. A x >= 1, x >= 0
            # Dual: max 1^T y  s.t. A^T y <= 1, y >= 0
            # Subgradient for dual: for each column j, if (A^T y)_j > 1, decrease y for rows in column j
            # Simpler: y_i *= exp(-alpha * (coverage_i - 1)) where coverage_i = sum of x_j for j covering i
            
            # Compute current "coverage" under uniform weighting of active blocks
            # Actually, let's just do: y_i = max(0, y_i - alpha * (num_blocks_covering_i - 1))
            # But we need x values. Let's use a simpler heuristic:
            # y_i proportional to 1 / (number of active blocks covering i)
            
            cov_count = np.zeros(n_tsubs, dtype=np.float64)
            for b in active_blocks:
                for ts in itertools.combinations(b, t):
                    idx = tsub_index.get(tuple(sorted(ts)))
                    if idx is not None:
                        cov_count[idx] += 1.0 / len(active_blocks)
            
            # y_i = 1 / max(cov_count_i, epsilon) — under-covered t-subs get higher weight
            y = 1.0 / np.maximum(cov_count, 1e-6)
            y /= y.sum() if y.sum() > 0 else 1.0
        
        return active_blocks

    def round_lp_to_cover(active_blocks, seed):
        """
        Given a pool of candidate blocks, find a minimum subset that covers all t-subs.
        Use greedy set cover on the pool, then deletion.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        selected = []
        
        # Precompute tsub indices for each block
        block_indices = []
        for b in active_blocks:
            idxs = block_tsub_indices(list(b))
            block_indices.append(idxs)
        
        # Greedy: repeatedly pick block covering most uncovered t-subs
        while uncovered.any() and active_blocks:
            best_gain = -1
            best_i = -1
            # Sample if pool is large
            if len(active_blocks) > 200:
                sample = rng.sample(range(len(active_blocks)), 200)
            else:
                sample = range(len(active_blocks))
            
            for i in sample:
                gain = 0
                for idx in block_indices[i]:
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            
            if best_i < 0 or best_gain == 0:
                break
            
            selected.append(active_blocks[best_i])
            for idx in block_indices[best_i]:
                uncovered[idx] = 0
        
        return selected

    def deletion_pass(blocks, dl):
        """Iteratively remove redundant blocks."""
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks:
            indices = block_tsub_indices(block)
            block_tsubs.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        blocks_list = [list(b) for b in blocks]
        improved = True
        while improved and time.time() < dl - 5:
            improved = False
            n_blocks = len(blocks_list)
            for i in range(n_blocks):
                if time.time() > dl - 5:
                    break
                # Check if block i is redundant
                redundant = True
                for idx in block_tsubs[i]:
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_tsubs[i]:
                        cov_count[idx] -= 1
                    blocks_list.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    break
        
        return blocks_list

    # Phase 1: Run LP column generation with different seeds
    best_blocks = None
    best_count = float('inf')
    
    # First get a quick greedy solution as baseline
    quick_blocks = greedy_cover(np.ones(n_tsubs, dtype=np.int32), 0, randomize=False)
    best_blocks = quick_blocks
    best_count = len(quick_blocks)
    
    # LP column generation with multiple seeds
    for seed in range(5):
        if time.time() > deadline - 40:
            break
        
        pool = lp_column_generation(seed, max_rounds=1000)
        
        # Round to integer cover
        cover = round_lp_to_cover(pool, seed)
        
        # Deletion pass
        cover = deletion_pass(cover, deadline)
        
        if len(cover) < best_count:
            best_blocks = cover
            best_count = len(cover)
    
    # Phase 2: Local search (1-for-1 swaps) on best solution
    if time.time() < deadline - 20:
        rng = random.Random(99)
        blocks_list = [list(b) for b in best_blocks]
        
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks_list:
            indices = block_tsub_indices(block)
            block_tsubs.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        n_blocks = len(blocks_list)
        best_count = n_blocks
        best_snapshot = [list(b) for b in blocks_list]
        
        # Build candidate pool for swaps
        candidate_pool = []
        for _ in range(3000):
            cb = tuple(sorted(rng.sample(range(v), k)))
            candidate_pool.append(cb)
        
        step = 0
        no_improve = 0
        max_no_improve = 3000
        
        while step < 50000 and time.time() < deadline - 10:
            step += 1
            
            # Try to remove a block
            if n_blocks > 0:
                # Find a block with no critical t-subs
                removable = []
                for i in range(n_blocks):
                    critical = 0
                    for idx in block_tsubs[i]:
                        if cov_count[idx] == 1:
                            critical += 1
                    if critical == 0:
                        removable.append(i)
                
                if removable:
                    bi = rng.choice(removable)
                    for idx in block_tsubs[bi]:
                        cov_count[idx] -= 1
                    blocks_list.pop(bi)
                    block_tsubs.pop(bi)
                    n_blocks -= 1
                    no_improve = 0
                    if n_blocks < best_count:
                        best_count = n_blocks
                        best_snapshot = [list(b) for b in blocks_list]
                    continue
            
            # Try a swap: remove one, add one
            if n_blocks > 1 and candidate_pool:
                bi = rng.randrange(n_blocks)
                for idx in block_tsubs[bi]:
                    cov_count[idx] -= 1
                blocks_list.pop(bi)
                block_tsubs.pop(bi)
                n_blocks -= 1
                
                cb = rng.choice(candidate_pool)
                cb_list = list(cb)
                indices = block_tsub_indices(cb_list)
                blocks_list.append(cb_list)
                block_tsubs.append(indices)
                for idx in indices:
                    cov_count[idx] += 1
                n_blocks += 1
                no_improve += 1
            
            if no_improve > max_no_improve:
                blocks_list = [list(b) for b in best_snapshot]
                cov_count = np.zeros(n_tsubs, dtype=np.int32)
                block_tsubs = []
                for block in blocks_list:
                    indices = block_tsub_indices(block)
                    block_tsubs.append(indices)
                    for idx in indices:
                        cov_count[idx] += 1
                n_blocks = len(blocks_list)
                no_improve = 0
                max_no_improve = max(500, int(max_no_improve * 0.7))
                if max_no_improve < 500:
                    break
        
        best_blocks = best_snapshot
        best_count = len(best_blocks)
    
    # Final deletion pass
    if time.time() < deadline - 3:
        best_blocks = deletion_pass(best_blocks, deadline)
    
    # Verify coverage
    if best_blocks:
        final_cov = np.zeros(n_tsubs, dtype=np.int32)
        for block in best_blocks:
            for idx in block_tsub_indices(block):
                final_cov[idx] += 1
        if final_cov.min() < 1:
            # Fallback: add blocks to fix uncovered
            uncovered_idx = np.where(final_cov == 0)[0]
            for idx in uncovered_idx:
                ts = all_tsubs[idx]
                extra = [x for x in range(v) if x not in ts]
                block = list(ts) + extra[:k - t]
                block.sort()
                best_blocks.append(block)
    
    return best_blocks if best_blocks else greedy_cover(np.ones(n_tsubs, dtype=np.int32), 0, randomize=False)
