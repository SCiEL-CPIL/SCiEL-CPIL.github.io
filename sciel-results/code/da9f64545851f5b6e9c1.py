import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_tsubs = []
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            tsubs.append(idx)
            tsub_blocks[idx].append(bi)
        block_tsubs.append(tsubs)
        
    # Convert block_tsubs to sets for faster subset checks in local search
    block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]

    best_blocks = None
    best_len = float('inf')

    def time_left():
        return deadline - time.time()

    def remove_redundant(blocks, cover_count=None, cover_blocks=None):
        """
        Remove redundant blocks.
        If cover_count and cover_blocks are provided, use them. Otherwise compute.
        Returns (new_blocks, new_cover_count, new_cover_blocks)
        """
        if not blocks:
            return blocks, np.zeros(n_tsubsets, dtype=np.int32), [[] for _ in range(n_tsubsets)]
        
        if cover_count is None or cover_blocks is None:
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            cover_blocks = [[] for _ in range(n_tsubsets)]
            for b in blocks:
                for s_idx in block_tsubs[b]:
                    cover_count[s_idx] += 1
                    cover_blocks[s_idx].append(b)
        
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, b in enumerate(blocks):
                if not kept[i]:
                    continue
                redundant = True
                for s_idx in block_tsubs[b]:
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for s_idx in block_tsubs[b]:
                        cover_count[s_idx] -= 1
                        # Remove b from cover_blocks[s_idx]
                        # Since we only remove, and we iterate, we can just mark it.
                        # But for correctness of subsequent checks, we should update cover_blocks.
                        # However, cover_count is sufficient for the "is it unique" check.
                        # We only need cover_blocks for candidate generation in swaps.
                        # We will rebuild cover_blocks if needed or update it.
                        # For efficiency in this function, we just update cover_count.
                        # The caller will need to update cover_blocks if it uses them.
                    changed = True
        
        new_blocks = [blocks[i] for i in range(len(blocks)) if kept[i]]
        
        # Rebuild cover_blocks if we want to return it accurately for the caller
        # But the caller might have its own cover_blocks.
        # Let's just return the new blocks. The caller should update its state.
        return new_blocks, cover_count, cover_blocks

    def lp_guided_greedy(duals, seed=0):
        rng = random.Random(seed)
        covered = np.zeros(n_tsubsets, dtype=bool)
        picked = np.zeros(n_blocks, dtype=bool)
        blocks = []
        
        block_weights = np.zeros(n_blocks)
        for j in range(n_blocks):
            if block_tsubs[j]:
                block_weights[j] = np.sum(duals[block_tsubs[j]])
                
        heap = [(-block_weights[j], j) for j in range(n_blocks) if block_weights[j] > 1e-9]
        heapq.heapify(heap)
        
        while heap and time_left() > 10:
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-9:
                continue
            
            covers_uncovered = False
            for s_idx in block_tsubs[j]:
                if not covered[s_idx]:
                    covers_uncovered = True
                    break
            if not covers_uncovered:
                continue
                
            picked[j] = True
            blocks.append(j)
            
            for s_idx in block_tsubs[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in tsub_blocks[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                if block_weights[b_prime] > 1e-9:
                                    heapq.heappush(heap, (-block_weights[b_prime], b_prime))
            
            if not np.any(~covered):
                break
        
        if not np.all(covered):
            uncovered_set = set(np.where(~covered)[0])
            while uncovered_set and time_left() > 10:
                best_block = None
                best_count = -1
                # Sample blocks to speed up?
                # For now, scan all.
                for j in range(n_blocks):
                    if picked[j]: continue
                    count = 0
                    for s_idx in block_tsubs[j]:
                        if s_idx in uncovered_set:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_block = j
                if best_block is None or best_count == 0:
                    break
                blocks.append(best_block)
                picked[best_block] = True
                for s_idx in block_tsubs[best_block]:
                    if s_idx in uncovered_set:
                        uncovered_set.remove(s_idx)
                        
        return blocks

    # Solve LP
    duals = None
    try:
        rows, cols, data = [], [], []
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
            duals = -res.ineqlin.marginals
            duals = np.maximum(duals, 0)
    except Exception:
        duals = np.ones(n_tsubsets)

    def local_search(blocks):
        """
        blocks: list of block indices
        """
        blocks = list(blocks)
        
        # Initialize cover state
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        cover_blocks = [[] for _ in range(n_tsubsets)]
        for b in blocks:
            for s_idx in block_tsubs[b]:
                cover_count[s_idx] += 1
                cover_blocks[s_idx].append(b)
        
        # Helper to check if all covered
        def is_covered():
            return np.all(cover_count > 0)
        
        # Helper to update state on add/remove
        def add_block(b):
            blocks.append(b)
            for s_idx in block_tsubs[b]:
                cover_count[s_idx] += 1
                cover_blocks[s_idx].append(b)
        
        def remove_block(b):
            blocks.remove(b)
            for s_idx in block_tsubs[b]:
                cover_count[s_idx] -= 1
                cover_blocks[s_idx].remove(b)
        
        # Initial redundancy removal
        while True:
            new_blocks, cover_count, _ = remove_redundant(blocks, cover_count, cover_blocks)
            if len(new_blocks) == len(blocks):
                break
            blocks = new_blocks
            # Rebuild cover state for removed blocks? 
            # remove_redundant returns updated cover_count.
            # But cover_blocks is stale.
            # Let's just rebuild cover_blocks from scratch if we change blocks significantly.
            cover_blocks = [[] for _ in range(n_tsubsets)]
            for b in blocks:
                for s_idx in block_tsubs[b]:
                    cover_blocks[s_idx].append(b)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()

        rng = random.Random(42)
        T = 1.0 # Temperature for simulated annealing style acceptance
        
        while time_left() > 10:
            # 1. Try to remove redundant blocks
            redundant_removed = False
            for b in list(blocks):
                redundant = True
                for s_idx in block_tsubs[b]:
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(b)
                    redundant_removed = True
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = blocks.copy()
            
            if not is_covered():
                # Repair: Add blocks to cover uncovered subsets
                uncovered = np.where(cover_count == 0)[0]
                if len(uncovered) == 0:
                    continue
                # Pick a random uncovered subset
                s_idx = rng.choice(uncovered.tolist())
                candidates = cover_blocks[s_idx]
                if not candidates:
                    # Should not happen if precomputation is correct
                    break
                # Pick candidate that covers most other uncovered subsets
                best_cand = None
                best_score = -1
                for b in candidates:
                    if b in blocks:
                        continue
                    score = 0
                    for s2 in block_tsubs[b]:
                        if cover_count[s2] == 0:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = b
                if best_cand is not None:
                    add_block(best_cand)
                continue

            # 2. If covered, try local moves
            # 2a. 2-for-1 move
            # Sample pairs
            if len(blocks) >= 2:
                b1, b2 = rng.sample(blocks, 2)
                # Calculate U_pair: subsets covered only by b1 or b2
                U_pair = []
                # We need to check all subsets covered by b1 or b2
                # To speed up, we can iterate over subsets of b1 and b2
                # But we need to check cover_count.
                # If cover_count[s] == 1 and s in b1, then s is unique to b1.
                # If cover_count[s] == 2 and s in b1 and b2, then s is unique to {b1, b2}.
                # If cover_count[s] == 1 and s in b2, then s is unique to b2.
                
                # Let's gather candidates for U_pair
                # We only care about subsets s such that cover_count[s] <= 2 and s is in b1 or b2.
                # We can iterate over all s in block_tsub_sets[b1] and block_tsub_sets[b2].
                candidates_s = block_tsub_sets[b1].union(block_tsub_sets[b2])
                for s in candidates_s:
                    if cover_count[s] <= 2:
                        # Check if s is actually covered by b1 or b2
                        # Since s is in the union, it is.
                        # But we need to ensure that the only blocks covering s are subset of {b1, b2}
                        # cover_count[s] == 1 implies the only block is b1 or b2.
                        # cover_count[s] == 2 implies the blocks are a subset of {b1, b2} of size 2?
                        # Not necessarily. It could be {b1, b_other}.
                        # So we must check cover_blocks[s].
                        if all(b in (b1, b2) for b in cover_blocks[s]):
                            U_pair.append(s)
                
                if U_pair:
                    # Find a block b_new that covers all s in U_pair
                    # Pick a random s0 from U_pair to limit candidates
                    s0 = rng.choice(U_pair)
                    found = False
                    for b_new in tsub_blocks[s0]:
                        if b_new in blocks:
                            continue
                        # Check if b_new covers all U_pair
                        valid = True
                        for s in U_pair:
                            if s not in block_tsub_sets[b_new]:
                                valid = False
                                break
                        if valid:
                            # Perform swap
                            # Remove b1, b2
                            remove_block(b1)
                            remove_block(b2)
                            # Add b_new
                            add_block(b_new)
                            found = True
                            if len(blocks) < best_len:
                                best_len = len(blocks)
                                best_blocks = blocks.copy()
                            break
                    # If not found, we could try other s0, but for now just skip.
            
            # 2b. 1-for-1 swap (to escape local minima)
            if blocks:
                b_old = rng.choice(blocks)
                U_old = []
                for s in block_tsub_sets[b_old]:
                    if cover_count[s] == 1:
                        U_old.append(s)
                
                if U_old:
                    s0 = rng.choice(U_old)
                    for b_new in tsub_blocks[s0]:
                        if b_new in blocks:
                            continue
                        valid = True
                        for s in U_old:
                            if s not in block_tsub_sets[b_new]:
                                valid = False
                                break
                        if valid:
                            # Swap
                            remove_block(b_old)
                            add_block(b_new)
                            # Try to remove redundant
                            # (Will be handled in next iteration's step 1)
                            break

    # Run initialization and local search
    num_trials = 10
    for i in range(num_trials):
        if time_left() < 20:
            break
        
        if i == 0:
            blocks = lp_guided_greedy(duals, seed=i)
        else:
            # Perturb duals
            perturbed_duals = duals * (1 + 0.2 * np.random.random(n_tsubsets))
            blocks = lp_guided_greedy(perturbed_duals, seed=i)
        
        if not blocks:
            continue
            
        # Convert to list of indices for local search
        local_search(blocks)

    if best_blocks is None:
        # Fallback
        duals = np.ones(n_tsubsets)
        blocks = lp_guided_greedy(duals, seed=0)
        best_blocks = blocks

    # Convert indices back to lists of points
    return [list(all_blocks[b]) for b in best_blocks]
