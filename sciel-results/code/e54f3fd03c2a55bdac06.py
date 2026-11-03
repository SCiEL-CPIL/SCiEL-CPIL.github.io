import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
        deadline = time.time() + 115.0
        
        # 1. Precompute
        all_tsubsets = list(itertools.combinations(range(v), t))
        n_tsubsets = len(all_tsubsets)
        tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
        
        all_blocks = list(itertools.combinations(range(v), k))
        n_blocks = len(all_blocks)
        
        # block_tsubs[bi] = list of tsub indices covered by block bi
        block_tsubs = []
        for block in all_blocks:
            tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
            block_tsubs.append(tsubs)
            
        # tsub_blocks[ti] = list of block indices covering tsub ti
        tsub_blocks = [[] for _ in range(n_tsubsets)]
        for bi, tsubs in enumerate(block_tsubs):
            for ts_idx in tsubs:
                tsub_blocks[ts_idx].append(bi)

        best_blocks = None
        best_len = float('inf')
        
        def greedy_cover(seed=0):
            rng = random.Random(seed)
            uncovered = np.ones(n_tsubsets, dtype=bool)
            blocks_used = []
            # Gain: number of uncovered tsubs covered by block i
            gain = np.array([sum(uncovered[ts] for ts in tsubs) for tsubs in block_tsubs], dtype=np.int32)
            
            while np.any(uncovered):
                if time.time() > deadline:
                    break
                max_gain = gain.max()
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                chosen = candidates[rng.randint(0, len(candidates) - 1)]
                blocks_used.append(all_blocks[chosen])
                
                # Update uncovered and gains
                for ts_idx in block_tsubs[chosen]:
                    if uncovered[ts_idx]:
                        uncovered[ts_idx] = False
                        for bi in tsub_blocks[ts_idx]:
                            if gain[bi] > 0: # Only if not chosen/processed?
                                gain[bi] -= 1
                gain[chosen] = -1
            return blocks_used

        def reduce_with_milp(initial_blocks):
            """Solve min set cover on the subset of initial_blocks."""
            if not initial_blocks:
                return []
            
            # Map block tuples to indices in all_blocks
            # To speed up, we can just work with the indices of the blocks in all_blocks
            block_indices_in_all = []
            for b in initial_blocks:
                # Find index in all_blocks. Since all_blocks is a list of tuples, we can use a dict if we built one, 
                # or just iterate. Better: build a dict once globally? 
                # For v=19, k=7, 50k blocks. Dict lookup is fast.
                pass
            
            # Let's build a block_to_idx dict for faster lookup in local search too
            # Actually, we can just use the indices from the greedy step if we returned indices.
            # Let's refactor greedy to return indices.
            
            pass # Placeholder

        # Refactor: Greedy returns indices
        def greedy_cover_indices(seed=0):
            rng = random.Random(seed)
            uncovered = np.ones(n_tsubsets, dtype=bool)
            blocks_used = []
            gain = np.array([sum(uncovered[ts] for ts in tsubs) for tsubs in block_tsubs], dtype=np.int32)
            
            while np.any(uncovered):
                if time.time() > deadline:
                    break
                max_gain = gain.max()
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                chosen = candidates[rng.randint(0, len(candidates) - 1)]
                blocks_used.append(chosen)
                
                for ts_idx in block_tsubs[chosen]:
                    if uncovered[ts_idx]:
                        uncovered[ts_idx] = False
                        for bi in tsub_blocks[ts_idx]:
                            if gain[bi] > 0:
                                gain[bi] -= 1
                gain[chosen] = -1
            return blocks_used

        def solve_reduced_milp(indices):
            """
            indices: list of block indices from all_blocks
            Returns: list of indices that form a minimal covering subset of indices
            """
            if not indices:
                return []
            
            # Build sparse matrix A_sub
            # Rows: n_tsubsets
            # Cols: len(indices)
            # A_sub[j, i] = 1 if indices[i] covers tsub j
            
            row_idx = []
            col_idx = []
            for i, bi in enumerate(indices):
                for ts_idx in block_tsubs[bi]:
                    row_idx.append(ts_idx)
                    col_idx.append(i)
            
            data = np.ones(len(row_idx), dtype=np.float64)
            A_sub = csr_matrix((data, (row_idx, col_idx)), shape=(n_tsubsets, len(indices)))
            
            c = np.ones(len(indices), dtype=np.float64)
            lb = np.ones(n_tsubsets, dtype=np.float64)
            ub = np.full(n_tsubsets, np.inf, dtype=np.float64)
            
            constraints = LinearConstraint(A_sub, lb, ub)
            integrality = np.ones(len(indices), dtype=np.int8)
            bounds = Bounds(0, 1)
            
            try:
                result = milp(
                    c=c,
                    constraints=constraints,
                    integrality=integrality,
                    bounds=bounds,
                    options={'time_limit': 5, 'mip_gap': 0.0} # Try to be optimal, small problem
                )
                if result.status == 0 or result.status == 1:
                    x = result.x
                    selected_local = [i for i in range(len(indices)) if x[i] > 0.5]
                    return [indices[i] for i in selected_local]
            except Exception:
                pass
            return indices # Fallback to original

        def local_search_swap(indices):
            """
            Try 1-for-1 swaps to improve 'quality' (not size directly, but enable 2-for-1)
            and 2-for-1 swaps to reduce size.
            indices: list of block indices
            """
            current = list(indices)
            n_cur = len(current)
            
            # Precompute coverage counts for current set
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            # Map local index to global block index
            # For speed, we work with the list of global indices
            
            # Helper to update cover counts
            def add_block(bi, sign=1):
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] += sign

            for bi in current:
                add_block(bi, 1)
            
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                
                # 1. Try to remove redundant blocks
                for i in range(len(current)):
                    bi = current[i]
                    redundant = True
                    for ts_idx in block_tsubs[bi]:
                        if cover_count[ts_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        add_block(bi, -1)
                        current.pop(i)
                        improved = True
                        break # Restart loop after modification
                
                if not improved:
                    # 2. Try 2-for-1 swaps
                    # For each pair (i, j), check if there exists a block b_new not in current
                    # such that union of unique coverage of current[i] and current[j] is covered by b_new
                    # Unique coverage of bi: tsubs where cover_count[ts] == 1 and bi covers ts.
                    
                    n_cur = len(current)
                    # We need to be efficient here.
                    # Precompute 'unique_cover' for each block in current?
                    # unique_cover[i] = set of ts_idx such that cover_count[ts_idx]==1 and block_tsubs[current[i]] contains ts_idx
                    
                    unique_covers = []
                    for i in range(n_cur):
                        bi = current[i]
                        uc = set()
                        for ts_idx in block_tsubs[bi]:
                            if cover_count[ts_idx] == 1:
                                uc.add(ts_idx)
                        unique_covers.append(uc)
                    
                    # For each pair (i, j), required_cover = unique_covers[i] | unique_covers[j]
                    # We need a block b_new that covers all ts in required_cover.
                    # Note: b_new must NOT be in current.
                    
                    # Optimization: 
                    # For a block b to cover a set of tsubs, it must contain all points of those tsubs?
                    # No, b is a k-subset. A t-subset is covered if it is a subset of b.
                    # So b must contain the union of the points in the t-subsets?
                    # If required_cover contains two t-subsets that share no points, b must contain all of them.
                    # If the union of points in required_cover has size > k, no such block exists.
                    
                    for i in range(n_cur):
                        if time.time() > deadline: break
                        for j in range(i+1, n_cur):
                            req_tsubs = unique_covers[i] | unique_covers[j]
                            if not req_tsubs:
                                continue
                            
                            # Check if there is a block covering all these tsubs
                            # A block covers a t-subset if the t-subset is contained in the block.
                            # So the block must contain the union of all points in req_tsubs.
                            # Let U be the union of points in all t-subsets in req_tsubs.
                            # If len(U) > k, impossible.
                            # If len(U) <= k, we need to find a block in all_blocks that contains U and is not in current.
                            
                            # Calculate U
                            # Since t=4, k=7, req_tsubs is a set of 4-sets.
                            # The union U can be large.
                            # If len(req_tsubs) is large, U is likely > k.
                            # We can prune: if len(req_tsubs) * t > k * something?
                            # Actually, just compute U.
                            
                            U = set()
                            for ts_idx in req_tsubs:
                                # We need the actual points of ts_idx.
                                # We have all_tsubsets list.
                                U.update(all_tsubsets[ts_idx])
                            
                            if len(U) > k:
                                continue
                            
                            # Find a block containing U
                            # We can iterate through all_blocks? 50k checks.
                            # Too slow for each pair.
                            # Precomputation: For each possible subset of points of size <= k, 
                            # list blocks containing it?
                            # Size of U is at most k.
                            # Number of possible U is sum_{j=0}^k C(19, j).
                            # C(19,7) is 50k. C(19,6) is 27k. Sum is manageable?
                            # But we only need U that are unions of t-subsets.
                            
                            # Alternative: 
                            # Iterate through candidate blocks b_new?
                            # We can filter blocks that contain U.
                            # How many blocks contain a specific set of points of size m?
                            # C(19-m, k-m).
                            # If m is small (e.g. 4), C(15, 3) = 455 blocks.
                            # If m is large (e.g. 7), C(12, 0) = 1 block.
                            
                            # So for each pair (i,j), we compute U.
                            # If len(U) <= k, we check blocks containing U.
                            # We need to check if any such block is NOT in current.
                            # And if we swap, does it maintain coverage for *other* tsubs?
                            # When we remove i and j, and add b_new:
                            # - tsubs covered only by i or j (and not others) MUST be covered by b_new. (This is satisfied by construction of req_tsubs).
                            # - tsubs covered by i or j AND others: still covered by others.
                            # - tsubs NOT covered by i or j: unaffected.
                            # - tsubs covered by b_new: 
                            #   - If they were covered by i or j, count decreases by 1 (or 2) and increases by 1.
                            #   - If they were NOT covered by i or j, count increases by 1.
                            #   - We must ensure no t-subset becomes uncovered.
                            #   - This only happens if a t-subset was covered ONLY by i or j, and b_new does NOT cover it.
                            #   - But we defined req_tsubs as exactly those tsubs covered ONLY by i or j (among current set).
                            #   - So if b_new covers req_tsubs, the coverage is maintained.
                            
                            # So the condition is simply: Find b_new not in current such that U is subset of b_new.
                            
                            # To speed up finding b_new:
                            # Precompute: for each subset U of points (size <= k), list of block indices containing U.
                            # This is expensive to build for all U.
                            # But we only encounter specific U.
                            
                            # Let's try a simpler heuristic for swap:
                            # Just check if there's ANY block containing U.
                            # If yes, and it's not in current, do the swap.
                            
                            # To avoid iterating 50k blocks for each pair:
                            # We can precompute a dictionary: point_tuple -> list of blocks containing it.
                            # For a set U, the blocks containing U are the intersection of blocks containing each point?
                            # No, blocks containing U = intersection of {blocks containing p for p in U}.
                            # This is still expensive.
                            
                            # Given the time limit, maybe skip 2-for-1 if it's too slow?
                            # The current program scores 1.0796.
                            # A good greedy + reduced MILP might get close to 140.
                            # 151 / 140 = 1.078.
                            # We need to beat 1.0796.
                            # So we need < 140 blocks? Or better greedy?
                            # If we get 139, score is 151/139 = 1.086.
                            
                            # Let's prioritize Reduced MILP on multiple Greedy runs.
                            pass
                    pass # End 2-for-1 loop
                pass # End while improved
            
            return current

        # Main Loop
        # Run multiple greedy seeds, reduce each with MILP, keep best.
        num_trials = 10
        for seed in range(num_trials):
            if time.time() > deadline - 20:
                break
            
            indices = greedy_cover_indices(seed)
            if not indices:
                continue
            
            # Reduce
            reduced = solve_reduced_milp(indices)
            
            # Local Search (Removal only first, then swaps if time)
            # For now, just removal to ensure validity and minimal size within the set.
            # The MILP already finds minimal subset, so removal should do nothing or verify.
            
            if len(reduced) < best_len:
                best_len = len(reduced)
                best_blocks = [all_blocks[i] for i in reduced]
                
            if best_len <= 135:
                break
                
        if best_blocks is None:
            # Fallback
            indices = greedy_cover_indices(0)
            best_blocks = [all_blocks[i] for i in indices]
            
        return best_blocks
