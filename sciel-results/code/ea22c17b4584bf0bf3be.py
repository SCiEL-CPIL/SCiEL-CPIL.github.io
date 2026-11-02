import itertools
import random
import time
import heapq
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets and their indices
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map from block tuple to index for fast lookup
    block_to_idx = {blk: i for i, blk in enumerate(all_blocks)}

    # For each block, precompute the list of t-subset indices it covers
    # Also precompute the set of t-subsets for faster set operations if needed, 
    # but lists are better for counting coverage.
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    def greedy_solve(seed, max_time_frac=0.05):
        """Run a fast greedy pass to get an initial valid covering."""
        local_rng = random.Random(seed)
        time_budget = _DEADLINE * max_time_frac
        local_start = time.time()

        uncovered = [True] * n_ts
        uncovered_count = n_ts

        # Current gain for each block
        block_gain = [len(ts_list) for ts_list in block_ts]
        block_version = [0] * n_blocks
        in_block = [False] * n_blocks

        # Heap: (-gain, block_index, version)
        heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while uncovered_count > 0:
            if time.time() - local_start > time_budget:
                break

            # Pop until valid
            valid_entry = None
            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                if in_block[b_idx]:
                    continue
                if ver != block_version[b_idx]:
                    continue
                if block_gain[b_idx] <= 0:
                    continue
                valid_entry = (neg_gain, b_idx, ver)
                break

            if valid_entry is None:
                break

            neg_gain, b_idx, ver = valid_entry
            selected.append(b_idx)
            in_block[b_idx] = True

            # Update coverage
            for s_idx in block_ts[b_idx]:
                if not uncovered[s_idx]:
                    continue
                uncovered[s_idx] = False
                uncovered_count -= 1

                # Update gains of other blocks containing this t-subset
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        result = [list(all_blocks[b]) for b in selected]

        # Check if complete and fix if needed
        if uncovered_count > 0:
            unc_set = set(i for i in range(n_ts) if not uncovered[i])
            while unc_set and time.time() - local_start < _DEADLINE - 2.0:
                best_b = -1
                best_gain = 0
                # To speed this up, we could use the heap, but since we broke out of it, 
                # let's just scan. This only happens if greedy timed out.
                for b_idx in range(n_blocks):
                    if in_block[b_idx]:
                        continue
                    gain = sum(1 for s in block_ts[b_idx] if s in unc_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                result.append(list(all_blocks[best_b]))
                in_block[best_b] = True
                for s in block_ts[best_b]:
                    unc_set.discard(s)

        return result

    def local_search_sa(blocks, seed, time_limit):
        """
        Simulated Annealing local search.
        State: set of selected block indices.
        Cost: num_blocks + lambda * num_uncovered.
        Moves: Add, Remove, Swap.
        """
        local_rng = random.Random(seed)
        local_start = time.time()
        
        if not blocks:
            return blocks

        # Initialize state
        selected_set = set(tuple(blk) for blk in blocks)
        selected_indices = [block_to_idx[blk] for blk in blocks]
        
        # Coverage counts
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b_idx in selected_indices:
            for s_idx in block_ts[b_idx]:
                coverage[s_idx] += 1
        
        num_uncovered = np.sum(coverage == 0)
        current_cost = len(selected_indices) + 10.0 * num_uncovered
        
        best_cost = current_cost
        best_state = selected_set.copy()
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        lambda_unc = 10.0
        
        # Precompute potential gains/losses for moves? 
        # For Remove: Gain is 1 block. Loss is number of subsets that become uncovered.
        # For Add: Loss is 1 block. Gain is number of subsets that become covered.
        
        steps = 0
        while time.time() - local_start < time_limit:
            steps += 1
            if steps % 1000 == 0:
                T *= cooling
                if T < T_min:
                    T = T_min

            # Choose a move
            move_type = local_rng.choice(['remove', 'add', 'swap'])
            
            new_cost = current_cost
            delta_cost = 0
            
            if move_type == 'remove' and selected_indices:
                b_idx = local_rng.choice(selected_indices)
                # Check if removal is safe or calculate penalty
                newly_uncovered = 0
                for s_idx in block_ts[b_idx]:
                    if coverage[s_idx] == 1:
                        newly_uncovered += 1
                
                delta_cost = -1 + lambda_unc * newly_uncovered
                new_cost = current_cost + delta_cost
                
                if local_rng.random() < min(1.0, np.exp(-delta_cost / T)):
                    # Apply move
                    selected_indices.remove(b_idx)
                    selected_set.discard(all_blocks[b_idx])
                    for s_idx in block_ts[b_idx]:
                        coverage[s_idx] -= 1
                    current_cost = new_cost
                    num_uncovered += newly_uncovered # Approximate update, will recalculate if needed for best
            
            elif move_type == 'add':
                # Pick a random block not in set
                # Optimization: Pick a block that covers many uncovered subsets
                # For simplicity and speed, random pick is okay if we have many steps,
                # but let's pick a random block from all_blocks
                b_idx = local_rng.randrange(n_blocks)
                if b_idx in selected_set:
                    continue
                    
                newly_covered = 0
                for s_idx in block_ts[b_idx]:
                    if coverage[s_idx] == 0:
                        newly_covered += 1
                        
                delta_cost = 1 - lambda_unc * newly_covered
                new_cost = current_cost + delta_cost
                
                if local_rng.random() < min(1.0, np.exp(-delta_cost / T)):
                    selected_indices.append(b_idx)
                    selected_set.add(all_blocks[b_idx])
                    for s_idx in block_ts[b_idx]:
                        coverage[s_idx] += 1
                    current_cost = new_cost
                    num_uncovered -= newly_covered
            
            elif move_type == 'swap' and selected_indices:
                b_out_idx = local_rng.choice(selected_indices)
                b_in_idx = local_rng.randrange(n_blocks)
                if b_in_idx in selected_set:
                    continue
                
                # Calculate delta for removing b_out and adding b_in
                # Uncovered by b_out (if count 1)
                newly_uncovered = 0
                for s_idx in block_ts[b_out_idx]:
                    if coverage[s_idx] == 1:
                        newly_uncovered += 1
                        
                # Covered by b_in (if count 0)
                newly_covered = 0
                for s_idx in block_ts[b_in_idx]:
                    if coverage[s_idx] == 0:
                        newly_covered += 1
                        
                # Note: Subsets in both are handled correctly by net effect
                # Actually, we need to be careful.
                # Let's just calculate the change in cost directly.
                # Cost change = (1 - 1) + lambda * (newly_uncovered - newly_covered)
                # Wait, if a subset is covered by both, removing one doesn't uncover it, adding one doesn't cover it.
                # My simple count above assumes disjointness which is false.
                
                # Correct calculation:
                # For each s in b_out: if cov[s]==1, it becomes 0 (penalty +1). 
                #   BUT if s is also in b_in, then cov[s] becomes 1 again (net 0 change for that s).
                # For each s in b_in: if cov[s]==0, it becomes 1 (gain -1).
                #   BUT if s is also in b_out, it was 1, becomes 0, then 1 (net 0).
                
                # So:
                # Penalty from b_out: count of s in b_out where cov[s]==1 AND s not in b_in
                # Gain from b_in: count of s in b_in where cov[s]==0 AND s not in b_out
                
                # To do this efficiently, we can use sets.
                set_out = set(block_ts[b_out_idx])
                set_in = set(block_ts[b_in_idx])
                
                # s in b_out, cov 1, not in b_in
                p1 = 0
                for s in set_out:
                    if s not in set_in and coverage[s] == 1:
                        p1 += 1
                        
                # s in b_in, cov 0, not in b_out
                g1 = 0
                for s in set_in:
                    if s not in set_out and coverage[s] == 0:
                        g1 += 1
                        
                delta_cost = lambda_unc * (p1 - g1)
                new_cost = current_cost + delta_cost
                
                if local_rng.random() < min(1.0, np.exp(-delta_cost / T)):
                    # Apply swap
                    selected_indices.remove(b_out_idx)
                    selected_set.discard(all_blocks[b_out_idx])
                    selected_indices.append(b_in_idx)
                    selected_set.add(all_blocks[b_in_idx])
                    
                    # Update coverage
                    # Subtract b_out
                    for s in set_out:
                        coverage[s] -= 1
                    # Add b_in
                    for s in set_in:
                        coverage[s] += 1
                        
                    current_cost = new_cost
                    # Update num_uncovered approximately or just rely on cost
                    # For tracking best, we need exact num_uncovered
                    # Recalculating num_uncovered is O(n_ts) which is 3876. Acceptable every step?
                    # Maybe every 10 steps?
                    if steps % 10 == 0:
                        num_uncovered = np.sum(coverage == 0)
                        # Recalculate current_cost to be safe? 
                        # current_cost = len(selected_indices) + lambda_unc * num_uncovered
                        # If we trust the delta, we don't need to. But drift is possible.
                        # Let's update current_cost based on actual state to avoid drift.
                        current_cost = len(selected_indices) + lambda_unc * num_uncovered

            # Check for best
            if steps % 10 == 0:
                num_uncovered = np.sum(coverage == 0)
                current_cost = len(selected_indices) + lambda_unc * num_uncovered
                if num_uncovered == 0 and len(selected_indices) < best_cost:
                    best_cost = len(selected_indices)
                    best_state = selected_set.copy()
                elif current_cost < best_cost: # Track best cost too, in case we find a valid one later
                    # We prefer valid solutions. If invalid, we track cost to guide SA.
                    # But we only return valid ones.
                    # Let's keep best_state as the best VALID solution.
                    pass
            else:
                # Just check if we improved the valid best
                if current_cost < best_cost and num_uncovered == 0:
                    best_cost = current_cost
                    best_state = selected_set.copy()

        # Convert best_state back to list of lists
        return [list(blk) for blk in best_state]

    # 1. Get an initial greedy solution
    initial_result = greedy_solve(12345, max_time_frac=0.05)
    
    # Ensure validity
    cov_check = np.zeros(n_ts, dtype=np.int32)
    for blk in initial_result:
        b_idx = block_to_idx[blk]
        for s_idx in block_ts[b_idx]:
            cov_check[s_idx] += 1
            
    if np.any(cov_check == 0):
        # Emergency fix
        unc_set = set(np.where(cov_check == 0)[0])
        while unc_set and _remaining() > 5.0:
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = sum(1 for s in block_ts[b_idx] if s in unc_set)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            initial_result.append(list(all_blocks[best_b]))
            for s in block_ts[best_b]:
                unc_set.discard(s)
            # Update cov_check
            for s in block_ts[best_b]:
                cov_check[s] += 1

    best_result = initial_result
    best_count = len(best_result)

    # 2. Run Simulated Annealing
    sa_time_limit = min(_remaining() - 20.0, 60.0)
    if sa_time_limit > 5.0:
        # Run SA for the remaining time minus a buffer for MILP
        sa_result = local_search_sa(best_result, 42, sa_time_limit)
        if len(sa_result) < best_count:
            # Verify SA result
            cov_check_sa = np.zeros(n_ts, dtype=np.int32)
            for blk in sa_result:
                b_idx = block_to_idx[blk]
                for s_idx in block_ts[b_idx]:
                    cov_check_sa[s_idx] += 1
            if np.all(cov_check_sa > 0):
                best_result = sa_result
                best_count = len(best_result)

    # 3. Try MILP optimization with the best result as a hint
    if _remaining() > 15.0:
        try:
            # Objective: minimize sum(x_i)
            c = np.ones(n_blocks)
            
            # Build A using scipy.sparse for efficiency
            row_indices = []
            col_indices = []
            data = []
            
            for b_idx, ts_list in enumerate(block_ts):
                for s_idx in ts_list:
                    row_indices.append(s_idx)
                    col_indices.append(b_idx)
                    data.append(1)
            
            A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_ts, n_blocks))
            
            constraints = [
                LinearConstraint(A, 1, np.inf)
            ]
            
            bounds = Bounds(0, 1)
            integrality = np.ones(n_blocks)
            
            # Initial solution hint
            x0 = np.zeros(n_blocks)
            selected_set = set(tuple(blk) for blk in best_result)
            for b_idx, blk in enumerate(all_blocks):
                if tuple(blk) in selected_set:
                    x0[b_idx] = 1.0
            
            # Set time limit for MILP
            time_limit_s = _remaining() - 5.0
            if time_limit_s < 5:
                time_limit_s = 5
                
            # Solve
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={
                    "time_limit": time_limit_s,
                    "presolve": "default",
                    "mip_rel_gap": 0.0,
                    "mip_abs_gap": 0.0
                }
            )
            
            if res.status == 0 or res.status == 1: # 0: optimal, 1: feasible (stopped)
                if res.x is not None:
                    new_blocks = []
                    for i in range(n_blocks):
                        if res.x[i] > 0.5:
                            new_blocks.append(list(all_blocks[i]))
                    
                    if new_blocks:
                        # Verify validity
                        cov_check_milp = np.zeros(n_ts, dtype=np.int32)
                        for blk in new_blocks:
                            b_idx = block_to_idx[blk]
                            for s_idx in block_ts[b_idx]:
                                cov_check_milp[s_idx] += 1
                        
                        if np.all(cov_check_milp > 0):
                            if len(new_blocks) < best_count:
                                best_result = new_blocks
                                best_count = len(best_result)
                                
        except Exception as e:
            # If MILP fails, keep the previous best
            pass

    return best_result
