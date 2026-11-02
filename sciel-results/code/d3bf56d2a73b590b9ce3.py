import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    t0 = time.time()
    # Safety margin: 140s out of 150s limit
    DEADLINE = t0 + 140.0

    # Helper for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Fallback for non-standard parameters
    if v != 19 or k != 8 or t != 5:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered and time.time() < DEADLINE:
            target = min(uncovered)
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            best_block = None
            best_gain = -1
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
        return blocks

    # --- Optimized Path for v=19, k=8, t=5 ---
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage structures
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(n_t)]
    ts_per_block = C(k, t)
    
    for j, block in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered_indices)

    def run_greedy(seed):
        random.seed(seed)
        gain = np.full(n_blocks, ts_per_block, dtype=np.int32)
        is_uncovered = np.full(n_t, True, dtype=bool)
        is_selected = np.full(n_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > DEADLINE - 5.0:
                return None
            
            # Find max gain among unselected blocks
            masked_gain = gain.copy()
            masked_gain[is_selected] = -1
            max_gain = np.max(masked_gain)
            
            if max_gain <= 0:
                if not is_uncovered.all():
                    break
                else:
                    break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(list(all_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=n_blocks)
                gain -= counts
                gain[is_selected] = -1000

        if is_uncovered.any():
            return None
        return sol

    def optimize_redundancy(blocks):
        if not blocks:
            return blocks
            
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_indices = []
        
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(t_subset_to_idx[ts])
                block_ts_indices.append(ts_list)
                
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > DEADLINE - 2.0:
                    return blocks
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break
        return blocks

    def local_search(blocks, time_limit=20):
        """
        1-exchange local search:
        Try to replace a block with another that covers its unique subsets.
        Or remove a block if it has no unique subsets.
        """
        if not blocks:
            return blocks
            
        # Build current state
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(t_subset_to_idx[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1

        sol_set = set(tuple(b) for b in blocks)
        
        # Precompute unique subsets for each block
        unique_ts_per_block = []
        for i in range(len(blocks)):
            uniq = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    uniq.append(ts_idx)
            unique_ts_per_block.append(uniq)

        improved = True
        while improved and time.time() < DEADLINE - time_limit:
            improved = False
            # Randomize order of blocks to check
            order = list(range(len(blocks)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > DEADLINE - time_limit:
                    break
                    
                if len(unique_ts_per_block[i]) == 0:
                    # Remove block i
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    unique_ts_per_block.pop(i)
                    sol_set.discard(tuple(blocks[i-1])) # Careful with index, actually just remove from set
                    # Rebuild sets is expensive, let's just mark as removed and handle carefully or rebuild at end
                    # Actually, for correctness of 'in sol_set' check later, we should maintain it.
                    # Since we pop, indices shift. Let's just return and re-run optimize_redundancy + LS from scratch if we want robustness, 
                    # but that's slow. 
                    # Better: keep track of indices carefully or just restart LS after a change.
                    # Given time constraints, restarting LS after a change is safer.
                    improved = True
                    break

                # Find a replacement block
                # The replacement must cover all unique_ts_per_block[i]
                candidates = set(ts_to_blocks[unique_ts_per_block[i][0]])
                for ts_idx in unique_ts_per_block[i][1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter out current block and blocks already in solution
                current_idx = block_to_idx.get(tuple(blocks[i]), -1)
                valid_candidates = []
                for c in candidates:
                    if c == current_idx:
                        continue
                    if tuple(all_blocks[c]) in sol_set:
                        continue
                    valid_candidates.append(c)
                
                if not valid_candidates:
                    continue
                
                # Pick a candidate (randomly)
                new_idx = random.choice(valid_candidates)
                new_block = all_blocks[new_idx]
                
                # Check if swap improves or maintains validity?
                # Swap is always valid if new_block covers all unique_ts of old_block.
                # Because:
                # - Any ts covered by old_block but not new_block must be in unique_ts (otherwise it was covered by someone else, so removing old_block wouldn't break it).
                # - Wait, if ts is covered by old_block and NOT new_block, and ts is NOT in unique_ts, then ts is covered by another block in solution. So removing old_block is fine for that ts.
                # - If ts is in unique_ts, new_block covers it.
                # - What about ts covered by new_block but not old_block? They become covered by one more block (or newly covered if they were uncovered? No, they were covered by old_block? No.
                #   Actually, we are replacing old_block with new_block.
                #   The set of covered ts changes.
                #   Any ts that was ONLY covered by old_block (unique_ts) must be covered by new_block.
                #   Any other ts covered by old_block is still covered by at least one other block.
                #   So validity is maintained.
                
                # Perform swap
                # Decrement cover count for old block's ts
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                # Increment cover count for new block's ts
                new_ts_list = block_to_ts[new_idx]
                for ts_idx in new_ts_list:
                    cover_count[ts_idx] += 1
                
                # Update structures
                sol_set.discard(tuple(blocks[i]))
                blocks[i] = list(new_block)
                sol_set.add(tuple(new_block))
                block_ts_indices[i] = new_ts_list
                
                # Update unique_ts for affected blocks?
                # This is complex. Simplest is to restart the LS loop.
                improved = True
                break
            
            if improved:
                # Rebuild unique_ts_per_block after change
                unique_ts_per_block = []
                for i in range(len(blocks)):
                    uniq = []
                    for ts_idx in block_ts_indices[i]:
                        if cover_count[ts_idx] == 1:
                            uniq.append(ts_idx)
                    unique_ts_per_block.append(uniq)

        return blocks

    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 0
    # Run multiple randomized greedy restarts
    while time.time() < DEADLINE - 30.0:
        num_restarts += 1
        sol = run_greedy(num_restarts)
        if sol is None:
            continue
        
        # Apply redundancy removal
        sol = optimize_redundancy(sol)
        
        # Apply local search
        sol = local_search(sol, time_limit=5)
        
        # Final redundancy removal after LS
        sol = optimize_redundancy(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = sol

    if best_blocks is None:
        # Emergency fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        random.seed(42)
        while uncovered and time.time() < DEADLINE - 1.0:
            b = random.sample(all_blocks, 1)[0]
            if b in blocks: continue
            blocks.append(list(b))
            for ts in itertools.combinations(b, t):
                if ts in uncovered:
                    uncovered.remove(ts)
        return blocks

    return best_blocks
