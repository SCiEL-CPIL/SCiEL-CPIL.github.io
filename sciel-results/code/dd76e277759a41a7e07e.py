import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

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

    if v != 19 or k != 8 or t != 5:
        # Fallback: Standard Greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
            best_block = None
            best_gain = -1
            remaining = [x for x in range(v) if x not in target]
            need = k - t
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
            if time.time() > deadline:
                break
        return blocks

    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage structures
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(n_t)]
    
    for j, block in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered_indices)
        
    # Convert to numpy arrays for speed
    block_ts_arrays = [np.array(lst, dtype=np.int32) for lst in block_to_ts]

    def run_greedy(seed):
        random.seed(seed)
        gain = np.full(n_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(n_t, True, dtype=bool)
        is_selected = np.full(n_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
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
                if time.time() > deadline - 2.0:
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

    def local_search_2swap(blocks, max_swaps=20):
        """
        Perform 2-for-2 exchanges to escape local minima.
        """
        if not blocks:
            return blocks
            
        # Map block tuple to index in current solution for quick lookup
        # Actually, we just work with the list of blocks.
        
        # Precompute ts coverage for current blocks
        # cover_count[ts_idx] = number of blocks in 'blocks' covering ts_idx
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
                
        n_sol = len(blocks)
        swaps_made = 0
        
        # We iterate until no more swaps or max_swaps reached
        while swaps_made < max_swaps and time.time() < deadline - 10.0:
            swap_found = False
            # Try all pairs
            for i in range(n_sol):
                if swap_found: break
                for j in range(i + 1, n_sol):
                    if time.time() > deadline - 10.0:
                        return blocks
                    # Calculate unique t-subsets for blocks i and j
                    # Unique if cover_count[ts] == 1 (only covered by this block) 
                    # But wait, if we remove i and j, a ts is uncovered if cover_count[ts] <= 2 AND it is in i or j.
                    # Specifically, ts is in U_ij if (ts in i OR ts in j) AND cover_count[ts] <= 2.
                    # If cover_count[ts] == 1, it's in exactly one of them.
                    # If cover_count[ts] == 2, it's in both.
                    
                    u_ij = set()
                    for ts_idx in block_ts_indices[i]:
                        if cover_count[ts_idx] <= 2:
                            u_ij.add(ts_idx)
                    for ts_idx in block_ts_indices[j]:
                        if cover_count[ts_idx] <= 2:
                            u_ij.add(ts_idx)
                            
                    if not u_ij:
                        # Both are redundant, handled by 1-swap, but if we are here, maybe 1-swap missed it?
                        # 1-swap should have removed them. So this shouldn't happen often.
                        continue
                        
                    # We need to find two blocks C1, C2 such that u_ij subset of C1 U C2
                    # And C1, C2 are not i or j (ideally, or if they are, it's a no-op swap)
                    # To keep it simple, we look for ANY two blocks.
                    
                    # Get candidates: blocks that cover at least one ts in u_ij
                    candidates = set()
                    for ts_idx in u_ij:
                        candidates.update(ts_to_blocks[ts_idx])
                    
                    # Remove i and j from candidates? 
                    # If we swap i,j with i,j, it's a no-op. 
                    # If we swap i,j with i,k, it's effectively 1-swap on j.
                    # Let's allow any pair, but prefer new blocks.
                    
                    candidates_list = list(candidates)
                    # Sort candidates by how much of u_ij they cover
                    def score_cand(c_idx):
                        c_ts = set(block_to_ts[c_idx])
                        return len(c_ts & u_ij)
                    
                    candidates_list.sort(key=score_cand, reverse=True)
                    
                    # Try pairs from top candidates
                    # Limit search to top 20 candidates to keep it fast
                    top_cands = candidates_list[:20]
                    
                    found_pair = None
                    for idx1 in range(len(top_cands)):
                        c1 = top_cands[idx1]
                        if c1 == i or c1 == j:
                            # If we pick i or j, we are effectively doing a 1-swap or no-op.
                            # Let's skip for now to focus on true 2-swaps.
                            continue
                        c1_ts = set(block_to_ts[c1])
                        remaining = u_ij - c1_ts
                        if not remaining:
                            # c1 alone covers everything. We need a second block.
                            # We can pick any block, say the first candidate.
                            # But we must ensure the new pair is valid.
                            # If c1 covers all u_ij, we just need any c2.
                            # Let's pick the next best candidate.
                            for idx2 in range(len(top_cands)):
                                if idx2 == idx1: continue
                                c2 = top_cands[idx2]
                                if c2 == i or c2 == j: continue
                                found_pair = (c1, c2)
                                break
                            if found_pair: break
                        else:
                            # Find c2 that covers 'remaining'
                            for idx2 in range(idx1 + 1, len(top_cands)):
                                c2 = top_cands[idx2]
                                if c2 == i or c2 == j: continue
                                c2_ts = set(block_to_ts[c2])
                                if remaining.issubset(c2_ts):
                                    found_pair = (c1, c2)
                                    break
                            if found_pair: break
                    
                    if found_pair:
                        c1, c2 = found_pair
                        # Perform swap
                        # Remove i and j
                        # Add c1 and c2
                        
                        # Update cover_count
                        for ts_idx in block_ts_indices[i]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_ts_indices[j]:
                            cover_count[ts_idx] -= 1
                        
                        # Add new blocks
                        c1_ts = block_to_ts[c1]
                        c2_ts = block_to_ts[c2]
                        for ts_idx in c1_ts:
                            cover_count[ts_idx] += 1
                        for ts_idx in c2_ts:
                            cover_count[ts_idx] += 1
                        
                        # Update blocks list
                        # Replace i and j with c1 and c2
                        # Be careful with indices if we pop
                        # Let's just rebuild the list
                        new_blocks = []
                        for idx, b in enumerate(blocks):
                            if idx == i or idx == j:
                                continue
                            new_blocks.append(b)
                        new_blocks.append(all_blocks[c1])
                        new_blocks.append(all_blocks[c2])
                        
                        blocks = new_blocks
                        n_sol = len(blocks)
                        
                        # Rebuild block_ts_indices and cover_count is expensive.
                        # Instead of full rebuild, we can update incrementally, but it's complex.
                        # Given the small size, let's rebuild cover_count and block_ts_indices
                        # for the next iteration of the while loop?
                        # No, the while loop uses cover_count.
                        # Let's just break and re-run 1-swap and rebuild state.
                        
                        # Actually, it's safer to just return and let the main loop re-evaluate.
                        # But we want to do multiple swaps.
                        # Let's update the internal structures.
                        
                        # Rebuild block_ts_indices
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
                                
                        # Rebuild cover_count
                        cover_count = np.zeros(n_t, dtype=np.int32)
                        for ts_list in block_ts_indices:
                            for ts_idx in ts_list:
                                cover_count[ts_idx] += 1
                                
                        swaps_made += 1
                        swap_found = True
                        break # Restart pair iteration with new state
                    
            if not swap_found:
                break
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        sol = run_greedy(i)
        if sol is None:
            continue
            
        sol = optimize_redundancy(sol)
        sol = local_search_2swap(sol)
        sol = optimize_redundancy(sol) # Final cleanup
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = [list(b) for b in sol]
            
        if best_len <= 208:
            break

    if best_blocks is None:
        best_blocks = run_greedy(0)
        if best_blocks is None:
            return []
        best_blocks = optimize_redundancy(best_blocks)

    return best_blocks
