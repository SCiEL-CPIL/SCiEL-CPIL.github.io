import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = [tset_index[ts] for ts in itertools.combinations(b, t)]
        block_tsets.append(ts_indices)
    
    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)
            
    # Convert block_tsets to numpy arrays for faster ops
    block_tsets_np = [np.array(ts, dtype=np.int32) for ts in block_tsets]
    
    # Greedy Set Cover: Pick the tset with the fewest covering blocks (most constrained),
    # then among those blocks, pick the one with the highest marginal gain.
    def run_greedy(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
            
        uncovered = np.ones(n_tsub, dtype=bool)
        remaining = n_tsub
        selected = []
        
        # gain[b] = number of currently uncovered tsets covered by block b
        gain = np.array([len(ts) for ts in block_tsets], dtype=np.int32)
        
        # static_crit[t] = number of blocks that cover tset t
        static_crit = np.array([len(tset_to_blocks[ti]) for ti in range(n_tsub)], dtype=np.int32)
        
        while remaining > 0:
            if time_left() < 10.0:
                break
                
            # Find uncovered tsets
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Get static criticality for these
            crit_vals = static_crit[uncovered_indices]
            min_crit = np.min(crit_vals)
            
            # Find all uncovered tsets with this min criticality
            mask = crit_vals == min_crit
            candidates = uncovered_indices[mask]
            
            # Pick one randomly if multiple
            target_tset = int(candidates[rng.randint(0, len(candidates) - 1)])
            
            # Find blocks covering this tset
            cand_blocks = tset_to_blocks[target_tset]
            
            # Pick block with max gain
            best_block = -1
            best_gain_val = -1
            for bj in cand_blocks:
                if gain[bj] > best_gain_val:
                    best_gain_val = gain[bj]
                    best_block = bj
                    
            if best_block == -1 or best_gain_val <= 0:
                # Should not happen if cover is possible
                break
                
            selected.append(best_block)
            
            # Update gains
            bts = block_tsets_np[best_block]
            for ti in bts:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    # Decrement gain for all blocks that cover this tset
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1
            gain[best_block] = 0 # Ensure it's not picked again
            
        return selected

    def reduce_solution(sel):
        if not sel:
            return []
        
        if time_left() < 5.0:
            return sel
        
        # Count coverage
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            cov[block_tsets_np[bi]] += 1
            
        new_sel = []
        for bi in sel:
            bts = block_tsets_np[bi]
            # If any tset in this block is covered ONLY by this block, we must keep it
            if np.any(cov[bts] == 1):
                new_sel.append(bi)
            else:
                # Otherwise, it's redundant. Remove it and update counts
                cov[bts] -= 1
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel

    def local_search(selected, max_rounds=50):
        if not selected or time_left() < 10.0:
            return selected
            
        best_selected = list(selected)
        best_size = len(selected)
        
        for rnd in range(max_rounds):
            if time_left() < 10.0:
                break
                
            selected = reduce_solution(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
                
            if not selected:
                break
                
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in selected:
                cov[block_tsets_np[bi]] += 1
                
            improved = False
            
            # Try 1-swap: remove one, add one
            n_tries = min(len(selected), 20)
            for _ in range(n_tries):
                if time_left() < 10.0:
                    break
                    
                idx_remove = random.randint(0, len(selected) - 1)
                bi_remove = selected[idx_remove]
                
                bts_remove = block_tsets_np[bi_remove]
                newly_uncovered = bts_remove[cov[bts_remove] == 1]
                
                if len(newly_uncovered) == 0:
                    # Redundant
                    cov[bts_remove] -= 1
                    selected.pop(idx_remove)
                    improved = True
                    continue
                    
                if len(newly_uncovered) > 3:
                    # Too many constraints, unlikely to find a single block covering all
                    continue
                    
                # Find candidate blocks that cover all newly_uncovered tsets
                cand_set = set(tset_to_blocks[int(newly_uncovered[0])])
                for ti in newly_uncovered[1:]:
                    cand_set &= set(tset_to_blocks[int(ti)])
                    
                cand_set.discard(bi_remove)
                cand_set -= set(selected) # Don't add a block already in set
                
                if not cand_set:
                    continue
                    
                cand_list = list(cand_set)
                if len(cand_list) > 5:
                    # Sort by gain (static size) or just random
                    random.shuffle(cand_list)
                    cand_list = cand_list[:5]
                    
                bi_add = random.choice(cand_list)
                
                # Tentative swap
                cov[bts_remove] -= 1
                bts_add = block_tsets_np[bi_add]
                cov[bts_add] += 1
                selected[idx_remove] = bi_add
                
                # Reduce
                reduced = reduce_solution(selected)
                
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_selected = reduced
                    selected = reduced
                    improved = True
                    # Rebuild cov for next iteration of loop if we continue
                    cov = np.zeros(n_tsub, dtype=np.int32)
                    for bi in selected:
                        cov[block_tsets_np[bi]] += 1
                else:
                    # Revert
                    cov[bts_remove] += 1
                    cov[bts_add] -= 1
                    selected[idx_remove] = bi_remove
                    
            if not improved:
                break
                
        return best_selected

    # Main
    best_overall = None
    best_size_overall = float('inf')
    
    # Run greedy multiple times with different seeds
    n_restarts = 0
    while time_left() > 20.0 and n_restarts < 10:
        n_restarts += 1
        seed = n_restarts * 12345
        sel = run_greedy(seed=seed)
        if not sel:
            continue
        sel = reduce_solution(sel)
        sel = local_search(sel)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel
            
    if best_overall is None:
        # Fallback
        best_overall = run_greedy(seed=1)
        best_overall = reduce_solution(best_overall)
        
    return [list(all_blocks[bi]) for bi in best_overall]
