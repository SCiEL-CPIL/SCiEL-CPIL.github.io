import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0
    
    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)
        
    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)
    
    # 2. Greedy Set Cover
    uncovered = np.ones(n_tsub, dtype=bool)
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = len(block_tsets[bi])
        
    selected = []
    remaining = n_tsub
    
    while remaining > 0:
        if time.time() > deadline - 40.0:
            break
        bi = int(np.argmax(gain))
        if gain[bi] == 0:
            break
        selected.append(bi)
        gain[bi] = 0
        
        for ti in block_tsets[bi]:
            if uncovered[ti]:
                uncovered[ti] = False
                remaining -= 1
                for bj in tset_to_blocks[ti]:
                    gain[bj] -= 1
                    
    # 3. Reduction Pass - iterative
    def reduce_solution(sel):
        if not sel: return []
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
                
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel
        
    selected = reduce_solution(selected)
    
    # 4. GRASP-style local search
    cov = np.zeros(n_tsub, dtype=np.int32)
    for bi in selected:
        for ti in block_tsets[bi]:
            cov[ti] += 1
            
    best_selected = list(selected)
    best_size = len(selected)
    
    # GRASP parameters
    alpha = 0.5  # randomness parameter for candidate list
    max_iterations = 2000
    
    for it in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
        if not selected:
            break
        
        # 1-swap local search: try removing each block and see if we can maintain coverage
        # with fewer or same blocks
        
        # First, try to remove redundant blocks
        selected = reduce_solution(selected)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
        
        # Try swaps: remove block i, add block j that covers the uncovered tsets
        # Pick a random block to try removing
        idx_remove = random.randint(0, len(selected) - 1)
        bi_remove = selected[idx_remove]
        
        # Find tsets that would become uncovered
        newly_uncovered = []
        for ti in block_tsets[bi_remove]:
            if cov[ti] == 1:
                newly_uncovered.append(ti)
        
        if not newly_uncovered:
            # Block is redundant, just remove it
            selected.pop(idx_remove)
            for ti in block_tsets[bi_remove]:
                cov[ti] -= 1
            selected = reduce_solution(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
            continue
        
        # Find candidate blocks that cover at least one of the newly_uncovered tsets
        candidates = set()
        for ti in newly_uncovered:
            candidates.update(tset_to_blocks[ti])
        candidates.discard(bi_remove)
        
        if not candidates:
            continue
        
        # Score candidates by how many of the newly_uncovered they cover
        # Use GRASP: select from top alpha-fraction of candidates
        cand_list = list(candidates)
        
        # Compute coverage gain for each candidate
        gains = []
        for bj in cand_list:
            g = 0
            for ti in block_tsets[bj]:
                if ti in newly_uncovered or (ti in set(newly_uncovered)):
                    # Check if this tset is actually uncovered after removing bi_remove
                    # We need to check if cov[ti] would be 0 after removing bi_remove
                    # Since we haven't removed it yet, check if cov[ti] == 1 and ti is in newly_uncovered
                    if ti in newly_uncovered:
                        g += 1
            gains.append((g, bj))
        
        # Sort by gain descending
        gains.sort(key=lambda x: -x[0])
        
        # Take top alpha-fraction
        n_cands = len(gains)
        top_n = max(1, int(alpha * n_cands))
        top_cands = gains[:top_n]
        
        # Randomly select from top candidates
        g, bi_add = random.choice(top_cands)
        
        # Check if bi_add is already in selected
        if bi_add in selected:
            continue
        
        # Save state
        saved_selected = list(selected)
        saved_cov = cov.copy()
        
        # Perform swap
        selected[idx_remove] = bi_add
        for ti in block_tsets[bi_remove]:
            cov[ti] -= 1
        for ti in block_tsets[bi_add]:
            cov[ti] += 1
        
        # Check if all tsets are still covered
        valid = True
        for ti in block_tsets[bi_remove]:
            if cov[ti] == 0:
                valid = False
                break
        
        if not valid:
            # Restore
            selected = saved_selected
            cov = saved_cov
            continue
        
        # Reduce
        selected = reduce_solution(selected)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
    
    # Final reduction
    best_selected = reduce_solution(best_selected)
    
    return [list(all_blocks[bi]) for bi in best_selected]
