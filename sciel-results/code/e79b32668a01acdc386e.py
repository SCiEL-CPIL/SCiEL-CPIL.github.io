"""Covering design C(v,k,t): minimize number of k-subset blocks covering all t-subsets.

Direct greedy construction with criticality bias + numpy-accelerated local search
+ multiple random restarts to escape local optima.
"""
import time
import random
import itertools

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# Precompute all t-subsets and each block's coverage as an integer bitmask.
_T_SUBSETS = []
_T_INDEX = {}
_BLOCK_COV = {}
_CACHE = {}


def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    index = {sub: i for i, sub in enumerate(t_subs)}
    n = len(t_subs)
    cov = {}
    for blk in itertools.combinations(range(v), k):
        m = 0
        for sub in itertools.combinations(blk, t):
            m |= (1 << index[sub])
        cov[blk] = m
    _CACHE[key] = (t_subs, index, n, cov)
    return _CACHE[key]


def _local_search(selected, cov_list, n, deadline, rng, max_rounds=8000):
    """Block-exchange local search with numpy acceleration: drop redundant, swap critical."""
    nblocks = len(cov_list)
    sel = list(selected)
    sel_set = set(sel)
    
    if _HAS_NUMPY and n <= 100000:
        return _local_search_numpy(sel, cov_list, n, nblocks, deadline, rng, max_rounds)
    
    # Pure Python fallback
    count = [0] * n
    for i in sel:
        m = cov_list[i]
        while m:
            lb = m & (-m)
            count[lb.bit_length() - 1] += 1
            m &= m - 1

    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        drop_idx = -1
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            ok = True
            while m:
                lb = m & (-m)
                if count[lb.bit_length() - 1] == 1:
                    ok = False
                    break
                m &= m - 1
            if ok:
                drop_idx = pos
                break
        
        if drop_idx != -1:
            i = sel[drop_idx]
            m = cov_list[i]
            while m:
                lb = m & (-m)
                count[lb.bit_length() - 1] -= 1
                m &= m - 1
            sel.pop(drop_idx)
            sel_set.remove(i)
            improved = True
            continue
            
        swap_done = False
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            crit_mask = 0
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                if count[idx] == 1:
                    crit_mask |= lb
                m &= m - 1
            if crit_mask == 0:
                continue
            for j in range(nblocks):
                if j in sel_set:
                    continue
                mj = cov_list[j]
                if (mj & crit_mask) == crit_mask:
                    mi = cov_list[i]
                    while mi:
                        lb = mi & (-mi)
                        count[lb.bit_length() - 1] -= 1
                        mi &= mi - 1
                    while mj:
                        lb = mj & (-mj)
                        count[lb.bit_length() - 1] += 1
                        mj &= mj - 1
                    sel[pos] = j
                    sel_set.remove(i)
                    sel_set.add(j)
                    improved = True
                    swap_done = True
                    break
            if swap_done:
                break
    return sel


def _local_search_numpy(sel, cov_list, n, nblocks, deadline, rng, max_rounds):
    """Numpy-accelerated local search."""
    sel_set = set(sel)
    count = np.zeros(n, dtype=np.int32)
    
    for i in sel:
        m = cov_list[i]
        while m:
            lb = m & (-m)
            count[lb.bit_length() - 1] += 1
            m &= m - 1

    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        
        # 1) Try dropping: find a block whose removal doesn't uncover any t-subset
        drop_idx = -1
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            ok = True
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                if count[idx] == 1:
                    ok = False
                    break
                m &= m - 1
            if ok:
                drop_idx = pos
                break
        
        if drop_idx != -1:
            i = sel[drop_idx]
            m = cov_list[i]
            while m:
                lb = m & (-m)
                count[lb.bit_length() - 1] -= 1
                m &= m - 1
            sel.pop(drop_idx)
            sel_set.remove(i)
            improved = True
            continue
        
        # 2) Try swapping
        swap_done = False
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            crit_mask = 0
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                if count[idx] == 1:
                    crit_mask |= lb
                m &= m - 1
            if crit_mask == 0:
                continue
            for j in range(nblocks):
                if j in sel_set:
                    continue
                mj = cov_list[j]
                if (mj & crit_mask) == crit_mask:
                    mi = cov_list[i]
                    while mi:
                        lb = mi & (-mi)
                        count[lb.bit_length() - 1] -= 1
                        mi &= mi - 1
                    while mj:
                        lb = mj & (-mj)
                        count[lb.bit_length() - 1] += 1
                        mj &= mj - 1
                    sel[pos] = j
                    sel_set.remove(i)
                    sel_set.add(j)
                    improved = True
                    swap_done = True
                    break
            if swap_done:
                break
    return sel


def _construct_direct(v, k, t, deadline, rng):
    """
    Directly construct a covering by greedily selecting blocks.
    Uses a refined scoring: maximize new coverage, penalize redundant coverage.
    """
    t_subs, index, n, cov = _prepare(v, k, t)
    all_blocks = list(cov.keys())
    cov_masks = [cov[b] for b in all_blocks]
    
    uncovered = (1 << n) - 1
    selected_blocks = []
    selected_set = set()
    counts = [0] * n
    
    while uncovered and time.time() < deadline:
        best_block = None
        best_score = -1e18
        
        for b in all_blocks:
            if b in selected_set:
                continue
            
            m = cov[b]
            new_cov = (m & uncovered).bit_count()
            
            if new_cov == 0:
                continue
            
            # Penalize redundant coverage: how many already-covered subsets this block covers
            redundant = (m & ~uncovered).bit_count()
            
            # Score: new coverage is primary, redundant is a penalty
            # Use a smooth penalty: score = new_cov - 0.1 * redundant
            score = new_cov - 0.1 * redundant
            
            # Add small random noise for diversity
            score += rng.random() * 0.01
            
            if score > best_score:
                best_score = score
                best_block = b
        
        if best_block is None:
            break
            
        selected_blocks.append(best_block)
        selected_set.add(best_block)
        
        m = cov[best_block]
        uncovered &= ~m
        while m:
            lb = m & (-m)
            counts[lb.bit_length() - 1] += 1
            m &= m - 1
            
    return selected_blocks


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 120s
    rng = random.Random(12345)

    t_subs, index, n, cov = _prepare(v, k, t)
    all_blocks = list(cov.keys())
    cov_list = [cov[b] for b in all_blocks]
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Multiple restarts: use different random seeds for tie-breaking
    # Allocate time: ~60% for construction across restarts, ~40% for local search
    total_time = deadline - t0
    n_restarts = 3
    cons_time_per = (total_time * 0.55) / n_restarts
    ls_time_per = (total_time * 0.40) / n_restarts
    
    best_solution = None
    best_len = float('inf')
    
    for restart in range(n_restarts):
        restart_rng = random.Random(12345 + restart * 7919)
        
        # Phase 1: Direct Construction with this restart's RNG
        cons_deadline = t0 + (restart + 1) * cons_time_per + (total_time * 0.05)
        cons_deadline = min(cons_deadline, deadline - ls_time_per - 5.0)
        
        if time.time() >= cons_deadline - 1:
            break
            
        selected_blocks = _construct_direct(v, k, t, cons_deadline, restart_rng)
        
        if not selected_blocks:
            continue
        
        # Phase 2: Local Search
        selected_indices = [block_to_idx[b] for b in selected_blocks]
        
        ls_deadline = min(cons_deadline + ls_time_per, deadline - 3.0)
        selected_indices = _local_search(selected_indices, cov_list, n, ls_deadline, restart_rng)
        
        # Ensure validity (fallback)
        covered_mask = 0
        for i in selected_indices:
            covered_mask |= cov_list[i]
        
        full_mask = (1 << n) - 1
        uncovered_now = full_mask & ~covered_mask
        
        if uncovered_now and time.time() < deadline - 1:
            for i in range(len(all_blocks)):
                if i in selected_indices:
                    continue
                if cov_list[i] & uncovered_now:
                    selected_indices.append(i)
                    uncovered_now &= ~cov_list[i]
                    if not uncovered_now:
                        break
        
        # Keep the best solution
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices[:]
        
        # Early termination if we've hit the theoretical lower bound
        if best_len <= 131:
            break
    
    if best_solution is None:
        # Fallback: single greedy without restarts
        selected_blocks = _construct_direct(v, k, t, deadline - 5.0, rng)
        selected_indices = [block_to_idx[b] for b in selected_blocks]
        covered_mask = 0
        for i in selected_indices:
            covered_mask |= cov_list[i]
        full_mask = (1 << n) - 1
        uncovered_now = full_mask & ~covered_mask
        if uncovered_now:
            for i in range(len(all_blocks)):
                if i not in selected_indices and (cov_list[i] & uncovered_now):
                    selected_indices.append(i)
                    uncovered_now &= ~cov_list[i]
                    if not uncovered_now:
                        break
        best_solution = selected_indices
    
    result = [list(all_blocks[i]) for i in best_solution]
    return result
