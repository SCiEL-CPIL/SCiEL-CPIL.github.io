import time
import itertools
import random
import sys

_DEADLINE = 115.0
_START = None

def _elapsed():
    return time.time() - _START

def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    t_lists = []
    coverage_bits = []
    t_to_blocks = [[] for _ in range(n_t)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        ts_list = []
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            mask |= (1 << idx)
            ts_list.append(idx)
            t_to_blocks[idx].append(i)
        t_lists.append(ts_list)
        coverage_bits.append(mask)
        
    return all_t, t_index, k_subsets, t_lists, coverage_bits, t_to_blocks

def _core_periphery_construct(v, k, t, coverage_bits, t_lists, t_to_blocks, rng):
    """
    Core-Periphery decomposition:
    1. Build a 'core' of blocks with minimal pairwise t-overlap (packing phase)
    2. Repair residual uncovered t-subsets with targeted blocks (periphery phase)
    """
    n_t = sum(1 for _ in itertools.combinations(range(v), t))
    uncovered = (1 << n_t) - 1
    
    # Phase 1: Core construction - greedy with overlap penalty
    # We want blocks that cover new t-subsets but minimize overlap with existing core
    core = []
    overlap_count = [0] * n_t  # how many times each t-subset is covered by core
    
    # Randomize order for diversity
    all_indices = list(range(len(coverage_bits)))
    rng.shuffle(all_indices)
    
    core_time_budget = 30  # seconds for core phase
    core_start = time.time()
    
    while uncovered and (time.time() - core_start) < core_time_budget and _elapsed() < _DEADLINE - 30:
        best_i = -1
        best_score = -float('inf')
        
        # Sample a subset of candidates to keep this fast
        # Use stratified sampling: some random, some from uncovered-heavy regions
        sample_size = min(2000, len(all_indices))
        candidates = rng.sample(all_indices, sample_size)
        
        for i in candidates:
            if i in core:
                continue
            
            mask = coverage_bits[i]
            new_covered = (mask & uncovered).bit_count()
            
            if new_covered == 0:
                continue
            
            # Overlap penalty: count how many t-subsets in this block are already covered by core
            overlap = 0
            for t_idx in t_lists[i]:
                if overlap_count[t_idx] > 0:
                    overlap += 1
            
            # Score: maximize new coverage, minimize overlap
            # Weight new coverage heavily, penalize overlap
            score = new_covered * 10 - overlap
            
            if score > best_score:
                best_score = score
                best_i = i
        
        if best_i < 0:
            break
            
        core.append(best_i)
        for t_idx in t_lists[best_i]:
            overlap_count[t_idx] += 1
        uncovered &= ~coverage_bits[best_i]
        
        # Early stop if coverage is high
        if uncovered.bit_count() < 50:
            break
    
    # Phase 2: Periphery repair - targeted greedy for remaining gaps
    periphery = []
    while uncovered and _elapsed() < _DEADLINE - 15:
        best_i = -1
        best_gain = 0
        
        # For repair, focus on blocks that cover the most uncovered t-subsets
        # Sample candidates weighted by potential gain
        for i in range(len(coverage_bits)):
            if i in core or i in periphery:
                continue
            
            mask = coverage_bits[i]
            gain = (mask & uncovered).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_i = i
        
        if best_i < 0 or best_gain == 0:
            break
            
        periphery.append(best_i)
        uncovered &= ~coverage_bits[best_i]
        
        if uncovered == 0:
            break
    
    return core + periphery

def _greedy_construct(v, k, t, coverage_bits, rng):
    n_t = sum(1 for _ in itertools.combinations(range(v), t))
    
    uncovered = (1 << n_t) - 1
    blocks = []
    
    while uncovered and _elapsed() < _DEADLINE - 5:
        best_i = -1
        best_gain = 0
        
        for i, mask in enumerate(coverage_bits):
            g = (mask & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_i = i
        
        if best_i < 0:
            break
            
        blocks.append(best_i)
        uncovered &= ~coverage_bits[best_i]
        
    return blocks

def _local_search(blocks, t_lists, t_to_blocks, n_t, time_budget):
    """
    Local search using count array and swap-to-enable-removal.
    """
    blocks = list(blocks)
    
    count = [0] * n_t
    for b in blocks:
        for t_idx in t_lists[b]:
            count[t_idx] += 1
            
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal Pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            removable = True
            for t_idx in t_lists[b]:
                if count[t_idx] == 1:
                    removable = False
                    break
            if removable:
                for t_idx in t_lists[b]:
                    count[t_idx] -= 1
                blocks.pop(i)
                improved = True
            else:
                i += 1
                
        # --- Swap Pass ---
        for i in range(len(blocks)):
            if _elapsed() >= time_budget:
                break
                
            b = blocks[i]
            critical_ts = [t_idx for t_idx in t_lists[b] if count[t_idx] == 1]
            
            if not critical_ts:
                continue
                
            critical_ts.sort(key=lambda x: len(t_to_blocks[x]))
            
            candidates = t_to_blocks[critical_ts[0]]
            
            found_swap = False
            for new_b in candidates:
                if new_b == b:
                    continue
                    
                covers_all = True
                new_t_set = set(t_lists[new_b])
                for t_idx in critical_ts:
                    if t_idx not in new_t_set:
                        covers_all = False
                        break
                        
                if not covers_all:
                    continue
                    
                for t_idx in t_lists[b]:
                    count[t_idx] -= 1
                for t_idx in t_lists[new_b]:
                    count[t_idx] += 1
                    
                blocks[i] = new_b
                found_swap = True
                improved = True
                break
            
            if found_swap:
                break
    
    return blocks

def construct(v, k, t):
    global _START
    _START = time.time()
    
    all_t, t_index, k_subsets, t_lists, coverage_bits, t_to_blocks = _precompute(v, k, t)
    n_t = len(all_t)
    
    best_blocks = None
    best_size = float('inf')
    
    # Multiple restarts with both strategies
    num_restarts = 8
    
    for r in range(num_restarts):
        if _elapsed() > _DEADLINE - 10:
            break
            
        rng = random.Random(12345 + r)
        
        # Alternate between core-periphery and standard greedy
        if r % 2 == 0:
            blocks = _core_periphery_construct(v, k, t, coverage_bits, t_lists, t_to_blocks, rng)
        else:
            blocks = _greedy_construct(v, k, t, coverage_bits, rng)
        
        ls_budget = _DEADLINE - 2
        blocks = _local_search(blocks, t_lists, t_to_blocks, n_t, ls_budget)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
            
            if best_size <= 131:
                break
                
    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy_construct(v, k, t, coverage_bits, rng)
        
    result = [list(k_subsets[b]) for b in best_blocks]
    
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
            
    return dedup
