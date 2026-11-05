"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TIME_LIMIT = 110.0  # safety margin below 120s evaluator limit
SEED_LENGTH = 104
FULL_MASK = (1 << 512) - 1

# EVOLVE-BLOCK-START

def _neighbors_mask(v):
    """Return a 512-bit mask with bits set for all N neighbors of v."""
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    return m

def _greedy_search_bitmask(rng, time_budget, lookahead_depth=0):
    """
    Fast greedy snake construction using bit-parallel 512-bit masks
    for used and exclusion sets.
    """
    path = [0]
    used_mask = 1  # bit 0 set
    exclusion_mask = 0
    
    while True:
        cur = path[-1]
        # Compute neighbor mask for current vertex
        nm = _neighbors_mask(cur)
        # Valid candidates: neighbors not in used and not in exclusion
        valid_mask = nm & ~used_mask & ~exclusion_mask
        
        if valid_mask == 0:
            return path
        
        # Extract candidate indices from valid_mask
        candidates = []
        vm = valid_mask
        while vm:
            # Get lowest set bit
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            vm ^= lsb
        
        if len(candidates) == 1:
            v = candidates[0]
        elif lookahead_depth > 0:
            # Score candidates with lookahead
            best_score = -1
            best_cands = []
            for c in candidates:
                score = _lookahead_score_bitmask(c, path, used_mask, exclusion_mask, lookahead_depth)
                if score > best_score:
                    best_score = score
                    best_cands = [c]
                elif score == best_score:
                    best_cands.append(c)
            v = rng.choice(best_cands)
        else:
            v = rng.choice(candidates)
        
        path.append(v)
        used_mask |= (1 << v)
        
        # Update exclusion: old terminal's neighbors become excluded
        exclusion_mask |= nm
        
        # Time check every ~100 steps to reduce overhead
        if len(path) % 100 == 0 and time.time() > time_budget:
            return path

def _lookahead_score_bitmask(candidate, path, used_mask, exclusion_mask, depth):
    """
    Estimate how many further steps are possible after choosing candidate.
    Uses bitmask representation for speed.
    """
    if depth <= 0:
        return 1
    
    new_used_mask = used_mask | (1 << candidate)
    old_term = path[-1]
    new_exclusion_mask = exclusion_mask | _neighbors_mask(old_term)
    
    count = 0
    nm = _neighbors_mask(candidate)
    valid_mask = nm & ~new_used_mask & ~new_exclusion_mask
    
    vm = valid_mask
    while vm:
        lsb = vm & (-vm)
        idx = lsb.bit_length() - 1
        count += 1
        if depth > 1:
            count += _lookahead_score_bitmask(idx, path + [candidate], new_used_mask, new_exclusion_mask, depth - 1)
        vm ^= lsb
    
    return count

def _try_extend_bitmask(path, rng, time_budget):
    """
    Try to extend an existing path further using greedy lookahead with bitmasks.
    """
    used_mask = 0
    for p in path:
        used_mask |= (1 << p)
    
    # Build exclusion from non-terminal vertices
    exclusion_mask = 0
    for p in path[:-1]:
        exclusion_mask |= _neighbors_mask(p)
    
    while True:
        cur = path[-1]
        nm = _neighbors_mask(cur)
        valid_mask = nm & ~used_mask & ~exclusion_mask
        
        if valid_mask == 0:
            return path
        
        candidates = []
        vm = valid_mask
        while vm:
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            vm ^= lsb
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            best_score = -1
            best_cands = []
            for c in candidates:
                score = _lookahead_score_bitmask(c, path, used_mask, exclusion_mask, 2)
                if score > best_score:
                    best_score = score
                    best_cands = [c]
                elif score == best_score:
                    best_cands.append(c)
            v = rng.choice(best_cands)
        
        path.append(v)
        used_mask |= (1 << v)
        exclusion_mask |= nm
        
        if len(path) % 50 == 0 and time.time() > time_budget:
            return path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = None
    best_len = 0
    
    rng = random.Random(42)
    
    # First run: deterministic greedy to get baseline using bitmask
    path = [0]
    used_mask = 1
    exclusion_mask = 0
    while True:
        cur = path[-1]
        nm = _neighbors_mask(cur)
        valid_mask = nm & ~used_mask & ~exclusion_mask
        if valid_mask == 0:
            break
        lsb = valid_mask & (-valid_mask)
        nxt = lsb.bit_length() - 1
        path.append(nxt)
        used_mask |= (1 << nxt)
        exclusion_mask |= nm
    
    best_path = path
    best_len = len(path)
    
    # Strategy 1: Fast randomized greedy (no lookahead) - many restarts
    while time.time() < deadline - 25.0:
        p = _greedy_search_bitmask(rng, deadline - 24.0, lookahead_depth=0)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Strategy 2: Lookahead depth 1
    while time.time() < deadline - 12.0:
        p = _greedy_search_bitmask(rng, deadline - 11.0, lookahead_depth=1)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Strategy 3: Lookahead depth 2 (fewer but potentially better)
    while time.time() < deadline - 2.0:
        p = _greedy_search_bitmask(rng, deadline - 1.0, lookahead_depth=2)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Final: try to extend the best path found
    if best_path is not None and time.time() < deadline - 1.0:
        extended = _try_extend_bitmask(list(best_path), rng, deadline - 0.5)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    return best_path
# EVOLVE-BLOCK-END
