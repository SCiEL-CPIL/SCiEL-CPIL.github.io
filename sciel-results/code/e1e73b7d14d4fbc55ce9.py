"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
TIME_LIMIT = 115.0  # safety margin below 120s evaluator limit
SEED_LENGTH = 104
TOTAL_VERTICES = 1 << N  # 512

# Precompute bit masks
BIT_MASKS = [1 << b for b in range(N)]

# EVOLVE-BLOCK-START

def _greedy_search(rng, time_budget, use_lookahead=True, lookahead_weight=0.3):
    """Run one greedy snake construction with optional look-ahead scoring."""
    path = [0]
    used = np.zeros(TOTAL_VERTICES, dtype=np.uint8)
    used[0] = 1
    # exclusion: vertices at Hamming distance 1 from any non-terminal vertex
    exclusion = np.zeros(TOTAL_VERTICES, dtype=np.uint8)
    
    while True:
        cur = path[-1]
        # Find valid next vertices
        candidates = []
        for b in range(N):
            v = cur ^ BIT_MASKS[b]
            if not used[v] and not exclusion[v]:
                candidates.append(v)
        
        if not candidates:
            break
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            if use_lookahead and len(candidates) <= 9:
                # Score candidates: prefer those with more future extensions
                # When we move to v, cur becomes non-terminal, so neighbors of cur get excluded
                # Score = number of neighbors of v that are valid after the move
                scored = []
                for v in candidates:
                    score = 0
                    for b2 in range(N):
                        nv = v ^ BIT_MASKS[b2]
                        if nv == cur:
                            continue  # going back
                        if not used[nv]:
                            # Check if nv would be excluded after move
                            # nv is excluded if it's a neighbor of any non-terminal
                            # After move, cur becomes non-terminal, so neighbors of cur are excluded
                            # But nv is a neighbor of v (which is the new terminal), so it's not excluded by v
                            # It IS excluded if it's a neighbor of cur (the old terminal, now non-terminal)
                            # Check: is nv a neighbor of cur?
                            if (nv ^ cur).bit_count() == 1:
                                continue  # would be excluded
                            score += 1
                    scored.append((score, v))
                max_score = max(s for s, _ in scored)
                best_cands = [v for s, v in scored if s >= max_score - 1]
                # Mix: with some probability pick from top, else random
                if rng.random() < lookahead_weight:
                    v = rng.choice(best_cands)
                else:
                    v = rng.choice(candidates)
            else:
                v = rng.choice(candidates)
        
        path.append(v)
        used[v] = 1
        
        # Update exclusion: cur becomes non-terminal, add its neighbors
        for b in range(N):
            nb = cur ^ BIT_MASKS[b]
            exclusion[nb] = 1
        
        if time.time() > time_budget:
            break
    
    return path

def _backtrack_extend(path, used, exclusion, rng, time_budget, max_backtrack=3):
    """Try to extend a path by backtracking up to max_backtrack steps and trying alternatives."""
    if len(path) < 2:
        return path
    
    # Work with a copy
    p = list(path)
    u = used.copy()
    ex = exclusion.copy()
    
    # Try backtracking from the end
    for bt in range(1, max_backtrack + 1):
        if len(p) <= bt:
            break
        # Remove last bt vertices
        removed = p[-bt:]
        p = p[:-bt]
        for v in removed:
            u[v] = 0
        
        # Need to rebuild exclusion for the new path
        # exclusion should contain neighbors of all non-terminal vertices
        # Rebuild from scratch for safety
        new_ex = np.zeros(TOTAL_VERTICES, dtype=np.uint8)
        for i in range(len(p) - 1):  # all except last
            v = p[i]
            for b in range(N):
                new_ex[v ^ BIT_MASKS[b]] = 1
        ex = new_ex
        
        # Now try to extend from p[-1] with random choices
        cur_path = list(p)
        cur_u = u.copy()
        cur_ex = ex.copy()
        
        while True:
            cur = cur_path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ BIT_MASKS[b]
                if not cur_u[v] and not cur_ex[v]:
                    candidates.append(v)
            
            if not candidates:
                break
            
            v = rng.choice(candidates)
            cur_path.append(v)
            cur_u[v] = 1
            for b in range(N):
                cur_ex[cur ^ BIT_MASKS[b]] = 1
            
            if time.time() > time_budget:
                break
        
        if len(cur_path) > len(path):
            path = cur_path
            used = cur_u
            exclusion = cur_ex
    
    return path

def _build_exclusion(path):
    """Build exclusion array from a path."""
    ex = np.zeros(TOTAL_VERTICES, dtype=np.uint8)
    for i in range(len(path) - 1):
        v = path[i]
        for b in range(N):
            ex[v ^ BIT_MASKS[b]] = 1
    return ex

def _build_used(path):
    """Build used array from a path."""
    u = np.zeros(TOTAL_VERTICES, dtype=np.uint8)
    for v in path:
        u[v] = 1
    return u

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_path = [0]
    best_len = 1
    
    rng = random.Random(12345)
    
    # Phase 1: Run many randomized greedy searches with look-ahead
    # Each search is fast (path length ~100-200, each step O(N))
    search_count = 0
    while time.time() < deadline - 30.0:
        p = _greedy_search(rng, deadline - 28.0, use_lookahead=True, lookahead_weight=0.4)
        search_count += 1
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            # Occasionally try backtrack extension on the best path found
            if search_count % 5 == 0 and time.time() < deadline - 20.0:
                used = _build_used(best_path)
                exclusion = _build_exclusion(best_path)
                ext_path = _backtrack_extend(best_path, used, exclusion, rng, deadline - 18.0, max_backtrack=3)
                if len(ext_path) > best_len:
                    best_len = len(ext_path)
                    best_path = ext_path
    
    # Phase 2: Use remaining time for more searches, now with different parameters
    remaining = deadline - time.time()
    if remaining > 5.0:
        while time.time() < deadline - 2.0:
            # Vary the lookahead weight
            lw = rng.random() * 0.6
            p = _greedy_search(rng, deadline - 1.0, use_lookahead=True, lookahead_weight=lw)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
    
    return best_path

# EVOLVE-BLOCK-END
