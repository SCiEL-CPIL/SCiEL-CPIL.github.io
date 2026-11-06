"""Combined program: Fast Warnsdorff Greedy with Limited Backtracking Refinement."""
import time
import random
import sys

N = 9
TIME_LIMIT = 118.0  # Safety margin below 120s evaluator limit
SEED_LENGTH = 104
TOTAL_VERTICES = 1 << N  # 512

# Precompute neighbors for all vertices in Q_9
# This avoids bit operations in the inner loop
NEIGHBORS = [[] for _ in range(TOTAL_VERTICES)]
for v in range(TOTAL_VERTICES):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# EVOLVE-BLOCK-START

def _get_valid_candidates(cur, used, exclusion):
    """Get list of valid next vertices from cur."""
    candidates = []
    for nb in NEIGHBORS[cur]:
        if nb not in used and nb not in exclusion:
            candidates.append(nb)
    return candidates

def _warnsdorff_score(v, used, exclusion):
    """
    Score a vertex v based on Warnsdorff's rule:
    The number of valid moves available from v.
    Lower score means 'tighter' vertex, which is preferred.
    """
    count = 0
    for nb in NEIGHBORS[v]:
        if nb not in used and nb not in exclusion:
            count += 1
    return count

def _greedy_warnsdorff(rng, time_budget, noise=0.0):
    """
    Run a greedy snake construction using Warnsdorff's heuristic.
    """
    path = [0]
    used = {0}
    exclusion = set()
    
    start_t = time.time()
    
    while True:
        cur = path[-1]
        candidates = _get_valid_candidates(cur, used, exclusion)
        
        if not candidates:
            break
            
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Score candidates
            scored = []
            for v in candidates:
                # Warnsdorff: prefer vertices with fewest future options
                s = _warnsdorff_score(v, used, exclusion)
                # Add noise to break ties and allow exploration
                if noise > 0:
                    s += rng.random() * noise
                scored.append((s, v))
            
            # Sort by score (ascending)
            scored.sort(key=lambda x: x[0])
            
            # Pick the best. If we want more diversity, we could pick from top 2-3.
            # But pure Warnsdorff is strong. Let's pick the best, but with a slight random chance to pick 2nd best if scores are close.
            if len(scored) > 1 and rng.random() < 0.1 and scored[0][0] == scored[1][0]:
                v = scored[1][1]
            else:
                v = scored[0][1]
        
        path.append(v)
        used.add(v)
        
        # Update exclusion: cur becomes non-terminal, so its neighbors are excluded
        for nb in NEIGHBORS[cur]:
            exclusion.add(nb)
            
        if time.time() > time_budget:
            break
            
    return path

def _backtrack_refine(path, rng, time_budget, max_depth=5):
    """
    Try to extend or improve the path by backtracking a few steps
    and trying alternative branches.
    """
    if len(path) < 2:
        return path
    
    best_path = list(path)
    best_len = len(path)
    
    # We will try backtracking from 1 to max_depth steps
    # For each backtrack depth, we try the alternative branches
    # To keep it efficient, we limit the number of branches tried
    
    for bt in range(1, max_depth + 1):
        if len(path) <= bt:
            break
        
        # State before the last bt moves
        prefix = path[:-bt]
        removed = path[-bt:]
        
        # Reconstruct state at prefix[-1]
        used = set(prefix)
        exclusion = set()
        for v in prefix[:-1]: # All except the last one in prefix
            for nb in NEIGHBORS[v]:
                exclusion.add(nb)
        
        cur = prefix[-1]
        
        # Get candidates at this point
        candidates = _get_valid_candidates(cur, used, exclusion)
        
        # We want to try candidates that are NOT the one originally taken (path[-bt])
        # Original next vertex was removed[0]
        orig_next = removed[0]
        
        # Score candidates
        scored = []
        for v in candidates:
            if v == orig_next:
                continue
            s = _warnsdorff_score(v, used, exclusion)
            scored.append((s, v))
            
        if not scored:
            continue
            
        scored.sort(key=lambda x: x[0])
        
        # Try the top few alternative candidates
        for _, v in scored[:3]: # Limit to top 3 alternatives
            if time.time() > time_budget:
                return best_path
                
            # Extend from v
            new_path = prefix + [v]
            new_used = set(new_path)
            new_exclusion = set(exclusion)
            # Add neighbors of cur to exclusion
            for nb in NEIGHBORS[cur]:
                new_exclusion.add(nb)
            
            # Continue greedily from here
            while True:
                c = new_path[-1]
                cands = _get_valid_candidates(c, new_used, new_exclusion)
                if not cands:
                    break
                if len(cands) == 1:
                    nv = cands[0]
                else:
                    sc = []
                    for nv in cands:
                        s = _warnsdorff_score(nv, new_used, new_exclusion)
                        sc.append((s, nv))
                    sc.sort(key=lambda x: x[0])
                    nv = sc[0][1]
                
                new_path.append(nv)
                new_used.add(nv)
                for nb in NEIGHBORS[c]:
                    new_exclusion.add(nb)
                
                if time.time() > time_budget:
                    break
            
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = new_path
                
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_path = [0]
    best_len = 1
    
    rng = random.Random(12345)
    
    # Phase 1: Fast Greedy Warnsdorff with multiple restarts
    # Each run is very fast, so we can do many.
    search_count = 0
    while time.time() < deadline - 10.0:
        # Vary noise slightly to explore different paths
        noise = rng.random() * 0.5
        p = _greedy_warnsdorff(rng, deadline - 8.0, noise=noise)
        search_count += 1
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
            # If we found a significantly better path, try to refine it
            if best_len > 150 and time.time() < deadline - 5.0:
                refined = _backtrack_refine(best_path, rng, deadline - 2.0, max_depth=3)
                if len(refined) > best_len:
                    best_len = len(refined)
                    best_path = refined
    
    # Phase 2: Use remaining time for more restarts or refinement
    remaining = deadline - time.time()
    if remaining > 2.0:
        # Try refining the current best path further
        if len(best_path) > 100:
            refined = _backtrack_refine(best_path, rng, deadline - 1.0, max_depth=5)
            if len(refined) > best_len:
                best_len = len(refined)
                best_path = refined
        else:
            # Do a few more greedy runs
            while time.time() < deadline - 1.0:
                p = _greedy_warnsdorff(rng, deadline - 0.5, noise=0.1)
                if len(p) > best_len:
                    best_len = len(p)
                    best_path = p
    
    return best_path

# EVOLVE-BLOCK-END
