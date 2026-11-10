"""
Strategy: Deterministic Conflict-Driven Local Search.
Instead of random flips, identify the triple involved in the most violations 
(red K4s or blue K5s) and flip it. This is much more efficient than random 
hill climbing for structured problems like hypergraph Ramsey.
"""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _get_triples(n):
    return list(combinations(range(n), 3))

def _init_coloring(n, strategy='random'):
    triples = _get_triples(n)
    if strategy == 'random':
        return np.array([random.randint(0, 1) for _ in range(len(triples))], dtype=np.int8)
    elif strategy == 'modular':
        # (a+b+c) % 3 == 0 -> red(0), else blue(1)
        return np.array([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
    else:
        return np.array([0] * len(triples), dtype=np.int8)

def _analyze_conflicts(n, colors):
    """
    Returns:
    - conflict_score: array of length M (number of triples), count of violations involving that triple.
    - violations: list of violation tuples. Each tuple is ('R', quad) or ('B', quint).
    """
    triples = _get_triples(n)
    m = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    conflict_score = np.zeros(m, dtype=np.int32)
    violations = []
    
    # 1. Check Red K4
    # A red K4 is a set of 4 vertices {a,b,c,d} where all 4 triples are red (0).
    # If a red K4 exists, all 4 triples in it contribute to the conflict.
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t1 = triple_index[(a, b, c)]
        t2 = triple_index[(a, b, d)]
        t3 = triple_index[(a, c, d)]
        t4 = triple_index[(b, c, d)]
        
        if colors[t1] == 0 and colors[t2] == 0 and colors[t3] == 0 and colors[t4] == 0:
            violations.append(('R', quad))
            conflict_score[t1] += 1
            conflict_score[t2] += 1
            conflict_score[t3] += 1
            conflict_score[t4] += 1
            
    # 2. Check Blue K5
    # A blue K5 is a set of 5 vertices {a,b,c,d,e} where all 10 triples are blue (1).
    # If a blue K5 exists, all 10 triples in it contribute to the conflict.
    for quint in combinations(range(n), 5):
        all_blue = True
        indices = []
        for tri in combinations(quint, 3):
            idx = triple_index[tri]
            indices.append(idx)
            if colors[idx] != 1:
                all_blue = False
                break
        
        if all_blue:
            violations.append(('B', quint))
            for idx in indices:
                conflict_score[idx] += 1
                
    return conflict_score, violations

def _local_search_deterministic(n, colors, time_limit):
    """
    Deterministic local search:
    1. Analyze conflicts.
    2. If no conflicts, return success.
    3. Find the triple with the highest conflict score.
    4. Flip it.
    5. Repeat until time limit or success.
    """
    colors = colors.copy()
    start_time = time.time()
    
    # Initial analysis
    conflict_score, violations = _analyze_conflicts(n, colors)
    
    if not violations:
        return colors, True
    
    # To avoid oscillation, we might need a tie-breaking mechanism or random perturbation
    # if stuck, but let's try pure greedy first.
    # We track the last few moves to avoid immediate undoing if possible, 
    # though greedy usually works well for this specific structure.
    
    while time.time() < time_limit:
        if not violations:
            return colors, True
            
        # Find the triple with max conflict score
        max_score = np.max(conflict_score)
        if max_score == 0:
            # Should not happen if violations exist, but safety check
            break
            
        # Find all triples with max score
        candidates = np.where(conflict_score == max_score)[0]
        
        # If multiple, pick one. Randomness here helps escape symmetric traps.
        # However, deterministic pick (first) is also fine. Let's use random for robustness.
        flip_idx = candidates[random.randint(0, len(candidates) - 1)]
        
        # Flip
        colors[flip_idx] = 1 - colors[flip_idx]
        
        # Re-analyze. 
        # Optimization: Instead of full re-analysis, we could update incrementally,
        # but for n < 30, full analysis is fast enough (C(30,4) ~ 27k, C(30,5) ~ 142k).
        # 142k * 10 checks is ~1.4M ops. In Python, this might take ~0.1s.
        # So we can do ~100-500 iterations in 120s. This is sufficient.
        
        conflict_score, violations = _analyze_conflicts(n, colors)
        
        # If we are stuck in a 2-cycle (flip A, flip B, flip A...), 
        # the greedy approach might oscillate. 
        # A simple fix: if we flip the same triple twice in a row, 
        # we are oscillating. We can force a different move.
        # For now, simple greedy is a strong baseline.
        
    return colors, len(violations) == 0

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    best_n = 1
    best_colors = [0]
    
    # Try increasing n
    # We know n=9 works with modular. 
    # Let's start from a reasonable n and go up.
    # If we fail to find a valid coloring for n, we stop (since larger n is harder? 
    # Actually, if n works, n-1 might work by restriction, but finding n from scratch is hard).
    # Strategy: Try to find valid coloring for n. If successful, try n+1.
    
    # Start with n=10
    # If we can't find n=10 quickly, we might drop to n=9.
    
    # However, the goal is to MAXIMIZE n.
    # Let's try a few target n values in parallel or sequentially with time budgets.
    
    # Since finding a valid coloring is the bottleneck, let's try to push n as high as possible.
    # We'll iterate n from 10 to 25 (or higher if fast).
    
    for n in range(10, 26):
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this n
        # Larger n needs more time? Or same time?
        # Let's give each n a budget of 5 seconds for local search.
        # If we find a solution for n, we record it and move to n+1.
        # If we fail, we move to n+1? No, if we fail for n, we likely fail for n+1.
        # But random restarts might help.
        
        # Actually, it's better to try to solve n=20, then 19, etc?
        # No, the property is monotonic? 
        # If there is a valid coloring for n, is there for n-1?
        # Yes, we can just restrict the coloring to the first n-1 vertices.
        # So if we find a valid coloring for n, we have a valid one for all k < n.
        # So we want to find the MAXIMUM n.
        
        # Let's try to find a valid coloring for n.
        # We use multiple restarts.
        
        found = False
        best_colors_for_n = None
        
        # Time budget for this n: 5 seconds
        t_start = time.time()
        budget = 5.0
        
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Use modular init if n is small, random otherwise?
            # Modular is a good heuristic.
            if restart % 2 == 0:
                colors = _init_coloring(n, 'modular')
            else:
                colors = _init_coloring(n, 'random')
            
            # Local search
            # Give it the remaining budget for this restart
            t_limit = min(deadline - 5, time.time() + budget / 2)
            
            colors, success = _local_search_deterministic(n, colors, t_limit)
            
            if success:
                found = True
                best_colors_for_n = colors.tolist()
                break
        
        if found:
            best_n = n
            best_colors = best_colors_for_n
            # Continue to n+1 to see if we can go higher
        else:
            # If we failed for n, should we try n+1?
            # It's possible that n is a "hard" barrier but n+1 is easier? Unlikely.
            # But if n=10 fails, n=11 will likely fail too.
            # However, if n=10 succeeds, we try 11.
            # If n=10 fails, we might want to retry n=10 with more time?
            # For simplicity, if we fail, we stop trying larger n, 
            # but we might want to keep the best we found so far.
            # Since we iterate 10, 11, 12..., if 10 fails, we break?
            # No, maybe 10 is just hard with our init. 
            # Let's break if we fail, assuming monotonic difficulty.
            break

    # If we didn't find anything >= 10, fallback to known n=9
    if best_n < 9:
        n = 9
        triples = _get_triples(n)
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        best_n = 9

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
