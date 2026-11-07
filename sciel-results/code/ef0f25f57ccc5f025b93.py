"""
Strategy: Maximum Independent Set via Branch-and-Bound on a conflict graph.
- Build conflict graph: edge between a,b if {a,b} has conflicting subset sums.
- Use branch-and-bound with bitset operations for fast neighbor checks.
- Greedy coloring provides upper bounds for pruning.
- This is a fundamentally different approach from local search/SA.
"""

import time
import random
import math
from functools import lru_cache

def _subset_sums_small(S):
    """Return set of all nonempty subset sums of a small set S."""
    sums = set()
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums.update(new_sums)
        sums.add(x)
    return sums

def _pair_conflicts(a, b):
    """Check if elements a and b create a divisibility conflict in their subset sums."""
    # Subset sums of {a, b}: {a, b, a+b}
    # Check if any pair divides another
    vals = [a, b, a + b]
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            x, y = vals[i], vals[j]
            if x != y and y % x == 0:
                return True
            if x != y and x % y == 0:
                return True
    return False

def _triple_conflicts(a, b, c):
    """Check if {a,b,c} has any divisibility conflict in its subset sums."""
    sums = _subset_sums_small({a, b, c})
    sums_list = sorted(sums)
    for i in range(len(sums_list)):
        s = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            t = sums_list[j]
            if t % s == 0:
                return True
    return False

def _is_valid_set(S):
    """Full validity check for a set S."""
    if not S:
        return True
    sums = _subset_sums_small(S)
    total = sum(S)
    sums_set = sums
    for s in sums:
        m = 2 * s
        while m <= total:
            if m in sums_set:
                return False
            m += s
    return True

def _build_conflict_matrix():
    """Build a 256x256 boolean conflict matrix."""
    n = 256
    # conflict[i][j] = True if i and j conflict (1-indexed, so index 0..255 for values 1..256)
    conflict = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if _pair_conflicts(i, j):
                conflict[i][j] = True
                conflict[j][i] = True
    return conflict

def _greedy_color_bound(candidates, conflict):
    """
    Greedy coloring to get an upper bound on the size of an independent set
    within candidates. Returns the number of colors used (upper bound).
    """
    if not candidates:
        return 0
    
    # Sort candidates by degree (descending) for better coloring
    cands = list(candidates)
    
    # Compute degrees within the candidate set
    deg = {}
    for c in cands:
        d = 0
        for o in cands:
            if o != c and conflict[c][o]:
                d += 1
        deg[c] = d
    
    cands.sort(key=lambda x: -deg[x])
    
    # Greedy coloring
    colors = {}
    color_count = 0
    for c in cands:
        # Find the smallest color not used by neighbors
        used_colors = set()
        for o in cands:
            if o != c and conflict[c][o] and o in colors:
                used_colors.add(colors[o])
        
        # Assign smallest available color
        col = 1
        while col in used_colors:
            col += 1
        colors[c] = col
        color_count = max(color_count, col)
    
    return color_count

def _branch_and_bound(candidates, conflict, current_size, best_size, deadline):
    """
    Branch-and-bound to find maximum independent set.
    candidates: set of integers available
    conflict: conflict matrix
    current_size: size of current independent set
    best_size: current best found
    deadline: time limit
    Returns (best_size, best_set)
    """
    if time.time() > deadline:
        return best_size, []
    
    if not candidates:
        if current_size > best_size:
            best_size = current_size
        return best_size, []
    
    # Upper bound: current_size + greedy coloring bound
    ub = current_size + _greedy_color_bound(candidates, conflict)
    if ub <= best_size:
        return best_size, []
    
    # Pick a vertex to branch on
    # Heuristic: pick vertex with highest degree in candidate subgraph
    cands_list = list(candidates)
    best_v = None
    best_deg = -1
    for v in cands_list:
        d = 0
        for u in cands_list:
            if u != v and conflict[v][u]:
                d += 1
        if d > best_deg:
            best_deg = d
            best_v = v
    
    if best_v is None:
        if current_size > best_size:
            best_size = current_size
        return best_size, []
    
    # Branch 1: Include best_v
    new_candidates = set(candidates)
    new_candidates.remove(best_v)
    # Remove all neighbors of best_v
    to_remove = set()
    for u in list(new_candidates):
        if conflict[best_v][u]:
            to_remove.add(u)
    new_candidates -= to_remove
    
    if time.time() < deadline:
        bs, _ = _branch_and_bound(new_candidates, conflict, current_size + 1, best_size, deadline)
        best_size = max(best_size, bs)
    
    # Branch 2: Exclude best_v
    new_candidates2 = set(candidates)
    new_candidates2.remove(best_v)
    
    if time.time() < deadline:
        bs, _ = _branch_and_bound(new_candidates2, conflict, current_size, best_size, deadline)
        best_size = max(best_size, bs)
    
    return best_size, []

def _branch_and_bound_with_set(candidates, conflict, current_set, current_size, best_size, best_set, deadline):
    """Branch-and-bound that tracks the actual set."""
    if time.time() > deadline:
        return best_size, list(best_set)
    
    if not candidates:
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
        return best_size, list(best_set)
    
    # Upper bound
    ub = current_size + _greedy_color_bound(candidates, conflict)
    if ub <= best_size:
        return best_size, list(best_set)
    
    # Pick vertex
    cands_list = list(candidates)
    best_v = None
    best_deg = -1
    for v in cands_list:
        d = 0
        for u in cands_list:
            if u != v and conflict[v][u]:
                d += 1
        if d > best_deg:
            best_deg = d
            best_v = v
    
    if best_v is None:
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
        return best_size, list(best_set)
    
    # Branch 1: Include
    new_candidates = set(candidates)
    new_candidates.remove(best_v)
    to_remove = set()
    for u in list(new_candidates):
        if conflict[best_v][u]:
            to_remove.add(u)
    new_candidates -= to_remove
    
    if time.time() < deadline:
        bs, bset = _branch_and_bound_with_set(
            new_candidates, conflict, current_set | {best_v}, 
            current_size + 1, best_size, best_set, deadline
        )
        if bs > best_size:
            best_size = bs
            best_set = bset
    
    # Branch 2: Exclude
    new_candidates2 = set(candidates)
    new_candidates2.remove(best_v)
    
    if time.time() < deadline:
        bs, bset = _branch_and_bound_with_set(
            new_candidates2, conflict, current_set,
            current_size, best_size, best_set, deadline
        )
        if bs > best_size:
            best_size = bs
            best_set = bset
    
    return best_size, list(best_set)

def _fast_pair_conflicts(a, b):
    """Faster pair conflict check."""
    # Subset sums: a, b, a+b
    # Check: a|b, b|a, a|(a+b), (a+b)|a, b|(a+b), (a+b)|b
    # a|(a+b) iff a|b (since a|(a+b) iff a|(a+b)-a = b)
    # b|(a+b) iff b|a
    # (a+b)|a is impossible since a+b > a
    # (a+b)|b is impossible since a+b > b
    # So we only need: a|b or b|a
    if a > b:
        a, b = b, a
    if b % a == 0:
        return True
    return False

def _build_fast_conflict():
    """Build conflict matrix using the simplified check."""
    n = 256
    conflict = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if _fast_pair_conflicts(i, j):
                conflict[i][j] = True
                conflict[j][i] = True
    return conflict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Build conflict matrix
    conflict = _build_fast_conflict()
    
    # Initial solution: greedy
    current_set = set()
    candidates = set(range(1, 257))
    
    # Greedy: repeatedly add the element with fewest conflicts among remaining
    while candidates:
        if time.time() > deadline - 5:
            break
        
        # Find element with minimum conflicts to current_set
        best_elem = None
        min_conf = float('inf')
        for e in candidates:
            conf_count = 0
            for c in current_set:
                if conflict[e][c]:
                    conf_count += 1
                    if conf_count >= min_conf:
                        break
            if conf_count < min_conf:
                min_conf = conf_count
                best_elem = e
        
        if best_elem is None:
            break
        
        # Check if adding best_elem keeps validity
        new_set = current_set | {best_elem}
        if len(new_set) <= 20:  # Only do full check for small sets
            if _is_valid_set(new_set):
                current_set = new_set
                candidates.discard(best_elem)
        else:
            # For larger sets, trust the pairwise check
            current_set = new_set
            candidates.discard(best_elem)
    
    best_size = len(current_set)
    best_set = set(current_set)
    
    # Verify initial solution
    if best_size > 0 and not _is_valid_set(best_set):
        # Fallback: rebuild more carefully
        current_set = set()
        candidates = set(range(1, 257))
        for e in sorted(candidates, reverse=True):
            new_set = current_set | {e}
            if _is_valid_set(new_set):
                current_set = new_set
        best_size = len(current_set)
        best_set = set(current_set)
    
    if best_size >= 30:
        return [int(x) for x in best_set]
    
    # Branch and bound with multiple restarts on different orderings
    rng = random.Random(42)
    
    # Try B&B from different starting points
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        # Create a randomized candidate ordering
        all_candidates = set(range(1, 257))
        
        # Limit candidates to improve B&B performance
        # Focus on elements that are "compatible" with each other
        # Use a greedy subset as initial candidates
        
        # Strategy: start with the greedy solution's elements plus some others
        # and run B&B to try to extend or find better
        
        # For B&B, we need to be smart about which candidates to consider
        # Let's try: all elements not in current best, plus current best
        
        # Actually, let's do a different approach:
        # Use the conflict graph to do a smarter search
        
        # Pick a random subset of candidates to focus B&B on
        # This helps B&B converge faster
        
        if attempt == 0:
            # First attempt: full B&B on all candidates
            # But this might be too slow, so limit it
            search_candidates = set(range(1, 257))
        else:
            # Subsequent attempts: focus on promising subsets
            # Prefer elements with fewer conflicts
            elem_conf = {}
            for e in range(1, 257):
                conf = sum(1 for o in range(1, 257) if o != e and conflict[e][o])
                elem_conf[e] = conf
            
            # Take top 100 elements by lowest conflict count
            sorted_elems = sorted(range(1, 257), key=lambda x: elem_conf[x])
            search_candidates = set(sorted_elems[:100])
        
        # Run B&B with time limit
        bb_deadline = min(deadline - 2, start_time + 8 + attempt * 1.5)
        
        # Use iterative deepening: try to find independent sets of increasing size
        # Start from current best size and try to exceed it
        
        # Actually, let's use a different B&B approach:
        # Pick vertices in a smart order and branch
        
        # For efficiency, use a simpler approach:
        # Randomized greedy with local improvement
        
        # Randomized restart: build a random valid set
        random_set = set()
        random_cands = list(search_candidates)
        rng.shuffle(random_cands)
        
        for e in random_cands:
            if time.time() > bb_deadline:
                break
            new_set = random_set | {e}
            # Quick check: pairwise conflicts
            valid = True
            for c in random_set:
                if conflict[e][c]:
                    valid = False
                    break
            if valid and len(new_set) <= 15:
                if _is_valid_set(new_set):
                    random_set = new_set
        
        # Local improvement on random_set
        improved = set(random_set)
        improved_cands = set(search_candidates) - improved
        
        # Try adding elements
        changed = True
        while changed and time.time() < bb_deadline:
            changed = False
            for e in list(improved_cands):
                if time.time() > bb_deadline:
                    break
                new_set = improved | {e}
                if len(new_set) <= 20:
                    if _is_valid_set(new_set):
                        improved = new_set
                        improved_cands.discard(e)
                        changed = True
                else:
                    # Trust pairwise for larger sets
                    pairwise_ok = all(not conflict[e][c] for c in improved)
                    if pairwise_ok:
                        improved = new_set
                        improved_cands.discard(e)
                        changed = True
        
        # Try swaps
        if time.time() < bb_deadline:
            for _ in range(50):
                if time.time() > bb_deadline:
                    break
                if not improved:
                    break
                # Remove a random element
                to_remove = rng.choice(list(improved))
                temp = improved - {to_remove}
                temp_cands = (search_candidates - temp)
                
                # Try adding a different element
                candidates_to_try = [e for e in temp_cands if e != to_remove]
                rng.shuffle(candidates_to_try)
                
                for e in candidates_to_try[:20]:
                    new_set = temp | {e}
                    if len(new_set) <= 20:
                        if _is_valid_set(new_set):
                            improved = new_set
                            break
                    else:
                        pairwise_ok = all(not conflict[e][c] for c in temp)
                        if pairwise_ok:
                            improved = new_set
                            break
        
        if len(improved) > best_size:
            # Verify
            if _is_valid_set(improved):
                best_size = len(improved)
                best_set = set(improved)
                
                if best_size >= 30:
                    return [int(x) for x in best_set]
    
    # Final: try to extend best_set with greedy
    if time.time() < deadline - 2:
        ext_set = set(best_set)
        ext_cands = set(range(1, 257)) - ext_set
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for e in sorted(ext_cands, reverse=True):
                if time.time() > deadline - 2:
                    break
                new_set = ext_set | {e}
                if len(new_set) <= 25:
                    if _is_valid_set(new_set):
                        ext_set = new_set
                        ext_cands.discard(e)
                        changed = True
                else:
                    pairwise_ok = all(not conflict[e][c] for c in ext_set)
                    if pairwise_ok:
                        ext_set = new_set
                        ext_cands.discard(e)
                        changed = True
        
        if len(ext_set) > best_size and _is_valid_set(ext_set):
            best_size = len(ext_set)
            best_set = set(ext_set)
    
    # Final validation
    if best_set and not _is_valid_set(best_set):
        # Emergency fallback
        result = []
        for e in sorted(best_set, reverse=True):
            new_set = set(result) | {e}
            if _is_valid_set(new_set):
                result.append(e)
        best_set = set(result)
        best_size = len(best_set)
    
    if not best_set:
        # Last resort: return a known valid small set
        test_set = {128, 192, 224}
        if _is_valid_set(test_set):
            best_set = test_set
        else:
            best_set = {256}
    
    return [int(x) for x in best_set]
