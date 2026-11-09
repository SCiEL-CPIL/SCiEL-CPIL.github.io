"""Rewritten solver: hypergraph independent set approach.
Vertices: 1..192. Hyperedge: {a,b,c,d} if a | (b+c+d).
Find max independent set (no hyperedge fully contained in S)."""

import time
import random
from collections import defaultdict

_DEADLINE = 85.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _check_validity(selected):
    """Return True if 'selected' satisfies the condition."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for a in selected:
        others = [x for x in selected if x != a]
        if len(others) < 3:
            continue
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                for k in range(j+1, len(others)):
                    if (others[i] + others[j] + others[k]) % a == 0:
                        return False
    return True

def _build_conflict_graph():
    """Build conflict structure: for each vertex a, store which triples of others
    cause a conflict (a divides their sum).
    Returns: conflicts[a] = set of frozensets of 3 elements that conflict with a.
    Also: triple_conflicts[sum_mod_info] for fast lookup.
    """
    # For each a in 1..192, find all triples {b,c,d} from 1..192 (b,c,d distinct, none = a)
    # such that (b+c+d) % a == 0.
    # Instead of storing all triples (too many), we'll compute on-the-fly.
    
    # Precompute: for each a, and for each pair (b,c) with b<c, b!=a, c!=a,
    # what values d (d != a, d != b, d != c) make (b+c+d) % a == 0?
    # d ≡ -(b+c) (mod a), so d = k*a - (b+c) for some k, must be in [1,192], distinct from a,b,c.
    
    # For greedy, we need: given current selected set S, and candidate x,
    # check if adding x creates any conflict.
    # Conflict types when adding x to S:
    # 1) For each a in S: does a divide (x + u + v) for some u,v in S\{a}, u≠v?
    # 2) Does x divide (u + v + w) for some u,v,w in S, all distinct?
    
    return None  # We'll use on-the-fly checks

def _is_safe_to_add(selected, sel_set, x):
    """Check if adding x to valid set keeps it valid.
    selected: list, sel_set: set for O(1) lookup, x: candidate.
    """
    n = len(selected)
    
    # Check 1: For each a in selected, does a | (x + u + v) for distinct u,v in selected\{a}?
    # We need u, v in selected, u ≠ v, u ≠ a, v ≠ a, and (x + u + v) % a == 0.
    # Equivalently: u + v ≡ -x (mod a).
    for i, a in enumerate(selected):
        # Find pairs (u,v) in selected, u≠v, both ≠ a, such that (u+v) ≡ -x (mod a)
        target = (-x) % a
        # Collect elements in selected excluding a
        others = []
        for j in range(n):
            if j != i:
                others.append(selected[j])
        if len(others) < 2:
            continue
        # Check all pairs in others
        for pi in range(len(others)):
            for pj in range(pi+1, len(others)):
                if (others[pi] + others[pj]) % a == target:
                    return False
    
    # Check 2: Does x divide (u + v + w) for distinct u,v,w in selected?
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        return False
    
    return True

def _is_safe_to_remove_then_add(selected, sel_set, remove_idx, x):
    """Check if removing selected[remove_idx] and adding x keeps validity."""
    new_sel = [selected[i] for i in range(len(selected)) if i != remove_idx]
    new_set = sel_set - {selected[remove_idx]}
    return _is_safe_to_add(new_sel, new_set, x)

def _degree_of_vertex(x, selected, sel_set):
    """Compute 'degree' of x: number of conflicts x would have if added to selected.
    Lower degree = safer to add. Used for greedy ordering.
    """
    n = len(selected)
    count = 0
    
    # Count conflicts of type 1: a in selected, a | (x + u + v)
    for i, a in enumerate(selected):
        target = (-x) % a
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 2:
            continue
        for pi in range(len(others)):
            for pj in range(pi+1, len(others)):
                if (others[pi] + others[pj]) % a == target:
                    count += 1
    
    # Count conflicts of type 2: x | (u + v + w) for u,v,w in selected
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        count += 1
    
    return count

def _greedy_degree_based():
    """Greedy: at each step, among remaining candidates, pick the one with
    minimum degree (fewest conflicts with current set). Ties broken randomly.
    """
    selected = []
    sel_set = set()
    remaining = set(range(1, 193))
    
    while remaining:
        if _time_left() < 2.0:
            break
        if len(selected) >= 48:
            break
        
        # For each candidate, compute degree
        best_x = None
        best_deg = float('inf')
        best_cands = []
        
        for x in remaining:
            deg = _degree_of_vertex(x, selected, sel_set)
            if deg < best_deg:
                best_deg = deg
                best_cands = [x]
            elif deg == best_deg:
                best_cands.append(x)
        
        # Pick randomly among best candidates
        x = random.choice(best_cands)
        
        # Verify it's actually safe (degree 0 means safe)
        if best_deg == 0:
            selected.append(x)
            sel_set.add(x)
            remaining.discard(x)
        else:
            # No safe candidate; try to pick the one with minimum degree anyway?
            # No, we can't add if there's a conflict. Stop.
            break
    
    return selected

def _local_search_swap(selected, max_iter=100):
    """Local search: try swapping one element out for another to enable more additions.
    Also try pure additions.
    """
    selected = list(selected)
    sel_set = set(selected)
    
    for _ in range(max_iter):
        if _time_left() < 3.0:
            break
        
        improved = False
        
        # First try pure additions
        for x in range(1, 193):
            if x in sel_set:
                continue
            if _is_safe_to_add(selected, sel_set, x):
                selected.append(x)
                sel_set.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add another
        for i in range(len(selected)):
            if _time_left() < 3.0:
                break
            removed = selected[i]
            new_sel = [selected[j] for j in range(len(selected)) if j != i]
            new_set = sel_set - {removed}
            
            for x in range(1, 193):
                if x in new_set:
                    continue
                if _is_safe_to_add(new_sel, new_set, x):
                    # Check if this swap is beneficial: we removed 'removed' and added 'x'
                    # Size stays same, but maybe enables future additions
                    # Only do swap if x > removed or if it enables additions
                    selected = new_sel + [x]
                    sel_set = new_set | {x}
                    improved = True
                    break
            
            if improved:
                break
    
    return selected

def construct():
    global _START
    _START = time.time()
    
    best = []
    best_size = 0
    
    # Strategy 1: Degree-based greedy
    random.seed(42)
    for restart in range(50):
        if _time_left() < 5.0:
            break
        
        cand = _greedy_degree_based()
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
        
        # Local search
        if _time_left() > 5.0 and len(cand) >= best_size - 2:
            improved = _local_search_swap(cand, max_iter=50)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
    
    # Strategy 2: Simple greedy largest first (baseline)
    if _time_left() > 5.0:
        selected = []
        sel_set = set()
        for x in range(192, 0, -1):
            if _time_left() < 3.0:
                break
            if _is_safe_to_add(selected, sel_set, x):
                selected.append(x)
                sel_set.add(x)
        if len(selected) > best_size:
            best = selected
            best_size = len(selected)
        
        if _time_left() > 5.0:
            improved = _local_search_swap(selected, max_iter=50)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
    
    # Strategy 3: Simple greedy smallest first
    if _time_left() > 5.0:
        selected = []
        sel_set = set()
        for x in range(1, 193):
            if _time_left() < 3.0:
                break
            if _is_safe_to_add(selected, sel_set, x):
                selected.append(x)
                sel_set.add(x)
        if len(selected) > best_size:
            best = selected
            best_size = len(selected)
        
        if _time_left() > 5.0:
            improved = _local_search_swap(selected, max_iter=50)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
    
    # Final local search on best
    if _time_left() > 5.0:
        improved = _local_search_swap(best, max_iter=100)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    return best

# Quick self-test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Valid: {_check_validity(res)}")
    print(res)
