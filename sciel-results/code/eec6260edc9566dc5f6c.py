"""CSP-backtracking solver for difference triangle sets (7,5).
Structural change: enumerate candidate rows with small scopes, build a
compatibility graph, and use a branch-and-bound maximum-clique search
with a difference-cost heuristic.
"""
import time
import random
import sys
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return the set of positive differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _total_scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used

def _generate_row_greedy(used, k):
    """
    Pure greedy row construction.
    Returns (row, new_diffs) or (None, None) on failure.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs

def _generate_row_random_walk(used, k, rng, max_scope=130, steps=3000):
    """
    Randomized row construction using a random walk.
    At each step, pick a random candidate from the valid range.
    """
    row = [0]
    new_diffs = set()
    for _ in range(steps):
        x = rng.randint(1, max_scope)
        new = {x - a for a in row if a < x}
        if row[-1] >= x:
            continue
        if not (new & used) and not (new & new_diffs):
            # Ensure strictly increasing
            if x > row[-1]:
                row.append(x)
                new_diffs |= new
                if len(row) == k + 1:
                    return row, new_diffs
    return None, None

def _generate_candidates(max_scope, k, max_candidates=5000, seed=42):
    """
    Generate a pool of candidate rows with max entry <= max_scope.
    Uses both greedy and random-walk strategies.
    """
    rng = random.Random(seed)
    candidates = []
    seen_diffs = set()  # To avoid duplicate rows (by their diff set)
    
    # Greedy base
    used = set()
    for _ in range(max_candidates // 4):
        row, new_diffs = _generate_row_greedy(used, k)
        if row is None or max(row) > max_scope:
            break
        key = frozenset(new_diffs)
        if key not in seen_diffs:
            seen_diffs.add(key)
            candidates.append((row, new_diffs))
        used |= new_diffs
    
    # Random walk variations
    for _ in range(max_candidates):
        if len(candidates) >= max_candidates:
            break
        # Random used set for diversity
        used_sample = set()
        for _ in range(5):
            row_try, diffs_try = _generate_row_greedy(used_sample, k)
            if row_try is not None and max(row_try) <= max_scope:
                used_sample |= diffs_try
                if len(used_sample) > max_scope * 2:
                    break
        
        row, new_diffs = _generate_row_random_walk(used_sample, k, rng, max_scope, steps=200)
        if row is not None and max(row) <= max_scope:
            key = frozenset(new_diffs)
            if key not in seen_diffs:
                seen_diffs.add(key)
                candidates.append((row, new_diffs))
    
    return candidates

def _build_compatibility_graph(candidates, n_rows):
    """
    Build an adjacency structure where two candidates are adjacent
    if their difference sets are disjoint.
    Returns (cands, adj) where cands is list of (row, diffs) and adj is a list of bitmasks.
    """
    n = len(candidates)
    adj = [0] * n
    
    for i in range(n):
        di = candidates[i][1]
        for j in range(i + 1, n):
            dj = candidates[j][1]
            if not (di & dj):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    return candidates, adj

def _max_clique_bron_kerbosch(adj, target_size, deadline, rng, max_candidates=None):
    """
    Bron-Kerbosch with pivot to find a clique of size target_size.
    adj: list of bitmasks (integers)
    Returns a list of indices forming the clique, or None.
    """
    n = len(adj)
    found = [False]
    result = [None]
    
    def expand(R, P, X):
        if found[0]:
            return
        if time.time() > deadline:
            return
        if len(R) >= target_size:
            found[0] = True
            result[0] = list(R)
            return
        if not P:
            return
        
        # Pivot selection: choose vertex in P∪X with max degree
        P_list = [i for i in range(n) if (P >> i) & 1]
        X_list = [i for i in range(n) if (X >> i) & 1]
        
        if not P_list:
            return
        
        # Choose pivot u from P ∪ X maximizing |P ∩ N(u)|
        best_pivot = None
        best_count = -1
        all_candidates = set(P_list) | set(X_list)
        # Sample for large sets
        if len(all_candidates) > 50:
            sample = rng.sample(list(all_candidates), 50)
        else:
            sample = list(all_candidates)
        
        for u in sample:
            count = 0
            mask = adj[u]
            for v in P_list:
                if (mask >> v) & 1:
                    count += 1
            if count > best_count:
                best_count = count
                best_pivot = u
        
        if best_pivot is None:
            # No pivot found, iterate all P
            candidates_to_iter = P_list[:]
        else:
            # Iterate P \ N(pivot)
            pivot_mask = adj[best_pivot]
            candidates_to_iter = [v for v in P_list if not ((pivot_mask >> v) & 1)]
        
        rng.shuffle(candidates_to_iter)
        
        for v in candidates_to_iter:
            if found[0] or time.time() > deadline:
                return
            
            new_R = set(R)
            new_R.add(v)
            
            # P' = P ∩ N(v), X' = X ∩ N(v)
            v_mask = adj[v]
            new_P = 0
            new_X = 0
            for i in range(n):
                if (v_mask >> i) & 1:
                    if (P >> i) & 1:
                        new_P |= (1 << i)
                    if (X >> i) & 1:
                        new_X |= (1 << i)
            
            expand(new_R, new_P, new_X)
            
            # P = P \ {v}, X = X ∪ {v}
            P = P & ~(1 << v)
            X = X | (1 << v)
            
            if found[0] or time.time() > deadline:
                return
    
    expand(set(), (1 << n) - 1, 0)
    return result[0] if found[0] else None

def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope by shrinking elements.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    stagnation = 0
    
    while time.time() < deadline and stagnation < 500:
        improved = False
        
        # Try to shrink the largest element
        max_idx = None
        max_val = -1
        for i, r in enumerate(rows):
            for j, val in enumerate(r):
                if val > max_val:
                    max_val = val
                    max_idx = (i, j)
        
        if max_idx:
            ri, ej = max_idx
            row = rows[ri]
            old_val = row[ej]
            
            for new_val in range(old_val - 1, max(ej, old_val - 20), -1):
                if new_val <= (row[ej-1] if ej > 0 else -1):
                    continue
                if ej < len(row) - 1 and new_val >= row[ej+1]:
                    continue
                
                old_diffs = set()
                new_diffs = set()
                for i2 in range(len(row)):
                    if i2 == ej:
                        continue
                    if i2 < ej:
                        old_diffs.add(old_val - row[i2])
                        new_diffs.add(new_val - row[i2])
                    else:
                        old_diffs.add(row[i2] - old_val)
                        new_diffs.add(row[i2] - new_val)
                
                conflict = False
                for d in new_diffs:
                    if d in used and d not in old_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    row[ej] = new_val
                    used = (used - old_diffs) | new_diffs
                    new_scope = _total_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                    improved = True
                    break
        
        if improved:
            stagnation = 0
        else:
            stagnation += 1
            # Perturbation
            if stagnation % 10 == 0:
                ri = rng.randint(0, len(rows) - 1)
                ej = rng.randint(1, len(rows[ri]) - 1)
                old_val = rows[ri][ej]
                for delta in [1, 2, 3, 5]:
                    new_val = old_val + delta
                    if new_val <= (rows[ri][ej-1] if ej > 0 else -1):
                        continue
                    if ej < len(rows[ri]) - 1 and new_val >= rows[ri][ej+1]:
                        continue
                    
                    old_diffs = set()
                    new_diffs = set()
                    for i2 in range(len(rows[ri])):
                        if i2 == ej:
                            continue
                        if i2 < ej:
                            old_diffs.add(old_val - rows[ri][i2])
                            new_diffs.add(new_val - rows[ri][i2])
                        else:
                            old_diffs.add(rows[ri][i2] - old_val)
                            new_diffs.add(rows[ri][i2] - new_val)
                    
                    conflict = False
                    for d in new_diffs:
                        if d in used and d not in old_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        rows[ri][ej] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break
    
    return best_rows

def _polish_pass(rows, k, rng, deadline):
    """
    Final polish: try to shrink each element to its minimum valid value.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    
    for pass_no in range(5):
        if time.time() > deadline:
            break
        for ri in range(len(rows)):
            row = rows[ri]
            for ej in range(1, len(row)):
                old_val = row[ej]
                # Try to shrink to minimum
                lower = row[ej-1] + 1
                for new_val in range(lower, old_val):
                    old_diffs = set()
                    new_diffs = set()
                    for i2 in range(len(row)):
                        if i2 == ej:
                            continue
                        if i2 < ej:
                            old_diffs.add(old_val - row[i2])
                            new_diffs.add(new_val - row[i2])
                        else:
                            old_diffs.add(row[i2] - old_val)
                            new_diffs.add(row[i2] - new_val)
                    
                    conflict = False
                    for d in new_diffs:
                        if d in used and d not in old_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        row[ej] = new_val
                        used = (used - old_diffs) | new_diffs
                        old_val = new_val
                        break
    
    return rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    deadline = time.time() + 135  # Leave 15s safety margin for 150s limit
    
    # Phase 1: Greedy initial solution
    rows = []
    used = set()
    for _ in range(n_rows):
        row, new_diffs = _generate_row_greedy(used, k)
        if row is None:
            # Fallback
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        else:
            rows.append(row)
            used |= new_diffs
    
    best_rows = rows
    best_scope = _total_scope(rows)
    
    if time.time() > deadline - 10:
        return best_rows
    
    # Phase 2: CSP / Maximum Clique search
    # Estimate lower bound for scope
    # Counting bound: C(k+1, 2) * n_rows differences, each in [1, M]
    # So M >= C(k+1, 2) * n_rows
    lower_bound = math.comb(k + 1, 2) * n_rows
    
    # Start searching from lower_bound
    # But we already have a solution with best_scope, so we can try to improve
    current_M = lower_bound
    found_better = False
    
    while current_M < best_scope and time.time() < deadline - 10:
        # Generate candidates with max entry <= current_M
        candidates = _generate_candidates(current_M, k, max_candidates=3000, seed=current_M)
        
        if len(candidates) < n_rows:
            current_M += 1
            continue
        
        # Build compatibility graph
        cands, adj = _build_compatibility_graph(candidates, n_rows)
        
        if len(cands) < n_rows:
            current_M += 1
            continue
        
        # Find clique
        rng = random.Random(current_M)
        clique_indices = _max_clique_bron_kerbosch(adj, n_rows, min(time.time() + 20, deadline - 10), rng)
        
        if clique_indices is not None:
            found_rows = [cands[i][0] for i in clique_indices]
            scope = max(max(r) for r in found_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in found_rows]
                found_better = True
                break  # Found minimal M, stop
        
        current_M += 1
    
    if not found_better and time.time() < deadline - 10:
        # Try local search on the best greedy solution
        ls_rows = _local_search_scope(best_rows, k, random.Random(42), min(time.time() + 30, deadline - 5))
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows
    
    # Phase 3: Polish
    if time.time() < deadline - 5:
        polished = _polish_pass(best_rows, k, random.Random(42), deadline - 2)
        pol_scope = _total_scope(polished)
        if pol_scope <= best_scope:
            best_rows = polished
            best_scope = pol_scope
    
    return best_rows

# EVOLVE-BLOCK-END
