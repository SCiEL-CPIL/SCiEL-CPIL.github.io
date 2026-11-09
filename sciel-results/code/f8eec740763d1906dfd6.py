"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    """
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _build_triple_index(triples, max_n):
    """
    Build:
      - elem_to_triples: element -> list of triple indices
      - For each triple, store its elements for quick access
    """
    elem_to_triples = {i: [] for i in range(1, max_n + 1)}
    for ti, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(ti)
    return elem_to_triples

def _build_conflict_graph(triples, max_n):
    """
    Build conflict graph: two elements conflict if they appear together in some bad triple.
    """
    adj = {i: set() for i in range(1, max_n + 1)}
    for t in triples:
        a, b, c = t
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    
    return adj

def _count_conflicts_fast(selected, triples):
    """
    Count bad triples fully contained in selected.
    """
    sel_set = selected
    count = 0
    for t in triples:
        if t[0] in sel_set and t[1] in sel_set and t[2] in sel_set:
            count += 1
    return count

def _greedy_max_independent_set(adj, max_n, seed=0):
    """
    Greedy: repeatedly pick the element with minimum degree (conflicts),
    add to independent set, remove it and its neighbors.
    """
    rng = random.Random(seed)
    remaining = set(range(1, max_n + 1))
    selected = set()
    
    # Work with a mutable degree count
    deg = {x: len(adj[x] & remaining) for x in remaining}
    
    while remaining:
        # Pick element with min degree, break ties randomly
        min_deg = min(deg[x] for x in remaining)
        candidates = [x for x in remaining if deg[x] == min_deg]
        x = rng.choice(candidates)
        
        selected.add(x)
        remaining.discard(x)
        
        # Collect all neighbors to remove
        neighbors_to_remove = set()
        for y in adj[x]:
            if y in remaining:
                neighbors_to_remove.add(y)
        
        # Remove neighbors
        for y in neighbors_to_remove:
            remaining.discard(y)
        
        # Update degrees: remaining elements lost x and possibly some neighbors
        for z in remaining:
            lost = 0
            if x in adj[z]:
                lost += 1
            for y in neighbors_to_remove:
                if y in adj[z]:
                    lost += 1
            deg[z] -= lost
    
    return selected

def _local_search(selected, triples, adj, max_n, rng, time_limit):
    """
    Local search with incremental conflict tracking.
    State: selected set (valid independent set).
    Moves: add, swap.
    """
    start = time.time()
    sel = set(selected)
    best_sel = set(sel)
    best_size = len(sel)
    
    # Precompute conflict pairs for each element
    x_conflict_pairs = {}
    for t in triples:
        a, b, c = t
        x_conflict_pairs.setdefault(a, []).append((b, c))
        x_conflict_pairs.setdefault(b, []).append((a, c))
        x_conflict_pairs.setdefault(c, []).append((a, b))
    
    def is_addable(x, S):
        """Check if adding x to S keeps it valid."""
        for (a, b) in x_conflict_pairs.get(x, []):
            if a in S and b in S:
                return False
        return True
    
    # Phase 1: Try to add elements
    excluded = set(range(1, max_n + 1)) - sel
    improved = True
    while improved:
        improved = False
        for x in sorted(excluded, key=lambda y: len(x_conflict_pairs.get(y, []))):
            if is_addable(x, sel):
                sel.add(x)
                excluded.discard(x)
                improved = True
                if len(sel) > best_size:
                    best_size = len(sel)
                    best_sel = set(sel)
        if time.time() - start > time_limit:
            break
    
    # Phase 2: Swap-based local search with incremental conflict check
    T = 3.0
    cooling = 0.999
    step = 0
    while T > 0.01 and time.time() - start < time_limit and step < 100000:
        step += 1
        x = rng.choice(list(sel))
        y = rng.choice(list(excluded))
        
        # Check if swap is valid: remove x, add y
        # After removing x, check if y is addable to sel - {x}
        sel_without_x = sel - {x}
        if is_addable(y, sel_without_x):
            sel = sel_without_x
            sel.add(y)
            excluded.discard(y)
            excluded.add(x)
            if len(sel) > best_size:
                best_size = len(sel)
                best_sel = set(sel)
        
        T *= cooling
    
    return best_sel

def _sa_with_incremental_conflicts(selected, triples, max_n, rng, time_limit):
    """
    SA with incremental conflict tracking.
    Work with selected set directly. Minimize conflicts, then maximize size.
    """
    start = time.time()
    all_elems = set(range(1, max_n + 1))
    sel = set(selected)
    
    # Precompute: for each triple, track how many of its elements are in sel
    # triple_active_count[ti] = number of elements of triple ti that are in sel
    triple_active = [0] * len(triples)
    for ti, t in enumerate(triples):
        cnt = 0
        for x in t:
            if x in sel:
                cnt += 1
        triple_active[ti] = cnt
    
    current_conflicts = sum(1 for c in triple_active if c == 3)
    
    # Precompute element -> list of triple indices
    elem_to_triples = {i: [] for i in range(1, max_n + 1)}
    for ti, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(ti)
    
    best_sel = set(sel)
    best_conflicts = current_conflicts
    best_size = len(sel)
    
    T = 5.0
    cooling = 0.9995
    step = 0
    while T > 0.01 and time.time() - start < time_limit and step < 200000:
        step += 1
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add an element not in sel
            candidates = list(all_elems - sel)
            if not candidates:
                continue
            x = rng.choice(candidates)
            # Update triple_active for triples containing x
            affected = elem_to_triples[x]
            for ti in affected:
                triple_active[ti] += 1
            new_conflicts = sum(1 for c in triple_active if c == 3)
            
            delta = (new_conflicts - current_conflicts) * 100 - 1  # -1 for size gain
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                sel.add(x)
                current_conflicts = new_conflicts
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(sel) > best_size):
                    best_conflicts = current_conflicts
                    best_size = len(sel)
                    best_sel = set(sel)
            else:
                for ti in affected:
                    triple_active[ti] -= 1
        elif move_type < 0.7:
            # Remove an element from sel
            if not sel:
                continue
            x = rng.choice(list(sel))
            affected = elem_to_triples[x]
            for ti in affected:
                triple_active[ti] -= 1
            new_conflicts = sum(1 for c in triple_active if c == 3)
            
            delta = (new_conflicts - current_conflicts) * 100 + 1  # +1 for size loss
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                sel.discard(x)
                current_conflicts = new_conflicts
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(sel) > best_size):
                    best_conflicts = current_conflicts
                    best_size = len(sel)
                    best_sel = set(sel)
            else:
                for ti in affected:
                    triple_active[ti] += 1
        else:
            # Swap: remove x from sel, add y not in sel
            if not sel:
                continue
            x = rng.choice(list(sel))
            candidates = list(all_elems - sel)
            if not candidates:
                continue
            y = rng.choice(candidates)
            
            # Apply both changes
            affected_x = elem_to_triples[x]
            affected_y = elem_to_triples[y]
            for ti in affected_x:
                triple_active[ti] -= 1
            for ti in affected_y:
                triple_active[ti] += 1
            new_conflicts = sum(1 for c in triple_active if c == 3)
            
            delta = (new_conflicts - current_conflicts) * 100  # size unchanged
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                sel.discard(x)
                sel.add(y)
                current_conflicts = new_conflicts
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(sel) > best_size):
                    best_conflicts = current_conflicts
                    best_size = len(sel)
                    best_sel = set(sel)
            else:
                for ti in affected_x:
                    triple_active[ti] += 1
                for ti in affected_y:
                    triple_active[ti] -= 1
        
        T *= cooling
    
    # If best has conflicts, fix them
    if best_conflicts > 0:
        fixed_sel = set(best_sel)
        while True:
            active = [t for t in triples if t[0] in fixed_sel and t[1] in fixed_sel and t[2] in fixed_sel]
            if not active:
                break
            freq = {}
            for t in active:
                for x in t:
                    freq[x] = freq.get(x, 0) + 1
            max_f = max(freq.values())
            cands = [x for x, f in freq.items() if f == max_f]
            x = rng.choice(cands)
            fixed_sel.discard(x)
        
        if len(fixed_sel) > best_size or (len(fixed_sel) == best_size):
            best_sel = fixed_sel
            best_size = len(fixed_sel)
    
    return best_sel

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s limit (90s kill, leave margin)
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
    
    adj = _build_conflict_graph(bad_triples, max_n)
    
    best_selected = None
    best_size = -1
    
    # Phase 1: Multiple greedy max independent set attempts
    num_greedy = 300
    for i in range(num_greedy):
        if time.time() > deadline - 20:
            break
        selected = _greedy_max_independent_set(adj, max_n, seed=i)
        size = len(selected)
        if size > best_size:
            best_size = size
            best_selected = selected
    
    if best_selected is None:
        # Fallback: single element
        return [1]
    
    # Phase 2: Local search polish on best greedy result
    if time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 10
        num_ls = max(1, int(remaining_time / 4))
        
        for i in range(num_ls):
            if time.time() > deadline - 8:
                break
            rng = random.Random(5000 + i)
            polished = _local_search(best_selected, bad_triples, adj, max_n, rng, 3.5)
            size = len(polished)
            if size > best_size:
                best_size = size
                best_selected = polished
    
    # Phase 3: SA polish with incremental conflicts
    if time.time() < deadline - 3:
        remaining_time = deadline - time.time() - 3
        num_sa = max(1, int(remaining_time / 6))
        
        for i in range(num_sa):
            if time.time() > deadline - 2:
                break
            rng = random.Random(10000 + i)
            polished = _sa_with_incremental_conflicts(best_selected, bad_triples, max_n, rng, 5.5)
            size = len(polished)
            if size > best_size:
                best_size = size
                best_selected = polished
    
    return sorted(best_selected)

# EVOLVE-BLOCK-END
