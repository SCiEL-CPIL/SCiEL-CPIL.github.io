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

def _build_conflict_graph(triples, max_n):
    """
    Build conflict graph: two elements conflict if they appear together in some bad triple.
    Also build element -> list of triple indices.
    """
    elem_to_triples = {i: [] for i in range(1, max_n + 1)}
    for ti, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(ti)
    
    # Conflict graph: adj[x] = set of elements that conflict with x
    adj = {i: set() for i in range(1, max_n + 1)}
    for t in triples:
        a, b, c = t
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    
    return adj, elem_to_triples

def _build_pair_index(triples, max_n):
    """
    For each element x, store the list of (a, b) pairs such that (a,b,x) is a bad triple.
    Also build a bitmask for each element's conflict neighbors.
    """
    # x -> list of (a, b) pairs from bad triples containing x
    x_conflict_pairs = {i: [] for i in range(1, max_n + 1)}
    for t in triples:
        a, b, c = t
        x_conflict_pairs[a].append((b, c))
        x_conflict_pairs[b].append((a, c))
        x_conflict_pairs[c].append((a, b))
    
    # Bitmask for each element: which other elements are in some bad triple with it
    # Using Python integers as bitsets (element i maps to bit (i-1))
    conflict_mask = [0] * (max_n + 1)
    for t in triples:
        a, b, c = t
        conflict_mask[a] |= (1 << (b-1)) | (1 << (c-1))
        conflict_mask[b] |= (1 << (a-1)) | (1 << (c-1))
        conflict_mask[c] |= (1 << (a-1)) | (1 << (b-1))
    
    return x_conflict_pairs, conflict_mask

def _greedy_max_independent_set(adj, max_n, seed=0):
    """
    Greedy: repeatedly pick the element with minimum degree (conflicts),
    add to independent set, remove it and its neighbors.
    Optimized with list-based tracking.
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
        
        # Remove neighbors from remaining and update degrees
        neighbors_to_remove = adj[x] & remaining
        for y in neighbors_to_remove:
            remaining.discard(y)
        
        # Update degrees: remaining elements lost x and all removed neighbors
        removed_set = {x} | neighbors_to_remove
        for z in remaining:
            # Count how many of z's neighbors were removed
            lost = len(adj[z] & removed_set)
            deg[z] -= lost
    
    return selected

def _local_search_fast(selected, triples, x_conflict_pairs, conflict_mask, max_n, rng, time_limit):
    """
    Local search with fast conflict checking using precomputed pair indices.
    """
    start = time.time()
    sel = set(selected)
    best_sel = set(sel)
    best_size = len(sel)
    
    # Build bitmask for selected set
    sel_mask = 0
    for x in sel:
        sel_mask |= (1 << (x - 1))
    
    def is_addable(x, S_mask):
        """Check if adding x to S keeps it valid using conflict pairs."""
        for (a, b) in x_conflict_pairs.get(x, []):
            if (S_mask >> (a-1)) & 1 and (S_mask >> (b-1)) & 1:
                return False
        return True
    
    # Phase 1: Try to add elements (greedy add)
    excluded = set(range(1, max_n + 1)) - sel
    # Sort by number of conflicts (fewer conflicts = easier to add)
    excluded_sorted = sorted(excluded, key=lambda y: len(x_conflict_pairs.get(y, [])))
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        for x in excluded_sorted:
            if x not in excluded:
                continue
            if is_addable(x, sel_mask):
                sel.add(x)
                sel_mask |= (1 << (x - 1))
                excluded.discard(x)
                improved = True
                if len(sel) > best_size:
                    best_size = len(sel)
                    best_sel = set(sel)
    
    # Phase 2: Swap-based local search with fast checks
    # For a swap (remove x, add y), we need to check:
    # - y is addable to (sel - {x})
    # This is fast: check y's conflict pairs against (sel_mask without x)
    
    T = 3.0
    cooling = 0.999
    step = 0
    while T > 0.01 and time.time() - start < time_limit and step < 50000:
        step += 1
        if not sel or not excluded:
            break
        
        x = rng.choice(list(sel))
        y = rng.choice(list(excluded))
        
        # Check if swap is valid: y addable to (sel - {x})
        sel_mask_without_x = sel_mask & ~(1 << (x - 1))
        valid = True
        for (a, b) in x_conflict_pairs.get(y, []):
            if (sel_mask_without_x >> (a-1)) & 1 and (sel_mask_without_x >> (b-1)) & 1:
                valid = False
                break
        
        if valid:
            # Perform swap
            sel.discard(x)
            sel.add(y)
            sel_mask = sel_mask_without_x | (1 << (y - 1))
            excluded.discard(y)
            excluded.add(x)
            
            if len(sel) > best_size:
                best_size = len(sel)
                best_sel = set(sel)
            
            # Try to add more elements after swap
            for z in list(excluded):
                if is_addable(z, sel_mask):
                    sel.add(z)
                    sel_mask |= (1 << (z - 1))
                    excluded.discard(z)
                    if len(sel) > best_size:
                        best_size = len(sel)
                        best_sel = set(sel)
        
        T *= cooling
    
    return best_sel

def _sa_polish_fast(selected, triples, x_conflict_pairs, conflict_mask, max_n, rng, time_limit):
    """
    SA polish with fast incremental conflict counting.
    Work with removed set; minimize |removed| subject to no conflicts in selected.
    """
    start = time.time()
    all_elems = set(range(1, max_n + 1))
    removed = all_elems - set(selected)
    sel = set(selected)
    
    # Build bitmask for selected
    sel_mask = 0
    for x in sel:
        sel_mask |= (1 << (x - 1))
    
    # Count current conflicts (should be 0 for valid input)
    def count_conflicts_from_mask(mask):
        """Count bad triples fully contained in the set represented by mask."""
        cnt = 0
        for t in triples:
            a, b, c = t
            if (mask >> (a-1)) & 1 and (mask >> (b-1)) & 1 and (mask >> (c-1)) & 1:
                cnt += 1
        return cnt
    
    current_conflicts = count_conflicts_from_mask(sel_mask)
    current_cost = len(removed) + 10000 * current_conflicts
    
    best_removed = set(removed)
    best_cost = current_cost
    best_conflicts = current_conflicts
    best_mask = sel_mask
    
    T = 5.0
    cooling = 0.9997
    step = 0
    while T > 0.01 and time.time() - start < time_limit and step < 100000:
        step += 1
        move_type = rng.random()
        
        if move_type < 0.4 and removed:
            # Remove from removed (add to selected)
            x = rng.choice(list(removed))
            new_mask = sel_mask | (1 << (x - 1))
            new_removed_count = len(removed) - 1
        elif move_type < 0.7 and len(removed) < max_n:
            # Add to removed
            candidates = [x for x in sel]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_mask = sel_mask & ~(1 << (x - 1))
            new_removed_count = len(removed) + 1
        else:
            # Swap
            if removed and sel:
                x_out = rng.choice(list(removed))
                x_in = rng.choice(list(sel))
                new_mask = (sel_mask & ~(1 << (x_in - 1))) | (1 << (x_out - 1))
                new_removed_count = len(removed)
            else:
                if removed:
                    x = rng.choice(list(removed))
                    new_mask = sel_mask | (1 << (x - 1))
                    new_removed_count = len(removed) - 1
                elif sel:
                    x = rng.choice(list(sel))
                    new_mask = sel_mask & ~(1 << (x - 1))
                    new_removed_count = len(removed) + 1
                else:
                    continue
        
        new_conflicts = count_conflicts_from_mask(new_mask)
        new_cost = new_removed_count + 10000 * new_conflicts
        
        delta = new_cost - current_cost
        if delta <= 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
            # Accept move
            # Update removed set
            if new_removed_count < len(removed):
                # We added to selected
                added_x = None
                for x in range(1, max_n + 1):
                    if (new_mask >> (x-1)) & 1 and not (sel_mask >> (x-1)) & 1:
                        added_x = x
                        break
                if added_x is not None:
                    removed.discard(added_x)
            elif new_removed_count > len(removed):
                # We removed from selected
                removed_x = None
                for x in range(1, max_n + 1):
                    if (sel_mask >> (x-1)) & 1 and not (new_mask >> (x-1)) & 1:
                        removed_x = x
                        break
                if removed_x is not None:
                    removed.add(removed_x)
            else:
                # Swap: find what changed
                added_bits = new_mask & ~sel_mask
                removed_bits = sel_mask & ~new_mask
                for x in range(1, max_n + 1):
                    if (added_bits >> (x-1)) & 1:
                        removed.discard(x)
                    if (removed_bits >> (x-1)) & 1:
                        removed.add(x)
            
            sel_mask = new_mask
            current_conflicts = new_conflicts
            current_cost = new_cost
            
            if new_conflicts == 0 and new_removed_count < len(best_removed):
                best_removed = set(removed)
                best_cost = new_removed_count
                best_conflicts = 0
                best_mask = new_mask
        
        T *= cooling
    
    # Use best found
    final_removed = best_removed
    final_sel = all_elems - final_removed
    
    # Fix if best has conflicts
    if best_conflicts > 0:
        active = [t for t in triples if (best_mask >> (t[0]-1)) & 1 and (best_mask >> (t[1]-1)) & 1 and (best_mask >> (t[2]-1)) & 1]
        while active:
            freq = {}
            for t in active:
                for x in t:
                    freq[x] = freq.get(x, 0) + 1
            max_f = max(freq.values())
            cands = [x for x, f in freq.items() if f == max_f]
            x = rng.choice(cands)
            final_removed.add(x)
            final_sel.discard(x)
            best_mask &= ~(1 << (x - 1))
            active = [t for t in active if x not in t]
        final_removed = set(final_removed)
    
    return final_removed

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit (90s kill, leave margin)
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
    
    adj, elem_to_triples = _build_conflict_graph(bad_triples, max_n)
    x_conflict_pairs, conflict_mask = _build_pair_index(bad_triples, max_n)
    
    best_selected = None
    best_size = -1
    
    # Phase 1: Multiple greedy max independent set attempts
    num_greedy = 300
    for i in range(num_greedy):
        if time.time() > deadline - 10:
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
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        num_ls = max(1, int(remaining_time / 4))
        
        for i in range(num_ls):
            if time.time() > deadline - 3:
                break
            rng = random.Random(5000 + i)
            polished = _local_search_fast(best_selected, bad_triples, x_conflict_pairs, conflict_mask, max_n, rng, 3.5)
            size = len(polished)
            if size > best_size:
                best_size = size
                best_selected = polished
    
    # Phase 3: SA polish
    if time.time() < deadline - 2:
        remaining_time = deadline - time.time() - 2
        num_sa = max(1, int(remaining_time / 6))
        
        for i in range(num_sa):
            if time.time() > deadline - 2:
                break
            rng = random.Random(10000 + i)
            removed = _sa_polish_fast(best_selected, bad_triples, x_conflict_pairs, conflict_mask, max_n, rng, 5.0)
            selected = set(range(1, max_n + 1)) - removed
            size = len(selected)
            if size > best_size:
                best_size = size
                best_selected = selected
    
    # Final verification and fix
    final_sel = set(best_selected)
    # Remove any remaining conflicts
    for t in bad_triples:
        a, b, c = t
        if a in final_sel and b in final_sel and c in final_sel:
            # Remove the element that appears in most other bad triples
            final_sel.discard(c)  # Remove the largest (arbitrary tiebreak)
    
    return sorted(final_sel)

# EVOLVE-BLOCK-END
