import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are all equal. Maximize cardinality.
    
    Strategy: 
    1. Identify all "bad triples" (triples with equal pairwise gcds).
    2. Frame the problem as finding a Maximum Independent Set in the 
       3-uniform hypergraph of bad triples, or equivalently, a Minimum 
       Hitting Set to remove to break all bad triples.
    3. Use a Greedy Hitting Set approach to start with a valid (but small) set.
    4. Use Local Search (1-swap, 2-swap) to expand the set by trying to 
       re-insert removed elements or swap them for others, guided by 
       conflict counts.
    """
    N = 128
    deadline = time.time() + 85.0  # Safety margin for 90s limit

    # 1. Precompute GCD table
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g

    # 2. Identify all bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    # Optimization: Group by gcd? 
    # If gcd(a,b)=g, then a,b are multiples of g.
    # Let's just iterate, N=128 is small. 128^3 / 6 is ~350k checks. Fast enough.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gt[a][b]
            for c in range(b + 1, N + 1):
                if gt[a][c] == gab and gt[b][c] == gab:
                    bad_triples.append((a, b, c))
    
    # Map element -> list of bad triple indices it participates in
    # This helps in "Hitting Set" logic
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)

    # 3. Greedy Hitting Set Construction
    # Start with all elements selected.
    # We want to REMOVE elements to break all bad triples.
    # Greedy: While any bad triple is "active" (all 3 elements present),
    # remove the element that is in the most active bad triples.
    
    def construct_via_hitting_set():
        selected = set(range(1, N + 1))
        # Track which triples are currently "broken" (at least one element removed)
        # and which are "active" (all 3 elements present)
        # Initially all are active.
        active_triples = set(range(len(bad_triples)))
        
        # To make this efficient, we maintain a count of how many elements 
        # from each active triple are still in 'selected'.
        # If count drops to 2 or less, the triple is broken.
        
        # For simplicity in the first pass, we just iterate.
        # Since N is small, we can re-evaluate or update carefully.
        
        # Let's use a priority queue approach or just repeat scanning.
        # Scanning 350k triples each step is too slow if we do many steps.
        # But we only remove ~100 elements max. 100 * 350k = 35M ops. OK in Python? Maybe slow.
        
        # Optimized Hitting Set:
        # Maintain for each element, the number of active triples it hits.
        # When we remove an element, we decrement the count for all its triples.
        # If a triple's count drops to 0 (meaning 0 elements removed from it? No.)
        
        # Let's redefine:
        # A triple is "satisfied" (bad) if all 3 are in selected.
        # We want to remove elements to minimize |selected| such that no triple is satisfied.
        # This is Min Vertex Cover in Hypergraph.
        
        # Greedy:
        # 1. Count for each element how many "currently bad" triples it is in.
        # 2. Pick the element with the max count.
        # 3. Remove it.
        # 4. Update counts for all triples containing it (they are no longer "currently bad" if they had this element? 
        #    No, a triple is "currently bad" only if ALL its elements are in selected.
        #    If we remove one element of a triple, that triple is no longer "currently bad".
        #    So we only care about triples that are *still fully present*.
        
        # Implementation:
        # active_triple_count[t] = number of elements of triple t still in selected.
        # Initially 3 for all.
        # If active_triple_count[t] < 3, triple t is broken.
        
        counts = [3] * len(bad_triples)
        selected = set(range(1, N + 1))
        
        # Pre-compute element -> triples map for fast updates
        # We already have elem_triples, but we need to know which triples are still active.
        
        while True:
            # Check if any active triples remain
            # We can track this with a count of active triples
            active_count = sum(1 for c in counts if c == 3)
            if active_count == 0:
                break
            
            # Find element to remove
            # We need the frequency of each element in the *active* triples.
            # This requires scanning all active triples.
            # To avoid O(N*T) every step, we can approximate or use a heap.
            # Given N=128, T~350k, and ~100 removals, 100 * 350k = 35M. 
            # Python might take ~1-2s per full scan. 100 scans = 100-200s. Too slow.
            
            # Alternative: Randomized Greedy Hitting Set
            # Pick a random active triple. Pick one of its elements at random (or weighted by global degree). Remove it.
            # This is faster.
            
            # Let's try a simpler heuristic: 
            # Remove elements that are in the most triples globally (static degree).
            # This is fast to compute once.
            
            if 'static_degree' not in locals():
                static_degree = [0] * (N + 1)
                for t in bad_triples:
                    for x in t:
                        static_degree[x] += 1
            
            # Find the element with max static_degree that is still in selected
            # And preferably one that is in many *currently active* triples.
            # Let's just pick the one with max static_degree among selected.
            # This is O(N) per step. 100 * 128 = 12800 ops. Very fast.
            
            best_elem = -1
            best_deg = -1
            for x in selected:
                if static_degree[x] > best_deg:
                    best_deg = static_degree[x]
                    best_elem = x
            
            if best_elem == -1:
                break
                
            selected.remove(best_elem)
            
            # Update counts for triples containing best_elem
            for t_idx in elem_triples[best_elem]:
                counts[t_idx] -= 1
                # Note: We don't need to check if it became 0, we just decrement.
                # The condition for "active" is count == 3.
                # If we remove one, it becomes 2, so it's no longer active.
                # If we remove another from the same triple, it becomes 1.
                # The logic holds: active iff count == 3.
                
        return selected

    # 4. Local Search to Expand
    # The hitting set approach minimizes removals, which maximizes size.
    # However, it's a greedy approximation. We can improve it.
    # We can try to ADD BACK elements that were removed, if they don't create a conflict.
    # Or swap.
    
    def is_valid(s):
        # Check if set s has no bad triples
        # For efficiency, we can check against the precomputed bad_triples
        # A bad triple exists in s if all 3 elements are in s.
        s_set = set(s)
        for a, b, c in bad_triples:
            if a in s_set and b in s_set and c in s_set:
                return False
        return True

    def polish(s):
        """Try to add any excluded element that fits."""
        s = set(s)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in s:
                    # Check if adding x creates a bad triple
                    # A bad triple (a,b,x) exists if a,b in s and (a,b,x) in bad_triples
                    # We can precompute for each x, the pairs (a,b) that form a bad triple with x.
                    # Let's just check all pairs in s? O(|s|^2). |s|~30. 900 checks. Fast.
                    can_add = True
                    for a in s:
                        for b in s:
                            if a < b:
                                # Check if (a,b,x) is a bad triple
                                # We can use a set of bad triples for O(1) lookup
                                # Or just gcd check
                                if gt[a][b] == gt[a][x] == gt[b][x]:
                                    can_add = False
                                    break
                        if not can_add:
                            break
                    if can_add:
                        s.add(x)
                        changed = True
        return s

    def local_search(s, max_iters=500):
        """
        Local search with 1-swap and 2-1 exchange.
        """
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # 1. Try to add any element
            for x in range(1, N + 1):
                if x not in s:
                    # Fast check
                    can_add = True
                    for a in s:
                        for b in s:
                            if a < b and gt[a][b] == gt[a][x] == gt[b][x]:
                                can_add = False
                                break
                        if not can_add:
                            break
                    if can_add:
                        s.add(x)
                        improved = True
                        break
                        
            # 2. 1-Swap: Remove one, add one
            if not improved:
                excluded = [x for x in range(1, N + 1) if x not in s]
                random.shuffle(excluded)
                found = False
                for r in list(s):
                    base = s - {r}
                    for y in excluded:
                        if y in base: continue
                        # Check if y can be added to base
                        can_add = True
                        for a in base:
                            for b in base:
                                if a < b and gt[a][b] == gt[a][y] == gt[b][y]:
                                    can_add = False
                                    break
                            if not can_add:
                                break
                        if can_add:
                            s = base | {y}
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue

            # 3. 2-1 Exchange: Remove one, add two
            if not improved:
                excluded = [x for x in range(1, N + 1) if x not in s]
                random.shuffle(excluded)
                found = False
                for r in list(s):
                    base = s - {r}
                    added = []
                    for y in excluded:
                        if len(added) >= 2:
                            break
                        if y in base or y in added:
                            continue
                        # Check if y can be added to base U added
                        temp_set = base | set(added)
                        can_add = True
                        for a in temp_set:
                            for b in temp_set:
                                if a < b and gt[a][b] == gt[a][y] == gt[b][y]:
                                    can_add = False
                                    break
                            if not can_add:
                                break
                        if can_add:
                            added.append(y)
                    if len(added) == 2:
                        s = base | set(added)
                        improved = True
                        found = True
                        break
                if found:
                    continue

            # 4. Random Perturbation
            if not improved and random.random() < 0.5 and len(s) > 1:
                r = random.choice(list(s))
                s.discard(r)
                improved = True

        return polish(s)

    # Main Execution
    random.seed(12345)
    
    best_set = set()
    best_size = 0

    def consider(s):
        nonlocal best_set, best_size
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)

    # Initial construction using Hitting Set
    initial_set = construct_via_hitting_set()
    consider(initial_set)
    
    # Polish the initial set
    polished_initial = polish(initial_set)
    consider(polished_initial)
    
    # Local Search on the polished initial set
    for _ in range(10):
        if time.time() > deadline:
            break
        s = local_search(polished_initial)
        consider(s)
        polished_initial = s # Refine on the new best

    # Try a few random restarts with different random seeds for the Hitting Set (if we made it randomized)
    # For now, the static degree greedy is deterministic.
    # Let's add a randomized hitting set variant.
    
    def construct_via_random_hitting_set():
        selected = set(range(1, N + 1))
        counts = [3] * len(bad_triples)
        
        # Pre-compute static degree for weighting
        static_degree = [0] * (N + 1)
        for t in bad_triples:
            for x in t:
                static_degree[x] += 1
        
        while True:
            # Find active triples
            active_indices = [i for i, c in enumerate(counts) if c == 3]
            if not active_indices:
                break
            
            # Pick a random active triple
            t_idx = random.choice(active_indices)
            a, b, c = bad_triples[t_idx]
            
            # Pick an element from this triple to remove
            # Weighted by static degree
            weights = [static_degree[a], static_degree[b], static_degree[c]]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            elem_to_remove = a
            for elem, w in zip([a, b, c], weights):
                cum += w
                if r <= cum:
                    elem_to_remove = elem
                    break
            
            selected.remove(elem_to_remove)
            
            # Update counts
            for t_idx2 in elem_triples[elem_to_remove]:
                counts[t_idx2] -= 1
                
        return selected

    for _ in range(20):
        if time.time() > deadline:
            break
        s = construct_via_random_hitting_set()
        s = polish(s)
        consider(s)
        # Short local search
        s = local_search(s, max_iters=50)
        consider(s)

    # Final Polish
    consider(polish(best_set))

    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
