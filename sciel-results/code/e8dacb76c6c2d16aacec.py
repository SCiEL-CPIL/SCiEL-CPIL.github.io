import numpy as np
import time
from itertools import combinations, product
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330
    
    # Precompute support sets for intersection checks
    sup_sets = [set(s) for s in supports]
    
    # Build incompatibility graph: two supports are incompatible if |intersection| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)
    
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
    
    # Build neighbor lists for faster access
    neighbors = [set(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    # Precompute incompatibility as a list of lists for fast membership tests
    incompat_lists = [list(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    # Precompute the set of incompatible indices as a numpy boolean matrix row view
    incompat_bool = incompat.astype(bool)
    
    def greedy_mis_min_degree(available, rng):
        """Greedy MIS using minimum degree heuristic (fast: maintain degree counts)."""
        selected = set()
        avail = set(available)
        # degree[u] = number of neighbors of u that are still in avail
        deg = {}
        for u in avail:
            deg[u] = len(neighbors[u] & avail)
        
        while avail:
            if time.time() > deadline - 15.0:
                break
            
            # Find node with minimum degree
            min_deg = float('inf')
            candidates = []
            for u in avail:
                d = deg[u]
                if d < min_deg:
                    min_deg = d
                    candidates = [u]
                elif d == min_deg:
                    candidates.append(u)
            
            # Pick randomly from minimum-degree candidates
            u = rng.choice(candidates)
            selected.add(u)
            
            # Remove u and its neighbors; update degrees of remaining avail nodes
            to_remove = {u} | (neighbors[u] & avail)
            for v in to_remove:
                avail.discard(v)
            # For each remaining node w, if u was a neighbor, decrement deg[w]
            for w in avail:
                if u in neighbors[w]:
                    deg[w] -= 1
            # Also need to account for removed neighbors other than u
            # Actually, we only removed u and its neighbors. For a remaining node w,
            # its neighbors that were removed are exactly {u} ∩ neighbors[w] plus any
            # neighbor of u that is also a neighbor of w.
            # Let's be more careful: for each removed node r (other than u), if r is
            # a neighbor of w, then deg[w] should be decremented.
            # But we already decremented for u. Let's redo:
            # Reset: for each w in avail, deg[w] = number of neighbors of w in avail
            # That's expensive. Instead, let's just recompute deg for all avail nodes
            # when the set changes significantly. Actually, the current approach of
            # only decrementing for u is incorrect because we also removed neighbors of u.
            
            # Let me use a simpler but still fast approach: recompute degrees only
            # when the set is small, otherwise use the original method.
            # Actually, let's just use the original set-intersection method but with
            # a precomputed incompatibility count via numpy.
            pass
        
        # The above is getting complex. Let me use a cleaner approach:
        # Use numpy to compute degrees in bulk.
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 15.0:
                break
            
            if len(avail) > 200:
                # Use numpy for bulk degree computation
                avail_arr = np.array(list(avail), dtype=np.int32)
                # degree of each avail node = number of incompatibilities with other avail nodes
                deg_arr = incompat_bool[avail_arr][:, avail_arr].sum(axis=1)
                min_deg = deg_arr.min()
                min_mask = deg_arr == min_deg
                candidates = avail_arr[min_mask]
                u = int(rng.choice(candidates))
            else:
                # Small set: use direct computation
                min_deg = float('inf')
                candidates = []
                for u in avail:
                    d = len(neighbors[u] & avail)
                    if d < min_deg:
                        min_deg = d
                        candidates = [u]
                    elif d == min_deg:
                        candidates.append(u)
                u = rng.choice(candidates)
            
            selected.add(u)
            
            # Remove u and its neighbors
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            
        return selected
    
    def local_search_add(selected, rng):
        """Try to add vertices that are compatible with all current selected vertices (vectorized)."""
        selected = set(selected)
        
        if not selected:
            return selected
        
        sel_arr = np.array(list(selected), dtype=np.int32)
        # A node x can be added if it has no incompatibilities with any selected node
        # incompat_bool[x, sel_arr] should be all False
        compatible_mask = ~incompat_bool[:, sel_arr].any(axis=1)
        # Exclude already-selected nodes
        compatible_mask[sel_arr] = False
        
        addable = np.where(compatible_mask)[0]
        for x in addable:
            if time.time() > deadline - 10.0:
                break
            selected.add(int(x))
        
        return selected
    
    def local_search_1_2_swap(selected, rng):
        """Improve MIS using 1-2 swaps: remove one vertex, add two compatible vertices."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            for v in list(selected):
                if time.time() > deadline - 10.0:
                    break
                
                # Find all nodes compatible with (selected - {v})
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    # x must be compatible with all y in selected - {v}
                    if not (neighbors[x] & (selected - {v})):
                        potential_adds.append(x)
                
                # Find a pair (u1, u2) in potential_adds that are compatible with each other
                found = False
                n_pot = len(potential_adds)
                for i in range(n_pot):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pot):
                        u2 = potential_adds[j]
                        if incompat[u1, u2] == 0:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                
                if improved:
                    break  # Restart the outer loop
        
        return selected
    
    def local_search_2_1_swap(selected, rng):
        """Improve MIS using 2-1 swaps: remove two vertices, add one compatible vertex."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            sel_list = list(selected)
            n_sel = len(sel_list)
            
            for i in range(n_sel):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, n_sel):
                    v1 = sel_list[i]
                    v2 = sel_list[j]
                    remaining = selected - {v1, v2}
                    
                    # Find a node x compatible with all in remaining
                    for x in range(N_sup):
                        if x in selected:
                            continue
                        if not (neighbors[x] & remaining):
                            selected.remove(v1)
                            selected.remove(v2)
                            selected.add(x)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return selected
    
    best_iset = set()
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    n_restarts = 120
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if r == 0:
            # First run: pure greedy min-degree
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
        elif r < 40:
            # Greedy + local search
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        elif r < 80:
            # Perturbed greedy: start with random subset removed
            avail = set(range(N_sup))
            n_remove = rng.integers(5, 30)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        else:
            # More aggressive perturbation
            avail = set(range(N_sup))
            n_remove = rng.integers(20, 60)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)
    
    # Final improvement attempts on best solution
    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
    
    # One more round of local search if time permits
    if time.time() < deadline - 5.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
    
    # Generate vectors: for each support in the MIS, include all 2^4 = 16 sign patterns
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    if not vectors:
        # Fallback: return a small valid set
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
