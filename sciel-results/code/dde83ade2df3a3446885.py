"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random
import numpy as np

def _generate_graph(n, p, seed):
    """Generate graph edges using the same RNG sequence as the verifier."""
    rng = _random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_bitmask(adj_list, n):
    """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot, bitmask version."""
    # Convert adjacency to bitmasks
    adj_mask = [0] * n
    for u in range(n):
        for v in adj_list[u]:
            adj_mask[u] |= (1 << v)
    
    cliques = []
    
    def bk(R, P, X):
        if P == 0 and X == 0:
            if bin(R).count('1') >= 2:
                cliques.append(R)
            return
        U = P | X
        if U == 0:
            return
        # Pick pivot with max degree in U
        best_pivot = -1
        best_deg = -1
        for v in range(n):
            if (U >> v) & 1:
                deg = bin(adj_mask[v] & U).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_pivot = v
        if best_pivot < 0:
            return
        candidates = P & ~adj_mask[best_pivot]
        while candidates:
            # Get lowest set bit
            v = (candidates & (-candidates)).bit_length() - 1
            candidates &= candidates - 1
            R_new = R | (1 << v)
            P_new = P & adj_mask[v]
            X_new = X & adj_mask[v]
            bk(R_new, P_new, X_new)
            P &= ~(1 << v)
            X |= (1 << v)
    
    bk(0, (1 << n) - 1, 0)
    return cliques

def _bitmask_to_set(mask):
    """Convert bitmask to set of vertices."""
    return {i for i in range(mask.bit_length()) if (mask >> i) & 1}

def _greedy_hitting_set_bitmask(clique_masks, n, rng=None):
    """Greedy set cover using bitmasks. Optionally randomized."""
    m = len(clique_masks)
    if m == 0:
        return 0
    
    # For each vertex, compute which cliques it covers (as bitmask over cliques)
    # But m can be large, so use list of clique indices instead
    vertex_cliques = [[] for _ in range(n)]
    for i, cm in enumerate(clique_masks):
        for v in range(n):
            if (cm >> v) & 1:
                vertex_cliques[v].append(i)
    
    uncovered = set(range(m))
    selected_mask = 0
    
    while uncovered:
        best_v = -1
        best_count = -1
        if rng is not None:
            # Randomized: among top-k candidates, pick one randomly
            counts = []
            for v in range(n):
                count = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if count > 0:
                    counts.append((count, v))
            if not counts:
                break
            counts.sort(reverse=True)
            k = min(5, len(counts))
            best_v = counts[_random.Random(rng.randint(0, 10**9)).randint(0, k-1)][1]
        else:
            for v in range(n):
                count = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
        if best_v < 0 or best_count <= 0:
            break
        selected_mask |= (1 << best_v)
        for ci in vertex_cliques[best_v]:
            uncovered.discard(ci)
    
    return selected_mask

def _local_search_bitmask(selected_mask, clique_masks, n, time_limit, rng=None):
    """Local search using bitmasks: deletions and 1-for-2 swaps."""
    start_time = time.time()
    selected_mask = selected_mask
    
    # Precompute: for each clique, which selected vertices are in it
    def get_selected_in_clique(cm):
        result = []
        for v in range(n):
            if (cm >> v) & 1 and (selected_mask >> v) & 1:
                result.append(v)
        return result
    
    # Phase 1: Remove unnecessary vertices
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            return selected_mask
        for v in range(n):
            if not ((selected_mask >> v) & 1):
                continue
            # Check if v is the only selected vertex in any clique
            needed = False
            for cm in clique_masks:
                if (cm >> v) & 1:
                    sel_in_c = [x for x in range(n) if (cm >> x) & 1 and (selected_mask >> x) & 1]
                    if len(sel_in_c) == 1:
                        needed = True
                        break
            if not needed:
                selected_mask &= ~(1 << v)
                changed = True
    
    # Phase 2: 1-for-2 swaps (remove two vertices, add one)
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            return selected_mask
        sel_list = [v for v in range(n) if (selected_mask >> v) & 1]
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                v1, v2 = sel_list[i], sel_list[j]
                # Find cliques that would become uncovered if we remove both v1 and v2
                # i.e., cliques where the only selected vertices are subset of {v1, v2}
                critical_cliques = []
                for cm in clique_masks:
                    sel_in_c = [x for x in range(n) if (cm >> x) & 1 and (selected_mask >> x) & 1]
                    if len(sel_in_c) <= 2:
                        # Check if removing v1 and v2 leaves this clique uncovered
                        remaining = [x for x in sel_in_c if x != v1 and x != v2]
                        if len(remaining) == 0:
                            critical_cliques.append(cm)
                
                if not critical_cliques:
                    continue
                
                # Find a vertex w not in selected that covers all critical cliques
                for w in range(n):
                    if (selected_mask >> w) & 1:
                        continue
                    covers_all = True
                    for cm in critical_cliques:
                        if not ((cm >> w) & 1):
                            covers_all = False
                            break
                    if covers_all:
                        new_mask = (selected_mask & ~(1 << v1) & ~(1 << v2)) | (1 << w)
                        # Verify all cliques still covered
                        valid = True
                        for cm in clique_masks:
                            if not any((cm >> x) & 1 and (new_mask >> x) & 1 for x in range(n)):
                                valid = False
                                break
                        if valid:
                            selected_mask = new_mask
                            changed = True
                            break
                if changed:
                    break
            if changed:
                break
    
    # Phase 3: 1-for-1 swaps (replace one vertex with another that allows removal of another)
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            return selected_mask
        sel_list = [v for v in range(n) if (selected_mask >> v) & 1]
        for v_out in sel_list:
            # Find cliques where v_out is the only selected vertex
            unique_cliques = []
            for cm in clique_masks:
                if (cm >> v_out) & 1:
                    sel_in_c = [x for x in range(n) if (cm >> x) & 1 and (selected_mask >> x) & 1]
                    if len(sel_in_c) == 1:
                        unique_cliques.append(cm)
            if not unique_cliques:
                continue
            # Try replacing v_out with w
            for w in range(n):
                if (selected_mask >> w) & 1:
                    continue
                # w must cover all unique cliques of v_out
                covers = True
                for cm in unique_cliques:
                    if not ((cm >> w) & 1):
                        covers = False
                        break
                if not covers:
                    continue
                new_mask = (selected_mask & ~(1 << v_out)) | (1 << w)
                # Check if this allows removing another vertex
                best_removal = -1
                for v_rem in range(n):
                    if v_rem == w or v_rem == v_out:
                        continue
                    if not ((new_mask >> v_rem) & 1):
                        continue
                    needed = False
                    for cm in clique_masks:
                        if (cm >> v_rem) & 1:
                            sel_in_c = [x for x in range(n) if (cm >> x) & 1 and (new_mask >> x) & 1]
                            if len(sel_in_c) == 1:
                                needed = True
                                break
                    if not needed:
                        best_removal = v_rem
                        break
                if best_removal >= 0:
                    final_mask = new_mask & ~(1 << best_removal)
                    # Verify
                    valid = True
                    for cm in clique_masks:
                        if not any((cm >> x) & 1 and (final_mask >> x) & 1 for x in range(n)):
                            valid = False
                            break
                    if valid:
                        selected_mask = final_mask
                        changed = True
                        break
            if changed:
                break
    
    return selected_mask

def _milp_solve(clique_masks, n, time_limit):
    """Try to solve the hitting set problem using MILP via scipy."""
    m = len(clique_masks)
    if m == 0:
        return 0
    
    # Build constraint matrix
    A = np.zeros((m, n), dtype=np.int8)
    for i, cm in enumerate(clique_masks):
        for v in range(n):
            if (cm >> v) & 1:
                A[i, v] = 1
    
    c = np.ones(n, dtype=np.float64)
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraints = LinearConstraint(A, 1, np.inf)
        bounds_obj = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                   bounds=bounds_obj, 
                   options={"time_limit": time_limit, "presolve": True})
        
        if res.x is not None:
            mask = 0
            for i in range(n):
                if res.x[i] > 0.5:
                    mask |= (1 << i)
            # Verify validity
            valid = True
            for cm in clique_masks:
                if not any((cm >> v) & 1 and (mask >> v) & 1 for v in range(n)):
                    valid = False
                    break
            if valid:
                return mask
    except Exception:
        pass
    
    return None

def _solve_instance(adj, n, time_budget, rng_seed):
    """Solve one instance: enumerate cliques, then use MILP + local search with restarts."""
    clique_masks = _bron_kerbosch_bitmask(adj, n)
    
    if not clique_masks:
        return 0
    
    best_mask = None
    best_size = n + 1
    
    # Try MILP first
    time_left = time.time()
    deadline = time_left + time_budget
    milp_budget = min(15.0, time_budget * 0.4)
    
    mask = _milp_solve(clique_masks, n, milp_budget)
    if mask is not None:
        size = bin(mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = mask
    
    # If MILP didn't solve or we have time, do greedy + local search with restarts
    time_left = deadline - time.time()
    if time_left > 2.0:
        # Number of restarts based on remaining time
        num_restarts = max(1, int(time_left / 3.0))
        num_restarts = min(num_restarts, 20)  # Cap at 20
        
        for restart in range(num_restarts):
            if time.time() > deadline - 1.0:
                break
            
            rng = _random.Random(rng_seed + restart * 7919)
            
            # Get initial solution: greedy (randomized for restarts > 0)
            if restart == 0:
                init_mask = _greedy_hitting_set_bitmask(clique_masks, n, rng=None)
            else:
                init_mask = _greedy_hitting_set_bitmask(clique_masks, n, rng=rng)
            
            if init_mask is None or init_mask == 0:
                # Fallback: all vertices with degree > 0
                init_mask = 0
                for v in range(n):
                    if len(adj[v]) > 0:
                        init_mask |= (1 << v)
            
            # Local search
            ls_time = max(0.5, min(2.0, (deadline - time.time()) * 0.3))
            polished = _local_search_bitmask(init_mask, clique_masks, n, ls_time, rng=rng)
            
            size = bin(polished).count('1')
            if size < best_size:
                best_size = size
                best_mask = polished
    
    if best_mask is None:
        # Last resort: all vertices with edges
        best_mask = 0
        for v in range(n):
            if len(adj[v]) > 0:
                best_mask |= (1 << v)
    
    return best_mask

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time: give more time to larger instances
    # Total budget: 140s, reserve 10s for safety
    available_time = 140.0 - 10.0  # 130s
    # Weights proportional to n^2 (roughly)
    weights = [30**2, 35**2, 40**2]
    total_weight = sum(weights)
    time_allocs = [available_time * w / total_weight for w in weights]
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() > deadline - 5.0:
            # Not enough time left; return trivial solution
            adj = _generate_graph(n, p, seed)
            touched = 0
            for u in range(n):
                for v in adj[u]:
                    touched |= (1 << u)
                    touched |= (1 << v)
            results.append(_bitmask_to_set(touched))
            continue
        
        adj = _generate_graph(n, p, seed)
        
        # Solve
        budget = min(time_allocs[idx], deadline - time.time() - 5.0)
        mask = _solve_instance(adj, n, budget, seed)
        
        results.append(_bitmask_to_set(mask))
    
    return [sorted(r) for r in results]
# EVOLVE-BLOCK-END
