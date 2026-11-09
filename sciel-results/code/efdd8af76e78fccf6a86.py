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

def _bron_kerbosch_bitmask(adj, n):
    """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with bitmasks."""
    # Convert adjacency to bitmasks
    adj_masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_masks[u] = mask
    
    cliques = []
    full_mask = (1 << n) - 1
    
    def bk(R_mask, P_mask, X_mask):
        if P_mask == 0 and X_mask == 0:
            if R_mask & (R_mask - 1):  # at least 2 bits set
                cliques.append(R_mask)
            return
        
        if P_mask == 0:
            return
        
        # Choose pivot: vertex in P|X with max degree in P|X
        U = P_mask | X_mask
        # Find pivot with maximum |adj[v] & U|
        best_pivot = -1
        best_deg = -1
        temp = U
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = bin(adj_masks[v] & U).count('1')
            if deg > best_deg:
                best_deg = deg
                best_pivot = v
        
        # Branch on vertices in P \ N(pivot)
        candidates = P_mask & ~adj_masks[best_pivot]
        
        temp = candidates
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            
            R_new = R_mask | (1 << v)
            P_new = P_mask & adj_masks[v]
            X_new = X_mask & adj_masks[v]
            
            bk(R_new, P_new, X_new)
            
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
    
    bk(0, full_mask, 0)
    return cliques

def _to_bitmasks(cliques, n):
    """Convert list of cliques (frozensets) to list of integer bitmasks."""
    masks = []
    for c in cliques:
        mask = 0
        for v in c:
            mask |= (1 << v)
        masks.append(mask)
    return masks

def _greedy_hitting_set_bitmask(clique_masks, n):
    """Greedy set cover using bitmasks with frequency-based ordering."""
    if not clique_masks:
        return set()
    
    m = len(clique_masks)
    selected = set()
    selected_mask = 0
    
    # Precompute for each vertex which cliques it hits
    vertex_cliques = [[] for _ in range(n)]
    for i, cm in enumerate(clique_masks):
        temp = cm
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            vertex_cliques[v].append(i)
    
    # Track uncovered cliques
    uncovered = [True] * m
    uncovered_count = m
    
    while uncovered_count > 0:
        best_v = -1
        best_new = 0
        for v in range(n):
            if v in selected:
                continue
            new_covered = 0
            for ci in vertex_cliques[v]:
                if uncovered[ci]:
                    new_covered += 1
            if new_covered > best_new:
                best_new = new_covered
                best_v = v
        
        if best_v == -1 or best_new == 0:
            break
        
        selected.add(best_v)
        for ci in vertex_cliques[best_v]:
            if uncovered[ci]:
                uncovered[ci] = False
                uncovered_count -= 1
    
    return selected

def _local_search_bitmask(selected, clique_masks, n, time_limit):
    """Local search using bitmasks with deletion, 1-for-1, and 2-for-1 swaps."""
    start_time = time.time()
    selected = set(selected)
    m = len(clique_masks)
    
    # Precompute vertex-clique incidence
    vertex_cliques = [[] for _ in range(n)]
    for i, cm in enumerate(clique_masks):
        temp = cm
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            vertex_cliques[v].append(i)
    
    def get_coverage(S):
        """Return list of uncovered clique indices for set S."""
        S_mask = 0
        for v in S:
            S_mask |= (1 << v)
        unc = []
        for i, cm in enumerate(clique_masks):
            if not (S_mask & cm):
                unc.append(i)
        return unc
    
    def is_valid(S):
        S_mask = 0
        for v in S:
            S_mask |= (1 << v)
        for cm in clique_masks:
            if not (S_mask & cm):
                return False
        return True
    
    # 1. Aggressive Deletion Pass
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            break
        for v in list(selected):
            if time.time() - start_time > time_limit:
                break
            if not is_valid(selected - {v}):
                continue
            # Check if removing v is still valid
            S_test = selected - {v}
            S_mask = 0
            for w in S_test:
                S_mask |= (1 << w)
            valid = True
            for cm in clique_masks:
                if not (S_mask & cm):
                    valid = False
                    break
            if valid:
                selected = S_test
                changed = True

    # 2. 1-for-1 Swap Pass (replace one vertex with another to maintain validity, potentially enabling future deletions)
    # Actually, 1-for-1 doesn't reduce size. Skip this.
    
    # 3. 2-for-1 Swap Pass
    for iteration in range(50):  # Multiple passes
        if time.time() - start_time > time_limit:
            break
        changed = False
        selected_list = list(selected)
        for i in range(len(selected_list)):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, len(selected_list)):
                if time.time() - start_time > time_limit:
                    break
                v1, v2 = selected_list[i], selected_list[j]
                S_base = selected - {v1, v2}
                
                # Find cliques not covered by S_base
                S_base_mask = 0
                for v in S_base:
                    S_base_mask |= (1 << v)
                
                unc_idx = []
                for idx, cm in enumerate(clique_masks):
                    if not (S_base_mask & cm):
                        unc_idx.append(idx)
                
                if not unc_idx:
                    # Can remove both
                    selected = S_base
                    changed = True
                    break
                    
                # Need a vertex w that covers all unc_idx cliques
                inter_mask = (1 << n) - 1
                for idx in unc_idx:
                    inter_mask &= clique_masks[idx]
                
                # w must be in inter_mask and not in S_base
                S_base_mask2 = 0
                for v in S_base:
                    S_base_mask2 |= (1 << v)
                
                candidates_mask = inter_mask & ~S_base_mask2
                
                if candidates_mask:
                    w = (candidates_mask & -candidates_mask).bit_length() - 1
                    if w < n:
                        selected = S_base | {w}
                        changed = True
                        break
            if changed:
                break
        if not changed:
            break

    return selected

def _milp_solve(cliques, n, time_limit):
    """Try to solve the hitting set problem using MILP via scipy."""
    m = len(cliques)
    if m == 0:
        return set()
    
    # Build constraint matrix
    A = np.zeros((m, n), dtype=np.int8)
    for i, clique in enumerate(cliques):
        for v in clique:
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
            sol = set(i for i in range(n) if res.x[i] > 0.5)
            # Verify validity
            valid = True
            for c_clique in cliques:
                if not any(v in c_clique for v in sol):
                    valid = False
                    break
            if valid:
                return sol
    except Exception:
        pass
    
    return None

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 30.0:
            # Not enough time left; return trivial solution for remaining
            adj = _generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        
        adj = _generate_graph(n, p, seed)
        
        # Enumerate maximal cliques using bitmask version
        cliques = _bron_kerbosch_bitmask(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # cliques are already bitmasks
        clique_masks = cliques
        
        best_solution = None
        
        # Try MILP if number of cliques is reasonable
        # MILP is exact, so if it finishes, it's the best possible.
        time_left = deadline - time.time() - 10.0
        milp_budget = min(20.0, time_left * 0.5)
        
        if len(cliques) < 800: # Higher threshold with faster BK
            # Convert bitmask cliques to frozensets for MILP
            cliques_frozen = []
            for cm in cliques:
                s = set()
                temp = cm
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    s.add(v)
                cliques_frozen.append(s)
            
            selected_milp = _milp_solve(cliques_frozen, n, milp_budget)
            if selected_milp is not None:
                best_solution = selected_milp
        
        # Run Heuristic
        time_left = deadline - time.time() - 10.0
        if time_left > 1.0:
            # Get initial solution
            if best_solution is None:
                selected_init = _greedy_hitting_set_bitmask(clique_masks, n)
            else:
                selected_init = best_solution
            
            # Polish with local search
            selected_ls = _local_search_bitmask(selected_init, clique_masks, n, time_left)
            
            if best_solution is None or len(selected_ls) < len(best_solution):
                best_solution = selected_ls
        
        if best_solution is None:
            # Fallback: greedy only
            best_solution = _greedy_hitting_set_bitmask(clique_masks, n)
            
        results.append(sorted(best_solution))
    
    return results
# EVOLVE-BLOCK-END
