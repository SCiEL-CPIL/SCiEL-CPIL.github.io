# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _maximal_cliques(adj, n):
    """Bron-Kerbosch with pivot to find all maximal cliques."""
    results = []
    def bronk(R, P_set, X):
        if P_set == 0 and X == 0:
            results.append(R)
            return
        # Pick pivot from P_set | X to minimize branching
        PX = P_set | X
        pivot = -1
        best_count = -1
        temp = PX
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp ^= (1 << v)
            cnt = bin(adj[v] & P_set).count('1')
            if cnt > best_count:
                best_count = cnt
                pivot = v
        # Branch on vertices in P_set not adjacent to pivot
        candidates = P_set & ~(adj[pivot] if pivot >= 0 else 0)
        while candidates:
            v = (candidates & -candidates).bit_length() - 1
            vm = 1 << v
            candidates ^= vm
            bronk(R | vm, P_set & adj[v], X & adj[v])
            P_set ^= vm
            X |= vm
    bronk(0, (1 << n) - 1, 0)
    return results

def _maximal_independent_sets(adj, n):
    """Maximal independent sets = maximal cliques in complement graph."""
    comp_adj = [0] * n
    for v in range(n):
        comp_adj[v] = ((1 << n) - 1) ^ (1 << v) ^ adj[v]
    return _maximal_cliques(comp_adj, n)

def _greedy_partition(adj):
    """Greedy heuristic to get an initial upper bound."""
    n = P['n']
    parts = []
    part_masks = []
    for v in range(n):
        vm = 1 << v
        best_idx = -1
        for i, pm in enumerate(part_masks):
            trial = pm | vm
            # Check if trial is clique or independent
            is_clique = True
            is_indep = True
            for u in range(n):
                if trial & (1 << u):
                    if (adj[u] & trial) != (trial ^ (1 << u)):
                        is_clique = False
                    if adj[u] & trial:
                        is_indep = False
                    if not is_clique and not is_indep:
                        break
            if is_clique or is_indep:
                best_idx = i
                break
        if best_idx >= 0:
            part_masks[best_idx] |= vm
        else:
            part_masks.append(vm)
    result = []
    for pm in part_masks:
        lst = [v for v in range(n) if pm & (1 << v)]
        result.append(lst)
    return result

def _solve_set_cover(sets, n, deadline, initial_bound):
    """
    Solve minimum set cover using Algorithm X / Dancing Links style backtracking.
    sets: list of bitmasks (candidate sets)
    n: number of vertices (universe size)
    deadline: time limit
    initial_bound: initial upper bound on number of sets
    Returns list of set indices forming the cover, or None.
    """
    full_mask = (1 << n) - 1
    num_sets = len(sets)
    
    # Precompute for each vertex which sets contain it
    vertex_to_sets = [0] * n  # bitmask of set indices
    for i, s in enumerate(sets):
        if i >= 64:
            # Fallback for large set counts (unlikely for n=14)
            pass
        for v in range(n):
            if s & (1 << v):
                if i < 64:
                    vertex_to_sets[v] |= (1 << i)
    
    # If any vertex can't be covered, fail
    for v in range(n):
        if vertex_to_sets[v] == 0:
            return None
    
    # Sort sets by size (larger first) for better pruning
    set_order = sorted(range(num_sets), key=lambda i: bin(sets[i]).count('1'), reverse=True)
    # Remap: new_index -> old_index
    # Actually, let's just work with the sorted order directly
    
    best_solution = None
    best_count = initial_bound
    
    # Use iterative deepening: try 1, 2, 3, ... up to initial_bound
    # But start from a good lower bound
    # Lower bound: max over vertices of 1 (trivial), or clique/independent number based
    
    # For Algorithm X style: pick the most constrained vertex (fewest candidate sets)
    # and branch on which set covers it
    
    solution_stack = []
    
    def search(uncovered, depth, max_depth):
        nonlocal best_solution, best_count
        
        if time.time() > deadline:
            return False
        
        if uncovered == 0:
            if depth < best_count:
                best_count = depth
                best_solution = list(solution_stack)
            return True
        
        if depth >= best_count:
            return False
        
        # Lower bound: need at least 1 more set
        if depth + 1 > best_count:
            return False
        
        # Pick the most constrained uncovered vertex
        # (fewest sets that can cover it among remaining)
        best_v = -1
        best_candidates = None
        best_count_cand = float('inf')
        
        temp = uncovered
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp ^= (1 << v)
            # Count sets in vertex_to_sets[v]
            cnt = bin(vertex_to_sets[v]).count('1')
            if cnt < best_count_cand:
                best_count_cand = cnt
                best_v = v
                best_candidates = vertex_to_sets[v]
                if cnt <= 1:
                    break
        
        if best_v < 0:
            return False
        
        # Branch on each set that covers best_v
        cand = best_candidates
        while cand:
            si = (cand & -cand).bit_length() - 1
            cand ^= (1 << si)
            
            if time.time() > deadline:
                return False
            
            solution_stack.append(si)
            if search(uncovered & ~sets[si], depth + 1, max_depth):
                solution_stack.pop()
                return True
            solution_stack.pop()
        
        return False
    
    # Try from 1 up to initial_bound
    for k in range(1, initial_bound + 1):
        best_count = k  # We're trying to find solution with exactly k sets
        solution_stack = []
        if search(full_mask, 0, k):
            return best_solution
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    adj = _build_graph()
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Get greedy upper bound
    greedy_parts = _greedy_partition(adj)
    greedy_count = len(greedy_parts)
    
    # Find all maximal cliques and maximal independent sets
    max_cliques = _maximal_cliques(adj, n)
    max_indeps = _maximal_independent_sets(adj, n)
    
    # Combine: candidate sets are all maximal cliques and maximal independent sets
    all_sets = []
    seen_masks = set()
    for mc in max_cliques:
        if mc not in seen_masks and mc != 0:
            all_sets.append(mc)
            seen_masks.add(mc)
    for mi in max_indeps:
        if mi not in seen_masks and mi != 0:
            all_sets.append(mi)
            seen_masks.add(mi)
    
    # Also add singletons as fallback (should be covered by maximal sets, but just in case)
    for v in range(n):
        vm = 1 << v
        if vm not in seen_masks:
            all_sets.append(vm)
            seen_masks.add(vm)
    
    # Solve set cover
    result_indices = _solve_set_cover(all_sets, n, deadline, greedy_count)
    
    if result_indices is not None:
        # Build partition from selected sets
        part_masks = [all_sets[i] for i in result_indices]
        result = []
        for pm in part_masks:
            lst = [v for v in range(n) if pm & (1 << v)]
            if lst:  # Nonempty
                result.append(lst)
    else:
        result = greedy_parts
    
    # Validation and cleanup
    if not isinstance(result, list):
        result = greedy_parts
    
    final_result = []
    seen = set()
    valid = True
    for part in result:
        if not isinstance(part, (list, tuple)):
            valid = False
            break
        if len(part) == 0:
            valid = False
            break
        for v in part:
            if not isinstance(v, int) or isinstance(v, bool):
                valid = False
                break
            if v in seen:
                valid = False
                break
            seen.add(v)
        if not valid:
            break
            
    if not valid or len(seen) != P['n']:
        result = greedy_parts
        final_result = []
        for part in result:
            final_result.append([int(v) for v in part])
    else:
        for part in result:
            final_result.append([int(v) for v in part])
            
    return final_result
# EVOLVE-BLOCK-END
