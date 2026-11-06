import time
import random
import sys
from collections import defaultdict

def _build_graph(n, p, seed):
    """Build adjacency lists from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        U = P | X
        if not U:
            return
        # Pick pivot with max degree in P (heuristic)
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
            
    bk(set(), set(range(n)), set())
    return cliques

def _greedy_hitting_set(vertex_covers, num_cliques):
    """Greedy set cover to get an initial hitting set."""
    n = len(vertex_covers)
    sol_mask = 0
    remaining = (1 << num_cliques) - 1
    
    while remaining:
        best_v = -1
        best_count = 0
        for v in range(n):
            if (sol_mask & (1 << v)) != 0:
                continue
            count = (vertex_covers[v] & remaining).bit_count()
            if count > best_count:
                best_count = count
                best_v = v
        if best_v == -1 or best_count == 0:
            break
        sol_mask |= (1 << best_v)
        remaining &= ~vertex_covers[best_v]
    
    return sol_mask

def _greedy_packing_lower_bound(clique_vertices, num_cliques):
    """
    Compute a lower bound on the hitting set size by greedily packing
    independent cliques (cliques that share no vertices).
    Each independent clique requires at least one distinct vertex in the hitting set.
    """
    remaining_cliques = set(range(num_cliques))
    count = 0
    
    while remaining_cliques:
        # Pick the clique with fewest vertices
        best_clique = min(remaining_cliques, key=lambda i: len(clique_vertices[i]))
        count += 1
        # Remove all cliques that share a vertex with best_clique
        used_vertices = set(clique_vertices[best_clique])
        to_remove = set()
        for c in remaining_cliques:
            if c == best_clique:
                continue
            if used_vertices & set(clique_vertices[c]):
                to_remove.add(c)
        remaining_cliques -= to_remove
        remaining_cliques.discard(best_clique)
    
    return count

def _local_search_polish(sol_mask, compact_covers, num_cliques, clique_vertices, max_iter=2000):
    """
    Try to improve a hitting set via local search:
    1. Remove redundant vertices
    2. Try 1-for-1 swaps that enable further removals
    """
    n = len(compact_covers)
    sol = set()
    for i in range(n):
        if (sol_mask & (1 << i)) != 0:
            sol.add(i)
    
    def is_valid(s):
        for i in range(num_cliques):
            has_hit = False
            for v in s:
                if (compact_covers[v] & (1 << i)) != 0:
                    has_hit = True
                    break
            if not has_hit:
                return False
        return True
    
    # Phase 1: Remove redundant vertices
    changed = True
    while changed:
        changed = False
        for v in list(sol):
            sol_test = sol - {v}
            if is_valid(sol_test):
                sol = sol_test
                changed = True
    
    # Phase 2: Try swaps to enable further removals
    for _ in range(max_iter):
        improved = False
        for v_out in list(sol):
            sol_test = sol - {v_out}
            if is_valid(sol_test):
                sol = sol_test
                improved = True
                break
            # Find uncovered cliques
            uncovered = []
            for i in range(num_cliques):
                has_hit = False
                for v in sol_test:
                    if (compact_covers[v] & (1 << i)) != 0:
                        has_hit = True
                        break
                if not has_hit:
                    uncovered.append(i)
            
            if not uncovered:
                sol = sol_test
                improved = True
                break
            
            # Try to find a vertex that covers all uncovered cliques
            for v_in in range(n):
                if v_in in sol_test:
                    continue
                covers_all = True
                for i in uncovered:
                    if (compact_covers[v_in] & (1 << i)) == 0:
                        covers_all = False
                        break
                if covers_all:
                    new_sol = sol_test | {v_in}
                    # Try to remove more from new_sol
                    for v_rem in list(new_sol):
                        if v_rem == v_in:
                            continue
                        sol_test2 = new_sol - {v_rem}
                        if is_valid(sol_test2):
                            sol = sol_test2
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            break
    
    return frozenset(sol)

def _solve_instance(n, p, seed, time_budget):
    """
    Solve the minimum hitting set problem for maximal cliques of a random graph.
    Uses branch-and-bound with constraint propagation and packing lower bound.
    """
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []

    num_cliques = len(cliques)
    
    # Precompute for each vertex, the bitmask of cliques it covers
    vertex_covers = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_covers[v] |= bit
    
    # Remove vertices that don't cover any clique
    active_vertices = [v for v in range(n) if vertex_covers[v] != 0]
    if not active_vertices:
        return []
    
    # Remap to compact indices
    v_to_compact = {v: i for i, v in enumerate(active_vertices)}
    compact_n = len(active_vertices)
    compact_covers = [0] * compact_n
    for v in active_vertices:
        compact_covers[v_to_compact[v]] = vertex_covers[v]
    
    # Get initial upper bound from greedy
    greedy_sol = _greedy_hitting_set(compact_covers, num_cliques)
    greedy_size = greedy_sol.bit_count()
    
    # If greedy is very small, try to prove optimality or improve
    if greedy_size <= 1:
        return [active_vertices[i] for i in range(compact_n) if (greedy_sol & (1 << i))]
    
    # Precompute: for each clique, list of vertices that cover it
    clique_vertices = [[] for _ in range(num_cliques)]
    for v in range(compact_n):
        c = compact_covers[v]
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            clique_vertices[i].append(v)
    
    # Compute packing lower bound
    packing_lb = _greedy_packing_lower_bound(clique_vertices, num_cliques)
    
    best_size = [greedy_size]
    best_sol = [greedy_sol]
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Precompute for faster lower bound computation
    # For each clique, precompute the set of vertices as a bitmask
    clique_vertex_masks = [0] * num_cliques
    for i in range(num_cliques):
        mask = 0
        for v in clique_vertices[i]:
            mask |= (1 << v)
        clique_vertex_masks[i] = mask
    
    def compute_packing_lb(remaining_cliques_mask, selected_mask):
        """Compute packing lower bound considering already selected vertices."""
        if remaining_cliques_mask == 0:
            return 0
        
        # Get list of remaining cliques
        remaining_list = []
        c = remaining_cliques_mask
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            remaining_list.append(i)
        
        if not remaining_list:
            return 0
        
        # Greedily pack independent cliques
        remaining_set = set(remaining_list)
        count = 0
        
        while remaining_set:
            # Pick the clique with fewest available (unselected) vertices
            best_clique = -1
            best_avail = float('inf')
            for i in remaining_set:
                avail = (clique_vertex_masks[i] & ~selected_mask).bit_count()
                if avail < best_avail:
                    best_avail = avail
                    best_clique = i
            
            if best_clique == -1 or best_avail == 0:
                return count + 1  # Conservative: at least 1 more needed
            
            count += 1
            used_vertices = clique_vertex_masks[best_clique]
            to_remove = set()
            for i in remaining_set:
                if i == best_clique:
                    continue
                if used_vertices & clique_vertex_masks[i]:
                    to_remove.add(i)
            remaining_set -= to_remove
            remaining_set.discard(best_clique)
        
        return count
    
    def bb(selected_mask, remaining_cliques_mask):
        """
        Branch and bound for minimum hitting set.
        selected_mask: bitmask of vertices already selected
        remaining_cliques_mask: bitmask of cliques not yet hit
        """
        if time.time() > deadline:
            return
        
        if remaining_cliques_mask == 0:
            size = selected_mask.bit_count()
            if size < best_size[0]:
                best_size[0] = size
                best_sol[0] = selected_mask
            return
        
        current_size = selected_mask.bit_count()
        
        # Stronger lower bound using packing
        lb = compute_packing_lb(remaining_cliques_mask, selected_mask)
        if current_size + lb >= best_size[0]:
            return
        
        # Find the remaining clique with fewest candidate vertices
        best_clique = -1
        best_count = float('inf')
        
        c = remaining_cliques_mask
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            count = (clique_vertex_masks[i] & ~selected_mask).bit_count()
            if count < best_count:
                best_count = count
                best_clique = i
                if count == 1:
                    break
        
        if best_clique == -1:
            return
        
        if best_count == 0:
            return
        
        # Branch on each vertex in the most constrained clique
        candidates = []
        avail_mask = clique_vertex_masks[best_clique] & ~selected_mask
        m = avail_mask
        while m:
            v = (m & -m).bit_length() - 1
            m &= m - 1
            candidates.append(v)
        
        # Sort candidates by how many remaining cliques they cover (most first)
        candidates.sort(key=lambda v: (compact_covers[v] & remaining_cliques_mask).bit_count(), reverse=True)
        
        for v in candidates:
            if time.time() > deadline:
                return
            
            new_selected = selected_mask | (1 << v)
            new_remaining = remaining_cliques_mask & ~compact_covers[v]
            
            if new_selected.bit_count() >= best_size[0]:
                continue
            
            bb(new_selected, new_remaining)
    
    # Run branch and bound
    bb(0, (1 << num_cliques) - 1)
    
    # Apply local search polish
    best_frozen = _local_search_polish(best_sol[0], compact_covers, num_cliques, clique_vertices, max_iter=1000)
    
    # Check if local search improved
    if len(best_frozen) < best_size[0]:
        new_mask = 0
        for v in best_frozen:
            new_mask |= (1 << v)
        best_size[0] = len(best_frozen)
        best_sol[0] = new_mask
    
    # Map back to original vertex indices
    result = []
    for i in range(compact_n):
        if (best_sol[0] & (1 << i)) != 0:
            result.append(active_vertices[i])
    
    return sorted(result)

def _local_search_improve(sol_set, cliques, adj, max_iter=5000):
    """
    Try to improve a hitting set via local search: remove vertices if possible,
    then swap one out for another if it reduces size.
    """
    sol = set(sol_set)
    
    # First, try to remove any vertex whose removal still keeps all cliques hit
    for v in list(sol):
        sol_test = sol - {v}
        ok = True
        for clique in cliques:
            if not (clique & sol_test):
                ok = False
                break
        if ok:
            sol = sol_test
    
    return sol

def construct():
    deadline = time.time() + 145  # Safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for i, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            # Fallback: just return greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            # Simple greedy
            sol = []
            for clique in cliques:
                if not any(v in sol for v in clique):
                    v = next(iter(clique))
                    sol.append(v)
            results.append(sorted(set(sol)))
            continue
         
        # Allocate time: give more to earlier instances, but ensure all get some
        if i == 0:
            alloc = min(remaining_time - 10, 60)
        elif i == 1:
            alloc = min(remaining_time - 5, 50)
        else:
            alloc = min(remaining_time - 2, 40)
        
        sol = _solve_instance(n, p, seed, alloc)
        results.append(sol)
        
    return results
