"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_edge_set():
    es = set()
    for u, v in P['edges']:
        es.add((min(u,v), max(u,v)))
    return es

def _is_biclique(L, R, edge_set):
    """Check if all cross pairs between L and R are edges."""
    for u in L:
        for v in R:
            if (min(u,v), max(u,v)) not in edge_set:
                return False
    return True

def _find_maximal_bicliques(edge_set, deadline, max_bicliques=5000):
    """Find many maximal bicliques using a greedy expansion approach from multiple seeds."""
    n = P['n']
    vertices = list(range(n))
    all_vertices_set = set(vertices)
    
    # Build adjacency for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    bicliques = []
    seen_bicliques = set()
    
    # Try expanding from various seed pairs
    seed_pairs = []
    for u, v in edge_set:
        seed_pairs.append((u, v))
    
    # Also try expanding from single vertices with high degree
    for v in vertices:
        if len(adj[v]) >= 2:
            for w in list(adj[v])[:5]:
                seed_pairs.append((v, w))
    
    random.shuffle(seed_pairs)
    
    for seed_u, seed_v in seed_pairs:
        if time.time() > deadline or len(bicliques) >= max_bicliques:
            break
        
        # Start with L={seed_u}, R={seed_v}
        L = {seed_u}
        R = {seed_v}
        
        # Expand: try to add vertices to L or R
        changed = True
        while changed:
            changed = False
            
            # Try adding to L: vertex x not in L∪R such that x connects to all in R
            for x in vertices:
                if x in L or x in R:
                    continue
                if all(x in adj[r] for r in R):
                    L.add(x)
                    changed = True
            
            # Try adding to R: vertex x not in L∪R such that x connects to all in L
            for x in vertices:
                if x in L or x in R:
                    continue
                if all(x in adj[l] for l in L):
                    R.add(x)
                    changed = True
        
        # Check if this is a valid biclique
        if L and R and _is_biclique(list(L), list(R), edge_set):
            key = (frozenset(L), frozenset(R))
            # Also check the swapped version
            key2 = (frozenset(R), frozenset(L))
            if key not in seen_bicliques and key2 not in seen_bicliques:
                seen_bicliques.add(key)
                seen_bicliques.add(key2)
                bicliques.append((frozenset(L), frozenset(R)))
        
        # Also try the swapped orientation
        L2, R2 = R, L
        key = (frozenset(L2), frozenset(R2))
        key2 = (frozenset(R2), frozenset(L2))
        if key not in seen_bicliques and key2 not in seen_bicliques:
            seen_bicliques.add(key)
            seen_bicliques.add(key2)
            bicliques.append((frozenset(L2), frozenset(R2)))
    
    return bicliques

def _greedy_biclique_partition(edge_set, deadline):
    """Greedy: repeatedly find largest biclique, remove its edges, recurse."""
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        # Find a good biclique from remaining edges
        best_biclique = None
        best_size = 0
        
        # Get all vertices involved in remaining edges
        verts = set()
        for u, v in remaining:
            verts.add(u)
            verts.add(v)
        verts = list(verts)
        
        # Try each vertex as a starting point for left side
        for start in verts:
            if time.time() > deadline:
                break
            # Collect all neighbors of start in remaining edges
            neighbors = set()
            for u, v in remaining:
                if u == start:
                    neighbors.add(v)
                elif v == start:
                    neighbors.add(u)
            
            if not neighbors:
                continue
            
            # Try to grow a biclique
            L = [start]
            R = list(neighbors)
            
            # Grow L
            changed = True
            while changed:
                changed = False
                candidates = set(verts) - set(L)
                for cand in list(candidates):
                    if cand in R:
                        continue
                    all_connect = True
                    for r in R:
                        if (min(cand, r), max(cand, r)) not in remaining:
                            all_connect = False
                            break
                    if all_connect:
                        L.append(cand)
                        changed = True
            
            # Grow R
            R_set = set(R)
            changed = True
            while changed:
                changed = False
                candidates = set(verts) - set(L)
                for cand in list(candidates):
                    if cand in R_set:
                        continue
                    all_connect = True
                    for l in L:
                        if (min(cand, l), max(cand, l)) not in remaining:
                            all_connect = False
                            break
                    if all_connect:
                        R_set.add(cand)
                        changed = True
            R = list(R_set)
            
            if len(L) * len(R) > best_size:
                best_size = len(L) * len(R)
                best_biclique = (L, R)
        
        if best_biclique is None or best_size == 0:
            u, v = next(iter(remaining))
            bicliques.append(([u], [v]))
            remaining.discard((u, v))
        else:
            L, R = best_biclique
            bicliques.append((L, R))
            for u in L:
                for v in R:
                    remaining.discard((min(u,v), max(u,v)))
    
    return bicliques

def _set_cover_search(edge_set, candidate_bicliques, deadline, initial_bound=None):
    """
    Solve minimum set cover: find minimum number of bicliques whose edge sets 
    partition all edges. Uses backtracking with memoization.
    """
    # Map each edge to an index
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    num_edges = len(edge_list)
    
    # For each candidate biclique, compute the set of edge indices it covers
    biclique_edges = []
    for L, R in candidate_bicliques:
        edges_covered = set()
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e in edge_to_idx:
                    edges_covered.add(edge_to_idx[e])
        if edges_covered:
            biclique_edges.append(frozenset(edges_covered))
    
    # Remove duplicate bicliques (by edge set)
    unique_bicliques = list(set(biclique_edges))
    
    # For each edge, find which bicliques cover it
    edge_to_bicliques = [set() for _ in range(num_edges)]
    for i, be in enumerate(unique_bicliques):
        for e in be:
            edge_to_bicliques[e].add(i)
    
    # Lower bound: at least 1 if there are edges
    # Upper bound from greedy
    def greedy_cover():
        covered = set()
        chosen = []
        remaining_edges = set(range(num_edges))
        while remaining_edges:
            best_bi = None
            best_count = 0
            for bi in range(len(unique_bicliques)):
                cnt = len(unique_bicliques[bi] & remaining_edges)
                if cnt > best_count:
                    best_count = cnt
                    best_bi = bi
            if best_bi is None:
                break
            chosen.append(best_bi)
            covered |= unique_bicliques[best_bi]
            remaining_edges -= unique_bicliques[best_bi]
        return len(chosen)
    
    greedy_bound = greedy_cover()
    if initial_bound is not None:
        greedy_bound = min(greedy_bound, initial_bound)
    
    best_solution = []
    best_size = greedy_bound
    
    # Memoization: state is frozenset of uncovered edges
    memo = {}
    
    def backtrack(uncovered, current_solution, depth):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
        
        if not uncovered:
            if depth < best_size:
                best_size = depth
                best_solution = list(current_solution)
            return
        
        # Pruning: if depth >= best_size, no point continuing
        if depth >= best_size:
            return
        
        # Lower bound: we need at least ceil(len(uncovered) / max_coverage) more bicliques
        max_cov = max(len(unique_bicliques[i] & uncovered) for i in range(len(unique_bicliques)) if unique_bicliques[i] & uncovered) if any(unique_bicliques[i] & uncovered for i in range(len(unique_bicliques))) else 0
        if max_cov == 0:
            return
        lb = (len(uncovered) + max_cov - 1) // max_cov
        if depth + lb >= best_size:
            return
        
        state = frozenset(uncovered)
        if state in memo:
            if memo[state] <= depth:
                return
        else:
            memo[state] = depth
        
        # Choose the edge with fewest covering bicliques (most constrained)
        min_count = float('inf')
        min_edge = None
        for e in uncovered:
            cnt = len(edge_to_bicliques[e] & set(range(len(unique_bicliques))))
            # Only count bicliques that actually cover this edge and are in our candidate set
            actual_cnt = 0
            for bi in edge_to_bicliques[e]:
                if bi < len(unique_bicliques) and e in unique_bicliques[bi]:
                    actual_cnt += 1
            if actual_cnt < min_count:
                min_count = actual_cnt
                min_edge = e
        
        if min_edge is None or min_count == 0:
            return
        
        # Try each biclique covering min_edge
        candidates = sorted(edge_to_bicliques[min_edge], key=lambda bi: -len(unique_bicliques[bi] & uncovered))
        
        for bi in candidates:
            if bi >= len(unique_bicliques):
                continue
            new_uncovered = uncovered - unique_bicliques[bi]
            current_solution.append(bi)
            backtrack(new_uncovered, current_solution, depth + 1)
            current_solution.pop()
    
    all_edges = frozenset(range(num_edges))
    backtrack(all_edges, [], 0)
    
    if best_solution:
        return [unique_bicliques[i] for i in best_solution]
    return None

def _edges_to_bicliques(edge_indices_list, edge_list, vertex_map=None):
    """Convert edge index sets back to biclique vertex sets."""
    # We need to reconstruct bicliques from edge sets
    # This is tricky; instead, we should keep track of the original biclique vertex sets
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    edge_set = _build_edge_set()
    num_edges = len(edge_set)
    
    # Phase 1: Greedy partition for initial upper bound
    greedy_result = _greedy_biclique_partition(edge_set, deadline)
    greedy_count = len(greedy_result)
    
    # Phase 2: Find many candidate bicliques
    candidates = _find_maximal_bicliques(edge_set, deadline, max_bicliques=3000)
    
    # Map candidates to edge index sets
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    biclique_edge_sets = []
    biclique_vertex_sets = []
    seen_edge_sets = set()
    for L, R in candidates:
        edges_covered = set()
        valid = True
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e in edge_to_idx:
                    edges_covered.add(edge_to_idx[e])
                else:
                    valid = False
                    break
            if not valid:
                break
        if valid and edges_covered:
            key = frozenset(edges_covered)
            if key not in seen_edge_sets:
                seen_edge_sets.add(key)
                biclique_edge_sets.append(key)
                biclique_vertex_sets.append((L, R))
    
    # Phase 3: Set cover search
    best_bicliques = None
    if biclique_edge_sets:
        # Build edge-to-biclique mapping
        num_cand = len(biclique_edge_sets)
        edge_to_bicliques = [set() for _ in range(num_edges)]
        for i, be in enumerate(biclique_edge_sets):
            for e in be:
                edge_to_bicliques[e].add(i)
        
        # Greedy upper bound
        def greedy_cover_idx():
            remaining = set(range(num_edges))
            chosen = []
            while remaining:
                best_bi = None
                best_count = 0
                for bi in range(num_cand):
                    cnt = len(biclique_edge_sets[bi] & remaining)
                    if cnt > best_count:
                        best_count = cnt
                        best_bi = bi
                if best_bi is None:
                    break
                chosen.append(best_bi)
                remaining -= biclique_edge_sets[best_bi]
            return chosen
        
        greedy_chosen = greedy_cover_idx()
        best_size = len(greedy_chosen)
        best_solution = list(greedy_chosen)
        
        memo = {}
        
        def backtrack(uncovered, current_solution, depth):
            nonlocal best_solution, best_size
            
            if time.time() > deadline:
                return
            
            if not uncovered:
                if depth < best_size:
                    best_size = depth
                    best_solution = list(current_solution)
                return
            
            if depth >= best_size:
                return
            
            # Lower bound
            max_cov = 0
            for i in range(num_cand):
                cov = len(biclique_edge_sets[i] & uncovered)
                if cov > max_cov:
                    max_cov = cov
            if max_cov == 0:
                return
            lb = (len(uncovered) + max_cov - 1) // max_cov
            if depth + lb >= best_size:
                return
            
            state = frozenset(uncovered)
            if state in memo:
                if memo[state] <= depth:
                    return
            else:
                memo[state] = depth
            
            # Most constrained edge
            min_count = float('inf')
            min_edge = None
            for e in uncovered:
                actual_cnt = len(edge_to_bicliques[e])
                if actual_cnt < min_count:
                    min_count = actual_cnt
                    min_edge = e
            
            if min_edge is None or min_count == 0:
                return
            
            candidates_for_edge = sorted(edge_to_bicliques[min_edge], 
                                         key=lambda bi: -len(biclique_edge_sets[bi] & uncovered))
            
            for bi in candidates_for_edge:
                new_uncovered = uncovered - biclique_edge_sets[bi]
                current_solution.append(bi)
                backtrack(new_uncovered, current_solution, depth + 1)
                current_solution.pop()
        
        all_edges_frozen = frozenset(range(num_edges))
        backtrack(all_edges_frozen, [], 0)
        
        if best_solution:
            best_bicliques = [biclique_vertex_sets[i] for i in best_solution]
    
    # Use the best result
    if best_bicliques and len(best_bicliques) < greedy_count:
        result_bicliques = best_bicliques
    else:
        result_bicliques = greedy_result
    
    # Convert to proper format
    result = []
    for L, R in result_bicliques:
        L_list = [int(v) for v in L]
        R_list = [int(v) for v in R]
        if L_list and R_list:
            result.append([L_list, R_list])
    
    return result
# EVOLVE-BLOCK-END
