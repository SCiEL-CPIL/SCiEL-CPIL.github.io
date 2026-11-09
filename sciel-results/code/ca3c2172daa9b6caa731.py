"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sets for fast operations
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    total_edges = len(edge_set)
    
    def get_biclique_edges(L, R):
        """Get the set of edges in a biclique (L, R)."""
        e_set = set()
        for u in L:
            for v in R:
                if u != v:
                    e_set.add((min(u, v), max(u, v)))
        return e_set
    
    def expand_biclique(u, v, adj):
        """Expand to maximal biclique containing seed edge (u, v)."""
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Expand L: add vertices connected to all of R
            new_L = []
            for w in range(n):
                if w not in L and w not in R:
                    if R.issubset(adj[w]):
                        new_L.append(w)
            if new_L:
                L.update(new_L)
                changed = True
            
            # Expand R: add vertices connected to all of L
            new_R = []
            for w in range(n):
                if w not in L and w not in R:
                    if L.issubset(adj[w]):
                        new_R.append(w)
            if new_R:
                R.update(new_R)
                changed = True
        return L, R
    
    def solve_greedy(deadline, seed_offset=0):
        """Greedy biclique partition: repeatedly pick max-coverage biclique."""
        remaining_edges = set(edge_set)
        used_edges = set()
        bicliques = []
        
        deg = [len(adj[i]) for i in range(n)]
        
        while remaining_edges:
            if time.time() > deadline:
                break
                
            best_coverage = 0
            best_L = None
            best_R = None
            
            # Sample seeds from remaining edges
            seed_list = list(remaining_edges)
            random.shuffle(seed_list)
            
            for seed in seed_list:
                u, v = seed
                L, R = expand_biclique(u, v, adj)
                E_B = get_biclique_edges(L, R)
                
                # Check conflict with used edges
                if E_B & used_edges:
                    continue
                
                coverage = len(E_B & remaining_edges)
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_L = L
                    best_R = R
            
            if best_L is not None:
                bicliques.append([sorted(best_L), sorted(best_R)])
                E_B = get_biclique_edges(best_L, best_R)
                remaining_edges -= E_B
                used_edges |= E_B
            else:
                # Fallback: single edge
                seed = next(iter(remaining_edges))
                u, v = seed
                bicliques.append([[u], [v]])
                remaining_edges.discard((min(u, v), max(u, v)))
                used_edges.add((min(u, v), max(u, v)))
        
        return bicliques
    
    def try_merge(bicliques, adj, edge_set, deadline):
        """Try to merge bicliques by finding a biclique that covers edges from multiple bicliques."""
        if not bicliques:
            return bicliques
        
        # Compute edges covered by each biclique
        bic_edges = []
        for L, R in bicliques:
            bic_edges.append(get_biclique_edges(L, R))
        
        total_covered = set()
        for e in bic_edges:
            total_covered |= e
        
        # Try to find merges: for each pair of bicliques, see if there's a larger biclique
        # that covers all edges from both (and possibly more from others)
        improved = True
        max_iter = 50
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            new_bicliques = []
            merged_flags = [False] * len(bicliques)
            
            for i in range(len(bicliques)):
                if merged_flags[i]:
                    continue
                
                # Try to merge biclique i with others
                best_merge = None
                best_merge_size = len(bic_edges[i])
                
                for j in range(len(bicliques)):
                    if i == j or merged_flags[j]:
                        continue
                    
                    # Union of edges from biclique i and j
                    union_edges = bic_edges[i] | bic_edges[j]
                    
                    # Find a biclique that covers all these edges
                    # Use the union of L and R as seed
                    L_union = set(bicliques[i][0]) | set(bicliques[j][0])
                    R_union = set(bicliques[i][1]) | set(bicliques[j][1])
                    
                    # Try expanding from various seeds in union_edges
                    found = False
                    for u, v in list(union_edges)[:20]:  # Limit seeds
                        L_exp, R_exp = expand_biclique(u, v, adj)
                        E_exp = get_biclique_edges(L_exp, R_exp)
                        
                        # Check if E_exp covers union_edges
                        if union_edges.issubset(E_exp):
                            # Check no conflict with other non-merged bicliques
                            conflict = False
                            for k in range(len(bicliques)):
                                if k != i and k != j and not merged_flags[k]:
                                    if E_exp & bic_edges[k]:
                                        conflict = True
                                        break
                            if not conflict:
                                new_size = len(E_exp & total_covered)
                                if new_size > best_merge_size:
                                    best_merge_size = new_size
                                    best_merge = (j, L_exp, R_exp, E_exp)
                                found = True
                                break
                    
                    if found:
                        break
                
                if best_merge is not None:
                    j, L_new, R_new, E_new = best_merge
                    merged_flags[j] = True
                    new_bicliques.append([sorted(L_new), sorted(R_new)])
                    # Update bic_edges for tracking
                    bic_edges[i] = E_new
                    improved = True
                else:
                    new_bicliques.append(bicliques[i])
                    bic_edges[i] = bic_edges[i]  # Keep as is
        
        # Reconstruct bic_edges
        if improved:
            bicliques = new_bicliques
            bic_edges = [get_biclique_edges(L, R) for L, R in bicliques]
        
        return bicliques
    
    def local_search(bicliques, adj, edge_set, deadline):
        """Local search: try moving vertices between sides or swapping bicliques."""
        if not bicliques:
            return bicliques
        
        best_count = len(bicliques)
        current = bicliques
        current_edges = [get_biclique_edges(L, R) for L, R in current]
        
        # Try to merge any pair of bicliques into one
        for _ in range(20):
            if time.time() > deadline:
                break
            
            improved = False
            new_current = [list(b) for b in current]
            new_edges = [set(e) for e in current_edges]
            
            for i in range(len(new_current)):
                for j in range(i + 1, len(new_current)):
                    L_i, R_i = new_current[i]
                    L_j, R_j = new_current[j]
                    
                    # Try all 4 combinations of merging
                    candidates = [
                        (set(L_i) | set(L_j), set(R_i) | set(R_j)),
                        (set(L_i) | set(R_j), set(R_i) | set(L_j)),
                        (set(L_i) | set(L_j), set(R_i) & set(R_j)),
                        (set(L_i) & set(L_j), set(R_i) | set(R_j)),
                    ]
                    
                    for L_new, R_new in candidates:
                        if not L_new or not R_new:
                            continue
                        E_new = get_biclique_edges(L_new, R_new)
                        
                        # Must cover all edges from both bicliques
                        if not (new_edges[i] | new_edges[j]).issubset(E_new):
                            continue
                        
                        # No conflict with others
                        conflict = False
                        for k in range(len(new_current)):
                            if k != i and k != j:
                                if E_new & new_edges[k]:
                                    conflict = True
                                    break
                        if conflict:
                            continue
                        
                        # Valid merge
                        new_current.pop(j)
                        new_edges.pop(j)
                        if i > j:
                            i -= 1
                        new_current[i] = [sorted(L_new), sorted(R_new)]
                        new_edges[i] = E_new
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                current = new_current
                current_edges = new_edges
                if len(current) < best_count:
                    best_count = len(current)
            else:
                break
        
        return current
    
    # Main search
    start_time = time.time()
    deadline = start_time + 70  # Safety margin
    
    best_result = None
    best_count = float('inf')
    
    # Run multiple restarts with different random seeds
    for trial in range(15):
        if time.time() > deadline:
            break
        
        random.seed(42 + trial)
        
        # Phase 1: Greedy extraction
        bicliques = solve_greedy(deadline, trial)
        
        if not bicliques:
            continue
        
        # Validate
        covered = set()
        valid = True
        for L, R in bicliques:
            if not L or not R:
                valid = False
                break
            E_B = get_biclique_edges(L, R)
            if E_B & covered:
                valid = False
                break
            if not E_B.issubset(edge_set):
                valid = False
                break
            covered |= E_B
        
        if not valid or covered != edge_set:
            continue
        
        # Phase 2: Try merges
        bicliques = try_merge(bicliques, adj, edge_set, deadline)
        
        # Re-validate after merge
        covered = set()
        valid = True
        for L, R in bicliques:
            if not L or not R:
                valid = False
                break
            E_B = get_biclique_edges(L, R)
            if E_B & covered:
                valid = False
                break
            if not E_B.issubset(edge_set):
                valid = False
                break
            covered |= E_B
        
        if not valid or covered != edge_set:
            continue
        
        # Phase 3: Local search
        bicliques = local_search(bicliques, adj, edge_set, deadline)
        
        # Final validation
        covered = set()
        valid = True
        for L, R in bicliques:
            if not L or not R:
                valid = False
                break
            E_B = get_biclique_edges(L, R)
            if E_B & covered:
                valid = False
                break
            if not E_B.issubset(edge_set):
                valid = False
                break
            covered |= E_B
        
        if not valid or covered != edge_set:
            continue
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_result = bicliques
    
    if best_result is None:
        # Emergency fallback: single edges
        best_result = []
        for u, v in edges:
            best_result.append([[u], [v]])
    
    # Convert to required format
    final_result = []
    for bic in best_result:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
    
    return final_result
# EVOLVE-BLOCK-END
