"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _generate_bicliques(adj, n, max_size=4):
    """
    Generate candidate bicliques up to a certain size.
    We focus on small bicliques (stars, paths, small cliques) because they are
    more likely to be part of an optimal partition in sparse-ish random graphs,
    and the number of candidates grows exponentially with size.
    """
    candidates = []
    edge_set = set(P['edges'])
    
    # 1. Single edges (trivial bicliques)
    for u, v in P['edges']:
        candidates.append(([u], [v]))
        
    # 2. Stars centered at each vertex (up to a limit to avoid explosion)
    # A star is a biclique [center, neighbors].
    # We can choose any subset of neighbors.
    # To keep it manageable, we generate stars of size 2 and 3 (total 3 vertices in biclique)
    # and maybe some larger ones if the degree is low.
    
    for center in range(n):
        neighbors = [v for v in adj[center] if v > center] # Avoid duplicates? No, direction matters for L/R but set is same.
        # Actually, for a star, L={center}, R={neighbors}.
        # We should consider all subsets of neighbors of size 2 and 3.
        deg = len(neighbors)
        if deg >= 2:
            # Subsets of size 2
            for i in range(deg):
                for j in range(i+1, deg):
                    r = [neighbors[i], neighbors[j]]
                    candidates.append(([center], r))
        if deg >= 3:
            # Subsets of size 3
            for i in range(deg):
                for j in range(i+1, deg):
                    for k in range(j+1, deg):
                        r = [neighbors[i], neighbors[j], neighbors[k]]
                        candidates.append(([center], r))
                        
    # 3. Small cliques (K3, K4)
    # A clique of size k is a biclique [V, V] ? No, a clique is a complete bipartite graph only if k=2.
    # Wait, a biclique K_{a,b} requires all cross edges.
    # A triangle (K3) is NOT a biclique unless one side is size 1 (star) or 2x2 (K4).
    # So we don't need to explicitly look for cliques, they are covered by stars if we allow arbitrary R.
    # However, a K4 is a biclique [2, 2].
    # Let's check for K4s.
    for u in range(n):
        for v in adj[u]:
            if v <= u: continue
            for w in adj[v]:
                if w <= v: continue
                if w in adj[u]:
                    # Triangle u-v-w. Not a biclique unless one is isolated in the subgraph.
                    # But if we have a 4th vertex x connected to u,v,w, then {u,v} x {w,x} is a biclique.
                    pass
    
    # Let's add specific K_{2,2} checks (cycles of length 4)
    # This might be too slow if done naively.
    # Given the deadline, we stick to Stars and Single Edges. 
    # Stars are very powerful for covering edges.
    
    # Deduplicate candidates
    unique_candidates = set()
    final_candidates = []
    for l, r in candidates:
        # Normalize: sort L, sort R. Ensure no overlap.
        l_sorted = tuple(sorted(l))
        r_sorted = tuple(sorted(r))
        if not set(l_sorted).isdisjoint(set(r_sorted)):
            continue
        key = (l_sorted, r_sorted)
        if key not in unique_candidates:
            unique_candidates.add(key)
            final_candidates.append((list(l_sorted), list(r_sorted)))
            
    return final_candidates

def _greedy_partition(adj, n):
    """
    Greedy fallback: Pick the largest available biclique, remove its edges, repeat.
    """
    remaining_edges = set(P['edges'])
    if not remaining_edges:
        return []
        
    # Pre-compute all possible bicliques? Too expensive.
    # Instead, greedily grow bicliques from remaining edges.
    bicliques = []
    
    while remaining_edges:
        # Pick an edge
        u, v = next(iter(remaining_edges))
        
        # Start with biclique {u}, {v}
        L = {u}
        R = {v}
        
        # Try to expand L and R
        # A simple heuristic: 
        # 1. Find common neighbors of L and R? No, that's for finding a maximal biclique.
        # Let's just find a maximal biclique containing (u,v) within remaining_edges.
        
        # Current biclique B = (L, R).
        # An edge (x, y) is in B if x in L, y in R.
        # We want to add vertices to L or R such that all cross edges exist in remaining_edges.
        
        # Iterative expansion
        changed = True
        while changed:
            changed = False
            # Try adding to L
            candidates_L = set()
            for w in range(n):
                if w in L or w in R: continue
                # w can be added to L if it is connected to all in R via remaining_edges
                can_add = True
                for r in R:
                    if (w, r) not in remaining_edges and (r, w) not in remaining_edges:
                        can_add = False
                        break
                if can_add and (len(R) > 0): # Must have edges to be useful? 
                    # Actually, if R is empty, it's just a vertex. But we start with edge.
                    candidates_L.add(w)
            
            # Pick the best candidate for L (highest degree in remaining?)
            if candidates_L:
                best_w = max(candidates_L, key=lambda w: sum(1 for x in range(n) if (w,x) in remaining_edges or (x,w) in remaining_edges))
                L.add(best_w)
                changed = True
                # Remove edges from remaining that are now covered? 
                # No, we only remove them when we finalize the biclique.
                # But we must ensure we don't use an edge that's already "reserved" for another biclique?
                # In greedy, we remove edges as we go.
                # So "remaining_edges" is the pool.
                # If we add w to L, we "claim" edges (w, r) for all r in R.
                # We should check if those edges are in remaining_edges.
                # If yes, we tentatively hold them.
                # This greedy approach is tricky to implement correctly without backtracking.
                
        # Simplified Greedy: 
        # Just take the star centered at u for all remaining neighbors?
        # Or take the edge (u,v) as a 1-edge biclique?
        # Let's try to find a maximal star.
        # Center u.
        # Neighbors in remaining: N_u = {v | (u,v) in remaining_edges or (v,u) in remaining_edges}
        # We can form a biclique [u, N_u].
        # This covers all edges incident to u.
        # This is a valid biclique.
        
        N_u = []
        for v in range(n):
            if (u, v) in remaining_edges:
                N_u.append(v)
            elif (v, u) in remaining_edges:
                N_u.append(v)
        
        if N_u:
            biclique = ([u], N_u)
            bicliques.append(biclique)
            # Remove all edges in this biclique from remaining
            for v in N_u:
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                if (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
        else:
            # Should not happen if we picked an edge from remaining
            break
            
    return bicliques

def construct():
    start_time = time.time()
    deadline = 60.0 # 60s limit, use 50s for safety
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Generate candidate bicliques
    # We will use a restricted set: Single edges and Stars of size 2,3,4 (center + 2,3,4 leaves)
    # This keeps the number of variables manageable for MILP.
    
    candidates = []
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    # Single edges
    for u, v in edges:
        candidates.append(([u], [v]))
        
    # Stars
    for center in range(n):
        neighbors = sorted(adj[center])
        deg = len(neighbors)
        if deg == 0: continue
        
        # Size 2 stars (3 vertices total)
        if deg >= 2:
            for i in range(deg):
                for j in range(i+1, deg):
                    candidates.append(([center], [neighbors[i], neighbors[j]]))
        
        # Size 3 stars
        if deg >= 3:
            for i in range(deg):
                for j in range(i+1, deg):
                    for k in range(j+1, deg):
                        candidates.append(([center], [neighbors[i], neighbors[j], neighbors[k]]))
                        
        # Size 4 stars
        if deg >= 4:
            for i in range(deg):
                for j in range(i+1, deg):
                    for k in range(j+1, deg):
                        for l in range(k+1, deg):
                            candidates.append(([center], [neighbors[i], neighbors[j], neighbors[k], neighbors[l]]))
                            
    # Deduplicate and validate
    unique_candidates = set()
    valid_candidates = []
    for l, r in candidates:
        l_t = tuple(sorted(l))
        r_t = tuple(sorted(r))
        if not set(l_t).isdisjoint(set(r_t)):
            continue
        # Check if all cross edges exist in graph
        valid = True
        for u in l_t:
            for v in r_t:
                if (min(u,v), max(u,v)) not in edge_set:
                    valid = False
                    break
            if not valid: break
        if valid:
            key = (l_t, r_t)
            if key not in unique_candidates:
                unique_candidates.add(key)
                valid_candidates.append((list(l_t), list(r_t)))
                
    # 2. Set up MILP
    # Variables: x_j for each candidate biclique j. x_j = 1 if used.
    # Objective: Minimize sum(x_j)
    # Constraints: For each edge e, sum(x_j for j where e in biclique_j) >= 1
    
    m = len(valid_candidates)
    if m == 0:
        return []
        
    # Map edges to indices
    edge_indices = {e: i for i, e in enumerate(edge_set)}
    num_edges = len(edge_set)
    
    # Build constraint matrix A (num_edges x m)
    # A[i, j] = 1 if edge i is in biclique j
    # Using sparse matrix for efficiency
    rows = []
    cols = []
    vals = []
    
    for j, (l, r) in enumerate(valid_candidates):
        for u in l:
            for v in r:
                e = (min(u,v), max(u,v))
                i = edge_indices[e]
                rows.append(i)
                cols.append(j)
                vals.append(1)
                
    A = csr_matrix((vals, (rows, cols)), shape=(num_edges, m))
    
    # Objective: c = [1, 1, ..., 1]
    c = np.ones(m)
    
    # Constraints: A x >= 1
    # milp requires A x <= b or A x >= b or A x == b
    # We can use LinearConstraint
    # A x >= 1  =>  -A x <= -1
    lb = np.ones(num_edges)
    ub = np.inf * np.ones(num_edges)
    constraints = LinearConstraint(A, lb, ub)
    
    # Integrality: x must be integer (0 or 1)
    integrality = np.ones(m)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
    
    best_result = None
    best_val = float('inf')
    
    # Run MILP
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            x = res.x
            # Extract bicliques
            bicliques = []
            for j in range(m):
                if x[j] > 0.5:
                    bicliques.append(valid_candidates[j])
            if bicliques:
                best_result = bicliques
                best_val = len(bicliques)
        else:
            # Fallback to greedy if MILP fails or times out
            pass
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 5:
        if best_result:
            return best_result
        else:
            return _greedy_partition(adj, n)
            
    # If MILP didn't find a good solution or failed, run greedy
    greedy_res = _greedy_partition(adj, n)
    if len(greedy_res) < best_val:
        best_result = greedy_res
        best_val = len(greedy_res)
        
    # Final check
    if best_result is None:
        # Fallback to trivial
        return [[[u],[v for a,v in edges if a==u]] for u in range(n) if any(a==u for a,v in edges)]
        
    return best_result
# EVOLVE-BLOCK-END
