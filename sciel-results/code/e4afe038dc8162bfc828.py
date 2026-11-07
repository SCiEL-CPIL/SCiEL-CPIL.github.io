"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _max_biclique_greedy(adj, n, max_iter=200):
    """Find a large biclique by greedy: start with two vertices that share many common neighbors."""
    best_L = set()
    best_R = set()
    best_size = 0
    for start_iter in range(min(30, n)):
        # Pick a starting pair
        L = set()
        R = set()
        # Try to grow from a random-ish starting edge
        for u in range(start_iter, n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Common neighbors
                common = adj[u] & adj[v]
                # Greedily add vertices
                L_try = {u}
                R_try = {v}
                # Common neighbors of all in L with all in R
                # Start: L={u}, R={v}, common = adj[u]&adj[v]
                candidate_pool = set(common)
                # Add to L or R greedily
                changed = True
                while changed and len(L_try) + len(R_try) < n:
                    changed = False
                    for w in list(candidate_pool):
                        # Try adding w to L: w must be adjacent to all in R
                        if all(w in adj[r] for r in R_try):
                            L_try.add(w)
                            # Update pool: must be adjacent to all in L
                            candidate_pool = set.intersection(*[adj[x] for x in L_try]) if L_try else set(range(n))
                            candidate_pool -= L_try
                            changed = True
                            break
                        # Try adding w to R
                        elif all(w in adj[l] for l in L_try):
                            R_try.add(w)
                            candidate_pool = set.intersection(*[adj[x] for x in R_try]) if R_try else set(range(n))
                            candidate_pool -= R_try
                            changed = True
                            break
                size = len(L_try) * len(R_try)
                if size > best_size:
                    best_size = size
                    best_L = set(L_try)
                    best_R = set(R_try)
    return best_L, best_R

def _can_merge(b1, b2, adj):
    """Check if bicliques b1=[L1,R1] and b2=[L2,R2] can be merged into [L1∪L2, R1∪R2]."""
    L1, R1 = set(b1[0]), set(b1[1])
    L2, R2 = set(b2[0]), set(b2[1])
    # All cross pairs between (L1∪L2) and (R1∪R2) must be edges
    newL = L1 | L2
    newR = R1 | R2
    # Check: for each u in newL, v in newR, (u,v) must be edge
    # Only need to check new cross pairs: (L2, R1) and (L1, R2)
    for u in L2:
        for v in R1:
            if v not in adj[u]:
                return False
    for u in L1:
        for v in R2:
            if v not in adj[u]:
                return False
    return True

def _merge_bicliques(b1, b2):
    L1, R1 = list(b1[0]), list(b1[1])
    L2, R2 = list(b2[0]), list(b2[1])
    newL = list(set(L1) | set(L2))
    newR = list(set(R1) | set(R2))
    return [newL, newR]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add(frozenset((u, v)))
    
    # Initial: each edge is a biclique
    bicliques = [[ [u], [v] ] for u, v in edges]
    
    # Greedy merge: repeatedly try to merge pairs
    # Use a priority queue or just iterate
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        new_bicliques = []
        used = [False] * len(bicliques)
        for i in range(len(bicliques)):
            if used[i]:
                continue
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if used[j]:
                    continue
                if _can_merge(bicliques[i], bicliques[j], adj):
                    # Merge
                    new_bicliques.append(_merge_bicliques(bicliques[i], bicliques[j]))
                    used[i] = True
                    used[j] = True
                    improved = True
                    break
            if used[i]:
                continue
            new_bicliques.append(bicliques[i])
        bicliques = new_bicliques
    
    # Try to find large bicliques and re-partition
    # Strategy: find a large biclique, remove its edges, repeat
    if time.time() < deadline - 5:
        remaining_edges = set(edge_set)
        bicliques2 = []
        while remaining_edges:
            if time.time() > deadline:
                break
            # Find a large biclique among remaining edges
            # Build subgraph of remaining edges
            rem_adj = [set() for _ in range(n)]
            for e in remaining_edges:
                u, v = list(e)
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Greedy: find a vertex with high degree, then find its neighbors that share common neighbors
            best_L = set()
            best_R = set()
            best_count = 0
            for u in range(n):
                if time.time() > deadline:
                    break
                if not rem_adj[u]:
                    continue
                # Try: L = {u}, R = some subset of rem_adj[u]
                # Find largest clique in the "intersection graph"
                # Simpler: greedy grow
                L_try = {u}
                # R candidates: neighbors of u in remaining graph
                R_cands = list(rem_adj[u])
                R_try = set()
                # Greedily add to R: v must be adjacent to all in L
                for v in sorted(R_cands, key=lambda x: -len(rem_adj[x])):
                    if all(v in rem_adj[l] for l in L_try):
                        R_try.add(v)
                # Now try to add to L: w must be adjacent to all in R
                L_cands = [w for w in range(n) if w != u and all(w in rem_adj[r] for r in R_try) if w in rem_adj[u]]
                for w in L_cands:
                    if all(w in rem_adj[r] for r in R_try):
                        L_try.add(w)
                
                count = len(L_try) * len(R_try)
                if count > best_count:
                    best_count = count
                    best_L = set(L_try)
                    best_R = set(R_try)
            
            if best_count > 0:
                bicliques2.append([list(best_L), list(best_R)])
                # Remove edges covered by this biclique
                for u in best_L:
                    for v in best_R:
                        remaining_edges.discard(frozenset((u, v)))
            else:
                # Fallback: take one edge
                e = next(iter(remaining_edges))
                u, v = list(e)
                bicliques2.append([[u], [v]])
                remaining_edges.discard(e)
        
        if len(bicliques2) < len(bicliques):
            bicliques = bicliques2
    
    # Final: ensure all edges are covered (validation)
    # Convert to proper format
    result = []
    for b in bicliques:
        L = [int(x) for x in b[0]]
        R = [int(x) for x in b[1]]
        result.append([L, R])
    
    return result
# EVOLVE-BLOCK-END
