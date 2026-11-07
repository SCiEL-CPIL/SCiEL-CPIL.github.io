import time
import sys
from random import Random

def construct():
    """
    Solve the minimum clique partition problem for three fixed split graphs.
    
    Strategy:
    1. Generate the split graph.
    2. Maximize the number of size-3 cliques {v, u, w} where v in I, u,w in K.
       This is equivalent to maximizing the number of internal edges in K assigned
       to vertices in I, subject to the constraint that edges assigned to a single
       v form a matching.
    3. Use random restarts with different orderings of I vertices to find a good packing.
    4. Cover remaining cross edges with size-2 cliques.
    5. Cover remaining internal edges with one large clique (or multiple if needed, but 
       one clique on the support of remaining edges suffices).
    """
    def instance(n, k, p, seed):
        rng = Random(seed)
        
        # Build adjacency lists
        neighbors_in_I = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors_in_I[v] = nbrs
        
        # Pre-compute internal edges
        all_internal_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_internal_edges.add((u, w))
        
        best_M = 0
        best_assignments = None
        
        # Number of restarts
        num_restarts = 100
        search_rng = Random(seed + 1000)
        
        I_vertices = list(range(k, n))
        
        def greedy_packing(order, local_rng):
            """Perform greedy packing given an ordering of I vertices."""
            used_edges = set()
            assignments = {}
            M = 0
            
            for v in order:
                nbrs = neighbors_in_I[v]
                if not nbrs:
                    continue
                
                # Shuffle neighbors for randomness
                local_nbrs = nbrs[:]
                local_rng.shuffle(local_nbrs)
                
                local_used_vertices = set()
                pairs_for_v = []
                
                i = 0
                while i < len(local_nbrs):
                    u = local_nbrs[i]
                    if u in local_used_vertices:
                        i += 1
                        continue
                    
                    found = False
                    j = i + 1
                    while j < len(local_nbrs):
                        w = local_nbrs[j]
                        if w in local_used_vertices:
                            j += 1
                            continue
                        
                        edge_key = (u, w) if u < w else (w, u)
                        if edge_key not in used_edges:
                            pairs_for_v.append((u, w))
                            used_edges.add(edge_key)
                            local_used_vertices.add(u)
                            local_used_vertices.add(w)
                            M += 1
                            found = True
                            break
                        j += 1
                    
                    if not found:
                        pass
                    i += 1
                
                if pairs_for_v:
                    assignments[v] = pairs_for_v
            
            return M, assignments
        
        # Run multiple restarts with different strategies
        for restart in range(num_restarts):
            # Create different orderings
            if restart % 3 == 0:
                # Random ordering
                order = I_vertices[:]
                search_rng.shuffle(order)
            elif restart % 3 == 1:
                # Order by degree (descending)
                order = sorted(I_vertices, key=lambda v: -len(neighbors_in_I[v]))
            else:
                # Order by degree (ascending)
                order = sorted(I_vertices, key=lambda v: len(neighbors_in_I[v]))
            
            M, assignments = greedy_packing(order, search_rng)
            
            if M > best_M:
                best_M = M
                best_assignments = assignments
        
        # Construct cliques
        cliques = []
        
        # 1. Size-3 cliques from best_assignments
        covered_cross = set()
        for v, pairs in best_assignments.items():
            for u, w in pairs:
                cliques.append([v, u, w])
                covered_cross.add((v, u))
                covered_cross.add((v, w))
        
        # 2. Remaining cross edges
        for v in range(k, n):
            for u in neighbors_in_I[v]:
                if (v, u) not in covered_cross:
                    cliques.append([v, u])
        
        # 3. Remaining internal edges
        used_edges_best = set()
        for v, pairs in best_assignments.items():
            for u, w in pairs:
                edge_key = (u, w) if u < w else (w, u)
                used_edges_best.add(edge_key)
        
        remaining_internal_edges = []
        for e in all_internal_edges:
            if e not in used_edges_best:
                remaining_internal_edges.append(e)
        
        if remaining_internal_edges:
            vertices_involved = set()
            for u, w in remaining_internal_edges:
                vertices_involved.add(u)
                vertices_involved.add(w)
            
            if len(vertices_involved) >= 2:
                cliques.append(list(vertices_involved))
        
        return cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        result.append(instance(*spec))
    
    return result
