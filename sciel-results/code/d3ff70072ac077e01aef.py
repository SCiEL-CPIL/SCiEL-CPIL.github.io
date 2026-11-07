import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def max_clique_tomita(vertices, adj):
        """
        Find a maximum clique in a graph defined by 'vertices' (list of nodes) 
        and 'adj' (dict of sets of neighbors in the original graph).
        We restrict to the subgraph induced by 'vertices'.
        """
        if not vertices:
            return []
        
        # Convert to list for Tomita's algorithm
        # Tomita's algorithm works on a set of candidates P
        # and a set of already selected vertices X
        # We want to find a clique in the subgraph induced by vertices.
        
        best_clique = []
        
        def expand(P, X):
            nonlocal best_clique
            if not P:
                if len(X) > len(best_clique):
                    best_clique = list(X)
                return
            
            # Ordering heuristic: sort P by degree in the subgraph induced by P
            # This is O(|P|^2) which is fine for |P| <= 30
            degs = {}
            for v in P:
                degs[v] = len(adj[v] & P)
            
            # Sort descending by degree
            P_sorted = sorted(P, key=lambda x: degs[x], reverse=True)
            
            for v in P_sorted:
                if len(X) + len(P) <= len(best_clique):
                    return # Pruning
                
                new_P = P & adj[v]
                new_X = X | {v}
                
                expand(new_P, new_X)
                
                P.remove(v)
                
                if len(X) + len(P) <= len(best_clique):
                    return

        expand(set(vertices), set())
        return best_clique

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Identify C and I
        C = list(range(k))
        I = list(range(k, n))
        
        # Available C-edges
        available_c_edges = set()
        for u in C:
            for v in C:
                if u < v and v in adj[u]:
                    available_c_edges.add((u, v))
        
        cliques = []
        
        # Process I-vertices
        # Heuristic: sort by degree in C descending
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=True)
        
        for v in I_sorted:
            # Neighbors of v in C
            neighbors_v = adj[v] & set(C)
            
            # We need to cover all edges (v, u) for u in neighbors_v
            # We will partition neighbors_v into cliques S such that all edges in S are in available_c_edges
            
            remaining_neighbors = set(neighbors_v)
            
            while remaining_neighbors:
                if time.time() > deadline:
                    break
                
                # Build auxiliary graph on remaining_neighbors
                # Vertices: remaining_neighbors
                # Edges: (u, w) in available_c_edges
                
                # Find a maximum clique in this auxiliary graph
                # Note: The vertices are from C, so adj is already built for C-C edges.
                # We just need to filter by available_c_edges.
                
                # For Tomita, we need a specific adjacency for the subgraph.
                # Let's create a local adjacency dict for the subgraph induced by remaining_neighbors
                # restricted to available_c_edges.
                
                sub_adj = {x: set() for x in remaining_neighbors}
                for x in remaining_neighbors:
                    for y in remaining_neighbors:
                        if x < y:
                            if (x, y) in available_c_edges:
                                sub_adj[x].add(y)
                                sub_adj[y].add(x)
                        elif y < x:
                            if (y, x) in available_c_edges:
                                sub_adj[x].add(y)
                                sub_adj[y].add(x)
                
                # Find max clique
                S = max_clique_tomita(list(remaining_neighbors), sub_adj)
                
                if not S:
                    # This shouldn't happen if remaining_neighbors is not empty and edges exist?
                    # If no edges in available_c_edges connect any pair in remaining_neighbors,
                    # then we can't form a clique of size > 1.
                    # We must cover the I-edges.
                    # If we can't form a clique of size >= 2 from C, we can still form a clique of size 1?
                    # A clique must have at least 2 vertices.
                    # If S is empty, it means no two vertices in remaining_neighbors are connected by an available C-edge.
                    # In that case, we cannot form a clique {v, u, w} with u,w in C.
                    # We must cover each I-edge (v, u) individually?
                    # But a clique {v, u} is valid (size 2).
                    # So we can just pick one vertex u, form clique {v, u}.
                    # This covers edge (v, u) and NO C-edges (since size 2 clique has 1 C-edge? No, {v, u} has edge (v, u) which is an I-edge. It has no C-edges because there is only one C-vertex).
                    # Wait, a clique {v, u} covers the edge (v, u). It does not cover any C-C edge.
                    # So we can always do this.
                    
                    # If S is empty, it means the max clique size is 1?
                    # Tomita returns empty if no clique found? 
                    # My Tomita implementation: if P is not empty, it will find at least one vertex.
                    # So S should have at least 1 vertex.
                    if not S:
                         # Fallback: pick any vertex
                         u = next(iter(remaining_neighbors))
                         S = [u]
                
                # Add clique {v} U S
                clique = [v] + S
                cliques.append(clique)
                
                # Remove C-edges covered by S
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        u, w = S[i], S[j]
                        if u < w:
                            available_c_edges.discard((u, w))
                        else:
                            available_c_edges.discard((w, u))
                
                # Remove S from remaining_neighbors
                remaining_neighbors -= set(S)
        
        # Cover remaining C-edges
        # Graph induced by C with edges in available_c_edges
        # We need to partition these edges into cliques.
        
        while available_c_edges:
            if time.time() > deadline:
                break
            
            # Pick an arbitrary edge
            u, v = next(iter(available_c_edges))
            
            # Find a maximal clique containing u and v in the graph of available_c_edges
            # Start with {u, v}
            current_clique = {u, v}
            
            # Candidates: vertices in C that are connected to all in current_clique via available edges
            # And the edge to them is in available_c_edges
            
            candidates = set()
            # Initial candidates: neighbors of u and v in available graph
            # Let's just iterate through all C vertices? Or just those in available edges?
            
            # Better: 
            # A vertex x can be added if:
            # 1. (x, y) is in available_c_edges for all y in current_clique.
            
            # Let's find all vertices x in C such that (x, u) and (x, v) are in available_c_edges
            # Then filter further.
            
            potential = set()
            for x in C:
                if x == u or x == v:
                    continue
                # Check if x is connected to u and v in available graph
                ok = True
                for y in current_clique:
                    if x < y:
                        if (x, y) not in available_c_edges:
                            ok = False
                            break
                    else:
                        if (y, x) not in available_c_edges:
                            ok = False
                            break
                if ok:
                    potential.add(x)
            
            # Now greedily add from potential
            # To maximize size, we can use a greedy heuristic similar to before
            # Or just add all that fit?
            # If we add x, it might block z.
            # Let's use a simple greedy: pick x with max degree in potential subgraph?
            
            # Actually, since we just want *a* maximal clique, we can just pick one that allows the most others?
            # Or just pick the first one?
            # To be safer and get larger cliques, let's use the same Tomita logic on the subgraph of potential?
            # No, we already have u, v. We just need to extend.
            
            # Let's try to find a maximum clique in the graph induced by potential + {u, v}
            # that contains u and v.
            # This is equivalent to finding a maximum clique in the intersection of neighborhoods?
            
            # Simpler: Just greedily add vertices from potential.
            # Sort potential by degree in the subgraph induced by potential?
            
            if potential:
                # Build subgraph on potential
                sub_adj_pot = {x: set() for x in potential}
                for x in potential:
                    for y in potential:
                        if x < y:
                            if (x, y) in available_c_edges:
                                sub_adj_pot[x].add(y)
                                sub_adj_pot[y].add(x)
                        elif y < x:
                            if (y, x) in available_c_edges:
                                sub_adj_pot[x].add(y)
                                sub_adj_pot[y].add(x)
                                
                # Find max clique in this subgraph
                ext = max_clique_tomita(list(potential), sub_adj_pot)
                current_clique.update(ext)
            
            # Add clique
            cliques.append(list(current_clique))
            
            # Remove edges
            v_list = list(current_clique)
            for i in range(len(v_list)):
                for j in range(i + 1, len(v_list)):
                    a, b = v_list[i], v_list[j]
                    if a < b:
                        available_c_edges.discard((a, b))
                    else:
                        available_c_edges.discard((b, a))
                        
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback to simple seed-like behavior if time runs out?
            # But the algorithm is fast.
            pass
            
        cliques = solve_split(*spec)
        result.append(cliques)
        
    return result
