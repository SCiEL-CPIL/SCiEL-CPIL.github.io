import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def dsatur_coloring(vertices, adj):
        """
        Color a graph using DSATUR heuristic.
        vertices: list of vertex labels
        adj: dict mapping vertex to set of neighbors
        Returns: list of lists, where each inner list is a color class (independent set)
        """
        if not vertices:
            return []
        
        uncolored = set(vertices)
        colors = {}
        # degree in the subgraph
        degree = {v: len(adj[v]) for v in vertices}
        # saturation degree: number of distinct colors in neighbors
        sat = {v: 0 for v in vertices}
        neighbor_colors = {v: set() for v in vertices}
        
        color_classes = []
        
        while uncolored:
            # Select vertex with highest saturation, then highest degree
            best_v = None
            best_key = (-1, -1)
            for v in uncolored:
                key = (sat[v], degree[v])
                if key > best_key:
                    best_key = key
                    best_v = v
            
            # Find smallest available color
            used_colors = neighbor_colors[best_v]
            c = 0
            while c in used_colors:
                c += 1
            
            colors[best_v] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(best_v)
            
            uncolored.remove(best_v)
            
            # Update neighbors
            for u in adj[best_v]:
                if u in uncolored:
                    if c not in neighbor_colors[u]:
                        neighbor_colors[u].add(c)
                        sat[u] += 1
        
        return color_classes

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))
        
        # used_mask[u] is a bitmask of vertices v in C such that edge (u,v) is covered
        # We only care about edges within C for the "used" status, 
        # but cross edges are covered immediately and don't need to be tracked in a persistent mask 
        # for the purpose of finding cliques in C, EXCEPT that we need to know which edges in C 
        # are already covered by previous cross-edge cliques.
        
        # Initialize used_mask for C vertices
        used_mask = [0] * k

        best_partition = None
        best_count = float('inf')

        # Define orderings for I
        # 1. Ascending degree in C
        order_asc = sorted(I, key=lambda v: len(adj[v] & set(C)))
        # 2. Descending degree in C
        order_desc = sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=True)
        # 3. Random
        rng = Random(seed + 1000)
        order_rand = I[:]
        rng.shuffle(order_rand)

        orderings = [order_asc, order_desc, order_rand]

        for order in orderings:
            # Reset used_mask for each trial
            current_used_mask = [0] * k
            partition = []

            for v in order:
                N_v = adj[v] & set(C)
                if not N_v:
                    continue

                # Build the "conflict" graph H on N_v
                # Edge (u, w) in H if edge (u, w) in C is ALREADY USED
                h_vertices = list(N_v)
                h_adj = {u: set() for u in h_vertices}
                
                for i, u in enumerate(h_vertices):
                    for j, w in enumerate(h_vertices):
                        if i == j: continue
                        # Check if edge (u, w) is used
                        if current_used_mask[u] & (1 << w):
                            h_adj[u].add(w)
                            h_adj[w].add(u)

                # Color H using DSATUR
                # Independent sets in H are cliques in the REMAINING graph of C
                color_classes = dsatur_coloring(h_vertices, h_adj)
                
                for S in color_classes:
                    if not S:
                        continue
                    # S is a clique in the remaining graph of C
                    # Create clique S U {v}
                    clique_list = sorted(list(S) + [v])
                    partition.append(clique_list)
                    
                    # Mark edges in S as used in C
                    for u in S:
                        for w in S:
                            if u != w:
                                current_used_mask[u] |= (1 << w)
                                current_used_mask[w] |= (1 << u)

            # Cover remaining edges in C
            # Find all edges (u, v) in C such that not (current_used_mask[u] & (1 << v))
            remaining_edges = []
            for u in range(k):
                for v in range(u + 1, k):
                    if not (current_used_mask[u] & (1 << v)):
                        remaining_edges.append((u, v))
            
            if remaining_edges:
                # Greedy clique cover for remaining edges
                # Build adjacency for remaining graph
                rem_adj = [set() for _ in range(k)]
                for u, v in remaining_edges:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
                
                # Keep a set of remaining edges for quick removal
                rem_edge_set = set(remaining_edges)
                
                while rem_edge_set:
                    # Find a maximal clique
                    # Pick vertex with max degree in remaining graph
                    max_deg_v = -1
                    max_deg = -1
                    for u in range(k):
                        if len(rem_adj[u]) > max_deg:
                            max_deg = len(rem_adj[u])
                            max_deg_v = u
                    
                    if max_deg_v == -1:
                        break

                    clique = {max_deg_v}
                    candidates = set(rem_adj[max_deg_v])
                    
                    # Greedy extension
                    while candidates:
                        # Pick candidate with max degree in the subgraph induced by candidates
                        best_cand = None
                        best_cand_deg = -1
                        for c in candidates:
                            # Degree of c within candidates
                            deg_in_cands = len(rem_adj[c] & candidates)
                            if deg_in_cands > best_cand_deg:
                                best_cand_deg = deg_in_cands
                                best_cand = c
                        
                        if best_cand is None:
                            break
                        
                        # Check if best_cand is connected to all in clique
                        can_add = True
                        for x in clique:
                            if x not in rem_adj[best_cand]:
                                can_add = False
                                break
                        if can_add:
                            clique.add(best_cand)
                            candidates.discard(best_cand)
                            # Update candidates: intersect with neighbors of best_cand
                            candidates = candidates & rem_adj[best_cand]
                        else:
                            # If can't add, remove it from candidates? 
                            # Actually, if we can't add it now, we might not be able to later 
                            # unless we remove some clique members, which we don't.
                            # So it can never be added to THIS clique.
                            candidates.discard(best_cand)

                    if len(clique) >= 2:
                        partition.append(sorted(list(clique)))
                    
                    # Remove edges of this clique from rem_edge_set and rem_adj
                    clique_list = list(clique)
                    for i in range(len(clique_list)):
                        u = clique_list[i]
                        for j in range(i + 1, len(clique_list)):
                            v = clique_list[j]
                            e = (min(u, v), max(u, v))
                            if e in rem_edge_set:
                                rem_edge_set.remove(e)
                                rem_adj[u].remove(v)
                                rem_adj[v].remove(u)

            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = partition

        return best_partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)

    return results
