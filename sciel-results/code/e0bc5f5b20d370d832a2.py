<<<<<<< SEARCH
    def greedy_partition(n, edges, seed_heur):
        neighbors = [set() for _ in range(n)]
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            # Find v with max degree in remaining graph
            max_deg = -1
            candidates = []
            for u in range(n):
                deg = len(neighbors[u])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [u]
                elif deg == max_deg and deg > 0:
                    candidates.append(u)
            
            if not candidates:
                break
            v = rng_heur.choice(candidates)
            
            C = [v]
            P = set(neighbors[v])
            
            while P:
                # Find u in P with max degree in induced subgraph P
                max_score = -1
                tie_candidates = []
                for u in P:
                    score = len(neighbors[u] & P)
                    if score > max_score:
                        max_score = score
                        tie_candidates = [u]
                    elif score == max_score:
                        tie_candidates.append(u)
                
                u = rng_heur.choice(tie_candidates)
                C.append(u)
                P = P & neighbors[u]
            
            # Remove edges of C from remaining_edges and neighbors
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v2 = C[i], C[j]
                    e = (u, v2) if u < v2 else (v2, u)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                    
                    neighbors[u].discard(v2)
                    neighbors[v2].discard(u)
                    
            cliques.append(C)
            
        return cliques
=======
    def greedy_partition(n, edges, seed_heur):
        neighbors = [set() for _ in range(n)]
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            # Find v with max degree in remaining graph
            max_deg = -1
            candidates = []
            for u in range(n):
                deg = len(neighbors[u])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [u]
                elif deg == max_deg and deg > 0:
                    candidates.append(u)
            
            if not candidates:
                break
            v = rng_heur.choice(candidates)
            
            C = [v]
            P = set(neighbors[v])
            
            while P:
                # Find u in P with max degree in induced subgraph P
                max_score = -1
                tie_candidates = []
                for u in P:
                    score = len(neighbors[u] & P)
                    if score > max_score:
                        max_score = score
                        tie_candidates = [u]
                    elif score == max_score:
                        tie_candidates.append(u)
                
                u = rng_heur.choice(tie_candidates)
                C.append(u)
                P = P & neighbors[u]
            
            # Remove edges of C from remaining_edges and neighbors
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v2 = C[i], C[j]
                    e = (u, v2) if u < v2 else (v2, u)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                    
                    neighbors[u].discard(v2)
                    neighbors[v2].discard(u)
                    
            cliques.append(C)
            
        return cliques

    def is_clique_union(C1, C2, edges_set):
        # Check if C1 U C2 is a clique
        # C1 and C2 are sets of vertices
        # We need to check all pairs (u, v) with u in C1, v in C2
        for u in C1:
            for v in C2:
                e = (u, v) if u < v else (v, u)
                if e not in edges_set:
                    return False
        return True

    def merge_cliques(cliques, edges_set, rng_heur):
        # Convert to sets for easier manipulation
        cliques_sets = [set(c) for c in cliques]
        changed = True
        while changed:
            changed = False
            # Shuffle indices to randomize merge order
            indices = list(range(len(cliques_sets)))
            rng_heur.shuffle(indices)
            
            for i_idx in indices:
                i = i_idx
                if i >= len(cliques_sets): continue
                C_i = cliques_sets[i]
                if not C_i: continue
                
                # Find a candidate j to merge with
                merged = False
                for j_idx in indices:
                    j = j_idx
                    if j == i or j >= len(cliques_sets): continue
                    C_j = cliques_sets[j]
                    if not C_j: continue
                    
                    if is_clique_union(C_i, C_j, edges_set):
                        # Merge C_j into C_i
                        C_i.update(C_j)
                        cliques_sets[j] = set() # Mark as empty
                        changed = True
                        merged = True
                        break
                
                if merged:
                    # Remove empty cliques
                    cliques_sets = [c for c in cliques_sets if c]
                    # Restart loop to check new possibilities
                    break
        
        return [list(c) for c in cliques_sets if c]
>>>>>>> REPLACE
