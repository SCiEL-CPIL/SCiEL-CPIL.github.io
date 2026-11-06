    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        
        if not edges:
            return []

        # Build adjacency list as sets for fast intersection
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        best_cliques = None
        best_count = float('inf')
        
        end_time = start_time + time_budget
        attempt_id = 0

        while time.time() < end_time - 2:
            attempt_id += 1
            rng = random.Random(seed + attempt_id)
            
            covered_edges = set()
            cliques = []
            
            # We need to efficiently find uncovered edges.
            # Since we only add to covered_edges, we can just iterate over all edges and check membership.
            # For n=64, E ~ 1000. This is fast.
            
            while True:
                if time.time() > end_time - 2:
                    break
                    
                # Find uncovered edges
                uncovered = [e for e in edges if e not in covered_edges]
                
                if not uncovered:
                    break
                
                # Find most constrained edge (min common neighbors in remaining graph)
                best_edge = None
                min_score = float('inf')
                candidates_for_best = []
                
                for u, v in uncovered:
                    common = adj[u].intersection(adj[v])
                    count = 0
                    for w in common:
                        if (min(u, w), max(u, w)) not in covered_edges and \
                           (min(v, w), max(v, w)) not in covered_edges:
                            count += 1
                    
                    if count < min_score:
                        min_score = count
                        candidates_for_best = [(u, v)]
                    elif count == min_score:
                        candidates_for_best.append((u, v))
                    
                    if min_score == 0:
                        # Can't get lower than 0. 
                        # We can break out of the loop over edges?
                        # Yes, 0 is the minimum possible score.
                        break
                
                if not candidates_for_best:
                    break
                
                u, v = rng.choice(candidates_for_best)
                
                # Build clique
                current_clique = [u, v]
                
                # Candidates for extension
                ext_candidates = set()
                common = adj[u].intersection(adj[v])
                for w in common:
                    if (min(u, w), max(u, w)) not in covered_edges and \
                       (min(v, w), max(v, w)) not in covered_edges:
                        ext_candidates.add(w)
                
                while ext_candidates:
                    # Pick w with max degree in ext_candidates (among uncovered edges)
                    best_w = None
                    max_deg = -1
                    cand_list = []
                    
                    for w in ext_candidates:
                        deg = 0
                        for z in ext_candidates:
                            if w == z: continue
                            if (min(w, z), max(w, z)) not in covered_edges:
                                deg += 1
                        if deg > max_deg:
                            max_deg = deg
                            cand_list = [w]
                        elif deg == max_deg:
                            cand_list.append(w)
                            
                    if not cand_list:
                        break
                    
                    best_w = rng.choice(cand_list)
                    current_clique.append(best_w)
                    ext_candidates.remove(best_w)
                    
                    # Update ext_candidates
                    new_ext = set()
                    for w in ext_candidates:
                        if (min(w, best_w), max(w, best_w)) not in covered_edges:
                            new_ext.add(w)
                    ext_candidates = new_ext
                
                cliques.append(current_clique)
                
                # Cover edges
                for i in range(len(current_clique)):
                    for j in range(i + 1, len(current_clique)):
                        e = (min(current_clique[i], current_clique[j]), max(current_clique[i], current_clique[j]))
                        covered_edges.add(e)
            
            # End of one greedy pass
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = [list(c) for c in cliques] # Deep copy

        return best_cliques if best_cliques is not None else []
