    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = []
        bron_kerbosch(adj, n, set(), set(range(n)), set(), cliques)
        
        valid_cliques = [c for c in cliques if len(c) >= 2]
        if not valid_cliques:
            return []
            
        M = len(valid_cliques)
        
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
                
        def greedy():
            uncovered_count = [1] * M # All uncovered initially? No, we track coverage.
            # Actually, let's just use the simple greedy as it's fast enough for n=40.
            # The bottleneck is local search.
            selected = set()
            covered = [False] * M
            for _ in range(M):
                best_v = -1
                best_gain = 0
                for v in range(n):
                    if v in selected: continue
                    gain = 0
                    for ci in vertex_to_cliques[v]:
                        if not covered[ci]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                selected.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    covered[ci] = True
            return list(selected)

        def local_search(init_sol):
            sol = set(init_sol)
            # Initialize cover counts
            cover_count = [0] * M
            for v in sol:
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] += 1
            
            # Check if valid
            for ci in range(M):
                if cover_count[ci] == 0:
                    return init_sol # Invalid, shouldn't happen
            
            improved = True
            while improved:
                improved = False
                verts = list(sol)
                random.shuffle(verts)
                for v in verts:
                    # Remove v
                    sol.remove(v)
                    new_uncovered = []
                    for ci in vertex_to_cliques[v]:
                        cover_count[ci] -= 1
                        if cover_count[ci] == 0:
                            new_uncovered.append(ci)
                    
                    if not new_uncovered:
                        improved = True
                        break
                    
                    # Try to repair with one vertex
                    # Find u not in sol such that it covers all new_uncovered
                    best_u = -1
                    # Pre-compute? No, just iterate
                    for u in range(n):
                        if u in sol: continue
                        ok = True
                        for ci in new_uncovered:
                            if ci not in vertex_to_cliques[u]:
                                ok = False
                                break
                        if ok:
                            best_u = u
                            break
                            
                    if best_u != -1:
                        sol.add(best_u)
                        for ci in vertex_to_cliques[best_u]:
                            cover_count[ci] += 1
                        # Swap performed. Structure changed.
                        improved = True
                        break
                    else:
                        # Put v back
                        sol.add(v)
                        for ci in vertex_to_cliques[v]:
                            cover_count[ci] += 1
                            
            return list(sol)

        best_sol = None
        best_size = float('inf')
        
        for _ in range(100):
            init = greedy()
            sol = local_search(init)
            if len(sol) < best_size:
                best_size = len(sol)
                best_sol = sol[:]
                
        return sorted(best_sol)
