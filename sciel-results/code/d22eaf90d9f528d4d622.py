<<<<<<< SEARCH
    def greedy_clique_cover(rem_edges, k, deadline):
        """Greedy set-cover: repeatedly pick clique covering most remaining edges."""
        cliques = []
        remaining = set(rem_edges)
        if not remaining:
            return cliques
        
        # Build adjacency
        adj = [set() for _ in range(k)]
        for (a, b) in remaining:
            adj[a].add(b)
            adj[b].add(a)
        
        while remaining and time.time() < deadline:
            # Find clique covering most remaining edges
            best_clique = None
            best_count = 0
            
            for start in range(k):
                if not adj[start]:
                    continue
                # Greedy expand, prioritizing vertices with high remaining degree
                clique = [start]
                candidates = sorted(adj[start], key=lambda w: len(adj[w]), reverse=True)
                for w in candidates:
                    if all(w in adj[x] for x in clique):
                        clique.append(w)
                
                # Count edges covered
                count = 0
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        if e in remaining:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None or best_count == 0:
                # Fallback: just take any edge
                e = next(iter(remaining))
                best_clique = list(e)
                best_count = 1
            
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining:
                        remaining.discard(e)
                        adj[a].discard(b)
                        adj[b].discard(a)
            
            cliques.append(best_clique)
        
        # If we ran out of time, cover remaining edges individually
        if remaining:
            for (a, b) in remaining:
                cliques.append([a, b])
        
        return cliques
=======
    def greedy_clique_cover(rem_edges, k, deadline):
        """Greedy set-cover: repeatedly pick clique covering most remaining edges."""
        cliques = []
        remaining = set(rem_edges)
        if not remaining:
            return cliques
        
        # Build adjacency
        adj = [set() for _ in range(k)]
        for (a, b) in remaining:
            adj[a].add(b)
            adj[b].add(a)
        
        while remaining and time.time() < deadline:
            # Find clique covering most remaining edges
            best_clique = None
            best_count = 0
            
            for start in range(k):
                if not adj[start]:
                    continue
                # Greedy expand, prioritizing vertices with high remaining degree
                clique = [start]
                candidates = sorted(adj[start], key=lambda w: len(adj[w]), reverse=True)
                for w in candidates:
                    if all(w in adj[x] for x in clique):
                        clique.append(w)
                
                # Count edges covered
                count = 0
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        if e in remaining:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None or best_count == 0:
                # Fallback: just take any edge
                e = next(iter(remaining))
                best_clique = list(e)
                best_count = 1
            
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining:
                        remaining.discard(e)
                        adj[a].discard(b)
                        adj[b].discard(a)
            
            cliques.append(best_clique)
        
        # If we ran out of time, cover remaining edges individually
        if remaining:
            for (a, b) in remaining:
                cliques.append([a, b])
        
        # Local improvement: try to merge cliques that form a larger clique
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_cliques = []
            used = [False] * len(cliques)
            for i in range(len(cliques)):
                if used[i]:
                    continue
                for j in range(i + 1, len(cliques)):
                    if used[j]:
                        continue
                    # Check if union forms a clique
                    union = set(cliques[i]) | set(cliques[j])
                    is_clique = True
                    for a in union:
                        for b in union:
                            if a >= b:
                                continue
                            if b not in adj[a]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        new_cliques.append(sorted(union))
                        used[i] = True
                        used[j] = True
                        improved = True
                        break
                if not used[i]:
                    new_cliques.append(cliques[i])
            if improved:
                cliques = new_cliques
        
        return cliques
>>>>>>> REPLACE

