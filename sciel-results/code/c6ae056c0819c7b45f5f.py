"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    all_edges_set = set()
    for u, v in edges:
        if u < v:
            all_edges_set.add((u, v))
        else:
            all_edges_set.add((v, u))
    
    def greedy_biclique_partition(remaining_edges, adj, n, rng):
        """Greedy: repeatedly extract the largest biclique from remaining edges."""
        remaining = set(remaining_edges)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Find the largest biclique covering remaining edges
            best_biclique = None
            best_count = 0
            
            # Try each vertex as a seed for the biclique
            vertices_in_remaining = set()
            for e in remaining:
                vertices_in_remaining.add(e[0])
                vertices_in_remaining.add(e[1])
            
            # Shuffle for diversity
            seed_vertices = list(vertices_in_remaining)
            rng.shuffle(seed_vertices)
            
            for seed in seed_vertices:
                # Build initial biclique: left={seed}, right=neighbors of seed in remaining
                # A biclique (L, R) requires all cross-pairs to be edges
                # Start: L = {seed}, R = {v : (seed,v) in remaining and v != seed}
                # But we need to be more clever.
                
                # Strategy: pick a seed vertex, set L={seed}, R=adj[seed] ∩ remaining vertices
                # Then iteratively: for each v in R, L must be subset of adj[v]
                # For each u in L, R must be subset of adj[u]
                
                # Actually, let's use a different approach:
                # Pick two sets L and R such that all (u,v) for u in L, v in R are in remaining
                # and maximize |L|*|R| (or just |L|+|R| or the number of edges covered)
                
                # Simple approach: start with L={seed}, R = all neighbors of seed that have edges in remaining
                # Then refine.
                
                L = {seed}
                # R = vertices v != seed such that (min(seed,v), max(seed,v)) in remaining
                R = set()
                for v in adj[seed]:
                    e = (min(seed, v), max(seed, v))
                    if e in remaining:
                        R.add(v)
                
                if not R:
                    continue
                
                # Iteratively refine: remove from L any vertex that doesn't connect to all of R
                # Remove from R any vertex that doesn't connect to all of L
                changed = True
                while changed:
                    changed = False
                    # Refine L: u in L must connect to all v in R
                    L_new = set()
                    for u in L:
                        all_connected = True
                        for v in R:
                            e = (min(u, v), max(u, v))
                            if e not in remaining:
                                all_connected = False
                                break
                        if all_connected:
                            L_new.add(u)
                    if L_new != L:
                        L = L_new
                        changed = True
                        if not L:
                            break
                    
                    # Refine R: v in R must connect to all u in L
                    if not L:
                        break
                    R_new = set()
                    for v in R:
                        all_connected = True
                        for u in L:
                            e = (min(u, v), max(u, v))
                            if e not in remaining:
                                all_connected = False
                                break
                        if all_connected:
                            R_new.add(v)
                    if R_new != R:
                        R = R_new
                        changed = True
                        if not R:
                            break
                
                if not L or not R:
                    continue
                
                # Count edges covered
                count = 0
                for u in L:
                    for v in R:
                        e = (min(u, v), max(u, v))
                        if e in remaining:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_biclique = (set(L), set(R))
            
            if best_biclique is None or best_count == 0:
                # Fallback: take one edge as a biclique
                e = next(iter(remaining))
                u, v = e
                bicliques.append(([int(u)], [int(v)]))
                remaining.discard(e)
            else:
                L, R = best_biclique
                bicliques.append((list(L), list(R)))
                # Remove all cross-edges from remaining
                for u in L:
                    for v in R:
                        e = (min(u, v), max(u, v))
                        remaining.discard(e)
        
        return bicliques, remaining
    
    def try_biclique_from_pair(remaining, adj, n, seed_u, seed_v, rng):
        """Try to build a large biclique containing edge (seed_u, seed_v)."""
        if (min(seed_u, seed_v), max(seed_u, seed_v)) not in remaining:
            return None, 0
        
        # L must contain vertices that connect to all of R, and R connects to all of L
        # Start: L = {seed_u}, R = {seed_v}
        # Expand: try to add vertices to L and R
        
        L = {seed_u}
        R = {seed_v}
        
        # Try to expand
        changed = True
        while changed:
            changed = False
            
            # Try adding to L: a vertex u can join L if it connects to all of R (in remaining)
            candidates_L = set()
            for u in adj[seed_u]:  # only consider neighbors
                if u in L:
                    continue
                all_conn = True
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e not in remaining:
                        all_conn = False
                        break
                if all_conn:
                    candidates_L.add(u)
            
            # Try adding to R: a vertex v can join R if it connects to all of L
            candidates_R = set()
            for v in adj[seed_v]:
                if v in R:
                    continue
                all_conn = True
                for u in L:
                    e = (min(u, v), max(u, v))
                    if e not in remaining:
                        all_conn = False
                        break
                if all_conn:
                    candidates_R.add(v)
            
            # Add all candidates (greedy expansion)
            if candidates_L:
                L.update(candidates_L)
                changed = True
            if candidates_R:
                R.update(candidates_R)
                changed = True
        
        # Final verification
        count = 0
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in remaining:
                    count += 1
        
        if count == 0:
            return None, 0
        
        # Verify it's truly a biclique (all cross pairs are in remaining)
        valid = True
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in remaining:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return None, 0
        
        return (L, R), count
    
    best_result = None
    best_count = len(all_edges_set) + 1
    
    # Multiple strategies
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(attempt * 12345 + 67890)
        
        if attempt % 2 == 0:
            # Strategy 1: Greedy max-biclique extraction
            bicliques, leftover = greedy_biclique_partition(all_edges_set, adj, n, rng)
            
            # If there's leftover, handle remaining edges
            if leftover:
                for e in leftover:
                    u, v = e
                    bicliques.append(([int(u)], [int(v)]))
            
            count = len(bicliques)
            if count < best_count:
                best_count = count
                best_result = bicliques
        
        else:
            # Strategy 2: Greedy with pair-based biclique building
            remaining = set(all_edges_set)
            bicliques = []
            
            while remaining:
                if time.time() > deadline:
                    break
                
                # Pick a random edge from remaining
                edge_list = list(remaining)
                rng.shuffle(edge_list)
                
                best_bc = None
                best_bc_count = 0
                
                # Try building bicliques from several candidate edges
                for e in edge_list[:20]:
                    u, v = e
                    bc, cnt = try_biclique_from_pair(remaining, adj, n, u, v, rng)
                    if bc and cnt > best_bc_count:
                        best_bc_count = cnt
                        best_bc = bc
                
                if best_bc is None or best_bc_count == 0:
                    e = next(iter(remaining))
                    bicliques.append(([int(e[0])], [int(e[1])]))
                    remaining.discard(e)
                else:
                    L, R = best_bc
                    bicliques.append((list(L), list(R)))
                    for u in L:
                        for v in R:
                            remaining.discard((min(u, v), max(u, v)))
            
            if leftover if False else remaining:
                for e in remaining:
                    bicliques.append(([int(e[0])], [int(e[1])]))
            
            count = len(bicliques)
            if count < best_count:
                best_count = count
                best_result = bicliques
    
    if best_result is None:
        # Fallback: star decomposition
        best_result = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                best_result.append(([int(u)], [int(v) for v in neighbors]))
    
    # Convert to proper format: list of [left_list, right_list]
    result = []
    for bic in best_result:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    
    return result

# EVOLVE-BLOCK-END
