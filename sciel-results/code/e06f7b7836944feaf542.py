"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit
    
    n = 40
    best_edges = 0
    best_adj = None
    best_edge_list = None
    
    # Initialize with a random dense graph and refine
    # Start with the 8 disjoint K5's as baseline (104 edges)
    # But we want to do better - aim for 105+
    
    # Strategy: Start with a random graph with ~105 edges, then do local search
    # to remove C6 cycles while trying to maintain high edge count.
    
    # Actually, let's use a smarter approach:
    # 1. Start with a known good structure or random
    # 2. Use local search: try adding edges that don't create C6
    # 3. If stuck, try swapping edges to free up capacity for more edges
    
    def count_c6(adj, u, v):
        """Count number of C6 cycles that would be created by adding edge (u,v).
        A C6 is created if there's a path of length 5 from u to v."""
        # This is expensive; use a faster check
        # Actually, adding edge (u,v) creates a C6 iff there exists a path u-w1-w2-w3-w4-v
        # with all distinct vertices and none equal to u or v (except endpoints)
        # Better: count paths of length 5 between u and v
        # Use matrix powers or BFS
        
        # Fast check: for each pair (a,b) where a is neighbor of u, b is neighbor of v
        # check if there's a path of length 3 from a to b avoiding u,v
        # This is still expensive. Let's use a different approach.
        
        # For efficiency, we'll precompute neighbor sets and use a quick rejection
        pass
    
    def has_c6_with_edge(adj, u, v, neighbors):
        """Check if adding edge (u,v) creates a C6.
        We need to check if there's a path of length 5 from u to v in G - {(u,v)}.
        Equivalently, is there a walk u-a-b-c-d-v with all distinct?
        
        Efficient approach: 
        For each a in N(u), for each d in N(v), check if there's a path of length 3 from a to d
        avoiding u and v. Path of length 3: a-b-c-d.
        """
        # This is O(deg(u)*deg(v)*n) which for n=40 is manageable
        nu = neighbors[u]
        nv = neighbors[v]
        for a in nu:
            if a == v:
                continue
            for d in nv:
                if d == u or d == a:
                    continue
                # Check if there's a path a-b-c-d with b,c distinct from u,v,a,d
                # b must be neighbor of a (not u), c must be neighbor of d (not v)
                # and b-c must be an edge
                for b in neighbors[a]:
                    if b == u or b == d:
                        continue
                    for c in neighbors[d]:
                        if c == v or c == a:
                            continue
                        if c == b:
                            continue
                        if adj[b][c]:
                            return True
        return False
    
    def random_graph(n, m):
        """Generate a random graph with m edges."""
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for i, j in all_possible[:m]:
            adj[i][j] = adj[j][i] = True
            edges.append([i, j])
        return adj, edges
    
    def build_neighbors(adj):
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        return neighbors
    
    def count_edges(adj):
        return int(np.sum(adj)) // 2
    
    def is_c6_free(adj):
        """Check if graph is C6-free."""
        neighbors = build_neighbors(adj)
        # Check all possible edges that are NOT in the graph for C6 creation
        # But we need to check if the current graph HAS a C6
        # A graph has a C6 iff there exist distinct vertices v1..v6 with edges
        # v1-v2-v3-v4-v5-v6-v1
        
        # Efficient check: for each pair of non-adjacent vertices, check if adding
        # them creates a C6 - but that's for the complement.
        
        # Direct check: enumerate all 6-cycles
        # O(n^6) is too slow for n=40 (40^6 = 4e9)
        
        # Better: use the fact that a C6 exists iff there's a path of length 5
        # between two non-adjacent vertices u,v such that... no wait.
        
        # Actually, let's just try to find a C6 by checking for each pair of
        # non-adjacent vertices if there's a path of length 5 between them.
        # If yes, then u...v (length 5) plus edge... but u and v aren't adjacent.
        # A C6 needs 6 vertices in a cycle. 
        
        # Let me think differently. A C6 is a cycle of length 6.
        # We can check: for each pair of vertices (u,v) that are NOT adjacent,
        # if there exists a path of length 5 from u to v, then we have a C6?
        # No! A path of length 5 from u to v gives 6 vertices u-w1-w2-w3-w4-w5-v,
        # but that's a path, not a cycle. For a cycle, we'd need v-u to be an edge,
        # but u,v are not adjacent.
        
        # Correct approach: A C6 exists iff there are vertices v1,...,v6 such that
        # vi-vi+1 are edges (cyclic). 
        # Equivalent: there exist non-adjacent u,v with a path of length 5 between them
        # AND... no, that's not right either.
        
        # Let me just do a direct search for 6-cycles using a smarter method.
        # For each pair of non-adjacent vertices (u,v), check if there's a path
        # of length 5 from u to v. If such a path exists, it forms a C6 with the
        # "virtual" edge uv... but uv isn't an edge, so it's not a C6 in the graph.
        
        # I'm overcomplicating this. Let me just check: does the graph contain
        # a 6-cycle? 
        # 
        # Method: For each ordered pair (u,v) with u<v and NOT adjacent,
        # count paths of length 5 from u to v. If count > 0, we have a 6-cycle
        # (the path plus... wait no, the path IS 5 edges connecting 6 vertices,
        # forming a path not a cycle).
        #
        # A 6-CYCLE needs 6 edges among 6 vertices forming a cycle.
        # 
        # Let me use: a graph has a C6 iff there exist distinct vertices
        # a,b,c,d,e,f such that ab,bc,cd,de,ef,fa are all edges.
        #
        # Efficient enumeration: 
        # For each pair of non-adjacent vertices (a,f), check if there's a path
        # a-b-c-d-e-f of length 5. If yes, then a-b-c-d-e-f-a is a C6.
        # Wait, that requires af to be an edge, but we said a,f are non-adjacent.
        # 
        # OK I think I've been confusing myself. Let me restart:
        # A C6 is 6 vertices v0,v1,v2,v3,v4,v5 with edges v0v1, v1v2, v2v3, v3v4, v4v5, v5v0.
        # 
        # To detect: pick two non-adjacent vertices u and v. If there's a path
        # of length 5 from u to v, say u-w1-w2-w3-w4-v, then u-w1-w2-w3-w4-v-u 
        # would be a 7-cycle if uv were an edge, but it's not. So that's not a C6.
        #
        # The right way: a C6 exists iff there are two vertices u,v (adjacent or not)
        # such that... hmm.
        #
        # Let me just brute force with some pruning. For n=40, the number of 6-cycles
        # could be large, but we just need to detect existence.
        #
        # Actually, the simplest correct check: 
        # For each pair of non-adjacent vertices (u, v), check if there exists
        # a path of length 5 from u to v. The union of this path and... no.
        #
        # I'll use a different approach: the graph is C6-free iff for every pair
        # of non-adjacent vertices u,v, the number of paths of length 5 between
        # u and v is 0. Because any such path u-w1-w2-w3-w4-v combined with edge uv
        # would form a C6, but uv isn't an edge. So that's wrong.
        #
        # CORRECT: The graph contains a C6 iff there exist vertices u,v (which ARE
        # adjacent) such that there is a path of length 5 from u to v in G - uv.
        # Because then u-w1-w2-w3-w4-v-u is a 6-cycle.
        #
        # So: for each edge (u,v) in E, check if there's a path of length 5 from u to v
        # in G with edge (u,v) removed.
        
        neighbors = build_neighbors(adj)
        for u in range(n):
            for v in neighbors[u]:
                if u >= v:
                    continue
                # Check if there's a path of length 5 from u to v avoiding direct edge
                # Path: u-a-b-c-d-v where a,b,c,d distinct from u,v and from each other
                # and a != v, d != u
                found = False
                for a in neighbors[u]:
                    if a == v:
                        continue
                    for d in neighbors[v]:
                        if d == u or d == a:
                            continue
                        # Need path a-b-c-d of length 3
                        for b in neighbors[a]:
                            if b == u or b == d:
                                continue
                            for c in neighbors[d]:
                                if c == v or c == a:
                                    continue
                                if c == b:
                                    continue
                                if adj[b][c]:
                                    # Check all distinct: u,a,b,c,d,v
                                    if len({u,a,b,c,d,v}) == 6:
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    return False
        return True
    
    # Let's use a more practical approach:
    # Start with 8 disjoint K5's (104 edges, C6-free)
    # Try to add edges between the K5's while maintaining C6-freeness
    # Use local search with edge additions and swaps
    
    def try_add_edge(adj, neighbors, u, v):
        """Try to add edge (u,v). Return True if successful."""
        if adj[u][v]:
            return False
        # Check if adding (u,v) creates a C6
        # Adding (u,v) creates a C6 iff there's a path of length 5 from u to v
        # in the current graph (without the new edge)
        # Path: u-a-b-c-d-v
        for a in neighbors[u]:
            for d in neighbors[v]:
                if d == a:
                    continue
                for b in neighbors[a]:
                    if b == u or b == d:
                        continue
                    for c in neighbors[d]:
                        if c == v or c == a or c == b:
                            continue
                        if adj[b][c]:
                            if len({u, a, b, c, d, v}) == 6:
                                return False
        # Safe to add
        adj[u][v] = adj[v][u] = True
        neighbors[u].add(v)
        neighbors[v].add(u)
        return True
    
    def try_remove_edge(adj, neighbors, u, v):
        """Remove edge (u,v)."""
        if not adj[u][v]:
            return
        adj[u][v] = adj[v][u] = False
        neighbors[u].discard(v)
        neighbors[v].discard(u)
    
    # Initialize with 8 disjoint K5's
    adj = np.zeros((n, n), dtype=bool)
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                adj[5*k+a][5*k+b] = adj[5*k+b][5*k+a] = True
    
    neighbors = build_neighbors(adj)
    current_edges = count_edges(adj)
    best_edges = current_edges
    best_adj = adj.copy()
    best_edge_list = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # Local search: try to add edges
    # Order pairs by some heuristic (e.g., between different K5 blocks)
    pairs_to_try = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                pairs_to_try.append((i, j))
    
    random.shuffle(pairs_to_try)
    
    improvement_found = True
    while improvement_found and time.time() - start_time < DEADLINE:
        improvement_found = False
        random.shuffle(pairs_to_try)
        for u, v in pairs_to_try:
            if time.time() - start_time > DEADLINE:
                break
            if not adj[u][v]:
                if try_add_edge(adj, neighbors, u, v):
                    current_edges += 1
                    improvement_found = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj.copy()
                        best_edge_list = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # Now try swaps: remove one edge, add two edges (net +1)
    # Or remove one, add one (net 0) to enable future additions
    if time.time() - start_time < DEADLINE:
        # Try to improve by swapping
        for _ in range(2000):  # limited iterations
            if time.time() - start_time > DEADLINE:
                break
            # Pick a random edge to remove
            edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
            if not edge_list:
                break
            ru, rv = random.choice(edge_list)
            # Try removing it and adding two new edges
            try_remove_edge(adj, neighbors, ru, rv)
            current_edges -= 1
            
            # Try to add two edges
            added = 0
            candidates = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            random.shuffle(candidates)
            for u, v in candidates:
                if try_add_edge(adj, neighbors, u, v):
                    current_edges += 1
                    added += 1
                    if added >= 2:
                        break
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
                best_edge_list = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
            
            # If we didn't gain, revert the removal
            if current_edges <= best_edges and not adj[ru][rv]:
                try_add_edge(adj, neighbors, ru, rv)
                current_edges += 1
    
    # Final: try more aggressive local search
    if time.time() - start_time < DEADLINE:
        # Rebuild neighbor list to be safe
        neighbors = build_neighbors(adj)
        for iteration in range(5000):
            if time.time() - start_time > DEADLINE:
                break
            # Randomly try to add an edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not adj[u][v]:
                if try_add_edge(adj, neighbors, u, v):
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj.copy()
                        best_edge_list = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # Return the best edge list found
    if best_edge_list is None:
        # Fallback to 8 disjoint K5's
        best_edge_list = []
        for k in range(8):
            best_edge_list += [[5*k+a, 5*k+b] for a in range(5) for b in range(a+1, 5)]
    
    return best_edge_list
# EVOLVE-BLOCK-END
