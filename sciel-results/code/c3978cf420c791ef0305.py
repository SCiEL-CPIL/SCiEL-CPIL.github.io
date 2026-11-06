import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    n = 20
    num_colors = 20
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    
    # Precompute for each pair (u,v) the list of "paths of length 3" between u and v
    # i.e., pairs (a,b) such that a != u,v and b != u,v and a != b,
    # and edges (u,a), (a,b), (b,v) could form a 4-cycle with (u,v).
    # Actually, for adding edge (u,v), we need to check all 4-cycles containing (u,v).
    # A 4-cycle containing (u,v) has the form u - a - b - v - u.
    # The other three edges are (u,a), (a,b), (b,v).
    # So for each pair (u,v), the relevant (a,b) pairs are those where:
    #   a is a neighbor of u, b is a neighbor of v, a != b, a != v, b != u,
    #   and (a,b) is an edge (or could be).
    # But we don't know which edges exist yet. So we precompute all possible (a,b) for each (u,v).
    
    # For each edge (u,v), precompute the list of (a,b) such that u-a-b-v forms a path
    # a, b are distinct, a != u,v, b != u,v, a != b.
    # There are (n-2)*(n-3) such pairs? No, a can be any of n-2 vertices (not u,v), 
    # b can be any of n-3 vertices (not u,v,a). So (n-2)*(n-3) = 18*17 = 306 per edge.
    # Total edges: 190. Total checks: 190 * 306 = 58140. That's fine to precompute.
    
    # Actually, we need to be careful: the 4-cycle is u-a-b-v-u, so edges are (u,a), (a,b), (b,v), (v,u).
    # When checking if we can add (u,v), we need to check all such (a,b) where (u,a), (a,b), (b,v) all exist.
    # The forbidden colors for (u,v) are the colors of (u,a), (a,b), (b,v) for all such existing triples.
    
    # Let's precompute for each ordered pair (u,v) the list of (a,b) candidates.
    # But since the graph is undirected, (u,v) and (v,u) are the same edge.
    # Let's just precompute for each unordered edge (u,v) with u < v, the list of (a,b) 
    # such that a != u,v, b != u,v, a != b, and we'll check both orientations? 
    # No, the path is u-a-b-v. So a is adjacent to u, b is adjacent to v, and (a,b) is an edge.
    # This is directional in the sense of the path, but the edge set is undirected.
    
    # Precompute: for each edge index (u,v), list of (a,b) pairs.
    # a ranges over all vertices except u,v.
    # b ranges over all vertices except u,v,a.
    # We'll store this as a list of lists: edge_paths[edge_idx] = [(a,b), ...]
    
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edge_paths = []
    for (u, v) in all_edges:
        paths = []
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                paths.append((a, b))
        edge_paths.append(paths)
    
    def is_valid_full(adj):
        """Check if the coloring is valid for all 4-cycles in the current graph."""
        for i, (u, v) in enumerate(all_edges):
            if adj[u][v] == -1:
                continue
            cu = adj[u][v]
            for (a, b) in edge_paths[i]:
                if adj[u][a] != -1 and adj[a][b] != -1 and adj[b][v] != -1:
                    ca = adj[u][a]
                    cb = adj[a][b]
                    cv = adj[b][v]
                    if len({cu, ca, cb, cv}) < 4:
                        return False
        return True
    
    def greedy_build(seed):
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        remaining = all_edges[:]
        rng.shuffle(remaining)
        
        for (u, v) in remaining:
            # Check if adding (u,v) is possible
            forbidden = set()
            feasible = True
            
            # For each path u-a-b-v, if (u,a), (a,b), (b,v) exist, 
            # then the color of (u,v) cannot be any of their colors.
            for (a, b) in edge_paths[edge_index[(u, v)]]:
                if adj[u][a] != -1 and adj[a][b] != -1 and adj[b][v] != -1:
                    forbidden.add(adj[u][a])
                    forbidden.add(adj[a][b])
                    forbidden.add(adj[b][v])
                    if len(forbidden) >= num_colors:
                        feasible = False
                        break
            
            if feasible and len(forbidden) < num_colors:
                candidates = [c for c in range(num_colors) if c not in forbidden]
                if candidates:
                    c = rng.choice(candidates)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
        
        return edges, adj, neighbors
    
    def try_add_edge(adj, neighbors, u, v, edges, rng):
        """Try to add edge (u,v) with a valid color. Returns True if added."""
        if adj[u][v] != -1:
            return True
        
        forbidden = set()
        for (a, b) in edge_paths[edge_index[(u, v)]]:
            if adj[u][a] != -1 and adj[a][b] != -1 and adj[b][v] != -1:
                forbidden.add(adj[u][a])
                forbidden.add(adj[a][b])
                forbidden.add(adj[b][v])
                if len(forbidden) >= num_colors:
                    return False
        
        candidates = [c for c in range(num_colors) if c not in forbidden]
        if candidates:
            c = rng.choice(candidates)
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
            return True
        return False
    
    def try_recolor(adj, neighbors, u, v, new_c, edges):
        """Check if recoloring (u,v) to new_c is valid. If so, apply and return True."""
        old_c = adj[u][v]
        if old_c == new_c:
            return True
        
        # Check all 4-cycles containing (u,v)
        # For each path u-a-b-v, the colors must be distinct.
        # After recoloring, (u,v) has color new_c.
        # The other three edges have colors ca, cb, cv.
        # We need {new_c, ca, cb, cv} to have size 4.
        # Since the graph was valid before, {old_c, ca, cb, cv} had size 4, 
        # so ca, cb, cv are distinct and none equals old_c.
        # We need new_c not in {ca, cb, cv}.
        
        for (a, b) in edge_paths[edge_index[(u, v)]]:
            if adj[u][a] != -1 and adj[a][b] != -1 and adj[b][v] != -1:
                ca = adj[u][a]
                cb = adj[a][b]
                cv = adj[b][v]
                if new_c in (ca, cb, cv):
                    return False
        
        # Also need to check 4-cycles where (u,v) is not the "first" edge?
        # No, the 4-cycle is defined by the edge set. The path u-a-b-v-u gives the cycle.
        # But (u,v) could also be part of a cycle v-a-b-u-v, which is the same cycle.
        # Our precomputation for edge (u,v) considers paths u-a-b-v, which covers all 4-cycles containing (u,v).
        # Wait, is that true? A 4-cycle containing edge (u,v) must have the form 
        # u - x - y - v - u or u - y - x - v - u.
        # In our precomputation, for edge (u,v), we consider all (a,b) such that 
        # the path u-a-b-v exists. This includes both orientations?
        # a is adjacent to u, b is adjacent to v. So the path is u-a-b-v.
        # This covers all 4-cycles containing (u,v) because in any such cycle,
        # the two neighbors of u in the cycle (other than v) and the two neighbors of v (other than u)
        # form a path of length 3.
        # Actually, in a 4-cycle u-v-x-y-u, the edges are (u,v), (v,x), (x,y), (y,u).
        # Here, a=x, b=y. Path u-x-y-v? No, the path is u-y-x-v or u-x-y-v?
        # The cycle is u-v-x-y-u. The path from u to v not using (u,v) is u-y-x-v.
        # So a=y, b=x. Our precomputation checks all (a,b) where a is any vertex, b is any vertex.
        # So (y,x) is included if adj[u][y] and adj[y][x] and adj[x][v] exist.
        # Yes, this is correct.
        
        # Apply recolor
        adj[u][v] = new_c
        adj[v][u] = new_c
        # Update edges list
        for e in edges:
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                e[2] = new_c
                break
        return True
    
    # Phase 1: Multiple greedy builds with different seeds
    best_edges = []
    best_adj = None
    best_neighbors = None
    best_count = 0
    
    # Try many seeds quickly
    for i in range(100):
        if time.time() > deadline - 30:
            break
        edges, adj, neighbors = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = [row[:] for row in adj]
            best_neighbors = [set(s) for s in neighbors]
    
    # Phase 2: Local search
    if best_adj is None:
        return [[0, 1, 0]]
    
    adj = [row[:] for row in best_adj]
    neighbors = [set(s) for s in best_neighbors]
    edges = best_edges[:]
    
    current_edge_set = set((min(u, v), max(u, v)) for u, v, c in edges)
    missing_edges = [e for e in all_edges if e not in current_edge_set]
    
    rng = random.Random(99999)
    
    improved = True
    stall_count = 0
    
    while time.time() < deadline - 10:
        if not improved:
            stall_count += 1
            if stall_count > 30:
                break
            improved = False
        else:
            stall_count = 0
        
        if not missing_edges:
            break
        
        # Try to add a random missing edge
        added = False
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            idx = rng.randrange(len(missing_edges))
            u, v = missing_edges[idx]
            if try_add_edge(adj, neighbors, u, v, edges, rng):
                current_edge_set.add((min(u, v), max(u, v)))
                missing_edges.pop(idx)
                added = True
                improved = True
                break
        
        if added:
            continue
        
        # If no edge could be added, try to recolor
        # Pick a random missing edge and try to recolor an edge in a blocking cycle
        target_u, target_v = rng.choice(missing_edges)
        
        # Find a blocking 4-cycle: a path target_u - a - b - target_v where all edges exist
        # and the forbidden colors cover all 20 colors.
        # We'll just try to recolor random edges and see if it helps.
        
        # Collect all edges that are in some 4-cycle with (target_u, target_v)
        blocking_edges = []
        for (a, b) in edge_paths[edge_index[(target_u, target_v)]]:
            if adj[target_u][a] != -1 and adj[a][b] != -1 and adj[b][target_v] != -1:
                blocking_edges.append((target_u, a))
                blocking_edges.append((a, b))
                blocking_edges.append((b, target_v))
        
        if blocking_edges:
            # Try to recolor one of these edges
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                x, y = rng.choice(blocking_edges)
                # Try random new colors
                for new_c in rng.sample(range(num_colors), min(5, num_colors)):
                    if try_recolor(adj, neighbors, x, y, new_c, edges):
                        # Try to add the target edge again
                        if try_add_edge(adj, neighbors, target_u, target_v, edges, rng):
                            current_edge_set.add((min(target_u, target_v), max(target_u, target_v)))
                            if (target_u, target_v) in missing_edges:
                                missing_edges.remove((target_u, target_v))
                            elif (target_v, target_u) in missing_edges:
                                missing_edges.remove((target_v, target_u))
                            improved = True
                            break
                if improved:
                    break
    
    # Final validation
    if is_valid_full(adj):
        return edges
    else:
        return best_edges

