import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        n_target = 320
        
        # Try optimized local search
        result = _local_search_optimized(n_target, g, deadline)
        if result is not None:
            return result

        # Fallback to LUW (will be worse, but valid)
        return _luw_construct(k, g)

    # General fallback
    return _luw_construct(k, g)

def _local_search_optimized(n, g, deadline):
    # Precompute adjacency structure helpers
    # We represent the graph as a list of lists (adjacency) and a set of edges for fast lookup/removal
    
    def generate_random_4reg(n):
        # Configuration model
        stubs = [i for i in range(n) for _ in range(4)]
        random.shuffle(stubs)
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None, None
            if (u, v) in edges or (v, u) in edges:
                return None, None
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        return adj, edges

    def check_girth_fast(adj, n, g):
        # Returns True if girth >= g
        # BFS from each node to depth g//2 - 1
        # If we find a cycle of length < g, return False
        
        # We only need to check cycles of length 4, 6, 8 (for g=10)
        # BFS depth 4 covers cycles up to length 9.
        max_depth = g // 2 - 1 # For g=10, max_depth=4
        
        for start in range(n):
            # dist[v] = distance from start
            # parent[v] = parent in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    else:
                        # If v is already visited, and v is not parent of u
                        if v == parent[u]:
                            continue
                        # Check if this edge creates a short cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        # But we need the LCA to be exact?
                        # If dist[u] + dist[v] + 1 < g, it's a candidate.
                        # However, if LCA is deeper, the cycle is shorter.
                        # If dist[u] + dist[v] + 1 >= g, it could still be < g if LCA is close to u/v?
                        # No, cycle_len = dist[u] + dist[v] + 1 - 2*dist[lca].
                        # Since dist[lca] >= 0, cycle_len <= dist[u] + dist[v] + 1.
                        # If dist[u] + dist[v] + 1 < g, we MUST check.
                        # If dist[u] + dist[v] + 1 >= g, the cycle is definitely >= g?
                        # No. Example: dist[u]=4, dist[v]=4, g=10. Sum=9.
                        # If lca is at depth 3, len = 4+4+1-6 = 3.
                        # So we must check if dist[u] + dist[v] + 1 < g is NOT sufficient to rule out.
                        # Wait, if dist[u] + dist[v] + 1 >= g, can cycle be < g?
                        # Yes, if lca is deep.
                        # But we are only BFS-ing to depth max_depth = 4.
                        # So dist[u] <= 4 and dist[v] <= 4.
                        # Max sum = 9.
                        # If g=10, any cycle found within depth 4 has length <= 9.
                        # So ANY back-edge to a visited node (that isn't the parent) within depth 4 implies a cycle of length < 10.
                        # Is that true?
                        # Cycle length = dist[u] + dist[v] + 1 - 2*dist[lca].
                        # Min length is 3 (triangle).
                        # Max length is dist[u] + dist[v] + 1 (if lca is root).
                        # Since dist[u], dist[v] <= 4, max length is 9.
                        # So yes, if we find a non-tree edge connecting two nodes in the BFS tree of depth <= 4, the cycle length is <= 9 < 10.
                        # Therefore, we just need to check if such an edge exists.
                        return False
        return True

    def find_short_cycle(adj, n, g):
        # Returns a list of edges in a short cycle, or None
        # Similar to check_girth_fast, but returns the cycle
        max_depth = g // 2 - 1
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    else:
                        if v == parent[u]:
                            continue
                        # Found a cycle
                        # Reconstruct path
                        # Path from u to start
                        path_u = []
                        cur = u
                        while cur != -1:
                            path_u.append(cur)
                            cur = parent[cur]
                        
                        # Path from v to start
                        path_v = []
                        cur = v
                        while cur != -1:
                            path_v.append(cur)
                            cur = parent[cur]
                        
                        # Find LCA
                        set_u = set(path_u)
                        lca = -1
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                            
                        if lca == -1:
                            continue # Should not happen

                        # Build cycle
                        # From u up to lca
                        cycle_nodes = []
                        cur = u
                        while cur != lca:
                            cycle_nodes.append(cur)
                            cur = parent[cur]
                        cycle_nodes.append(lca)
                        
                        # From lca up to v (reverse of path_v up to lca)
                        cur = v
                        while cur != lca:
                            cycle_nodes.append(cur)
                            cur = parent[cur]
                        
                        # Convert to edges
                        cycle_edges = []
                        for i in range(len(cycle_nodes)):
                            u1 = cycle_nodes[i]
                            v1 = cycle_nodes[(i+1) % len(cycle_nodes)]
                            # Ensure edge exists in adj
                            # It should, because it's either a tree edge or the back edge
                            # But the last edge (v1, u1) is the back edge (u, v) or (v, u)
                            # We need to be careful with the closing edge.
                            # The cycle is u1-v1-...-uk-u1.
                            # The edges are (u1,v1), (v1,v2)... (uk, u1).
                            # The last edge is (uk, u1) which corresponds to the back edge found.
                            # We just return the set of nodes, and the caller can verify.
                            pass
                        
                        # Simpler: return the edges directly
                        # Edges in path u->lca
                        edges = []
                        cur = u
                        while cur != lca:
                            edges.append((min(cur, parent[cur]), max(cur, parent[cur])))
                            cur = parent[cur]
                        
                        # Edges in path v->lca
                        cur = v
                        while cur != lca:
                            edges.append((min(cur, parent[cur]), max(cur, parent[cur])))
                            cur = parent[cur]
                        
                        # The closing edge (u, v)
                        edges.append((min(u, v), max(u, v)))
                        
                        return edges
        return None

    best_result = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        adj, edges = generate_random_4reg(n)
        if adj is None:
            continue
            
        if check_girth_fast(adj, n, g):
            return n, list(edges)
            
        # Local Search
        stuck_count = 0
        while time.time() < deadline:
            if stuck_count > 500:
                break
                
            # Find a short cycle
            short_cycle = find_short_cycle(adj, n, g)
            if short_cycle is None:
                # Should be caught by check_girth_fast, but just in case
                return n, list(edges)
                
            # Pick an edge from the cycle to break
            # Randomly pick one
            edge_to_break = random.choice(short_cycle)
            
            # Find another edge to swap with
            # To maintain 4-regularity, we do a 2-switch.
            # We need to pick another edge (x, y) such that the new edges are valid.
            # To increase chances of breaking the cycle without creating a new short one,
            # we should pick an edge far from the cycle.
            # For now, pick a random edge.
            
            all_edges = list(edges)
            # Remove the edge we want to break from the candidate list to avoid self-loop/duplicate issues easily?
            # No, we pick e2 from all edges.
            e2 = random.choice(all_edges)
            if e2 == edge_to_break:
                continue
                
            u1, v1 = edge_to_break
            u2, v2 = e2
            
            # Try the two possible 2-switches
            # Option 1: (u1, u2), (v1, v2)
            # Option 2: (u1, v2), (v1, u2)
            
            candidates = [
                ((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))),
                ((min(u1, v2), max(u1, v2)), (min(v1, u2), max(v1, u2)))
            ]
            
            # Filter valid candidates
            valid_candidates = []
            for ne1, ne2 in candidates:
                if ne1[0] == ne1[1] or ne2[0] == ne2[1]:
                    continue
                if ne1 in edges or ne2 in edges:
                    continue
                if ne1 == ne2:
                    continue
                valid_candidates.append((ne1, ne2))
                
            if not valid_candidates:
                stuck_count += 1
                continue
                
            ne1, ne2 = random.choice(valid_candidates)
            
            # Apply swap
            edges.remove(edge_to_break)
            edges.remove(e2)
            edges.add(ne1)
            edges.add(ne2)
            
            # Update adjacency
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].append(ne1[1] if ne1[0] == u1 else ne1[0])
            adj[ne1[1] if ne1[0] == u1 else ne1[0]].append(u1)
            
            adj[v1].append(ne2[1] if ne2[0] == v1 else ne2[0])
            adj[ne2[1] if ne2[0] == v1 else ne2[0]].append(v1)
            
            # Check girth
            if check_girth_fast(adj, n, g):
                return n, list(edges)
                
            stuck_count += 1
            
    return None

def _luw_construct(k, g):
    # Fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
