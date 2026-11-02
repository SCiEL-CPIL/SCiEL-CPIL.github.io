import time
import random
import sys

def construct(k, g):
    # We focus on k=3, g=16 as per the task description hint.
    # If k or g are different, we fall back to LUW or a simple random search.
    # But the task says "evaluator passes this task's degree k and girth g".
    # The seed uses LUW for general k, g.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    if k != 3 or g != 16:
        # Fallback to LUW for other parameters to ensure validity
        # (Though the prompt implies k=3, g=16 is the target)
        return _luw_construct(k, g)

    # Target n
    # Known best is 936. Let's aim for 936 to 1200.
    # If we can't find girth 16, we return the best effort.
    
    best_graph = None
    best_girth = 0
    best_n = float('inf')
    
    # Try a few different n values
    # 936 is the known optimum.
    # Random 3-regular graphs on 936 vertices might be hard to fix.
    # Larger n is easier to fix.
    # Let's try 1000 first.
    
    for n_attempt in [1000, 1200, 1500]:
        if time.time() > deadline:
            break
            
        # Generate random 3-regular graph
        graph = _generate_random_3_reg(n_attempt)
        if graph is None:
            continue
            
        # Try to improve girth
        improved_graph, girth = _improve_girth(graph, n_attempt, g, deadline)
        
        if girth > best_girth or (girth == best_girth and n_attempt < best_n):
            best_girth = girth
            best_n = n_attempt
            best_graph = improved_graph
            
        if best_girth >= g:
            # Found a valid graph!
            # We might try to reduce n? 
            # No, n is fixed in the generation.
            # We can return immediately if we found one.
            # But maybe a smaller n was possible?
            # We tried 1000. If we found one, it's good.
            # Let's return.
            return _graph_to_edges(best_graph, best_n)

    # If no graph with girth >= g found, return the best one we have
    if best_graph is not None:
        return _graph_to_edges(best_graph, best_n)
        
    # Fallback
    return _luw_construct(k, g)

def _graph_to_edges(adj, n):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return n, edges

def _generate_random_3_reg(n):
    # Pairing model
    # 3n half-edges
    # n must be even
    if n % 2 != 0:
        n += 1
        
    adj = [set() for _ in range(n)]
    half_edges = []
    for i in range(n):
        for _ in range(3):
            half_edges.append(i)
            
    random.shuffle(half_edges)
    
    # Pair them up
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            # Loop, reject this graph
            return None
        if v in adj[u]:
            # Multi-edge, reject
            return None
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degrees
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    return adj

def _improve_girth(adj, n, target_girth, deadline):
    # BFS to find girth
    # And rewiring
    
    girth = _calc_girth(adj, n, target_girth)
    
    while time.time() < deadline - 5.0:
        if girth >= target_girth:
            break
            
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, target_girth)
        if not cycle:
            # Girth calculation might have been optimistic?
            # Or cycle was broken by previous step but girth not updated?
            girth = _calc_girth(adj, n, target_girth)
            if girth >= target_girth:
                break
            else:
                # Should not happen if _find_short_cycle is correct
                break
        
        # Try to break the cycle
        success = False
        for _ in range(100): # Try 100 random swaps
            # Pick an edge in the cycle
            e_idx = random.randint(0, len(cycle)-1)
            u = cycle[e_idx]
            v = cycle[(e_idx+1) % len(cycle)]
            
            # Pick a random other edge (w, x)
            # Ensure w, x are not u, v
            # And (u,w), (v,x) are not existing edges
            while True:
                w = random.randint(0, n-1)
                if w == u or w == v:
                    continue
                x = random.choice(list(adj[w]))
                if x == u or x == v:
                    continue
                # Check if (u,w) or (v,x) exist
                if w in adj[u] or x in adj[v]:
                    continue
                # Check if (u,x) or (v,w) exist? 
                # We are swapping (u,v) and (w,x) to (u,w) and (v,x).
                # We need to ensure (u,w) and (v,x) are NEW edges.
                # We already checked that.
                # We also need to ensure we don't create multi-edges with OTHER neighbors?
                # No, we are just adding two edges.
                # But we must remove (u,v) and (w,x).
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(x)
                adj[x].add(v)
                
                success = True
                break
            
        if not success:
            # Could not find a valid swap, try next cycle or give up
            continue
            
        # Recalculate girth
        girth = _calc_girth(adj, n, target_girth)
        
    return adj, girth

def _calc_girth(adj, n, max_check):
    # BFS from each node
    # We only care about cycles < max_check
    # So we stop BFS at depth max_check // 2
    max_depth = max_check // 2
    min_cycle = max_check
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # But this is only valid if the paths are disjoint.
                    # In a tree, they are disjoint.
                    # Since we are doing BFS, the first time we hit a visited node
                    # (that is not parent), we found a cycle.
                    # However, this cycle might be long.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        
        if min_cycle == 3: # Can't get lower than 3 for simple graph
            break
            
    return min_cycle

def _find_short_cycle(adj, n, max_check):
    # Similar to calc girth, but returns the cycle
    max_depth = max_check // 2
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < max_check:
                        # Reconstruct cycle
                        # Path s -> ... -> u
                        # Path s -> ... -> v
                        # Cycle is u -> ... -> s -> ... -> v -> u
                        path1 = [u]
                        cur = u
                        while cur != s:
                            cur = parent[cur]
                            path1.append(cur)
                        
                        path2 = [v]
                        cur = v
                        while cur != s:
                            cur = parent[cur]
                            path2.append(cur)
                        
                        # path1 is u...s, path2 is v...s
                        # Cycle: u ... s ... v u
                        # Note: s is in both.
                        # The cycle is path1 + path2[1:] + [u]?
                        # No, path1 ends at s. path2 ends at s.
                        # Cycle: u -> ... -> s -> ... -> v -> u
                        # path1: u, ..., s
                        # path2: v, ..., s
                        # Reverse path2: s, ..., v
                        # Combine: path1 + path2[1:] (which is s...v without s? No, path2[0] is v)
                        # Let's just return the list of nodes in the cycle.
                        # The nodes are: path1 (u to s) and path2 (v to s).
                        # The cycle is path1 + path2[1:] + [u] ?
                        # Wait, path2[1:] is s...v? No, path2 is v...s.
                        # So path2 reversed is s...v.
                        # path1 is u...s.
                        # Cycle: u...s...v...u
                        # Nodes: path1 + path2[1:] (which is s...v? No, path2[1] is parent of v)
                        # Let's just collect unique nodes.
                        cycle_nodes = set(path1) | set(path2)
                        # We need the order.
                        # Let's return the list of nodes in the cycle in order.
                        # path1: u -> p(u) -> ... -> s
                        # path2: v -> p(v) -> ... -> s
                        # Cycle: u -> ... -> s -> ... -> v -> u
                        # So: path1 (reversed? No, u is start)
                        # path1 is [u, ..., s]
                        # path2 is [v, ..., s]
                        # We need to go from s to v.
                        # Reverse path2: [s, ..., v]
                        # So cycle: path1 + path2[1:] (which is s...v? No, path2[0] is v, path2[-1] is s)
                        # path2 reversed: [s, ..., v]
                        # So cycle: path1 + path2_reversed[1:] + [u]
                        # path1 ends with s.
                        # path2_reversed starts with s.
                        # So: path1 + path2_reversed[1:]
                        # And close the loop by adding u?
                        # path1[0] is u.
                        # path2_reversed[-1] is v.
                        # Edge (v, u) exists.
                        # So the cycle is path1 + path2_reversed[1:]
                        
                        p2_rev = path2[::-1]
                        cycle = path1 + p2_rev[1:]
                        return cycle
                        
    return None

def _luw_construct(k, g):
    # Original seed logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
