import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n = 20
    num_colors = 20
    
    # Initialize dense random graph
    p = 0.85
    edges = set()
    edge_colors = {}
    
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                edges.add((u, v))
                edge_colors[(u, v)] = random.randint(0, num_colors - 1)
                
    # Adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_color(u, v):
        if u > v:
            u, v = v, u
        return edge_colors.get((u, v), -1)
        
    def is_rainbow(c1, c2, c3, c4):
        return len({c1, c2, c3, c4}) == 4
        
    def count_bad_cycles_for_edge(u, v):
        """
        Count bad 4-cycles containing edge (u,v).
        Cycle: u-v-x-y-u
        """
        bad_count = 0
        c_uv = get_color(u, v)
        for x in adj[v]:
            if x == u:
                continue
            c_vx = get_color(v, x)
            for y in adj[u]:
                if y == v or y == x:
                    continue
                if (min(x, y), max(x, y)) in edges:
                    c_xy = get_color(x, y)
                    c_yu = get_color(y, u)
                    if not is_rainbow(c_uv, c_vx, c_xy, c_yu):
                        bad_count += 1
        return bad_count

    def total_bad_cycles():
        total = 0
        for u, v in edges:
            total += count_bad_cycles_for_edge(u, v)
        return total // 4

    total_bad = total_bad_cycles()
    
    # Heuristic: If initial graph is too bad, maybe restart with different p?
    # But let's try to repair first.
    
    while total_bad > 0 and time.time() < deadline:
        # Find edge with max bad cycles
        max_bad = 0
        target_edge = None
        for u, v in edges:
            b = count_bad_cycles_for_edge(u, v)
            if b > max_bad:
                max_bad = b
                target_edge = (u, v)
                
        if target_edge is None:
            break
            
        u, v = target_edge
        current_bad = max_bad
        
        best_color = None
        best_bad_count = current_bad
        second_best_color = None
        second_best_count = current_bad
        
        for c in range(num_colors):
            if c == edge_colors[(u, v)]:
                continue
            # Temporarily set
            old_color = edge_colors[(u, v)]
            edge_colors[(u, v)] = c
            new_bad = count_bad_cycles_for_edge(u, v)
            edge_colors[(u, v)] = old_color
            
            if new_bad < best_bad_count:
                second_best_color = best_color
                second_best_count = best_bad_count
                best_color = c
                best_bad_count = new_bad
            elif new_bad < second_best_count:
                second_best_color = c
                second_best_count = new_bad
                
        # Decide action
        if best_color is not None and best_bad_count < current_bad:
            # Strict improvement
            edge_colors[(u, v)] = best_color
            total_bad += best_bad_count - current_bad
        elif second_best_color is not None and second_best_count == current_bad:
            # Plateau move: keep same count, change color to explore
            # Only do this occasionally or if stuck?
            # Let's do it if we are stuck (no strict improvement found)
            # But we need to be careful not to oscillate.
            # Given time limit, it's probably fine.
            edge_colors[(u, v)] = second_best_color
            # total_bad unchanged
        else:
            # No improvement or plateau move, remove edge
            edges.remove((u, v))
            del edge_colors[(u, v)]
            adj[u].remove(v)
            adj[v].remove(u)
            total_bad -= current_bad

    # Build result
    result = []
    for u, v in edges:
        c = edge_colors[(u, v)]
        result.append([u, v, c])
        
    return result
