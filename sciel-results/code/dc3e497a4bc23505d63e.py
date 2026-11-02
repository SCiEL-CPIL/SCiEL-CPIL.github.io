import random
import time
import sys

def construct(k, g):
    # Fallback to seed logic if we fail or don't have time
    # But we want to try to improve.
    
    # Parameters
    target_k = k
    target_g = g
    
    # If k, g are not 4, 10, we might not have a specific strategy.
    # The task is specific to 4,10? "evaluator passes this task's degree k and girth g".
    # It implies k=4, g=10.
    
    if k != 4 or g != 10:
        # Fallback to seed for other parameters
        return _seed_construct(k, g)

    # Try to construct a better graph for 4,10
    # Known cage is 320.
    # We try to find a graph with n <= 320.
    
    # Time management
    start_time = time.time()
    time_limit = 120  # 150s wall clock, keep margin
    
    # We will try n values.
    # Since 320 is known to exist, we can try 320 first.
    # Then try smaller n?
    # If we find one at 320, score is 2048/320 = 6.4.
    # If we find one at 300, score is 2048/300 = 6.8.
    
    # Let's try a range.
    # We can't spend too much time on each n.
    # Let's try n = 320, 310, 300, 290, 280, 270, 260, 250, 244.
    # If we find a valid graph, return it.
    # If we run out of time, return the best found so far?
    # Or return seed?
    # If we find NO graph, return seed.
    
    best_n = None
    best_edges = None
    
    # Try n from 320 down to 244
    # But searching 244 might take forever.
    # Maybe just try 320 and 300?
    # Let's try 320 first.
    
    candidates = [320, 310, 300, 290, 280, 270, 260, 250, 244]
    
    for n in candidates:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try to construct graph of size n
        # Use multiple restarts
        max_restarts = 50
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            edges, success = _greedy_construct(n, k, g, restart)
            if success:
                # Verify girth exactly?
                # The greedy check is necessary but might miss some?
                # No, if we check dist < g-1 before adding, we ensure no cycle < g.
                # We must ensure the graph is connected.
                if _is_connected(edges, n):
                    best_n = n
                    best_edges = edges
                    found = True
                    break
        if found:
            # We found a graph with this n.
            # Can we try smaller?
            # Yes, but we have limited time.
            # Let's try to find a smaller one if we have time.
            # But the score improvement from 320 to 310 is small.
            # 2048/320 = 6.4
            # 2048/310 = 6.6
            # Let's try to find the smallest possible.
            # But if we spend too much time, we might fail to find 320.
            # So we return the first one found?
            # Or we can try a few smaller ones if time permits.
            
            # Let's try to find a smaller one.
            # We will break the loop and try smaller n's.
            # But we need to keep the current best.
            
            # Actually, if we find 320, we can try 310, 300...
            # But we must be careful with time.
            
            # Let's just return the first one found to be safe?
            # No, the goal is to minimize n.
            # Let's try to find smaller ones.
            
            # We'll continue the loop to try smaller n.
            # But we should prioritize finding *a* graph.
            
            # Let's modify the loop:
            # Try 320. If found, save.
            # Try 310. If found, save (overwrites).
            # ...
            # If time runs out, return the best saved.
            
            # So we don't break immediately.
            # But we should limit time per n.
            pass

    if best_n is not None:
        return best_n, best_edges
    
    # If no graph found, return seed
    return _seed_construct(k, g)

def _greedy_construct(n, k, g, seed_offset):
    random.seed(42 + seed_offset)
    
    # Initialize
    edges = []
    deg = [0] * n
    adj = [set() for _ in range(n)]
    
    # List of all potential edges
    # To avoid generating all pairs (which is O(n^2)), we can just pick random pairs.
    # But we need to ensure we cover all possibilities.
    # Generating all pairs for n=320 is ~50k. That's fine.
    
    all_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            all_pairs.append((u, v))
            
    random.shuffle(all_pairs)
    
    # We need to add n*k/2 = 2n edges.
    target_edges = n * k // 2
    
    # Greedy addition
    for u, v in all_pairs:
        if len(edges) >= target_edges:
            break
            
        if deg[u] >= k or deg[v] >= k:
            continue
            
        # Check if adding (u,v) creates a cycle of length < g
        # This is equivalent to checking if dist(u,v) < g-1 in current graph
        # If dist(u,v) <= g-2, then adding edge creates cycle of length dist+1 <= g-1.
        # We want no cycle < g. So cycle length must be >= g.
        # So we must not have dist(u,v) <= g-2.
        # So we require dist(u,v) >= g-1.
        
        # BFS from u to v, limit depth to g-2
        # If found, skip.
        
        # Optimization: If u and v are in different components, dist is infinite, so ok.
        # We can maintain components? Or just BFS.
        
        if _has_short_path(adj, u, v, g - 2):
            continue
            
        # Add edge
        edges.append((u, v))
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    # Check if we reached target edges
    if len(edges) < target_edges:
        return None, False
        
    # Check if graph is connected
    # (Done in caller)
    
    return edges, True

def _has_short_path(adj, u, v, max_depth):
    # BFS from u up to max_depth
    # Return True if v is reached within max_depth
    
    if u == v:
        return True # Self loop? No, simple graph. But u!=v in pairs.
        
    visited = {u: 0}
    queue = [(u, 0)]
    
    while queue:
        node, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for neighbor in adj[node]:
            if neighbor == v:
                return True
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
                
    return False

def _is_connected(edges, n):
    if n == 0:
        return True
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    visited = [False] * n
    queue = [0]
    visited[0] = True
    
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                
    return all(visited)

def _seed_construct(k, g):
    # Copy of the original seed logic
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
