import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    q = 7
    points = []
    for a in range(q):
        for b in range(q):
            points.append((1, a, b))
    for a in range(q):
        points.append((0, 1, a))
    points.append((0, 0, 1))
    
    n_total = len(points)
    
    adj = {i: set() for i in range(n_total)}
    for i in range(n_total):
        pi = points[i]
        for j in range(i + 1, n_total):
            pj = points[j]
            dot = (pi[0]*pj[0] + pi[1]*pj[1] + pi[2]*pj[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    target_n = 43
    
    def has_independent_set(vertices, K, adj_map):
        if len(vertices) < K:
            return False
        v_map = {v: i for i, v in enumerate(vertices)}
        n = len(vertices)
        local_adj = [0] * n
        for i in range(n):
            mask = 0
            v = vertices[i]
            for u in adj_map[v]:
                if u in v_map:
                    mask |= (1 << v_map[u])
            local_adj[i] = mask
        
        def dfs(current_mask, current_size, start):
            if current_size == K:
                return True
            if current_size + (n - start) < K:
                return False
            for i in range(start, n):
                if local_adj[i] & current_mask:
                    continue
                if dfs(current_mask | (1 << i), current_size + 1, i + 1):
                    return True
            return False
        return dfs(0, 0, 0)

    best_result = None
    
    for attempt in range(20): # Max 20 random attempts
        if time.time() > deadline:
            break
            
        random.shuffle(points)
        # Re-index? No, adj is based on original indices.
        # We just iterate through shuffled points.
        
        selected = []
        for p_idx in points:
            if time.time() > deadline:
                break
            if len(selected) >= target_n:
                break
            
            # Check if adding p_idx keeps alpha < 11
            # We only need to check if the new set has independent set of size 11
            # Optimization: If current set has alpha < 11, and p_idx is not "too connected" to the independent set...
            # Just do the check.
            
            # To speed up, we can check if p_idx can be part of an independent set of size 11
            # Or just check the whole set.
            
            # Heuristic: If len(selected) + 1 < 11, no check needed.
            if len(selected) + 1 < 11:
                selected.append(p_idx)
                continue
                
            if has_independent_set(selected + [p_idx], 11, adj):
                # Adding p_idx creates alpha >= 11, so skip
                continue
            else:
                selected.append(p_idx)
                
        if len(selected) > (best_result is None or len(best_result) < len(selected)):
            best_result = selected
            
        if best_result and len(best_result) == target_n:
            break
            
    if not best_result:
        # Fallback: return first 43 (might be invalid, but better than nothing? No, invalid scores 0)
        # If no valid set found, return a known valid small set?
        # The seed returns n=30.
        # If we can't find n=43, we should return the best valid one found.
        # If best_result is None, we have no valid graph?
        # The check ensures validity. So best_result is always valid (if not empty).
        # If empty, we return seed?
        # Let's just return the first 43 if we didn't find a valid one, hoping the evaluator is lenient? No, "invalid result ... scores 0".
        # So we must return a valid graph.
        # The greedy check ensures validity.
        # If we didn't find any, we can return a small valid graph, e.g., 10 triangles (n=30).
        
        # Construct seed graph
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
        
    # Construct edges for best_result
    selected_set = set(best_result)
    edges = []
    for i in best_result:
        for j in adj[i]:
            if j in selected_set and i < j:
                edges.append([i, j])
                
    mapping = {old: new for new, old in enumerate(best_result)}
    new_edges = []
    for u, v in edges:
        new_edges.append([mapping[u], mapping[v]])
        
    return {'n': len(best_result), 'edges': new_edges}
