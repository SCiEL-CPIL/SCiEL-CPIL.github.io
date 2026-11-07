import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        if dist[v] > 4:
                            return max_dist
                        queue.append(v)
            if -1 in dist:
                return float('inf')
        return max_dist
    
    def is_valid(added_edges):
        adj = build_adj(added_edges)
        return get_diameter(adj) <= 4
    
    def is_triangle_free(added_edges):
        # Check bipartite: all edges must connect even to odd
        for u, v in added_edges:
            if u % 2 == v % 2:
                return False
        return True
    
    # Precompute all valid (bipartite-compatible) non-base edges
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            if u % 2 != v % 2:
                valid_edges.append((u, v))
    
    # Constructive approach: design a small set of edges that achieves diameter 4
    # Strategy: create "jump" edges that fold the path
    # 
    # Key idea: If we add edges that let us jump ~5-6 positions, combined with
    # the path edges, we can cover the 24-vertex path within diameter 4.
    #
    # Let's try a structured pattern:
    # Add edges (i, i+5) for appropriate i, and (i, i+3) for others
    # 
    # Actually, let me think more carefully.
    # With diameter 4, from any vertex we can reach any other in ≤ 4 steps.
    # The path already gives us ±1 moves. Added edges give us longer jumps.
    #
    # A good pattern: add edges that create a "grid" of jumps
    # Try: edges at offsets 5 and 7 (both odd, so bipartite-compatible)
    
    # Let me try a specific small solution and verify:
    # Candidate: 6 edges
    # (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)
    # This creates a "ladder" structure in the first 11 vertices
    # But we need to cover all 24 vertices...
    
    # Better approach: use a systematic construction
    # Divide the path into segments and add "bridge" edges
    
    # Let me try: add edges that create shortcuts of length 5
    # (0,5), (5,10), (10,15), (15,20) - these are the "main" jumps
    # Plus some cross-edges to reduce diameter further
    
    # Actually, let me use a different structural approach:
    # Create a "complete bipartite" substructure on selected vertices
    
    # Let me try a greedy construction with a smarter scoring function
    # that uses the actual diameter reduction, not just far-pair count
    
    def greedy_construct():
        if time.time() > deadline:
            return set()
        
        added = set()
        adj = build_adj(added)
        
        # Precompute initial distances
        def compute_all_dists(adj):
            dists = []
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                queue = deque([src])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                dists.append(dist)
            return dists
        
        dists = compute_all_dists(adj)
        
        while True:
            if time.time() > deadline:
                break
            
            # Check if diameter <= 4
            max_d = 0
            for d in dists:
                for val in d:
                    if val > max_d:
                        max_d = val
            if max_d <= 4:
                break
            
            # Find pairs with distance > 4
            far_pairs = set()
            for src in range(n):
                for dst in range(src + 1, n):
                    if dists[src][dst] > 4:
                        far_pairs.add((src, dst))
            
            if not far_pairs:
                break
            
            # For each candidate edge, compute how many far pairs it would resolve
            best_edge = None
            best_improvement = -1
            
            for u, v in valid_edges:
                if (u, v) in added:
                    continue
                
                # Temporarily add edge
                new_adj = build_adj(added | {(u, v)})
                new_dists = compute_all_dists(new_adj)
                
                new_far = 0
                for src in range(n):
                    for dst in range(src + 1, n):
                        if new_dists[src][dst] > 4:
                            new_far += 1
                
                improvement = len(far_pairs) - new_far
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_edge = (u, v)
                
                if new_far == 0:
                    best_edge = (u, v)
                    break
            
            if best_edge is None or best_improvement <= 0:
                break
            
            added.add(best_edge)
            adj = build_adj(added)
            dists = compute_all_dists(adj)
        
        return added
    
    def local_search_reduce(added_edges):
        if time.time() > deadline:
            return added_edges
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for edge in list(added_edges):
                if time.time() > deadline:
                    break
                new_set = added_edges - {edge}
                if is_valid(new_set):
                    added_edges = new_set
                    improved = True
                    break
        return added_edges
    
    # Try multiple constructive approaches and pick the best
    
    best_result = None
    best_count = float('inf')
    
    # Approach 1: Pure greedy
    res = greedy_construct()
    if is_valid(res):
        res = local_search_reduce(res)
        if len(res) < best_count:
            best_count = len(res)
            best_result = res
    
    # Approach 2: Structured pattern - add edges at regular intervals
    # Try: (i, i+5) for i = 0,1,2,3,4,5 and (i, i+7) for i = 0,1,2,3
    for pattern_name, edges_list in [
        ("offset5", [(i, i+5) for i in range(n-5) if i % 2 != (i+5) % 2]),
        ("offset7", [(i, i+7) for i in range(n-7) if i % 2 != (i+7) % 2]),
        ("offset5_7", [(i, i+5) for i in range(n-5) if i % 2 != (i+5) % 2] + 
                      [(i, i+7) for i in range(n-7) if i % 2 != (i+7) % 2]),
        ("offset3_5", [(i, i+3) for i in range(n-3) if i % 2 != (i+3) % 2] + 
                      [(i, i+5) for i in range(n-5) if i % 2 != (i+5) % 2]),
        ("offset3_7", [(i, i+3) for i in range(n-3) if i % 2 != (i+3) % 2] + 
                      [(i, i+7) for i in range(n-7) if i % 2 != (i+7) % 2]),
    ]:
        edge_set = set()
        for e in edges_list:
            if e not in base_edges:
                edge_set.add(e)
        if is_valid(edge_set):
            edge_set = local_search_reduce(edge_set)
            if len(edge_set) < best_count:
                best_count = len(edge_set)
                best_result = edge_set
    
    # Approach 3: Hub-based - select hub vertices and connect to others
    # Hubs should be spread out to cover the path
    hub_sets = [
        [0, 6, 12, 18],
        [1, 7, 13, 19],
        [0, 5, 10, 15, 20],
        [2, 7, 12, 17, 22],
        [0, 4, 8, 12, 16, 20],
        [1, 5, 9, 13, 17, 21],
    ]
    
    for hubs in hub_sets:
        hub_edges = set()
        for h in hubs:
            for v in range(n):
                if v == h:
                    continue
                if (min(h,v), max(h,v)) in base_edges:
                    continue
                if h % 2 != v % 2:
                    hub_edges.add((min(h,v), max(h,v)))
        
        if is_valid(hub_edges):
            hub_edges = local_search_reduce(hub_edges)
            if len(hub_edges) < best_count:
                best_count = len(hub_edges)
                best_result = hub_edges
    
    # Approach 4: Randomized restarts with greedy
    import random
    random.seed(123)
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        added = set()
        adj = build_adj(added)
        
        def compute_all_dists(adj):
            dists = []
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                queue = deque([src])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                dists.append(dist)
            return dists
        
        dists = compute_all_dists(adj)
        
        while True:
            if time.time() > deadline:
                break
            
            max_d = max(max(d) for d in dists)
            if max_d <= 4:
                break
            
            far_pairs = set()
            for src in range(n):
                for dst in range(src + 1, n):
                    if dists[src][dst] > 4:
                        far_pairs.add((src, dst))
            
            if not far_pairs:
                break
            
            candidates = [e for e in valid_edges if e not in added]
            if not candidates:
                break
            
            random.shuffle(candidates)
            best_edge = None
            best_improvement = -1
            
            for u, v in candidates:
                new_adj = build_adj(added | {(u, v)})
                new_dists = compute_all_dists(new_adj)
                
                new_far = 0
                for src in range(n):
                    for dst in range(src + 1, n):
                        if new_dists[src][dst] > 4:
                            new_far += 1
                
                improvement = len(far_pairs) - new_far
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_edge = (u, v)
                
                if new_far == 0:
                    best_edge = (u, v)
                    break
            
            if best_edge is None or best_improvement <= 0:
                break
            
            added.add(best_edge)
            adj = build_adj(added)
            dists = compute_all_dists(adj)
        
        if is_valid(added):
            added = local_search_reduce(added)
            if len(added) < best_count:
                best_count = len(added)
                best_result = added
    
    if best_result is None:
        # Fallback: dense solution
        dense_set = set(valid_edges)
        dense_set = local_search_reduce(dense_set)
        return [[int(u), int(v)] for u, v in dense_set]
    
    return [[int(u), int(v)] for u, v in best_result]
