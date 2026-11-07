import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    start_time = time.time()
    deadline = start_time + 80
    
    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_diam_leq_4(adj):
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
            if -1 in dist:
                return False
        return True
    
    def get_all_dists(adj):
        all_dists = []
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
            all_dists.append(dist)
        return all_dists
    
    def can_add_edge(adj, u, v):
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True
    
    def greedy_solve(randomize=False):
        edge_set = set(base_edges)
        adj = build_adj(edge_set)
        
        while not is_diam_leq_4(adj):
            if time.time() > deadline:
                break
            
            all_dists = get_all_dists(adj)
            
            # Find pairs with distance > 4
            far_pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    if all_dists[i][j] > 4:
                        far_pairs.append((i, j, all_dists[i][j]))
            
            if not far_pairs:
                break
            
            # Generate candidates: edges that might help reduce far distances
            candidates = set()
            for (i, j, d) in far_pairs:
                # Try direct edge
                candidates.add((i, j))
                # Try edges that create shortcuts on the path between i and j
                # For now, just use direct edge
            
            # Also add some random candidates for diversity
            if randomize:
                for _ in range(20):
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i > j:
                        i, j = j, i
                    candidates.add((i, j))
            
            # Filter candidates
            valid_candidates = []
            for (i, j) in candidates:
                if (i, j) in edge_set or (i, j) in base_edges:
                    continue
                if not can_add_edge(adj, i, j):
                    continue
                valid_candidates.append((i, j))
            
            if not valid_candidates:
                break
            
            if randomize:
                random.shuffle(valid_candidates)
            
            best_edge = None
            best_score = -1
            
            for (i, j) in valid_candidates:
                if time.time() > deadline:
                    break
                new_set = edge_set | {(i, j)}
                new_adj = build_adj(new_set)
                if is_diam_leq_4(new_adj):
                    # Count how many far pairs are resolved
                    new_dists = get_all_dists(new_adj)
                    resolved = 0
                    for (a, b, _) in far_pairs:
                        if new_dists[a][b] <= 4:
                            resolved += 1
                    if resolved > best_score:
                        best_score = resolved
                        best_edge = (i, j)
                    if best_score == len(far_pairs):
                        break
                else:
                    # Still, track if it reduces the number of far pairs
                    new_dists = get_all_dists(new_adj)
                    new_far = sum(1 for a in range(n) for b in range(a+1, n) if new_dists[a][b] > 4)
                    old_far = len(far_pairs)
                    if new_far < old_far:
                        if -new_far > best_score:
                            best_score = -new_far
                            best_edge = (i, j)
            
            if best_edge is None:
                # Fallback: try all candidates
                for i in range(n):
                    for j in range(i+1, n):
                        if (i, j) in edge_set or (i, j) in base_edges:
                            continue
                        if not can_add_edge(adj, i, j):
                            continue
                        new_set = edge_set | {(i, j)}
                        new_adj = build_adj(new_set)
                        if is_diam_leq_4(new_adj):
                            best_edge = (i, j)
                            break
                        new_dists = get_all_dists(new_adj)
                        new_far = sum(1 for a in range(n) for b in range(a+1, n) if new_dists[a][b] > 4)
                        if new_far < len(far_pairs):
                            if best_edge is None:
                                best_edge = (i, j)
                    if best_edge is not None:
                        break
            
            if best_edge is None:
                break
            
            edge_set.add(best_edge)
            adj = build_adj(edge_set)
        
        added = [(u, v) for u, v in edge_set if (u, v) not in base_edges]
        return added
    
    def local_search(added_edges):
        edge_set = set(base_edges) | set(added_edges)
        adj = build_adj(edge_set)
        
        if not is_diam_leq_4(adj):
            return added_edges
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for edge in list(added_edges):
                if time.time() > deadline:
                    break
                u, v = edge
                new_set = edge_set - {(u, v)}
                new_adj = build_adj(new_set)
                if is_diam_leq_4(new_adj):
                    edge_set = new_set
                    adj = new_adj
                    added_edges = [(a, b) for a, b in added_edges if (a, b) != (u, v)]
                    improved = True
                    break
        
        return added_edges
    
    best_result = None
    best_count = float('inf')
    
    random.seed(77777)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        result = greedy_solve(randomize=(attempt > 0))
        if result is None:
            continue
        
        result = local_search(result)
        
        if len(result) < best_count:
            best_count = len(result)
            best_result = result
        
        if best_count <= 10:
            break
    
    if best_result is None:
        return [[u, v] for u in range(n) for v in range(u+1, n) if (u-v) % 2 != 0 and (u, v) not in base_edges]
    
    return [[int(u), int(v)] for u, v in best_result]
