"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 85.0
    random.seed(42)
    
    n = P['n']
    edges = P['edges']
    
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def time_check():
        return time.time() > deadline
    
    # Find max clique
    max_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal max_clique
        if time_check():
            return
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        if P_set or X:
            pivot = max(list(P_set | X), key=lambda v: len(adj[v] & P_set))
            candidates = [v for v in P_set if v not in adj[pivot]]
        else:
            candidates = list(P_set)
        for v in candidates:
            bron_kerbosch(R + [v], P_set & adj[v], X & adj[v])
            P_set.discard(v)
            X.add(v)
    
    bron_kerbosch([], set(range(n)), set())
    
    if not max_clique:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    base_clique = [int(v) for v in max_clique]
    base_k = len(base_clique)
    
    def find_paths_for_branches(B, max_orderings=20):
        """Try multiple orderings of pairs to find disjoint paths."""
        B_set = set(B)
        pairs = list(combinations(B, 2))
        
        def bfs_path(u, v, forbidden):
            """Find shortest path from u to v avoiding forbidden (except endpoints)."""
            if u == v:
                return [u]
            visited = {u}
            queue = deque([(u, [u])])
            while queue:
                node, path = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor in visited:
                        continue
                    if neighbor in forbidden and neighbor != v:
                        continue
                    if neighbor == v:
                        return path + [v]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            return None
        
        # Try multiple orderings
        best_result = None
        
        # Ordering 1: by shortest path length
        def ordering_by_shortest():
            result = []
            for u, v in pairs:
                path = bfs_path(u, v, B_set)
                result.append((len(path) if path else float('inf'), u, v))
            result.sort(key=lambda x: x[0])
            return [(u, v) for _, u, v in result]
        
        # Ordering 2: by degree (low degree first)
        degree = {i: len(adj[i]) for i in range(n)}
        def ordering_by_degree():
            return sorted(pairs, key=lambda p: degree[p[0]] + degree[p[1]])
        
        # Ordering 3: random
        def ordering_random():
            p = list(pairs)
            random.shuffle(p)
            return p
        
        orderings = [ordering_by_shortest(), ordering_by_degree()]
        for _ in range(max_orderings - 2):
            orderings.append(ordering_random())
        
        for ordering in orderings:
            if time_check():
                break
            used_interiors = set()
            found_paths = {}
            success = True
            
            for u, v in ordering:
                path = bfs_path(u, v, B_set | used_interiors)
                if path is None:
                    success = False
                    break
                interior = path[1:-1]
                if any(x in B_set for x in interior):
                    success = False
                    break
                found_paths[(u, v)] = path
                used_interiors.update(interior)
            
            if success:
                res_paths = []
                for u, v in pairs:
                    if (u, v) in found_paths:
                        res_paths.append(found_paths[(u, v)])
                    elif (v, u) in found_paths:
                        res_paths.append(found_paths[(v, u)][::-1])
                    else:
                        success = False
                        break
                if success:
                    best_result = res_paths
                    break
        
        return best_result
    
    best_branches = base_clique
    best_paths = [[int(u), int(v)] for u, v in combinations(base_clique, 2)]
    
    # Local search: try to grow the set
    degree = {i: len(adj[i]) for i in range(n)}
    sorted_vertices = sorted(range(n), key=lambda v: -degree[v])
    
    # Try adding vertices one at a time
    current_set = set(base_clique)
    improved = True
    while improved and not time_check():
        improved = False
        for v in sorted_vertices:
            if time_check():
                break
            if v in current_set:
                continue
            new_set = current_set | {v}
            B = sorted(new_set)
            paths = find_paths_for_branches(B, max_orderings=10)
            if paths is not None:
                # Validate
                valid = True
                all_interiors = set()
                for path in paths:
                    for x in path[1:-1]:
                        if x in current_set | {v}:
                            valid = False
                            break
                        if x in all_interiors:
                            valid = False
                            break
                        all_interiors.add(x)
                    for i in range(len(path) - 1):
                        if path[i+1] not in adj[path[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_branches = B
                    best_paths = paths
                    current_set = new_set
                    improved = True
                    break
    
    # Also try random restarts with larger sets
    for attempt in range(50):
        if time_check():
            break
        k = random.randint(base_k + 1, min(n, base_k + 5))
        B = random.sample(range(n), k)
        paths = find_paths_for_branches(B, max_orderings=5)
        if paths is not None:
            B_set = set(B)
            valid = True
            all_interiors = set()
            for path in paths:
                for x in path[1:-1]:
                    if x in B_set:
                        valid = False
                        break
                    if x in all_interiors:
                        valid = False
                        break
                    all_interiors.add(x)
                for i in range(len(path) - 1):
                    if path[i+1] not in adj[path[i]]:
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(B) > len(best_branches):
                best_branches = [int(v) for v in B]
                best_paths = paths
    
    if len(best_branches) < 2:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return {'branches': best_branches, 'paths': best_paths}

# EVOLVE-BLOCK-END
