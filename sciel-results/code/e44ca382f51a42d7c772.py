"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import collections
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit for 60s total

    n = P['n']
    edges = P['edges']
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    adj = {i: list(set(adj[i])) for i in range(n)}  # Ensure no duplicates and consistent types
    
    # Precompute shortest paths (unweighted)
    dist = [[n+1 for _ in range(n)] for _ in range(n)]
    paths = [[None for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
        paths[i][i] = [i]
        queue = collections.deque([(i, [i])])
        while queue:
            node, path = queue.popleft()
            for nei in adj[node]:
                if dist[i][nei] == n+1:
                    dist[i][nei] = dist[i][node] + 1
                    new_path = path + [nei]
                    paths[i][nei] = new_path
                    queue.append((nei, new_path))

    # Function to find a K_k subdivision for a specific set of branches
    def find_subdivision(branches):
        k = len(branches)
        if k < 2:
            return None
        
        # Map branch vertex to index in branches list
        branch_to_idx = {b: i for i, b in enumerate(branches)}
        
        # Pairs of branch indices
        pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
        
        # Sort pairs by shortest path length to constrain search
        pair_order = sorted(pairs, key=lambda p: dist[branches[p[0]]][branches[p[1]]])

        def get_valid_path(u, v, used_internal):
            # BFS to find a path from u to v avoiding used_internal
            # u and v are branch vertices, so they are allowed.
            visited = set()
            visited.add(u)
            parent = {u: None}
            queue = collections.deque([u])
            
            while queue:
                curr = queue.popleft()
                if curr == v:
                    # Reconstruct path
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                for nei in adj[curr]:
                    if nei in visited:
                        continue
                    # Internal vertices cannot be ANY branch.
                    if nei in branch_to_idx and nei != v:
                        continue
                    if nei in used_internal:
                        continue
                        
                    visited.add(nei)
                    parent[nei] = curr
                    queue.append(nei)
            return None

        # Backtracking search
        def backtrack(idx, used_internal, assigned_paths):
            if idx == len(pair_order):
                return True
            
            i, j = pair_order[idx]
            u = branches[i]
            v = branches[j]
            
            # Find a path
            path = get_valid_path(u, v, used_internal)
            if path is None:
                return False
            
            # Mark internal vertices as used
            new_used = set(used_internal)
            for node in path[1:-1]:
                new_used.add(node)
            
            assigned_paths[(i, j)] = path
            
            if backtrack(idx + 1, new_used, assigned_paths):
                return True
            
            return False

        used_internal = set()
        assigned_paths = {}
        if backtrack(0, used_internal, assigned_paths):
            # Convert assigned_paths to the required format
            result_paths = []
            for i in range(k):
                for j in range(i+1, k):
                    result_paths.append(assigned_paths[(i, j)])
            return {'branches': list(branches), 'paths': result_paths}
        return None

    # Strategy:
    # 1. Find max clique.
    # 2. Try to find K_k subdivisions for k > max_clique.
    
    # Find Max Clique (Bron-Kerbosch or simple backtracking for n=24)
    # Simple backtracking for max clique
    def find_max_clique():
        max_clique = []
        candidates = list(range(n))
        
        def expand(clique, candidates):
            nonlocal max_clique
            if len(clique) > len(max_clique):
                max_clique = clique[:]
            if len(candidates) == 0:
                return
            # Pick a vertex
            v = candidates[0]
            # Include v
            new_candidates = [u for u in candidates[1:] if u in adj[v]]
            expand(clique + [v], new_candidates)
            # Exclude v
            expand(clique, candidates[1:])
            
        expand([], candidates)
        return max_clique

    max_clique = find_max_clique()
    best_result = {'branches': max_clique, 'paths': [[max_clique[i], max_clique[j]] for i in range(len(max_clique)) for j in range(i+1, len(max_clique))]}
    best_k = len(max_clique)
    
    # Try to find larger subdivisions
    # We will try k from best_k + 1 up to n
    # Candidate generation:
    # 1. If we have a clique of size k, try to add vertices.
    # 2. Or, generate random sets of size k and test.
    
    # Better: Use the structure of the graph.
    # High degree vertices are good candidates for branches.
    deg = [len(adj[i]) for i in range(n)]
    sorted_by_deg = sorted(range(n), key=lambda x: -deg[x])
    
    # Try to find K_{best_k+1}
    # We will test sets of size k.
    # To limit search space, we will only test sets that contain the max clique?
    # No, a subdivision might not contain the max clique.
    
    # Let's try a few random sets of high-degree vertices.
    random.seed(42)
    
    # We'll try a wider range of k and more random samples
    for k in range(best_k + 1, min(n, 15) + 1): # Limit to 15 to be safe
        if time.time() > deadline:
            break
            
        # Generate candidates
        # 1. Top k degree vertices
        # 2. Random samples from top 2k degree vertices
        # 3. If we have a valid set of size k-1, try to extend it?
        
        candidates_to_test = set()
        
        # Add top k degrees
        top_k = sorted_by_deg[:k]
        candidates_to_test.add(tuple(sorted(top_k)))
        
        # Add random sets from top 1.5k
        pool_size = min(n, int(k * 2.0) + 1)
        pool = sorted_by_deg[:pool_size]
        
        # Increase random samples
        for _ in range(100): 
            if time.time() > deadline:
                break
            sample = random.sample(pool, k)
            candidates_to_test.add(tuple(sorted(sample)))
            
        # Test each candidate
        for cand in candidates_to_test:
            if time.time() > deadline:
                break
            res = find_subdivision(list(cand))
            if res:
                best_result = res
                best_k = k
                break # Found a valid set of size k, move to k+1
        
    # Ensure types are correct
    if best_result:
        # Convert branches to list of ints
        best_result['branches'] = [int(b) for b in best_result['branches']]
        # Convert paths to list of lists of ints
        new_paths = []
        for p in best_result['paths']:
            new_paths.append([int(x) for x in p])
        best_result['paths'] = new_paths
        
    return best_result

# EVOLVE-BLOCK-END
