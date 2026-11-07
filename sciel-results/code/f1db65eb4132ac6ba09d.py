import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort vertices by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We will try to find a clique subdivision of size k
    # Start with high k and go down
    # Max possible is n, but realistically much lower.
    # Let's try k from 6 down to 3.
    
    best_result = None
    
    # Candidate pool: top 12 vertices by degree
    candidates = vertices_by_degree[:12]
    
    for k in range(6, 2, -1):
        if time.time() > deadline:
            break
            
        # Iterate through combinations of candidates of size k
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
                
            branches = list(combo)
            branches_set = set(branches)
            
            # Pairs to connect
            pairs = list(combinations(branches, 2))
            
            # Try to find disjoint paths
            # We use backtracking
            used_vertices = set()
            paths = [None] * len(pairs)
            
            def find_paths(idx):
                if time.time() > deadline:
                    return False
                
                if idx == len(pairs):
                    return True
                
                u, v = pairs[idx]
                
                # BFS to find a path from u to v avoiding used_vertices and branches (except u,v)
                # We want a simple path.
                # To help disjointness, we prefer shorter paths.
                
                # Standard BFS for shortest path
                parent = [-1] * n
                visited = [False] * n
                queue = deque()
                queue.append(u)
                visited[u] = True
                
                found = False
                while queue and not found:
                    curr = queue.popleft()
                    if curr == v:
                        found = True
                        break
                    for nei in adj[curr]:
                        if not visited[nei]:
                            # Check if nei is allowed
                            # If nei is a branch, it must be v (since u is start, v is target)
                            # Actually, path interiors cannot contain any branch.
                            # So if nei is in branches_set and nei != v, it's invalid.
                            # Also, if nei is in used_vertices, it's invalid.
                            if nei in branches_set and nei != v:
                                continue
                            if nei in used_vertices:
                                continue
                                
                            visited[nei] = True
                            parent[nei] = curr
                            queue.append(nei)
                
                if not found:
                    return False
                
                # Reconstruct path
                path = []
                curr = v
                while curr != -1:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                
                # Check if path is valid (simple, no other branches in interior)
                # BFS ensures simple path.
                # We already checked branches in BFS.
                
                # Mark interior vertices as used
                interior = set(path[1:-1])
                
                # Check if interior is disjoint from current used_vertices
                # It should be, because we checked `nei in used_vertices` in BFS.
                
                # However, BFS finds *a* path. It might not be the *best* one for disjointness.
                # For now, let's assume the first found path works. If it leads to a dead end, backtracking will handle it?
                # Wait, simple backtracking with "first found path" might fail if a different path would have worked.
                # To be robust, we should try multiple paths or use a more sophisticated search.
                # But for small graphs, "shortest path" is a strong heuristic.
                
                # Optimization: If we fail, we might need to try another path.
                # Let's implement a simple recursion that tries the shortest path first.
                # If that fails, we can try to find another path by blocking the current one?
                # That's expensive.
                
                # Let's stick to one path per BFS for speed, but be aware of its limitations.
                # If this fails to find a K5/K6, we might miss it.
                
                # To improve success rate, we can shuffle the order of pairs or try different BFS tie-breakers.
                
                # For now, let's try this.
                
                paths[idx] = path
                used_vertices.update(interior)
                
                if find_paths(idx + 1):
                    return True
                
                # Backtrack
                used_vertices.difference_update(interior)
                paths[idx] = None
                return False
            
            if find_paths(0):
                # Found a valid configuration
                # Construct the result
                result_paths = [p for p in paths if p is not None]
                # Ensure types are correct
                return {
                    'branches': [int(b) for b in branches],
                    'paths': [list(map(int, p)) for p in result_paths]
                }
            
            # If we are at k=3 and failed, we should have a fallback
            # But the seed guarantees a triangle exists.
            
    # Fallback: Find any triangle
    # This should not be reached if k=3 loop runs, but just in case
    # Re-run k=3 loop without deadline pressure? No, just return best found.
    
    # If loop finished without returning, find a max clique (size 3) as fallback
    es = {tuple(e) for e in edges}
    for row in combinations(range(n), 3):
        if all(e in es for e in combinations(row, 2)):
            return {
                'branches': [int(u) for u in row],
                'paths': [list(map(int, e)) for e in combinations(row, 2)]
            }
    
    # Absolute fallback: 2 vertices
    u, v = edges[0]
    return {
        'branches': [int(u), int(v)],
        'paths': [[int(u), int(v)]]
    }
