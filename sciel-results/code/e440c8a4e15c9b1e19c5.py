import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 40
    best_edges = set()
    best_count = 0
    
    # Helper: Check if adding (u, v) creates a C6
    # A C6 is formed if there is a simple path of length 5 between u and v.
    def has_path_length_5(adj, u, v):
        # First, check shortest path distance
        # If dist > 5, no path of length 5
        # If dist == 5, yes
        # If dist < 5, we need to search for a specific simple path of length 5
        
        # BFS for shortest path
        from collections import deque
        dist = {u: 0}
        q = deque([u])
        while q:
            curr = q.popleft()
            if curr == v:
                break
            for nei in adj[curr]:
                if nei not in dist:
                    dist[nei] = dist[curr] + 1
                    if dist[nei] > 5:
                        continue
                    q.append(nei)
        
        if v not in dist:
            return False
        if dist[v] > 5:
            return False
        if dist[v] == 5:
            return True
        
        # dist[v] < 5
        # We need to find a simple path of length 5.
        # DFS from u, depth 5, avoid visited
        visited = set([u])
        
        def dfs(curr, depth):
            if depth == 5:
                return curr == v
            for nei in adj[curr]:
                if nei not in visited:
                    visited.add(nei)
                    if dfs(nei, depth + 1):
                        return True
                    visited.remove(nei)
            return False
        
        return dfs(u, 0)

    def find_c6(adj):
        # Find any C6
        # Iterate over all pairs (u, v)
        # If has_path_length_5(u, v), then we have a C6
        # To be efficient, we can just pick a random subset of pairs
        # Or iterate systematically
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if u in adj[v]: # already edge, skip
                    continue
                if has_path_length_5(adj, u, v):
                    # We found a C6. We need to return the edges of the cycle to remove one.
                    # has_path_length_5 only returns bool. 
                    # We need the path to remove an edge.
                    # Let's modify has_path_length_5 to return the path if found.
                    pass
        return None

    # Revised: We need to actually find the cycle to repair.
    # Let's combine check and find.
    
    def find_c6_path(adj, u, v):
        """Returns a list of vertices forming a path of length 5 between u and v, or None."""
        from collections import deque
        
        # BFS for shortest path
        dist = {u: 0}
        parent = {u: None}
        q = deque([u])
        found = False
        while q and not found:
            curr = q.popleft()
            if curr == v:
                found = True
                break
            for nei in adj[curr]:
                if nei not in dist:
                    dist[nei] = dist[curr] + 1
                    parent[nei] = curr
                    if dist[nei] <= 5:
                        q.append(nei)
        
        if found and dist[v] == 5:
            # Reconstruct path
            path = []
            curr = v
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return path
        
        if not found or dist[v] > 5:
            return None
            
        # dist[v] < 5, need DFS for simple path of length 5
        visited = set([u])
        
        def dfs(curr, depth, path_so_far):
            if depth == 5:
                if curr == v:
                    return path_so_far + [v]
                return None
            for nei in adj[curr]:
                if nei not in visited:
                    visited.add(nei)
                    res = dfs(nei, depth + 1, path_so_far + [nei])
                    if res:
                        return res
                    visited.remove(nei)
            return None
        
        return dfs(u, 0, [u])

    def repair_graph(adj):
        """Removes edges until no C6 exists. Returns True if success."""
        while True:
            # Find a C6
            c6_found = False
            vertices = list(range(n))
            random.shuffle(vertices)
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    if u in adj[v]:
                        continue
                    path = find_c6_path(adj, u, v)
                    if path:
                        # path is [u, ..., v] length 6 vertices, 5 edges
                        # Remove a random edge from this path
                        edge_idx = random.randint(0, 4)
                        a, b = path[edge_idx], path[edge_idx+1]
                        adj[a].remove(b)
                        adj[b].remove(a)
                        c6_found = True
                        break
                if c6_found:
                    break
            if not c6_found:
                return True

    def random_init():
        adj = {i: set() for i in range(n)}
        # Target ~80 edges
        m = random.randint(70, 90)
        attempts = 0
        while m > 0 and attempts < 5000:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                m -= 1
            attempts += 1
        return adj

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Main Loop
    while time.time() < deadline:
        adj = random_init()
        if not repair_graph(adj):
            continue
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get all possible edges not in adj
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
            random.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if time.time() > deadline:
                    break
                # Check if adding (u,v) creates C6
                # We can reuse find_c6_path logic, but we just need to know if path exists
                # Actually, find_c6_path returns the path if exists.
                # So if find_c6_path(adj, u, v) is not None, it's invalid.
                path = find_c6_path(adj, u, v)
                if not path:
                    # Safe to add
                    adj[u].add(v)
                    adj[v].add(u)
                    improved = True
                    # Do not break, try to add more in this pass
                    # But note: adding (u,v) changes adj, so subsequent checks are valid.
        
        current_count = sum(len(adj[i]) for i in range(n)) // 2
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edge_list(adj)

    return best_edges
# EVOLVE-BLOCK-END
