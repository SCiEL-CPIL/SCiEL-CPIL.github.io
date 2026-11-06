import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback to trivial solution if time runs out
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(n, p, seed)
        num_edges = len(edges)
        
        # If no edges, return empty
        if not edges:
            results.append([])
            continue

        # Helper to find a cycle in a graph represented by a set of edges
        def find_cycle(edge_set):
            if not edge_set:
                return None
            
            # Build adjacency list
            adj = {i: [] for i in range(n)}
            for u, v in edge_set:
                adj[u].append(v)
                adj[v].append(u)
                
            # Shuffle neighbors to introduce randomness for restarts
            for u in adj:
                random.shuffle(adj[u])
                
            visited = {False} * n
            parent = [-1] * n
            cycle = []
            
            def dfs(u, start):
                visited[u] = True
                for v in adj[u]:
                    if not visited[v]:
                        parent[v] = u
                        if dfs(v, start):
                            return True
                    elif v != parent[u] and v == start:
                        # Found a cycle back to start
                        # Reconstruct path from u to start
                        path = [u]
                        cur = u
                        while cur != start:
                            cur = parent[cur]
                            path.append(cur)
                        cycle.extend(path)
                        return True
                    elif v != parent[u]:
                        # Back edge to non-parent, non-start? 
                        # In undirected graph DFS, any back edge to an ancestor 
                        # that is not the parent indicates a cycle.
                        # However, since we start from 'start', and we only close 
                        # the loop if we hit 'start', we might miss cycles 
                        # that don't involve 'start'. 
                        # To find ANY cycle, we can just look for any back edge.
                        # If v is visited and v != parent[u], then u-v-parent[u]... 
                        # forms a cycle? No, u-v is an edge. v is visited.
                        # If v is an ancestor, we have a cycle.
                        # Let's just return True if we find any such edge, 
                        # but we need the cycle vertices.
                        # This simple DFS only finds cycles involving the start node 
                        # if we restrict to v==start. 
                        # To find any cycle, we can do:
                        # If visited[v] and v != parent[u]:
                        #    Cycle is u -> v -> ... -> u? 
                        #    We need to trace from u up to LCA(u,v) and from v up to LCA(u,v).
                        #    This is more complex.
                        # 
                        # Alternative: Just pick a random edge (u,v) and see if there 
                        # is a path from u to v in the graph without using (u,v).
                        # If yes, that path + (u,v) is a cycle.
                        # This is BFS/DFS per edge, which is O(E*(V+E)). 
                        # For V=30, E~100, this is 100*130 ~ 13k ops. Very fast.
                        # Let's use this method instead, it's more robust for finding ANY cycle.
                        return False
                return False
            
            # Use the "path between u and v" method for better cycle finding
            for u, v in list(edge_set):
                # Remove edge (u,v) temporarily to find path
                temp_edges = set(edge_set)
                temp_edges.discard((u, v))
                temp_edges.discard((v, u)) # ensure sorted tuples
                
                # Build adj for temp_edges
                adj_temp = {i: [] for i in range(n)}
                for a, b in temp_edges:
                    adj_temp[a].append(b)
                    adj_temp[b].append(a)
                    
                # BFS from u to v
                visited = {False} * n
                parent = [-1] * n
                queue = [u]
                visited[u] = True
                
                while queue:
                    curr = queue.pop(0)
                    if curr == v:
                        # Found path
                        path = []
                        c = v
                        while c != -1:
                            path.append(c)
                            c = parent[c]
                        path.reverse() # u to v
                        # Cycle is path + [u] (closing edge v-u)
                        # Vertices: path[0] ... path[-1] (which is v) ... u
                        # But u is path[0]. So cycle is path + [u]? 
                        # No, path is u, ..., v. Edge v-u closes it.
                        # Cycle vertices: u, ..., v.
                        return path
                    for nei in adj_temp[curr]:
                        if not visited[nei]:
                            visited[nei] = True
                            parent[nei] = curr
                            queue.append(nei)
            return None

        best_pieces = None
        best_count = float('inf')
        
        # Run multiple restarts
        # Time budget per graph: ~30 seconds? Total 110s for 3 graphs.
        # Let's allow ~30s per graph.
        graph_deadline = time.time() + 30.0
        
        restarts = 0
        while time.time() < graph_deadline:
            restarts += 1
            current_pieces = []
            remaining = set(edges)
            
            while remaining:
                # Find a cycle
                c = find_cycle(remaining)
                if c:
                    # c is a list of vertices [v1, ..., vk]
                    # Edges are (v1,v2), ..., (vk-1,vk), (vk,v1)
                    # Add cycle piece
                    current_pieces.append(c)
                    # Remove these edges
                    for i in range(len(c)):
                        u = c[i]
                        v = c[(i + 1) % len(c)]
                        e1 = tuple(sorted((u, v)))
                        remaining.discard(e1)
                else:
                    # No cycles left, add all remaining as single edges
                    for e in remaining:
                        current_pieces.append(list(e))
                    break
            
            count = len(current_pieces)
            if count < best_count:
                best_count = count
                best_pieces = current_pieces
                
            # If we found a perfect packing (all cycles), we can't do better than 
            # number_of_cycles. But we don't know the max cycles easily.
            # If best_count is low, we might stop early? 
            # For now, keep searching until deadline.
            
        if best_pieces is None:
            # Fallback
            best_pieces = [list(e) for e in edges]
            
        results.append(best_pieces)

    return results
