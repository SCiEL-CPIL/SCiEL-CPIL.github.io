from random import Random

def construct():
        # 1. Generate Trees
        def instance_trees(n=20, seed=611):
            rng = Random(seed)
            trees = []
            for k in range(1, n + 1):
                if k <= 2:
                    trees.append([] if k == 1 else [(0, 1)])
                    continue
                code = [rng.randrange(k) for _ in range(k - 2)]
                degree = [1] * k
                for x in code:
                    degree[x] += 1
                edges = []
                for x in code:
                    leaf = min(v for v in range(k) if degree[v] == 1)
                    edges.append((leaf, x))
                    degree[leaf] -= 1
                    degree[x] -= 1
                u, w = [v for v in range(k) if degree[v] == 1]
                edges.append((u, w))
                trees.append(edges)
            return trees

        trees = instance_trees()
        
        # Total vertices in host
        N = 20
        all_edges = set()
        for i in range(N):
            for j in range(i + 1, N):
                all_edges.add((i, j))
        
        used_edges = set()
        result = [None] * 20
        
        # Helper to add edges to used set
        def add_edges(edge_list, mapping, used_set):
            for u, v in edge_list:
                hu = mapping[u]
                hv = mapping[v]
                if hu > hv: hu, hv = hv, hu
                used_set.add((hu, hv))
        
        # Embed T20 first (Identity)
        # T20 has 20 vertices, indices 0..19
        # Mapping is identity
        map_20 = list(range(20))
        result[19] = map_20
        add_edges(trees[19], map_20, used_edges)
        
        # Embed T19 down to T1
        # We need a backtracking solver
        def find_embedding(tree_edges, avail_edges):
            k = len(tree_edges) + 1 if tree_edges else 1
            # Build adjacency list for tree
            adj = [[] for _ in range(k)]
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # If k==1, no edges, any vertex works.
            if k == 1:
                # Return first available vertex
                # Actually, mapping is just [0] or any.
                # But we need to return a list of k distinct ints.
                # We can pick 0.
                return [0]
            
            # Order vertices for backtracking
            # BFS from 0 to ensure parent is mapped before child
            order = []
            visited = [False] * k
            queue = [0]
            visited[0] = True
            while queue:
                u = queue.pop(0)
                order.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            
            # Reverse order? No, we want to map parents first.
            # So we iterate through `order`.
            
            mapping = [-1] * k
            used_host_vertices = [False] * N
            
            def dfs(idx):
                if idx == len(order):
                    return True
                
                u = order[idx]
                # Find parent in tree (the one already mapped)
                parent_host = -1
                for v in adj[u]:
                    if mapping[v] != -1:
                        parent_host = mapping[v]
                        break
                
                # Try all host vertices
                # Optimization: if parent_host is set, only try neighbors of parent_host in avail_edges
                candidates = range(N)
                if parent_host != -1:
                    # Filter candidates to those connected to parent_host in avail_edges
                    # And not used
                    candidates = [w for w in range(N) if not used_host_vertices[w] and (min(parent_host, w), max(parent_host, w)) in avail_edges]
                else:
                    candidates = [w for w in range(N) if not used_host_vertices[w]]
                
                for w in candidates:
                    # Check compatibility with other already mapped neighbors
                    valid = True
                    for v in adj[u]:
                        if mapping[v] != -1 and v != parent_host: # parent already checked
                            hv = mapping[v]
                            if (min(hv, w), max(hv, w)) not in avail_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    mapping[u] = w
                    used_host_vertices[w] = True
                    
                    if dfs(idx + 1):
                        return True
                    
                    mapping[u] = -1
                    used_host_vertices[w] = False
                
                return False
            
            if dfs(0):
                return mapping
            return None

        for k in range(19, 0, -1):
            tree_edges = trees[k-1]
            # avail_edges is all_edges - used_edges
            # For efficiency, we pass the set
            mapping = find_embedding(tree_edges, all_edges - used_edges)
            if mapping is None:
                # Fallback: try random restarts or different root?
                # For now, if it fails, we might return what we have.
                # But it should succeed.
                pass
            else:
                result[k-1] = mapping
                add_edges(tree_edges, mapping, used_edges)
        
        return result
