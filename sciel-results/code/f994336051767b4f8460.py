    def embed_tree(tree_idx, used_edges_global):
        tree = trees[tree_idx]
        k = len(tree) + 1
        
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
            
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
                    
        host_map = [-1] * k
        used_in_tree = set()
        
        def dfs(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
                
            tv = order[idx]
            p_tv = parent[tv]
            
            candidates = []
            if p_tv == -1:
                # Root: any host not used in this tree
                candidates = [h for h in range(n) if h not in used_in_tree]
            else:
                p_hv = host_map[p_tv]
                for h in range(n):
                    if h in used_in_tree:
                        continue
                    e = (min(p_hv, h), max(p_hv, h))
                    if e not in used_edges_global:
                        candidates.append(h)
                        
            # Optional: Shuffle candidates to avoid getting stuck in bad local minima
            # But deterministic is faster if it works.
            
            for h in candidates:
                host_map[tv] = h
                used_in_tree.add(h)
                
                # Forward Check: 
                # Check children
                feasible = True
                for child_tv in adj[tv]:
                    if parent[child_tv] == tv:
                        # Child must have at least one valid host
                        # Valid host for child: not in used_in_tree, edge (h, child_h) not in used_edges_global
                        has_opt = False
                        for ch in range(n):
                            if ch in used_in_tree:
                                continue
                            e = (min(h, ch), max(h, ch))
                            if e not in used_edges_global:
                                has_opt = True
                                break
                        if not has_opt:
                            feasible = False
                            break
                            
                if feasible:
                    if dfs(idx + 1):
                        return True
                
                host_map[tv] = -1
                used_in_tree.remove(h)
                
            return False
            
        if dfs(0):
            # Add edges to global used set
            for u, v in tree:
                hu = host_map[u]
                hv = host_map[v]
                used_edges_global.add((min(hu, hv), max(hu, hv)))
            return host_map[:]
        else:
            return None
