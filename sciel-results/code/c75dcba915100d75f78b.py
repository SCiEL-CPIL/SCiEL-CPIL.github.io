    def find_longest_cycle(adj, edges):
        # adj is dict of sets
        # edges is set of edges
        if not edges:
            return None
        
        best_cycle = None
        best_len = 0
        
        # We need to find all simple cycles?
        # We can use DFS from each node.
        # To avoid duplicates, we can restrict the start node to be the smallest index in the cycle.
        
        nodes = sorted(adj.keys())
        
        for start in nodes:
            if len(adj[start]) < 2:
                continue
            
            # DFS
            path = [start]
            visited = {start}
            
            def dfs(u, current_path, current_visited):
                nonlocal best_cycle, best_len
                # Pruning: if max possible length is less than best_len, return
                # Max possible length is len(current_path) + (number of unvisited nodes reachable)
                # This is hard to estimate quickly.
                
                for v in sorted(adj[u]):
                    if v == start and len(current_path) >= 3:
                        # Found a cycle
                        if len(current_path) > best_len:
                            best_len = len(current_path)
                            best_cycle = list(current_path)
                    elif v not in current_visited:
                        # Only allow visiting nodes with index > start to avoid duplicate cycles
                        # where the "start" is not the smallest node in the cycle.
                        # This ensures each cycle is found exactly once.
                        if v > start:
                            current_path.append(v)
                            current_visited.add(v)
                            dfs(v, current_path, current_visited)
                            current_path.pop()
                            current_visited.remove(v)
            
            dfs(start, path, visited)
        
        return best_cycle
    