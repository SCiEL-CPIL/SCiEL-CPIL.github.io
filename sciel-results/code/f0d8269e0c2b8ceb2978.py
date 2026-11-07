    def find_shortest_cycle(rem_edges):
        if not rem_edges:
            return None
            
        rem_adj = defaultdict(list)
        for u, v in rem_edges:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
            
        best_cycle = None
        best_len = float('inf')
        
        for u, v in list(rem_edges):
            if best_len == 3:
                break
                
            rem_edges.discard((u, v))
            
            visited = {u}
            queue = deque([(u, [u])])
            found_path = None
            
            while queue:
                node, path = queue.popleft()
                if node == v:
                    found_path = path
                    break
                for neighbor in rem_adj[node]:
                    if neighbor not in visited:
                        e = (min(node, neighbor), max(node, neighbor))
                        if e in rem_edges:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                            
            rem_edges.add((u, v))
            
            if found_path:
                cycle_len = len(found_path)
                if cycle_len < best_len:
                    best_len = cycle_len
                    best_cycle = found_path
                    
        return best_cycle
