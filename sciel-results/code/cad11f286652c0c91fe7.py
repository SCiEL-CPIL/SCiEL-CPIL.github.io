    def find_path(K):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if pos[(i, j)] <= K:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        visited = 1 << 0
        path = [0]
        start_time = time.time()
        
        def dfs(v, visited_mask, path):
            if time.time() - start_time > 50: # 50s limit for this phase
                return None
            if len(path) == n:
                if (adj[v] >> 0) & 1:
                    return path[:]
                return None
            unvisited = adj[v] & ~visited_mask
            while unvisited:
                u = (unvisited & -unvisited).bit_length() - 1
                unvisited ^= (1 << u)
                res = dfs(u, visited_mask | (1 << u), path + [u])
                if res:
                    return res
            return None
            
        return dfs(0, visited, path)
