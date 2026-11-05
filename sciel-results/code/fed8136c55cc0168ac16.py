    def check_girth_circulant_fast(n, S, k, g):
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        q = deque([0])
        
        while q:
            x = q.popleft()
            for s in S:
                y = (x + s) % n
                if y == x: continue
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    parent[y] = x
                    q.append(y)
                else:
                    if parent[x] != y:
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < g:
                            return False
        return True
