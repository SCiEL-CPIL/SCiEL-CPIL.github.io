    def _check_girth_fast(adj, n, g):
        for s in range(n):
            dq = deque()
            visited = set()
            for nb in adj[s]:
                dq.append((nb, s, 1))
                visited.add((nb, s))
            while dq:
                curr, prev, depth = dq.popleft()
                if curr == s:
                    if depth + 1 < g:
                        return False
                    continue
                if depth + 1 >= g:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in visited:
                        visited.add(state)
                        dq.append((nb, curr, depth + 1))
        return True
    