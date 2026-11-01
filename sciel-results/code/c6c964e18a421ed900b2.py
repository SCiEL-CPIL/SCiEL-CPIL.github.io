<<<<<<< SEARCH
    def girth_check_fast(n, adj, g, time_limit):
        """
        Fast girth check: verify no cycle shorter than g exists.
        Uses BFS from each vertex with non-backtracking constraint.
        Returns True if girth >= g, False otherwise.
        Checks time limit periodically.
        """
        check_time = time.time()
        for start in range(n):
            if time.time() - check_time > time_limit:
                return None  # timeout, unknown result
            
            # BFS over non-backtracking walks
            # dist[(u, prev)] = shortest non-backtracking walk length from start to u via prev
            visited = set()
            queue = deque()
            
            for v in adj[start]:
                state = (v, start)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, start, 1))
            
            while queue:
                if time.time() - check_time > time_limit:
                    return None  # timeout
                
                u, prev, depth = queue.popleft()
                
                if u == start:
                    if depth < g:
                        return False
                    continue
                
                if depth >= g - 1:
                    continue
                
                for w in adj[u]:
                    if w == prev:
                        continue
                    new_state = (w, u)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((w, u, depth + 1))
        
        return True
=======
    def girth_check_fast(n, adj, g, time_limit):
        """
        Fast girth check: verify no cycle shorter than g exists.
        Uses BFS from each vertex with non-backtracking constraint.
        Returns True if girth >= g, False otherwise.
        Checks time limit periodically.
        """
        check_time = time.time()
        for start in range(n):
            if time.time() - check_time > time_limit:
                return None  # timeout, unknown result
            
            # BFS over non-backtracking walks
            # Use dict for visited: (u, prev) -> depth
            visited = {}
            queue = deque()
            
            for v in adj[start]:
                state = (v, start)
                visited[state] = 1
                queue.append((v, start, 1))
            
            while queue:
                if time.time() - check_time > time_limit:
                    return None  # timeout
                
                u, prev, depth = queue.popleft()
                
                if u == start:
                    if depth < g:
                        return False
                    continue
                
                if depth >= g - 1:
                    continue
                
                for w in adj[u]:
                    if w == prev:
                        continue
                    new_state = (w, u)
                    if new_state not in visited:
                        visited[new_state] = depth + 1
                        queue.append((w, u, depth + 1))
        
        return True
>>>>>>> REPLACE

