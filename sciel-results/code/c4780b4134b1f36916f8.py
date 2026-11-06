<<<<<<< SEARCH
    def reachable_count(current, visited):
        """Count unvisited nodes in comp_set reachable from current via unvisited nodes."""
        if current not in comp_set:
            return 0
        stack = [current]
        seen = {current}
        count = 0
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in visited or v not in comp_set:
                    continue
                if v not in seen:
                    seen.add(v)
                    count += 1
                    stack.append(v)
        return count
=======
    def max_reachable_unvisited(start, visited):
        """Find the size of the connected component of unvisited nodes containing start."""
        if start not in comp_set or start in visited:
            return 0
        seen = {start}
        stack = [start]
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for v in adj[u]:
                if v in visited or v not in comp_set:
                    continue
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return count
>>>>>>> REPLACE

