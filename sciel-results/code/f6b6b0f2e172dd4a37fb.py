    def find_c6():
        for u in range(n):
            for v in adj[u]:
                if u >= v: continue
                # BFS from v to u, depth 5
                visited = {v: 0}
                parent = {v: None}
                queue = deque([(v, 0)])
                
                while queue:
                    node, depth = queue.popleft()
                    if depth == 5:
                        if node == u:
                            # Reconstruct path
                            path = [u]
                            curr = node
                            while curr != v:
                                curr = parent[curr]
                                path.append(curr)
                            path.reverse()
                            return path
                        continue # Don't expand depth 5 nodes
                    
                    for neighbor in adj[node]:
                        if neighbor == u:
                            if depth + 1 == 5:
                                # Found path v ... u of length 5
                                # Wait, if neighbor is u and depth+1==5, we found it.
                                # But we are in the loop for neighbors of 'node'.
                                # If we find it here, we can return immediately.
                                path = [u]
                                curr = node
                                while curr != v:
                                    curr = parent[curr]
                                    path.append(curr)
                                path.reverse()
                                return path
                            else:
                                continue # Too short, or just skip u as intermediate
                        if neighbor == v:
                            continue # Can't go back to v
                        if neighbor not in visited:
                            visited[neighbor] = depth + 1
                            parent[neighbor] = node
                            if depth + 1 < 5: # Only add if we can still reach depth 5?
                                # Actually, we can add up to depth 4, then check neighbors at depth 5.
                                # Or add up to depth 5 and check when popping.
                                # Adding up to depth 4 is sufficient to find depth 5 neighbors.
                                # If we add depth 4 nodes, their neighbors (depth 5) will be checked.
                                # So we add if depth + 1 <= 4?
                                # No, if we add depth 4 nodes, we pop them, and check their neighbors.
                                # So we should add nodes if depth + 1 < 5.
                                queue.append((neighbor, depth + 1))
        return None
