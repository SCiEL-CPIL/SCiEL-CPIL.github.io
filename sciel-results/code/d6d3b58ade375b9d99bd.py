    def has_short_cycle(order, index, rules, min_girth):
        # BFS from each vertex
        n = len(order)
        for u in range(n):
            if time.time() > deadline:
                return True # Assume bad if timeout
            dist = [-1] * n
            parent = [-1] * n
            dist[index[order[u]]] = 0
            queue = [index[order[u]]]
            while queue:
                v = queue.pop(0)
                if dist[v] >= min_girth // 2:
                    continue
                # Get neighbors of order[v]
                # Note: order[v] is the vertex object
                for nb in neighbours(order[v], rules):
                    nb_idx = index.get(nb)
                    if nb_idx is None:
                        continue
                    if dist[nb_idx] == -1:
                        dist[nb_idx] = dist[v] + 1
                        parent[nb_idx] = v
                        queue.append(nb_idx)
                    else:
                        if nb_idx != parent[v]:
                            # Found a cycle
                            # Cycle length is at least dist[v] + 1 + dist[nb_idx]?
                            # Actually, if dist[v] + 1 < min_girth and dist[nb_idx] <= dist[v], 
                            # it's a short cycle.
                            if dist[v] + 1 < min_girth:
                                return True
        return False
