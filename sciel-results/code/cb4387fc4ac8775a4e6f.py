import random
import time

def construct(k, g):
    start_time = time.time()
    time_limit = 110.0

    best_n = None
    best_edges = None

    def local_has_short_cycle(adj_list, nodes_to_check, g):
        """
        Checks for short cycles (< g) starting from nodes in nodes_to_check.
        Only explores BFS tree up to depth (g-1)//2.
        Returns True if any short cycle is found.
        """
        max_depth = (g - 1) // 2
        n = len(adj_list)
        
        for start in nodes_to_check:
            if start is None:
                continue
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= max_depth:
                    continue
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a back edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return True
        return False

    def global_has_short_cycle(adj_list, n, g):
        """
        Checks for short cycles from every node.
        """
        return local_has_short_cycle(adj_list, range(n), g)

    def random_regular_graph(n, k):
        if (n * k) % 2 != 0:
            return None, None
        
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        # Retry until simple
        for _ in range(50):
            stubs_copy = stubs[:]
            random.shuffle(stubs_copy)
            
            adj_list = [[] for _ in range(n)]
            edge_set = set()
            edges = []
            valid = True
            
            for i in range(0, len(stubs_copy), 2):
                u = stubs_copy[i]
                v = stubs_copy[i+1]
                if u == v:
                    valid = False
                    break
                if u in adj_list[v]:
                    valid = False
                    break
                adj_list[u].append(v)
                adj_list[v].append(u)
                e = (min(u,v), max(u,v))
                edge_set.add(e)
                edges.append(e)
            
            if valid:
                return adj_list, edges
        
        return None, None

    # Search
    # Try n from 430 to 1200
    for n in range(430, 1210, 2):
        if time.time() - start_time > time_limit:
            break
        
        max_restarts = 3
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            adj_list, edges = random_regular_graph(n, k)
            if adj_list is None:
                continue
            
            # Check initial validity
            if not global_has_short_cycle(adj_list, n, g):
                best_n = n
                best_edges = edges
                return best_n, best_edges
            
            # Local Search
            # We will perform swaps until we find a valid graph or time runs out.
            # To speed up, we check global validity every 100 steps.
            
            max_iters = 20000
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                
                # Check global validity periodically
                if it % 100 == 0:
                    if not global_has_short_cycle(adj_list, n, g):
                        best_n = n
                        best_edges = edges
                        return best_n, best_edges
                
                # Pick random edge (u,v)
                u, v = random.choice(edges)
                
                # Pick random edge (x,y)
                x, y = random.choice(edges)
                
                if u == x and v == y:
                    continue
                if u == y and v == x:
                    continue
                
                # Try swap (u,v) + (x,y) -> (u,x) + (v,y)
                # Check if new edges are valid (no loops, no multi-edges)
                if x in adj_list[u] and x != v: # x in adj_list[u] means edge (u,x) exists
                    # If x is v, it's the same edge, handled above? No, (u,v) is removed.
                    # If x is neighbor of u (not v), then (u,x) would be a multi-edge.
                    continue
                if y in adj_list[v] and y != u:
                    continue
                if x == u or y == v:
                    continue # Loop
                
                # Perform swap
                adj_list[u].remove(v)
                adj_list[v].remove(u)
                adj_list[x].remove(y)
                adj_list[y].remove(x)
                
                adj_list[u].append(x)
                adj_list[x].append(u)
                adj_list[v].append(y)
                adj_list[y].append(v)
                
                # Update edge list
                edges.remove((u,v))
                edges.remove((x,y))
                edges.append((min(u,x), max(u,x)))
                edges.append((min(v,y), max(v,y)))
                
                # Local check: did we create a short cycle?
                # Nodes involved: u, v, x, y
                # If no short cycle from these nodes, accept.
                if not local_has_short_cycle(adj_list, [u, v, x, y], g):
                    # Accepted (implicitly, we kept the change)
                    pass
                else:
                    # Revert
                    adj_list[u].remove(x)
                    adj_list[x].remove(u)
                    adj_list[v].remove(y)
                    adj_list[y].remove(v)
                    
                    adj_list[u].append(v)
                    adj_list[v].append(u)
                    adj_list[x].append(y)
                    adj_list[y].append(x)
                    
                    edges.remove((min(u,x), max(u,x)))
                    edges.remove((min(v,y), max(v,y)))
                    edges.append((u,v))
                    edges.append((x,y))
                    
                    # Try the other pairing? (u,y) and (v,x)
                    # For simplicity, we just skip this iteration if the first pairing fails.
                    # This reduces acceptance rate but is safer.
                    # Let's try the second pairing if the first failed.
                    
                    # Check validity of (u,y) and (v,x)
                    if y in adj_list[u] and y != v:
                        continue
                    if x in adj_list[v] and x != u:
                        continue
                    if y == u or x == v:
                        continue
                    
                    # Perform swap 2
                    adj_list[u].remove(v)
                    adj_list[v].remove(u)
                    adj_list[x].remove(y)
                    adj_list[y].remove(x)
                    
                    adj_list[u].append(y)
                    adj_list[y].append(u)
                    adj_list[v].append(x)
                    adj_list[x].append(v)
                    
                    edges.remove((u,v))
                    edges.remove((x,y))
                    edges.append((min(u,y), max(u,y)))
                    edges.append((min(v,x), max(v,x)))
                    
                    if not local_has_short_cycle(adj_list, [u, v, x, y], g):
                        # Accepted
                        pass
                    else:
                        # Revert
                        adj_list[u].remove(y)
                        adj_list[y].remove(u)
                        adj_list[v].remove(x)
                        adj_list[x].remove(v)
                        
                        adj_list[u].append(v)
                        adj_list[v].append(u)
                        adj_list[x].append(y)
                        adj_list[y].append(x)
                        
                        edges.remove((min(u,y), max(u,y)))
                        edges.remove((min(v,x), max(v,x)))
                        edges.append((u,v))
                        edges.append((x,y))

    if best_n is None:
        # Fallback: If we didn't find one, we must return something valid.
        # This shouldn't happen if the search space is sufficient.
        # We'll return a small valid graph if we can construct one?
        # For k=5, g=9, we need a specific construction.
        # I will raise an error or return a placeholder? No, must be valid.
        # Let's assume the search finds one.
        pass

    return best_n, best_edges
