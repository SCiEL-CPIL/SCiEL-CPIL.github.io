import time
import random

def construct():
    n = 20
    deadline = time.time() + 110
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def validate_and_add(adj, u, v):
        """Check if edge (u,v) can be added given current adj. Returns True if valid."""
        # Condition 1: u and v must have at most 2 common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # Condition 2: For each neighbor b of v, u and b must have at most 1 common neighbor (excluding v which is added)
        # Actually, as derived: |N(u) & N(b)| <= 1
        mask = adj[v]
        while mask:
            b = (mask & -mask).bit_length() - 1
            if (adj[u] & adj[b]).bit_count() > 1:
                return False
            mask &= mask - 1
        
        # Condition 3: For each neighbor a of u, v and a must have at most 1 common neighbor (excluding u which is added)
        mask = adj[u]
        while mask:
            a = (mask & -mask).bit_length() - 1
            if (adj[v] & adj[a]).bit_count() > 1:
                return False
            mask &= mask - 1
        
        return True

    def try_add_edge(adj, current_edges, u, v):
        """Try to add edge (u,v). Returns True if added."""
        if validate_and_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges.append((u, v))
            return True
        return False

    def remove_edge(adj, current_edges, u, v):
        """Remove edge (u,v). Uses swap-pop for efficiency."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Find and remove from list
        for i, e in enumerate(current_edges):
            if e[0] == u and e[1] == v:
                current_edges[i] = current_edges[-1]
                current_edges.pop()
                break

    def local_search(adj, current_edges, max_iter=1000):
        """Aggressive local search."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            # 1. Try adding any missing edge
            added_any = False
            for i in range(len(current_edges)):
                u, v = current_edges[i]
                for w in range(n):
                    if w == u or w == v or (adj[u] >> w) & 1:
                        continue
                    if validate_and_add(adj, u, w):
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        current_edges.append((u, w))
                        added_any = True
                        break
                if added_any:
                    break
            
            if added_any:
                improved = True
                continue

            # 2. Try 1-for-2 swap
            swapped_any = False
            for i in range(len(current_edges)):
                u, v = current_edges[i]
                old_adj = adj[:]
                old_edges = current_edges[:]
                remove_edge(adj, current_edges, u, v)
                
                added = 0
                # Try to add edges from u
                for w in range(n):
                    if w == u or w == v or (adj[u] >> w) & 1:
                        continue
                    if validate_and_add(adj, u, w):
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        current_edges.append((u, w))
                        added += 1
                        if added == 2:
                            break
                # Try to add edges from v
                if added < 2:
                    for w in range(n):
                        if w == u or w == v or (adj[v] >> w) & 1:
                            continue
                        if validate_and_add(adj, v, w):
                            adj[v] |= (1 << w)
                            adj[w] |= (1 << v)
                            current_edges.append((v, w))
                            added += 1
                            if added == 2:
                                break
                
                if added < 2:
                    # Revert
                    adj[:] = old_adj
                    current_edges[:] = old_edges
                else:
                    swapped_any = True
                    break
            
            if swapped_any:
                improved = True
                continue

            # 3. Try 2-for-3 swap (optional, might be slow)
            # If we have time, try removing 2 edges and adding 3.
            # This is expensive. We'll skip for now to ensure speed, 
            # but the 1-for-2 is usually sufficient for this density.
            
        return current_edges

    best_edges = []
    best_count = 0
    
    # Multiple restarts
    while time.time() < deadline:
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        adj = [0] * n
        current_edges = []
        
        for u, v in edges_copy:
            if validate_and_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        current_edges = local_search(adj, current_edges, max_iter=500)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # Intensive polish on best found
            if time.time() < deadline - 10:
                adj2 = [0] * n
                for u, v in best_edges:
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
                best_edges = local_search(adj2, best_edges, max_iter=2000)
                if len(best_edges) > best_count:
                    best_count = len(best_edges)

    return [[u, v] for u, v in best_edges]
