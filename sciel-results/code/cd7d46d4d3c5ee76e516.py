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

            # 3. Try 2-for-3 swap
            if time.time() < deadline - 5:
                swapped23 = False
                for i in range(len(current_edges)):
                    u1, v1 = current_edges[i]
                    for j in range(i + 1, len(current_edges)):
                        u2, v2 = current_edges[j]
                        # Check if edges are distinct
                        if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                            continue
                        
                        old_adj = adj[:]
                        old_edges = current_edges[:]
                        
                        # Remove both edges
                        remove_edge(adj, current_edges, u1, v1)
                        remove_edge(adj, current_edges, u2, v2)
                        
                        # Try to add up to 3 edges
                        added = 0
                        # Try all possible new edges
                        for a in range(n):
                            if added >= 3:
                                break
                            for b in range(a + 1, n):
                                if added >= 3:
                                    break
                                # Skip if edge already exists
                                if (adj[a] >> b) & 1:
                                    continue
                                # Skip if this is one of the removed edges
                                if (a == u1 and b == v1) or (a == v1 and b == u1):
                                    continue
                                if (a == u2 and b == v2) or (a == v2 and b == u2):
                                    continue
                                
                                if validate_and_add(adj, a, b):
                                    adj[a] |= (1 << b)
                                    adj[b] |= (1 << a)
                                    current_edges.append((a, b))
                                    added += 1
                                    if added == 3:
                                        break
                        
                        if added < 3:
                            # Revert
                            adj[:] = old_adj
                            current_edges[:] = old_edges
                        else:
                            swapped23 = True
                            break
                    if swapped23:
                        break
                
                if swapped23:
                    improved = True
                    continue
            
        return current_edges

    def final_polish(adj, current_edges):
        """Final high-precision polish pass with more iterations."""
        if time.time() > deadline - 2:
            return current_edges
        
        # Try more aggressive local search
        current_edges = local_search(adj, current_edges, max_iter=5000)
        
        # Try to find any remaining addable edges
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            added_any = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if validate_and_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges.append((u, v))
                        added_any = True
                        break
                if added_any:
                    break
            if not added_any:
                break
        
        return current_edges

    best_edges = []
    best_count = 0
    
    # Multiple restarts
    while time.time() < deadline - 15:
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

    # Final polish on the best solution
    if best_edges and time.time() < deadline - 2:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        best_edges = final_polish(adj_final, best_edges)
        best_count = len(best_edges)

    return [[u, v] for u, v in best_edges]
