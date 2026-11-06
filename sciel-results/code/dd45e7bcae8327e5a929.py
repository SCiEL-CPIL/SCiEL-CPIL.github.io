import time
import random

def construct():
    n = 20
    deadline = time.time() + 140
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    def validate_and_add(adj, u, v):
        """Check if edge (u,v) can be added given current adj. Returns True if valid."""
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        # Check that adding edge doesn't create K2,3 with existing neighbors
        # For each neighbor a of u: after adding (u,v), v's neighbors gain u.
        # Need: for all a in N(u), |N(v) ∩ N(a)| <= 2 after adding u to N(v)
        # i.e., |N(v) ∩ N(a) ∩ {u}| + |N(v) ∩ N(a) \ {u}| <= 2
        # Since u not in N(v) yet, this is |N(v) ∩ N(a)| <= 2 (u adds at most 1 if a~v)
        # Wait: if a~v, then u becomes common neighbor of v and a. So |N(v)∩N(a)| + 1 <= 2
        # i.e., |N(v)∩N(a)| <= 1 if a~v, else |N(v)∩N(a)| <= 2
        # Similarly for neighbors of v.
        mask = adj[u]
        while mask:
            b = (mask & -mask).bit_length() - 1
            if b == v:
                mask &= mask - 1
                continue
            if (adj[v] & adj[b]).bit_count() > (1 if (adj[u] >> b) & 1 else 2):
                return False
            mask &= mask - 1
        mask = adj[v]
        while mask:
            a = (mask & -mask).bit_length() - 1
            if a == u:
                mask &= mask - 1
                continue
            if (adj[u] & adj[a]).bit_count() > (1 if (adj[v] >> a) & 1 else 2):
                return False
            mask &= mask - 1
        return True

    def can_add_fast(adj, u, v):
        """Fast check: just verify the direct constraint."""
        return (adj[u] & adj[v]).bit_count() <= 2

    def local_search(adj, current_edges, max_iter=2000):
        """Aggressive local search with add and swap moves."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            # Try to add edges
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
            
            if added_any:
                improved = True
                continue

            # Try 1-for-2 swaps: remove edge (u,v), try to add two new edges
            swapped_any = False
            for i in range(len(current_edges)):
                u, v = current_edges[i]
                # Save state
                old_adj = adj[:]
                
                # Remove edge (u,v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges[i] = current_edges[-1]
                current_edges.pop()
                
                # Try to add two edges
                added = 0
                # First try adding edges incident to u
                for w in range(n):
                    if w == u or w == v or (adj[u] >> w) & 1:
                        continue
                    if validate_and_add(adj, u, w):
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        current_edges.append((u, w))
                        added += 1
                        break
                
                # Try adding edges incident to v
                for w in range(n):
                    if w == u or w == v or (adj[v] >> w) & 1:
                        continue
                    if validate_and_add(adj, v, w):
                        adj[v] |= (1 << w)
                        adj[w] |= (1 << v)
                        current_edges.append((v, w))
                        added += 1
                        break
                
                # Try adding any other edge
                if added < 2:
                    for a in range(n):
                        for b in range(a + 1, n):
                            if (adj[a] >> b) & 1:
                                continue
                            if a == u or a == v or b == u or b == v:
                                continue
                            if validate_and_add(adj, a, b):
                                adj[a] |= (1 << b)
                                adj[b] |= (1 << a)
                                current_edges.append((a, b))
                                added += 1
                                break
                        if added >= 2:
                            break
                
                if added < 2:
                    # Revert
                    adj[:] = old_adj
                    current_edges.append((u, v))
                    # Re-sort isn't needed, we just need the edge back
                else:
                    swapped_any = True
                    break
            
            if swapped_any:
                improved = True
                continue

        return current_edges

    best_edges = []
    best_count = 0
    
    # Phase 1: Multiple randomized restarts with greedy insertion
    num_restarts = 50
    for trial in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        random.seed(trial * 7919 + 13)  # Deterministic per trial for reproducibility
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        adj = [0] * n
        current_edges = []
        
        for u, v in edges_copy:
            if validate_and_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        # Local search polish
        current_edges = local_search(adj, current_edges, max_iter=500)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
    
    # Phase 2: Intensive polish on best solution
    if best_edges and time.time() < deadline - 5:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        best_edges = local_search(adj_final, best_edges, max_iter=10000)
        best_count = len(best_edges)
    
    # Phase 3: Final addable edge scan (multiple passes)
    if best_edges and time.time() < deadline - 1:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        for _ in range(100):
            if time.time() > deadline - 0.5:
                break
            added_any = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_final[u] >> v) & 1:
                        continue
                    if validate_and_add(adj_final, u, v):
                        adj_final[u] |= (1 << v)
                        adj_final[v] |= (1 << u)
                        best_edges.append((u, v))
                        added_any = True
                        break
                if added_any:
                    break
            if not added_any:
                break
        best_count = len(best_edges)

    return [[u, v] for u, v in best_edges]
