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
        mask = adj[v]
        while mask:
            b = (mask & -mask).bit_length() - 1
            if (adj[u] & adj[b]).bit_count() > 1:
                return False
            mask &= mask - 1
        mask = adj[u]
        while mask:
            a = (mask & -mask).bit_length() - 1
            if (adj[v] & adj[a]).bit_count() > 1:
                return False
            mask &= mask - 1
        return True

    def local_search(adj, current_edges, max_iter=1000):
        """Aggressive local search."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
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

            swapped_any = False
            for i in range(len(current_edges)):
                u, v = current_edges[i]
                old_adj = adj[:]
                old_edges = current_edges[:]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                for k, e in enumerate(current_edges):
                    if e[0] == u and e[1] == v:
                        current_edges[k] = current_edges[-1]
                        current_edges.pop()
                        break
                
                added = 0
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
                    adj[:] = old_adj
                    current_edges[:] = old_edges
                else:
                    swapped_any = True
                    break
            
            if swapped_any:
                improved = True
                continue

        return current_edges

    def mcis_branch_and_bound(adj, current_edges, max_iter=100000):
        """
        MCIS formulation: Each node in conflict graph = a potential edge.
        Two edges conflict if they can't coexist in a K2,3-free graph.
        Find maximum independent set.
        
        We use branch and bound with greedy coloring upper bound.
        """
        # Build conflict graph among all possible edges
        # Edge i and edge j conflict if adding both would violate K2,3-free
        m = len(all_edges)
        
        # Precompute conflict graph: conflict[i] is a bitmask of edges conflicting with edge i
        # Two edges (u,v) and (a,b) conflict if:
        # 1. They share a vertex: the shared vertex and the other two vertices would have 3 common neighbors
        #    Actually need to be more careful. Let me think...
        # 
        # If we have edges (u,v) and (a,b), they conflict if there exists a K2,3 subgraph
        # that uses both edges. A K2,3 has two vertices with 3 common neighbors.
        # 
        # Actually, the simplest conflict: if edges share a vertex, say (u,v) and (u,w),
        # then v and w must not have 3 common neighbors in the final graph. But that's
        # not a direct conflict between just these two edges - it depends on other edges too.
        #
        # The real conflict: edges (u,v) and (a,b) directly conflict if {u,v,a,b} contains
        # a K2,3 pattern. This happens when the 4 vertices form a structure where two of them
        # have the other two as common neighbors plus one more... 
        #
        # Actually, let me reconsider. The constraint is: no two vertices have 3 common neighbors.
        # Two edges e1=(u,v) and e2=(a,b) are "incompatible" if there's no way to have both
        # in any K2,3-free graph. But that's not quite right either - they might be compatible
        # depending on other edges.
        #
        # The correct MCIS formulation: We want to select a maximum set of edges such that
        # no two selected vertices have 3 common neighbors among selected edges.
        #
        # This is NOT a standard independent set problem because the constraint is not pairwise.
        # Two edges might be individually fine but three together create a violation.
        #
        # However, we CAN formulate it as: two edges conflict if they MUST create a violation
        # together. The simplest case: if edges (u,v) and (u,w) exist, and there's a vertex x
        # connected to both v and w, then u and x have v,w as common neighbors. If there's
        # also another vertex y connected to both v and w, then u and x have v,w,y as common
        # neighbors -> K2,3 violation.
        #
        # This is complex. Let me use a different approach: branch and bound on the edge set
        # directly, using the validation check.
        
        # Simpler approach: Use the existing local search but with more sophisticated moves
        # and a better initial solution from structured constructions.
        
        # Try circulant-based structured initial solutions
        best_local = current_edges[:]
        best_count = len(current_edges)
        
        # Try different structured constructions
        for start in range(1, 10):
            if time.time() > deadline - 5:
                break
            adj_struct = [0] * n
            edges_struct = []
            # Add edges at distances start, 2*start, etc.
            for dist in range(1, n // 2 + 1):
                if dist % start == 0 or (dist * start) % n < start:
                    for u in range(n):
                        v = (u + dist) % n
                        if u < v:
                            if validate_and_add(adj_struct, u, v):
                                adj_struct[u] |= (1 << v)
                                adj_struct[v] |= (1 << u)
                                edges_struct.append((u, v))
            
            if len(edges_struct) > best_count:
                best_count = len(edges_struct)
                best_local = edges_struct[:]
            
            # Polish the structured solution
            if len(edges_struct) >= best_count - 2:
                polished = local_search(adj_struct, edges_struct, max_iter=200)
                if len(polished) > best_count:
                    best_count = len(polished)
                    best_local = polished[:]
        
        return best_local

    best_edges = []
    best_count = 0
    
    # Phase 1: Try structured constructions
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        adj = [0] * n
        current_edges = []
        
        for u, v in edges_copy:
            if validate_and_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        current_edges = local_search(adj, current_edges, max_iter=300)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
    
    # Phase 2: Intensive polish on best
    if best_edges and time.time() < deadline - 5:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        best_edges = local_search(adj_final, best_edges, max_iter=5000)
        best_count = len(best_edges)
    
    # Phase 3: Final addable edge scan
    if best_edges and time.time() < deadline - 1:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        for _ in range(50):
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
