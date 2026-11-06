import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()

    def has_c4(n, adj):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common & (common - 1):
                    return True
        return False

    def find_c4(n, adj):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common & (common - 1):
                    k = (common & -common).bit_length() - 1
                    common ^= (1 << k)
                    l = (common & -common).bit_length() - 1
                    return (i, j, k, l)
        return None

    def max_independent_set(n, adj):
        best = [0]
        
        def branch(candidates_mask, current_size):
            if not candidates_mask:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            if current_size + bin(candidates_mask).count('1') <= best[0]:
                return
            
            # Heuristic: pick vertex with highest degree in subgraph
            # Approximation: pick lowest set bit for speed, or pre-computed order
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            
            # Include v
            new_candidates = candidates_mask & ~adj[v] & ~(1 << v)
            branch(new_candidates, current_size + 1)
            
            # Exclude v
            candidates_mask ^= (1 << v)
            branch(candidates_mask, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    n = 39
    best_graph = None

    for attempt in range(500):
        if time_left() < 5:
            break
            
        # 1. Generate random graph
        p = 0.25 # Density
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    add_edge(adj, i, j)
                    
        # 2. Remove edges to break C4s
        # Limit iterations to avoid infinite loop
        c4_removals = 0
        while c4_removals < 200:
            c4 = find_c4(n, adj)
            if not c4:
                break
            i, j, k, l = c4
            # Remove one edge from the C4
            # Options: (i,j), (j,k), (k,l), (l,i)
            # Note: The C4 is i-k-j-l-i. Edges are (i,k), (k,j), (j,l), (l,i).
            # My find_c4 returns i, j with common neighbors k, l.
            # So edges are (i,k), (j,k), (i,l), (j,l).
            # The cycle is i-k-j-l-i.
            # Edges in cycle: (i,k), (k,j), (j,l), (l,i).
            edge_options = [(i, k), (k, j), (j, l), (l, i)]
            u, v = random.choice(edge_options)
            remove_edge(adj, u, v)
            c4_removals += 1
            
        if c4_removals >= 200:
            continue
            
        # 3. Check Alpha
        alpha = max_independent_set(n, adj)
        
        if alpha < 11:
            # Success
            final_edges = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    v = (mask & -mask).bit_length() - 1
                    if u < v:
                        final_edges.append([u, v])
                    mask ^= (1 << v)
            best_graph = {'n': n, 'edges': final_edges}
            break
            
        # 4. Try to improve by adding edges
        # Find a max independent set
        # We don't have the set, just the size. 
        # Let's modify max_independent_set to return the set too? 
        # Or just try random pairs of non-adjacent vertices.
        
        # Better: Find a specific independent set of size alpha.
        # Re-run B&B to get the set.
        
        def find_independent_set(n, adj, target_size):
            # Returns a set of vertices of size target_size if exists, else None
            # This is slow.
            pass
        
        # Instead, let's just try to add random edges between non-adjacent vertices
        # that might reduce alpha.
        # If we add an edge, alpha might decrease.
        # We only add if no C4 is created.
        
        for _ in range(100):
            if time_left() < 2:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if adj[u] & (1 << v):
                continue # Already connected
            
            # Check if adding (u, v) creates a C4
            common = adj[u] & adj[v]
            if common & (common - 1):
                continue # Would create C4
            
            # Add it
            add_edge(adj, u, v)
            
            # Check if alpha decreased
            new_alpha = max_independent_set(n, adj)
            if new_alpha < alpha:
                alpha = new_alpha
                if alpha < 11:
                    final_edges = []
                    for uu in range(n):
                        mask = adj[uu]
                        while mask:
                            vv = (mask & -mask).bit_length() - 1
                            if uu < vv:
                                final_edges.append([uu, vv])
                            mask ^= (1 << vv)
                    best_graph = {'n': n, 'edges': final_edges}
                    break
            else:
                # Revert if it didn't help? Or keep it?
                # Keeping it might help later, but might block other edges.
                # Let's keep it to increase density.
                pass
        
        if best_graph:
            break

    if best_graph:
        return best_graph
        
    # Fallback
    return {'n': 1, 'edges': []}
