import time
import random
import math
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def get_neighbors(adj, u):
        temp = adj[u]
        lst = []
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            lst.append(v)
            temp -= lsb
        return lst

    def valid_add(adj, u, v):
        # Check if edge (u,v) can be added
        if adj[u] & (1 << v):
            return False
        # u and v must have < 3 common neighbors
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        # For every neighbor w of u, w and v must have < 3 common neighbors (after adding u-v)
        # Currently common(w, v) = popcount(adj[w] & adj[v])
        # After adding u-v, u becomes a common neighbor of w and v? No, u is already neighbor of w.
        # If we add (u,v), then for any neighbor w of u:
        #   common(w, v) increases by 1 if u is a neighbor of v? No, u is not neighbor of v yet.
        #   Wait. If we add (u,v), then u becomes a neighbor of v.
        #   So for any neighbor w of u, does u become a common neighbor of w and v?
        #   w is neighbor of u. v is neighbor of u (new). So u is a common neighbor of w and v?
        #   Yes, if w is neighbor of u and v is neighbor of u, then u is a common neighbor of w and v?
        #   No. Common neighbors of w and v are vertices x such that x~w and x~v.
        #   If we add (u,v), then u becomes a neighbor of v.
        #   If u is already a neighbor of w, then u is a common neighbor of w and v.
        #   So common(w, v) increases by 1 if u is a neighbor of w.
        #   So we need: for all w in neighbors(u), popcount(adj[w] & adj[v]) + 1 < 3 => popcount(adj[w] & adj[v]) <= 1.
        for w in get_neighbors(adj, u):
            if popcount(adj[w] & adj[v]) > 1:
                return False
        # Similarly for neighbors of v
        for w in get_neighbors(adj, v):
            if popcount(adj[w] & adj[u]) > 1:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_fill(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if valid_add(adj, u, v):
                            add_edge(adj, u, v)
                            changed = True

    def local_search(adj, max_iters=500):
        current_edges = count_edges(adj)
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            improved = False
            # Try 1-swap: remove (a,b), add (c,d)
            # To keep it fast, we just try random swaps
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append((u, v))
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        non_edges.append((u, v))
            
            if not edges or not non_edges:
                break
                
            for _ in range(10): # Try a few random swaps per iteration
                if time.time() - start_time > time_limit:
                    break
                a, b = random.choice(edges)
                c, d = random.choice(non_edges)
                
                # Check if swap is valid
                # Remove (a,b)
                remove_edge(adj, a, b)
                
                # Check if (c,d) can be added
                if valid_add(adj, c, d):
                    add_edge(adj, c, d)
                    # Check if (a,b) can be re-added? No, we want to keep the swap if it doesn't decrease edges.
                    # Since we removed 1 and added 1, edge count is same.
                    # We accept if it leads to a state where we can add more edges?
                    # Simple hill climbing: accept if valid.
                    improved = True
                    break
                else:
                    # Revert
                    add_edge(adj, a, b)
            
            if improved:
                # Try greedy fill after a successful swap
                greedy_fill(adj)
            else:
                # If no swap worked, try just adding edges
                greedy_fill(adj)
                if count_edges(adj) > current_edges:
                    current_edges = count_edges(adj)
                else:
                    # No improvement, break early to save time
                    break
        return adj

    best_adj = None
    best_edges = 0

    # Initial construction: 5 disjoint K4s
    base_adj = [0] * n
    for k in range(5):
        base = 4 * k
        for a in range(4):
            for b in range(a + 1, 4):
                add_edge(base_adj, base + a, base + b)
    
    greedy_fill(base_adj)
    best_adj = base_adj[:]
    best_edges = count_edges(base_adj)

    # Multi-restart
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb best solution
        adj = best_adj[:]
        # Remove 2 random edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append((u, v))
        
        if len(edges) >= 2:
            idxs = random.sample(range(len(edges)), 2)
            remove_edge(adj, edges[idxs[0]][0], edges[idxs[0]][1])
            remove_edge(adj, edges[idxs[1]][0], edges[idxs[1]][1])
        
        greedy_fill(adj)
        
        # Local search
        adj = local_search(adj)
        
        e = count_edges(adj)
        if e > best_edges:
            best_edges = e
            best_adj = adj[:]
        
        # Also try random start occasionally
        if r % 10 == 5:
            rand_adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.2:
                        if valid_add(rand_adj, u, v):
                            add_edge(rand_adj, u, v)
            greedy_fill(rand_adj)
            rand_adj = local_search(rand_adj)
            e = count_edges(rand_adj)
            if e > best_edges:
                best_edges = e
                best_adj = rand_adj[:]

    # Final Polish: Try to add any missing edges
    if best_adj:
        greedy_fill(best_adj)
        # Check if we can do better with local search again
        if time.time() - start_time < time_limit - 5:
            best_adj = local_search(best_adj, max_iters=100)
            greedy_fill(best_adj)

    # Strict Validation
    if not is_k23_free(best_adj):
        # Fallback to a known valid graph: 5 disjoint K4s
        fallback_adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(fallback_adj, base + a, base + b)
        best_adj = fallback_adj

    # Convert to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    
    return result
