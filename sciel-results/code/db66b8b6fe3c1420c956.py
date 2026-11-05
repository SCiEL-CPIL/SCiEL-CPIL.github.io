import random
import time
import sys

def construct():
    n = 20
    # Bitmask representation of edges
    # Edge index for (u, v) with u < v
    def edge_idx(u, v):
        if u > v: u, v = v, u
        return u * n - u * (u + 1) // 2 + v - u - 1
        
    # Precompute edge indices
    edge_indices = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices[(u, v)] = idx
            idx += 1
    total_edges = idx
            
    # Initialize with 5 disjoint K4s
    adj = [0] * n  # Bitmask of neighbors
    edges_mask = 0
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                u = 4 * k + a
                v = 4 * k + b
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges_mask |= 1 << edge_indices[(u, v)]
                
    best_edges = edges_mask
    best_count = bin(edges_mask).count('1')
    
    def is_valid(mask, adj):
        # Check if any pair has > 2 common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    return False
        return True

    def add_edge(mask, adj, u, v):
        idx = edge_indices[(u, v)]
        mask |= 1 << idx
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        return mask, adj

    def remove_edge(mask, adj, u, v):
        idx = edge_indices[(u, v)]
        mask &= ~(1 << idx)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return mask, adj

    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    current_mask = best_edges
    current_adj = [x for x in adj]
    current_count = best_count
    
    while time.time() < deadline:
        # Try to add a random missing edge
        found_improvement = False
        # Pick a random missing edge
        # We can iterate through all missing edges or pick random
        # Picking random is faster if graph is dense
        for _ in range(100): # Try 100 random edges per step
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            if (current_adj[u] >> v) & 1: continue # Already exists
            
            # Try adding
            new_mask, new_adj = add_edge(current_mask, current_adj, u, v)
            
            # Check validity
            # Only need to check pairs involving u or v?
            # No, adding (u,v) only affects common neighbors of pairs involving u or v?
            # Actually, |N(x) \cap N(y)| changes only if x or y is u or v.
            # So we only need to check pairs (u, w) and (v, w) for all w.
            valid = True
            for w in range(n):
                if w == u or w == v: continue
                # Check pair (u, w)
                common_uw = new_adj[u] & new_adj[w]
                if bin(common_uw).count('1') > 2:
                    valid = False
                    break
                # Check pair (v, w)
                common_vw = new_adj[v] & new_adj[w]
                if bin(common_vw).count('1') > 2:
                    valid = False
                    break
            if not valid:
                # Try to fix by removing an edge
                # Identify conflicts
                # For each w, if conflict, find common neighbors
                # This is complex. Let's just restart or try a different edge.
                # Or, if we want to be smarter, we can try to remove one of the conflicting edges.
                pass
            else:
                # Valid improvement
                current_mask = new_mask
                current_adj = new_adj
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges = current_mask
                found_improvement = True
                break
        
        if not found_improvement:
            # Try edge swaps to escape local optima?
            # Or just restart with a different random seed?
            # Let's try a few random swaps
            for _ in range(10):
                # Pick a random edge to remove
                u, v = random.choice(list(edge_indices.keys()))
                if not ((current_adj[u] >> v) & 1): continue
                # Pick a random non-edge to add
                x, y = random.choice(list(edge_indices.keys()))
                if (current_adj[x] >> y) & 1: continue
                
                # Remove (u,v) and add (x,y)
                new_mask, new_adj = remove_edge(current_mask, current_adj, u, v)
                new_mask, new_adj = add_edge(new_mask, new_adj, x, y)
                
                # Check validity
                valid = True
                for w in range(n):
                    if w in (u, v, x, y): continue
                    # Check pairs involving u, v, x, y
                    for a in (u, v, x, y):
                        common = new_adj[a] & new_adj[w]
                        if bin(common).count('1') > 2:
                            valid = False
                            break
                    if not valid: break
                if not valid: continue
                
                # Also check pairs among u,v,x,y
                for a in range(4):
                    for b in range(a+1, 4):
                        p = [u, v, x, y][a]
                        q = [u, v, x, y][b]
                        common = new_adj[p] & new_adj[q]
                        if bin(common).count('1') > 2:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    current_mask = new_mask
                    current_adj = new_adj
                    # Count remains same, but structure changed
                    # This might allow future additions
                    break

    # Convert best_edges back to list of edges
    result = []
    for (u, v), idx in edge_indices.items():
        if (best_edges >> idx) & 1:
            result.append([u, v])
    return result
