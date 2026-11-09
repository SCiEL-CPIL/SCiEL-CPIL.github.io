import time
import random
from itertools import combinations

def _is_c4_free(n, adj):
    """Check if graph is C4-free using bitsets for speed. adj is list of bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common = adj[u] & adj[v]
            # Count bits: if >= 2, C4 exists
            c = common
            if c & (c - 1):  # More than one bit set
                return False
    return True

def _min_degree(n, adj):
    """Compute minimum degree from adjacency bitmasks."""
    min_deg = n
    for u in range(n):
        d = bin(adj[u]).count('1')
        if d < min_deg:
            min_deg = d
    return min_deg

def _can_add_edge(u, v, adj, n):
    """Check if adding edge (u, v) creates a C4. Returns True if safe."""
    nu = adj[u]
    nv = adj[v]
    # Common neighbors of u and v
    common = nu & nv
    # If there is more than 0 common neighbors, adding (u,v) creates a C4
    # Wait, C4 is u-x-v-y-u. If u and v are connected, and share a neighbor x, that's a triangle.
    # A C4 involving new edge (u,v) would be u-v-w-x-u.
    # This requires u and v to be connected to w and x? No.
    # The cycle is u - v - w - x - u.
    # So u is connected to v and x. v is connected to u and w. w is connected to v and x. x is connected to w and u.
    # If we add (u,v), we need to ensure there is no path of length 3 from u to v in the existing graph.
    # Path: u - a - b - v.
    # This is equivalent to: exists a in N(u), b in N(v) such that (a,b) is an edge.
    # Let's check:
    # For each a in N(u):
    #   For each b in N(v):
    #     if (a,b) is edge -> C4.
    
    # Using bitmasks:
    # Let A = N(u) \ {v}
    # Let B = N(v) \ {u}
    # We need to check if there is an edge between A and B.
    # This is not a simple bitwise AND.
    # However, we can iterate over the smaller set.
    
    # Optimization: If degree(u) + degree(v) is small, this is fast.
    # Here degrees are moderate.
    
    # Let's stick to the previous logic which was:
    # creates_c4 = False
    # w_mask = nv
    # while w_mask:
    #    ... w is neighbor of v
    #    x_mask = nu & ~w_bit & ~(1 << v)
    #    if x_mask & adj[w]: creates_c4 = True
    # This checks: for w in N(v), is there x in N(u) \ {v, w} such that (w,x) is edge?
    # This corresponds to path u-x-w-v. Yes.
    
    nu_mask = nu & ~(1 << v)
    nv_mask = nv & ~(1 << u)
    
    # Iterate over neighbors of v
    w_mask = nv_mask
    while w_mask:
        w_bit = w_mask & (-w_mask)
        w = w_bit.bit_length() - 1
        w_mask ^= w_bit
        
        # Check if any neighbor of u (excluding v and w) is connected to w
        x_mask = nu_mask & ~w_bit
        if x_mask & adj[w]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_n = 39
    best_edges = []
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = target_n - 39
        
        found = False
        
        for seed in range(2000):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            
            n = target_n
            adj = [0] * n
            deg = [0] * n
            
            # Start with a random graph with moderate density
            p = 0.25
            
            all_edges = list(combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if random.random() < p:
                    if _can_add_edge(u, v, adj, n):
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        deg[u] += 1
                        deg[v] += 1
            
            # Local search with edge swaps to balance degrees
            # We want min_deg >= min_deg_needed
            # Strategy: Find vertex u with min degree. Find vertex v with high degree.
            # Try to swap an edge (v, w) to (u, w) if possible.
            
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                # Check if we've reached target
                min_deg = _min_degree(n, adj)
                if min_deg >= min_deg_needed:
                    # Verify C4-free
                    if _is_c4_free(n, adj):
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if (adj[u] >> v) & 1:
                                    edges.append([u, v])
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = edges
                        found = True
                        break
                
                # Find vertex with minimum degree
                min_deg_vertex = 0
                min_deg_val = n
                max_deg_vertex = 0
                max_deg_val = 0
                for u in range(n):
                    d = deg[u]
                    if d < min_deg_val:
                        min_deg_val = d
                        min_deg_vertex = u
                    if d > max_deg_val:
                        max_deg_val = d
                        max_deg_vertex = u
                
                u = min_deg_vertex
                v = max_deg_vertex
                
                if min_deg_val >= min_deg_needed:
                     # Already good?
                     pass
                elif max_deg_val - min_deg_val < 2:
                    # Degrees are balanced but too low. Try to add edges to u.
                    added = False
                    candidates = [w for w in range(n) if w != u and not ((adj[u] >> w) & 1)]
                    random.shuffle(candidates)
                    for w in candidates:
                        if _can_add_edge(u, w, adj, n):
                            adj[u] |= 1 << w
                            adj[w] |= 1 << u
                            deg[u] += 1
                            deg[w] += 1
                            added = True
                            break
                    if not added:
                        break
                else:
                    # Try to swap edge (v, w) to (u, w)
                    # Conditions:
                    # 1. (v, w) is an edge.
                    # 2. (u, w) is NOT an edge.
                    # 3. deg[v] - 1 >= min_deg_needed (v stays valid)
                    # 4. Adding (u, w) is C4-free.
                    # 5. Removing (v, w) is always C4-free.
                    
                    swapped = False
                    # Candidates for w: neighbors of v
                    w_candidates = []
                    mask = adj[v]
                    while mask:
                        w_bit = mask & (-mask)
                        w = w_bit.bit_length() - 1
                        mask ^= w_bit
                        if w == u:
                            continue
                        if not ((adj[u] >> w) & 1):
                            w_candidates.append(w)
                    
                    random.shuffle(w_candidates)
                    
                    for w in w_candidates:
                        # Check if v can afford to lose this edge
                        if deg[v] - 1 < min_deg_needed:
                            continue
                        
                        # Check if adding (u, w) is safe
                        if _can_add_edge(u, w, adj, n):
                            # Perform swap
                            # Remove (v, w)
                            adj[v] &= ~(1 << w)
                            adj[w] &= ~(1 << v)
                            deg[v] -= 1
                            deg[w] -= 1
                            
                            # Add (u, w)
                            adj[u] |= 1 << w
                            adj[w] |= 1 << u
                            deg[u] += 1
                            deg[w] += 1
                            
                            swapped = True
                            break
                    
                    if not swapped:
                        # Try adding edges to u directly if swap failed
                        added = False
                        candidates = [w for w in range(n) if w != u and not ((adj[u] >> w) & 1)]
                        random.shuffle(candidates)
                        for w in candidates:
                            if _can_add_edge(u, w, adj, n):
                                adj[u] |= 1 << w
                                adj[w] |= 1 << u
                                deg[u] += 1
                                deg[w] += 1
                                added = True
                                break
                        if not added:
                            break
            
            if found:
                break
        
        if found:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
