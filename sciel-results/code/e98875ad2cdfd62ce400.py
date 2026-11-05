import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 115

def _can_add_edge(u, v, adj):
    """
    Check if adding edge (u, v) creates a C6.
    Equivalent to: is there a path of length 4 from u to v in G \ {u, v}?
    Path: u - a - b - c - d - v
    a, b, c, d distinct and not u, v.
    
    Using bitmasks for speed.
    """
    # Neighbors of u excluding v
    nu = adj[u] & ~(1 << v)
    if nu == 0:
        return False
    
    # We want to find a path u-a-b-c-d-v.
    # Let S1 = N(u) \ {v}
    # Let S2 = (union of N(a) for a in S1) \ {u, v}
    # Let S3 = (union of N(b) for b in S2) \ {u, v, S1}  -- actually we just need distinctness of a,b,c,d
    # The condition is that there exists a in S1, b in N(a)\{u,v}, c in N(b)\{u,v,a}, d in N(c)\{u,v,b} such that d in N(v).
    # Or more simply: 
    # Let A = N(u) \ {v}
    # Let B = (union_{a in A} N(a)) \ {u, v}
    # Let C = (union_{b in B} N(b)) \ {u, v}
    # Let D = (union_{c in C} N(c)) \ {u, v}
    # If N(v) intersects D, does that guarantee a simple path?
    # Not necessarily, because we need a, b, c, d to be distinct.
    # However, for C6-free check, a simple path is required.
    # The "expansion" method might overcount if paths intersect.
    # But for a quick necessary condition, if (N(v) & D) != 0, it's *possible*.
    # If (N(v) & D) == 0, it's impossible.
    # If (N(v) & D) != 0, we must verify a simple path exists.
    
    # Let's try the expansion first. It's very fast with bitmasks.
    # A = N(u) \ {v}
    # B = union N(a) for a in A, excluding u, v
    # C = union N(b) for b in B, excluding u, v
    # D = union N(c) for c in C, excluding u, v
    
    # If N(v) & D == 0, return False (no C6).
    # If N(v) & D != 0, we need to check if a simple path exists.
    # Given the sparsity (approx 5.25 degree), the expansion is likely small.
    # But wait, if the graph is dense, expansion is large.
    # With 105 edges on 40 vertices, avg degree 5.25.
    # |A| ~ 5.
    # |B| ~ 5*5 = 25.
    # |C| ~ 25*5 = 125 (capped at 38).
    # |D| ~ 38*5 = 190 (capped at 38).
    # So D is likely all vertices.
    # Thus N(v) & D is almost always non-zero.
    # So the expansion check is not sufficient to rule out C6. It only rules out if D is small.
    
    # We need a more precise check or a faster DFS.
    # Let's use a BFS/DFS with visited masks, but optimized.
    # Since depth is only 4, and branching factor is low, it should be fast.
    
    # Optimization: 
    # If u and v are not at distance 4, we return False.
    # We can check distance 2, 3, 4.
    # If dist < 4, adding edge creates C3, C4, or C5? No, C6 requires cycle length 6.
    # Adding edge (u,v) creates a C6 iff there is a path of length 5 between u and v?
    # No. A cycle of length 6 involving (u,v) consists of edge (u,v) and a path of length 5 from u to v?
    # Path: u - a - b - c - d - e - v. Length 5.
    # Wait.
    # Cycle: u - v - e - d - c - b - a - u.
    # Edges: (u,v), (v,e), (e,d), (d,c), (c,b), (b,a), (a,u).
    # This is a 7-cycle?
    # Let's count vertices: u, v, e, d, c, b, a. 7 vertices.
    # So adding edge (u,v) creates a C6 if there is a path of length 5 from u to v in G \ {u,v}?
    # Let's re-read the definition.
    # C6: 6 vertices.
    # If we add edge (u,v), and there is a path u-a-b-c-d-v of length 5 (5 edges, 4 internal nodes),
    # then the cycle is u-v-d-c-b-a-u.
    # Vertices: u, v, d, c, b, a. 6 vertices.
    # Yes.
    # So we need to check if there is a path of length 5 from u to v in G \ {u, v}.
    # My previous code checked for path of length 4?
    # "Path: u - a - b - c - d - v"
    # Edges: (u,a), (a,b), (b,c), (c,d), (d,v). 5 edges.
    # Internal nodes: a, b, c, d. 4 nodes.
    # Cycle: u-v-d-c-b-a-u.
    # Edges in cycle: (u,v), (v,d), (d,c), (c,b), (b,a), (a,u). 6 edges.
    # Yes.
    # So we need to check for a path of length 5 from u to v.
    # Which means DFS depth 5?
    # Start at u.
    # Step 1: neighbors of u.
    # Step 2: neighbors of step 1.
    # Step 3: neighbors of step 2.
    # Step 4: neighbors of step 3.
    # Step 5: neighbors of step 4. If any is v, return True.
    
    # My previous code:
    # stack.append((a, 1, ...))  # a is neighbor of u. Depth 1.
    # if depth == 4: check if neighbor is v.
    # This checks for path u-a-b-c-d-v.
    # u (0) -> a (1) -> b (2) -> c (3) -> d (4).
    # Then check if d is neighbor of v.
    # This is a path of length 5 from u to v.
    # So the previous logic was correct.
    
    # Let's implement a faster version.
    # Use bitmasks.
    
    # Level 1: N(u) \ {v}
    l1 = adj[u] & ~(1 << v)
    if l1 == 0:
        return False
    
    # Level 2: union N(a) for a in l1, excluding u, v, and l1?
    # We need distinct nodes.
    # If we just take union, we might allow a=b.
    # But if a=b, then u-a-a-... is not a simple path.
    # However, if there is a non-simple walk, there might not be a simple path.
    # But if there is a simple path, it will be found by a proper BFS/DFS.
    # The bitmask expansion (union) finds *walks*, not necessarily *simple paths*.
    # If a walk exists, a simple path might not.
    # Example: u-a-b-a-v. Walk of length 4. But no simple path of length 4?
    # u-a-b-v is length 3.
    # So expansion is a necessary condition (if no walk, no path).
    # But not sufficient.
    
    # Given the small size, let's do a proper DFS but with bitmasks for visited.
    # To make it fast, we can use a recursive function or a stack.
    # Stack: (node, depth, visited_mask)
    # We prune if depth > 5.
    
    # Further optimization:
    # If l1 & adj[v] != 0, then there is a path u-a-v of length 2.
    # This doesn't create a C6.
    # We need length 5.
    
    # Let's try a bidirectional BFS?
    # From u to depth 2, and from v to depth 3?
    # Or u to 2, v to 2, and check intersection?
    # Path u-a-b-c-d-v.
    # Middle is b-c.
    # If we find a path u-a-b and c-d-v, and b-c is an edge, and all distinct?
    # This is complicated.
    
    # Let's stick to DFS but optimize the inner loop.
    
    # Pre-compute neighbors lists? No, adj is already fast.
    
    # One trick: if (adj[u] & adj[v]) != 0, we have a triangle.
    # Triangles don't form C6 directly, but they reduce the space for long paths?
    # Not really.
    
    # Let's implement the DFS with a stack of integers representing (node, depth, mask).
    # To save memory and time, we can pack node and depth into one int?
    # Or just use a list of tuples.
    
    # Actually, for depth 5, the number of paths can explode.
    # But we can prune aggressively.
    
    # Alternative: 
    # Count the number of paths of length 5?
    # If count > 0, return True.
    # We can compute this using matrix multiplication?
    # (A^5)[u][v] > 0?
    # This counts walks, not simple paths.
    # But for C6-free graphs, the number of walks is small?
    # If (A^5)[u][v] > 0, it's likely a simple path exists unless there's a lot of backtracking.
    # Given the sparsity, backtracking is limited.
    # Let's try this as a fast filter.
    # If (A^5)[u][v] == 0, return False.
    # If > 0, do the DFS to confirm.
    
    # Computing A^5 is expensive if done from scratch each time.
    # But we can update it? No, graph changes.
    
    # Let's just optimize the DFS.
    
    # We can use a "frontier" approach.
    # F0 = {u}
    # F1 = N(u) \ {v}
    # F2 = (union N(a) for a in F1) \ {u, v} \ F1
    # F3 = (union N(b) for b in F2) \ {u, v} \ F1 \ F2
    # F4 = (union N(c) for c in F3) \ {u, v} \ F1 \ F2 \ F3
    # If F4 & N(v) != 0, then there is a walk.
    # Does it imply a simple path?
    # Yes, if the walk is simple.
    # The construction of Fi by removing previous layers ensures that the *layers* are disjoint.
    # But within a layer, nodes are distinct.
    # So a path u-a-b-c-d-v where a in F1, b in F2, c in F3, d in F4 is a simple path?
    # Yes, because all layers are disjoint and u, v are excluded.
    # So if we construct F1, F2, F3, F4 this way, and F4 intersects N(v), then there IS a simple path of length 5.
    # Why?
    # Because we can pick d in F4 ∩ N(v).
    # d has a neighbor c in F3 (by definition of F4).
    # c has a neighbor b in F2.
    # b has a neighbor a in F1.
    # a has a neighbor u.
    # And all nodes u, a, b, c, d, v are distinct.
    # So this "layered" BFS finds a simple path if and only if one exists?
    # It finds *a* simple path if the intersection is non-empty.
    # Is it possible that a simple path exists but this method misses it?
    # No, because any simple path u-a-b-c-d-v will have a in F1, b in F2 (since b not in F1, u, v), c in F3, d in F4.
    # So if a simple path exists, F4 will intersect N(v).
    # So this method is correct!
    # And it's very fast with bitmasks.
    
    mask_uv = (1 << u) | (1 << v)
    
    f1 = adj[u] & ~mask_uv
    if f1 == 0:
        return False
    
    # Compute union of neighbors of f1
    # We can do this by iterating bits in f1
    # Or using a precomputed "neighbor union" table? No.
    # Just iterate.
    
    # To speed up union of neighbors, we can use a trick:
    # If we represent the graph as a list of bitmasks, we can just OR them.
    
    union1 = 0
    temp = f1
    while temp:
        bit = temp & -temp
        idx = bit.bit_length() - 1
        union1 |= adj[idx]
        temp ^= bit
    
    f2 = union1 & ~mask_uv & ~f1
    if f2 == 0:
        return False
        
    union2 = 0
    temp = f2
    while temp:
        bit = temp & -temp
        idx = bit.bit_length() - 1
        union2 |= adj[idx]
        temp ^= bit
        
    f3 = union2 & ~mask_uv & ~f1 & ~f2
    if f3 == 0:
        return False
        
    union3 = 0
    temp = f3
    while temp:
        bit = temp & -temp
        idx = bit.bit_length() - 1
        union3 |= adj[idx]
        temp ^= bit
        
    f4 = union3 & ~mask_uv & ~f1 & ~f2 & ~f3
    if f4 == 0:
        return False
        
    # Check if any node in f4 is connected to v
    if f4 & adj[v]:
        return True
        
    return False

def _greedy_with_local_search(n=40, seed=0):
    rng = random.Random(seed)
    adj = [0] * n
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if not _can_add_edge(u, v, adj):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add((u, v))
            
    # Local Search
    improved = True
    max_iter = 500
    
    while improved and time.time() < _DEADLINE - 5:
        improved = False
        
        # Phase 1: Try to add
        non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
        rng.shuffle(non_edges)
        
        added = False
        for u, v in non_edges:
            if not _can_add_edge(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
                added = True
                improved = True
                break
                
        if added:
            continue
            
        # Phase 2: Try to swap
        edge_list = list(edges)
        rng.shuffle(edge_list)
        
        swapped = False
        for u, v in edge_list:
            # Remove u, v
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.remove((u, v))
            
            # Try to add a non-edge
            non_edges_curr = [(x, y) for x, y in all_edges if (x, y) not in edges]
            rng.shuffle(non_edges_curr)
            
            added_back = False
            for x, y in non_edges_curr:
                if not _can_add_edge(x, y, adj):
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    edges.add((x, y))
                    added_back = True
                    swapped = True
                    improved = True
                    break
                    
            if not added_back:
                # Revert swap
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
            
            if swapped:
                break
                
    return list(edges)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 115
    
    best_edges = []
    best_count = 0
    
    for seed in range(1000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            edges = _greedy_with_local_search(40, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if count >= 110:
                    break
        except Exception:
            continue
            
    return [[u, v] for u, v in best_edges]
