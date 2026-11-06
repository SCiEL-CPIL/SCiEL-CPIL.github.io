import random
import time

def construct():
    deadline = time.time() + 140  # 10s margin
    
    def is_c4_free(adj, n):
        # adj is list of int bitmasks
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    # not adjacent
                    common = adj[u] & adj[v]
                    # count bits
                    if bin(common).count('1') >= 2:
                        return False
        return True

    def find_c4(adj, n):
        # Returns (u, v, w, x) or None
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    if bin(common).count('1') >= 2:
                        # get two bits
                        bits = []
                        c = common
                        while c:
                            lsb = c & -c
                            idx = lsb.bit_length() - 1
                            bits.append(idx)
                            c -= lsb
                            if len(bits) == 2:
                                break
                        w, x = bits
                        return (u, v, w, x)
        return None

    def can_add_edge(adj, u, v, n):
        # Check if adding (u, v) creates a C4
        # C4 would be u-a-b-v-u
        # a in N(u), b in N(v), a~b
        # Note: u and v are not adjacent
        # We need to check if there is an edge between N(u) and N(v)
        # Also, we must ensure that adding the edge doesn't violate the "non-adjacent common neighbors" rule for other pairs?
        # Actually, if the graph was C4-free before, adding an edge (u,v) only creates C4s involving (u,v).
        # Because any C4 not involving (u,v) was already a C4.
        # So we just check for C4s involving (u,v).
        # A C4 involving (u,v) is u-a-b-v-u.
        # This requires a in N(u), b in N(v), and a~b.
        # Also a!=v, b!=u (guaranteed since u,v not adjacent and simple graph).
        # Also a!=b.
        
        N_u = adj[u]
        N_v = adj[v]
        
        # Check if any neighbor of u is connected to any neighbor of v
        # Iterate neighbors of u
        c = N_u
        while c:
            lsb = c & -c
            a = lsb.bit_length() - 1
            c -= lsb
            
            # Check if a is connected to any neighbor of v
            # adj[a] & N_v
            if adj[a] & N_v:
                return False
        
        return True

    def try_construct(n, min_deg):
        random.seed(42)
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Initialize
            adj = [0] * n
            # Random graph p=0.15
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.15:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Step 1: Remove C4s
            # We remove edges until no C4 exists
            # To be efficient, we can just loop
            changed = True
            while changed and time.time() < deadline:
                changed = False
                c4 = find_c4(adj, n)
                if c4:
                    u, v, w, x = c4
                    # Remove a random edge from the C4
                    # Edges: (u,w), (w,v), (v,x), (x,u)
                    edges_to_remove = [(u, w), (w, v), (v, x), (x, u)]
                    e_u, e_v = random.choice(edges_to_remove)
                    adj[e_u] &= ~(1 << e_v)
                    adj[e_v] &= ~(1 << e_u)
                    changed = True
            
            # Check if C4 free
            if not is_c4_free(adj, n):
                continue
                
            # Step 2: Boost degree
            # Calculate degrees
            deg = [bin(adj[i]).count('1') for i in range(n)]
            
            # Try to add edges to increase min degree
            # We will try to add edges until min degree is satisfied or we get stuck
            success = True
            for _ in range(100): # Max iterations
                if min(deg) >= min_deg:
                    break
                if time.time() > deadline:
                    success = False
                    break
                    
                # Find vertex with min degree
                v_min = deg.index(min(deg))
                # Try to add an edge from v_min to some u
                # u must not be adjacent to v_min
                # And adding edge must not create C4
                candidates = []
                for u in range(n):
                    if u != v_min and not (adj[v_min] >> u) & 1:
                        if can_add_edge(adj, v_min, u, n):
                            candidates.append(u)
                            
                if not candidates:
                    # Stuck
                    # Try to remove an edge from a high degree vertex to free up space?
                    # Or just fail this attempt
                    success = False
                    break
                else:
                    u_new = random.choice(candidates)
                    # Add edge
                    adj[v_min] |= (1 << u_new)
                    adj[u_new] |= (1 << v_min)
                    deg[v_min] += 1
                    deg[u_new] += 1
                    
            if success and min(deg) >= min_deg:
                # Construct edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
        return None

    # Try n=46
    res = try_construct(46, 7)
    if res:
        return res
        
    # Try n=45
    res = try_construct(45, 6)
    if res:
        return res
        
    # Fallback: Empty graph
    return {'n': 39, 'edges': []}
