import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    # --- Step 1: Construct PG(2, 5) Polarity Graph ---
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, q)
                    points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    points.append((0, 1, (z * inv_y) % q))
                else:
                    points.append((0, 0, 1))
    
    points = list(set(points))
    n_base = len(points) # 31
    
    # Bilinear form: u0*v2 + u1*v1 + u2*v0
    def bilinear(u, v):
        return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % q

    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear(points[i], points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    n = n_base
    # We will store adjacency as a list of bitmasks
    # We need to dynamically add vertices.
    # Let's use a list `adj_list` where adj_list[i] is a bitmask of neighbors.
    # When adding a new vertex, we append to adj_list and update existing bitmasks.
    
    # Helper: Check C4-free property for a new vertex with neighbor set S
    # Condition: For any u, v in S, N(u) and N(v) must be disjoint (in the current graph).
    # Note: N(u) includes the new vertex? No, we are checking before adding.
    # So N(u) is the current neighborhood.
    # If u and v share a neighbor w, then adding x connected to u and v creates C4 x-u-w-v-x.
    # So we require N(u) & N(v) == 0 for all u != v in S.
    
    def is_valid_neighbor_set(S_mask):
        # S_mask is a bitmask of vertices in the current graph
        # We need to check all pairs in S_mask
        verts = []
        m = S_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m -= lsb
            
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u = verts[i]
                v = verts[j]
                if adj[u] & adj[v]:
                    return False
        return True

    # Helper: Fast MIS using Branch and Bound with Bitmasks
    def max_independent_set_size(n, adj, time_limit):
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')
            
        def branch(candidates_mask, current_size):
            if time.time() > time_limit:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with max degree in candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    # We will try to extend the graph up to 43 vertices.
    # We use a randomized greedy approach.
    
    # To allow backtracking or multiple attempts, we can run this in a loop with different seeds.
    # But since the base is fixed, we just randomize the choice of S.
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
            
        # Reset graph to base
        current_adj = adj[:]
        current_n = n_base
        
        # Try to add vertices one by one
        success = True
        for target_n in range(current_n + 1, 44):
            if time.time() > deadline - 5:
                success = False
                break
                
            # Find a valid neighbor set S for the new vertex
            # We want S to be "large" to reduce alpha, but valid.
            # Let's find a maximal independent set in the conflict graph.
            # Conflict graph: vertices are 0..current_n-1. Edge (u,v) if N(u) & N(v) != 0.
            # We need an independent set in this conflict graph.
            # This is equivalent to finding a set S such that N(u) & N(v) == 0 for all u,v in S.
            
            # Greedy construction of S:
            # Randomly order vertices.
            # Add vertex v to S if it doesn't conflict with any vertex already in S.
            
            order = list(range(current_n))
            random.shuffle(order)
            
            S = []
            S_mask = 0
            for v in order:
                # Check if v conflicts with any u in S
                conflict = False
                for u in S:
                    if current_adj[u] & current_adj[v]:
                        conflict = True
                        break
                if not conflict:
                    S.append(v)
                    S_mask |= (1 << v)
            
            if not S:
                # Couldn't add any neighbors? Then the new vertex is isolated.
                # An isolated vertex increases alpha by 1.
                # We should check if alpha(current) + 1 <= 10.
                # If current alpha is 9, we can add an isolated vertex.
                # If current alpha is 10, we cannot.
                pass
            
            # Check if S is valid (it is by construction)
            # But we also need to ensure that adding this vertex doesn't make alpha > 10.
            # Let's tentatively add the vertex and check alpha.
            
            # Tentative addition
            new_v = current_n
            # Update current_adj for existing vertices
            for u in range(current_n):
                if S_mask & (1 << u):
                    current_adj[u] |= (1 << new_v)
            # Add new vertex
            current_adj.append(S_mask)
            current_n += 1
            
            # Check alpha
            # We can use a faster check: 
            # alpha(G_new) <= max(alpha(G_old), 1 + alpha(G_old - N[new_v] - {new_v}))
            # But calculating exact alpha is safe.
            # To save time, we can first check if a greedy MIS is <= 10.
            # Or just run the B&B with a short time limit.
            
            alpha_val = max_independent_set_size(current_n, current_adj, time.time() + 2)
            
            if alpha_val > 10:
                # Rollback
                current_n -= 1
                current_adj.pop()
                for u in range(current_n):
                    if S_mask & (1 << u):
                        current_adj[u] &= ~(1 << new_v)
                
                # Try a different S?
                # For simplicity in this loop, we might just break and try a new attempt.
                # Or we can try to find a smaller S.
                # Let's try to find a smaller S by removing vertices from S.
                # But that's complex. Let's just break this target_n attempt and restart the whole extension with a new seed.
                success = False
                break
            else:
                # Success, keep the vertex
                continue
        
        if success and current_n >= 39:
            # We found a valid graph with n >= 39
            # Extract edges
            edge_list = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if current_adj[i] & (1 << j):
                        edge_list.append([i, j])
            return {'n': current_n, 'edges': edge_list}
        
        # If we didn't reach 39, we might still return the best found if it's valid.
        # But we want to maximize n.
        # Let's check if current_n (from the last successful attempt or base) is valid.
        # Actually, the loop breaks on failure. So current_n might be less than target.
        # Let's just try the next attempt.

    # Fallback: Return the base graph (31 vertices) which is definitely valid.
    # Or try to return a smaller valid graph if we managed to get some extensions.
    # Since we reset in the loop, we need to keep track of the best found.
    
    # Let's restructure to keep the best found.
    
    best_n = n_base
    best_adj = adj[:]
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
            
        current_adj = best_adj[:]
        current_n = best_n
        
        # Try to add vertices one by one
        for target_n in range(current_n + 1, 44):
            if time.time() > deadline - 5:
                break
                
            order = list(range(current_n))
            random.shuffle(order)
            
            S = []
            S_mask = 0
            for v in order:
                conflict = False
                for u in S:
                    if current_adj[u] & current_adj[v]:
                        conflict = True
                        break
                if not conflict:
                    S.append(v)
                    S_mask |= (1 << v)
            
            new_v = current_n
            for u in range(current_n):
                if S_mask & (1 << u):
                    current_adj[u] |= (1 << new_v)
            current_adj.append(S_mask)
            current_n += 1
            
            alpha_val = max_independent_set_size(current_n, current_adj, time.time() + 2)
            
            if alpha_val > 10:
                # Rollback
                current_n -= 1
                current_adj.pop()
                for u in range(current_n):
                    if S_mask & (1 << u):
                        current_adj[u] &= ~(1 << new_v)
                break
            else:
                # Keep it
                continue
        
        if current_n > best_n:
            best_n = current_n
            best_adj = current_adj[:]
            
            if best_n >= 39:
                edge_list = []
                for i in range(best_n):
                    for j in range(i + 1, best_n):
                        if best_adj[i] & (1 << j):
                            edge_list.append([i, j])
                return {'n': best_n, 'edges': edge_list}

    # Return the best found
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])
    return {'n': best_n, 'edges': edge_list}
