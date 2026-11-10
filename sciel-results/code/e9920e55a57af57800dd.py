import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n = 31
    target_edges = 91
    
    # Build polarity graph of PG(2,5) as initial structure
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    def count_c4s(adj_matrix):
        # C4 = trace(A4)/8 - S2/4 + S1/8
        deg = np.sum(adj_matrix, axis=1)
        S1 = np.sum(deg)
        S2 = np.sum(deg**2)
        A2 = adj_matrix @ adj_matrix
        A4 = A2 @ A2
        trace_A4 = np.trace(A4)
        return int(trace_A4 // 8 - S2 // 4 + S1 // 8)

    current_c4s = count_c4s(adj)
    
    # Helper to count new C4s when adding edge (u, v)
    def count_new_c4s_add(u, v, adj_matrix):
        # New C4s using edge (u,v)
        # This is the number of paths of length 2 from u to v that do not use (u,v)
        # But wait, the formula counts all C4s.
        # If we add (u,v), the change in C4 count is the number of C4s that include (u,v).
        # A C4 including (u,v) has form u-v-a-b-u.
        # This requires a~v, b~u, a~b.
        # This is equivalent to counting pairs (a,b) such that a in N(v), b in N(u), a~b, a!=b.
        # Note: a,b cannot be u or v.
        # Let's compute this efficiently.
        # N(v) and N(u).
        # For each a in N(v), count neighbors in N(u) excluding v and a.
        count = 0
        # Using matrix rows
        row_v = adj_matrix[v]
        row_u = adj_matrix[u]
        # a must be neighbor of v
        # b must be neighbor of u
        # a must be neighbor of b
        # a != b
        # a != u, b != v
        # Since u,v not connected initially, N(v) doesn't contain u? 
        # In polarity graph, if u,v not connected, they are not neighbors.
        # So a in N(v) implies a != u.
        # b in N(u) implies b != v.
        # So we just need a~v, b~u, a~b, a!=b.
        
        # Iterate a in N(v)
        # for a in np.where(row_v > 0)[0]:
        #     # count b in N(u) such that b~a and b != a
        #     # b~a means adj[a,b] == 1
        #     # b in N(u) means adj[u,b] == 1
        #     # b != a
        #     common = np.where((row_u > 0) & (adj_matrix[a] > 0))[0]
        #     if a in common: common.remove(a) # or just subtract 1 if a is in common
        #     # a is in N(v). Is a in N(u)? If a in N(u) and a in N(v), then a is common neighbor.
        #     # If a is common neighbor, then a~a is false (no self loop). So a is not counted in adj[a]>0.
        #     # So we just count common.
        #     count += len(common)
        # 
        # This is O(n^2) per edge add.
        # We can vectorize.
        # But for n=31, O(n^2) is fine.
        
        for a in np.where(row_v > 0)[0]:
            # a is neighbor of v
            # find b in N(u) such that b is neighbor of a
            # b in N(u) -> row_u[b] == 1
            # b neighbor of a -> adj[a,b] == 1
            # b != a (guaranteed by no self loop)
            # b != v (guaranteed by b in N(u) and u,v not adjacent? 
            # If v in N(u), then u,v adjacent. But we only add non-edges.
            # So v not in N(u). So b != v.)
            # b != u (guaranteed by no self loop)
            # a != u (guaranteed by a in N(v) and u,v not adjacent)
            
            # Count b
            # b must satisfy row_u[b]==1 and adj[a,b]==1
            # This is dot product of row_u and adj[a]
            # But we must exclude b=a? No, adj[a,a]=0.
            # So count = np.sum(row_u * adj[a])
            count += np.sum(row_u * adj_matrix[a])
            
        return count

    # Helper to count C4s removed when removing edge (u, v)
    def count_c4s_removed(u, v, adj_matrix):
        # Same as count_new_c4s_add
        return count_new_c4s_add(u, v, adj_matrix)

    # Greedy optimization: add edges to reach 91, minimizing C4s
    while np.sum(adj) // 2 < target_edges:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_cost = float('inf')
        
        # Try all non-edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    cost = count_new_c4s_add(i, j, adj)
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = (i, j)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
    
    # If we have too many edges, remove greedily
    while np.sum(adj) // 2 > target_edges:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_saving = -1 # We want to remove edge with max C4s removed
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    saving = count_c4s_removed(i, j, adj)
                    if saving > best_saving:
                        best_saving = saving
                        best_edge = (i, j)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u, v] = 0
        adj[v, u] = 0
        
    # Local search: try edge swaps to reduce C4s
    # Use the fast C4 count for evaluation
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_c4s = count_c4s(adj)
        
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 1]
        non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        
        # To speed up, we can limit the number of swaps tried or use a heuristic
        # But with fast count, we can try more.
        # Let's try all swaps but break early if time is up.
        
        for (u, v) in edges_list:
            if time.time() > deadline:
                break
            for (x, y) in non_edges_list:
                if (u, v) == (x, y) or (u, v) == (y, x):
                    continue
                
                # Try swap
                adj[u, v] = 0
                adj[v, u] = 0
                adj[x, y] = 1
                adj[y, x] = 1
                
                new_c4s = count_c4s(adj)
                
                if new_c4s < current_c4s:
                    current_c4s = new_c4s
                    improved = True
                else:
                    # Revert
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[x, y] = 0
                    adj[y, x] = 0
        
        if time.time() > deadline:
            break
    
    # Extract edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])
    
    return result
