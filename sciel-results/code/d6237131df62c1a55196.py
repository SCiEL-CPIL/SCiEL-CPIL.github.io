"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    q = 5
    n = 31
    
    # 1. Build PG(2,5) polarity graph (90 edges, 0 C4s)
    points = []
    for z in range(q):
        for y in range(q):
            for x in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    rep = (1, (y*inv_x) % q, (z*inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    rep = (0, 1, (z*inv_y) % q)
                else:
                    rep = (0, 0, 1)
                if rep not in points:
                    points.append(rep)
    
    assert len(points) == 31
    
    def dot(p1, p2):
        return (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
    
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # 2. Find best edge to add to get 91 edges
    # Current edge count is 90. We need to add 1 edge.
    # Score is C4 count. Adding edge (u,v) adds C(common(u,v), 2) new C4s.
    # We want to minimize this.
    
    def count_c4s(adj_matrix):
        # M = adj @ adj gives common neighbors counts
        # M_ii is degree. M_ij is common neighbors for i != j.
        # Number of C4s = 1/2 * sum_{i < j} C(M_ij, 2)
        # C(M, 2) = M*(M-1)/2
        # Sum over i < j of M_ij*(M_ij-1)/2
        # Since M is symmetric, sum_{i != j} M_ij*(M_ij-1)/2 = 2 * sum_{i < j} ...
        # So total C4s = 1/2 * (1/2) * sum_{i != j} M_ij*(M_ij-1) = 1/4 * sum_{i != j} M_ij*(M_ij-1)
        # Or simply:
        M = adj_matrix @ adj_matrix
        # Zero out diagonal to avoid self-loops in counting? 
        # M_ii is degree, C(deg, 2) is not part of C4 count directly in this formula?
        # The formula sum_{i<j} C(M_ij, 2) works for i != j.
        # Let's just compute the sum for i < j.
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                m = M[i, j]
                count += m * (m - 1) // 2
        return count // 2 # Wait, does the formula sum_{i<j} C(M_ij, 2) count each C4 exactly once or twice?
        # A C4 is u-v-w-x-u. The pairs of opposite vertices are (u,w) and (v,x).
        # For pair (u,w), common neighbors are v and x. C(2,2)=1.
        # For pair (v,x), common neighbors are u and w. C(2,2)=1.
        # So each C4 is counted twice (once for each diagonal).
        # So yes, divide by 2.
        
    # Optimization: Use vectorized operations for count_c4s
    def count_c4s_fast(adj_matrix):
        M = adj_matrix @ adj_matrix
        # We need sum_{i < j} M[i,j]*(M[i,j]-1)//2
        # Create upper triangle mask
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        vals = M[mask]
        # Calculate sum of v*(v-1)/2
        # Since we are dividing by 2 at the end for the double counting, 
        # and the formula sum_{i<j} C(M_ij, 2) yields 2 * (#C4s), 
        # we return sum // 2.
        total = np.sum(vals * (vals - 1)) // 2
        return total // 2

    # Find best edge to add
    best_edge = None
    min_cost = float('inf')
    current_c4 = count_c4s_fast(adj) # Should be 0
    
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v] == 0:
                # Cost is number of new C4s created
                # New C4s = C(common_neighbors(u,v), 2)
                common = np.sum(adj[u] & adj[v])
                cost = common * (common - 1) // 2
                if cost < min_cost:
                    min_cost = cost
                    best_edge = (u, v)
    
    if best_edge:
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Now we have 91 edges.
    # Perform local search: swap an edge with a non-edge to reduce C4 count.
    
    # Precompute list of edges and non-edges for faster access? 
    # Or just iterate. n=31 is small.
    
    current_score = count_c4s_fast(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    # Simulated Annealing parameters
    temp = 10.0
    cooling = 0.999
    
    # Time limit check
    while time.time() < deadline - 5:
        # Randomly pick an edge to remove
        # And a non-edge to add
        
        # Get list of edges
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if not edges or not non_edges:
            break
            
        e_remove = random.choice(edges)
        e_add = random.choice(non_edges)
        
        # Check if they share vertices? 
        # If they share a vertex, it's a valid swap (edge swap).
        # If they are disjoint, it's also a valid swap.
        # Any swap of (u,v) with (x,y) maintains degree sum? 
        # Yes, degree of u decreases by 1, v decreases by 1.
        # x increases by 1, y increases by 1.
        # Total edges remains 91.
        
        u1, v1 = e_remove
        u2, v2 = e_add
        
        # Make sure we don't add an edge that already exists (checked) 
        # and remove an edge that exists (checked).
        # Also ensure (u2,v2) is not (u1,v1) or (v1,u1) - guaranteed by edge/non-edge split.
        
        # Apply swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        new_score = count_c4s_fast(adj)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        else:
            # Revert
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            # Simulated annealing acceptance
            if random.random() < np.exp(-(new_score - current_score) / temp):
                # Accept
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                current_score = new_score
            else:
                # Revert was already done, just keep state
                pass
                
        temp *= cooling
        if temp < 0.1:
            temp = 0.1 # Don't let it go to 0 too fast, or restart?
            # Let's keep it simple, just hill climbing with some randomness if stuck
            # Actually, let's just do hill climbing with random restarts if stuck
            # But for now, simple SA is fine.
            
    # Restore best state
    adj = best_adj
    
    # Build edge list
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges_list.append([i, j])
                
    assert len(edges_list) == 91, f"Expected 91 edges, got {len(edges_list)}"
    
    return edges_list
# EVOLVE-BLOCK-END
