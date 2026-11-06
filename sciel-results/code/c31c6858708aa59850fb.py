import numpy as np
import time

def _count_c4s(adj_matrix):
    """Count 4-cycles in a graph given by its adjacency matrix."""
    # A 4-cycle is a set of 4 vertices {a, b, c, d} such that edges (a,b), (b,c), (c,d), (d,a) exist.
    # Number of C4s = (1/8) * sum_{u,v} (A^2[u,v])^2 - (1/4) * sum_{u,v} (A^2[u,v]) * A[u,v] ... wait, simpler formula:
    # C4s = (1/8) * sum_{u != v} ( (A^2)_{u,v} )^2 - (1/4) * sum_{u} ( (A^2)_{u,u} )^2 ? No.
    # Let M = A^2. M[u,v] is the number of paths of length 2 from u to v.
    # The number of 4-cycles is (1/8) * sum_{u,v} M[u,v]^2 - (1/4) * sum_{u} M[u,u]^2? 
    # Actually, a standard formula is:
    # C4 = (1/8) * ( sum_{u,v} (A^2[u,v])^2 - sum_{u} (A^2[u,u])^2 ) - (1/2) * sum_{u,v} A[u,v] * A^2[u,v] ?
    # Let's use the property: Number of C4s = (1/8) * sum_{u,v} ( (A^2)_{u,v} )^2 - (1/4) * sum_{u} ( (A^2)_{u,u} )^2 is not quite right for simple graphs.
    # Correct formula:
    # Let N(u) be the set of neighbors of u.
    # For each pair of vertices {u, v}, let c(u,v) be the number of common neighbors.
    # The number of 4-cycles is (1/8) * sum_{u < v} c(u,v)^2? No.
    # Each 4-cycle (a,b,c,d) contributes 1 to c(a,c) and 1 to c(b,d).
    # So sum_{u < v} c(u,v)^2 counts each 4-cycle twice?
    # If we have a 4-cycle a-b-c-d-a, common neighbors of (a,c) include b and d. So c(a,c) >= 2.
    # The term c(a,c)^2 includes the pair (b,d) and (d,b) and (b,b) and (d,d).
    # It is easier to just iterate. But for 31 vertices, O(n^4) is 31^4 ~ 1M, which is fast enough for a baseline check.
    # However, for the search, we need delta.
    
    pass # This function is not used in the optimized path, kept for clarity if needed.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Step 1: Generate the 31 points of PG(2, 5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    p = (1, (y * pow(x, 4, 5)) % 5, (z * pow(x, 4, 5)) % 5)
                elif y != 0:
                    p = (0, 1, (z * pow(y, 4, 5)) % 5)
                else:
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    n = 31
    
    # Step 2: Construct Adjacency Matrix for Polarity Graph
    # Polarity defined by M = diag(1, 1, -1) = diag(1, 1, 4) in F_5
    # P ~ Q iff P^T M Q == 0 and P != Q
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            val = (p1[0]*p2[0] + p1[1]*p2[1] + 4*p1[2]*p2[2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    # Convert to bitmask representation for fast common neighbor counting
    # adj_bits[i] is a 32-bit integer where bit j is 1 if edge (i,j) exists
    adj_bits = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if A[i, j]:
                mask |= (1 << j)
        adj_bits[i] = mask
        
    # Function to count C4s from adj_bits
    def count_c4s(bits):
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = bits[i] & bits[j]
                c = bin(common).count('1')
                c4_count += c * c
        # Each C4 is counted 4 times in sum c(i,j)^2?
        # Let's verify.
        # A C4 is a-b-c-d-a.
        # Common neighbors of (a,c): b, d. c=2.
        # Common neighbors of (b,d): a, c. c=2.
        # Sum includes 2^2 for (a,c) and 2^2 for (b,d). Total 8.
        # So C4s = sum / 8.
        return c4_count // 8

    current_c4s = count_c4s(adj_bits)
    
    # Initial state: Polarity graph has 90 edges. We need 91.
    # The polarity graph is C4-free? 
    # Polarity graphs of PG(2,q) for q=5 have specific properties.
    # Actually, the task says ex(31, C4) = 90. The polarity graph has 90 edges.
    # If it were C4-free, current_c4s would be 0.
    # Let's check.
    # If current_c4s > 0, the polarity graph itself has C4s.
    # Wait, the problem states "one more than ex(31;C4) = 90".
    # This implies there exists a C4-free graph with 90 edges.
    # The polarity graph might not be that graph, or it might be.
    # If the polarity graph has C4s, we start with a higher count.
    # We must ensure we have exactly 91 edges.
    
    # If the polarity graph has 90 edges, we need to add 1.
    # But if the polarity graph already has C4s, adding an edge increases them.
    # Maybe we should start with a different C4-free graph?
    # The He-Ma-Yang theorem doesn't cover q=5.
    # However, the prompt implies the polarity graph is a good starting point.
    # Let's assume the polarity graph is C4-free or close to it.
    # Actually, for q=5, the polarity graph of PG(2,5) with M=diag(1,1,-1) is the generalized quadrangle GQ(4,4)? No, GQ(4,4) has 45 points.
    # The polarity graph here is a graph on 31 vertices.
    # Let's just run the search.
    
    # We need to add 1 edge to get to 91.
    # Let's find the edge that adds the fewest C4s.
    # Delta C4 for adding edge (u,v) is:
    # New C4s = number of paths of length 3 between u and v?
    # No, adding (u,v) creates a C4 for every path u-x-y-v.
    # So delta = A^3[u,v] (number of length 3 paths).
    
    # Let's calculate A^3 for the initial polarity graph.
    A2 = A @ A
    A3 = A2 @ A
    
    best_add = None
    min_delta = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                val = A3[i, j]
                if val < min_delta:
                    min_delta = val
                    best_add = (i, j)
                    
    # Add the best edge
    u, v = best_add
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)
    A[u, v] = 1
    A[v, u] = 1
    
    current_c4s += min_delta
    
    # Now we have 91 edges.
    # Perform local search: swap edges.
    # Remove edge (a,b), add edge (c,d).
    # Constraint: (a,b) must be present, (c,d) must be absent.
    # Also, we must maintain simple graph properties (no self loops, distinct edges).
    
    # Precompute list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                edges.append((i, j))
                
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not A[i, j]:
                non_edges.append((i, j))
                
    # To speed up delta calculation:
    # Delta for removing (a,b):
    # C4s destroyed = number of paths a-x-y-b.
    # This is A^3[a,b] in the current graph?
    # Yes, if we remove (a,b), the C4s that are broken are those using (a,b).
    # A C4 is a-b-c-d-a. It uses (a,b).
    # The other path is a-d-c-b. Length 3.
    # So # C4s using (a,b) = A^3[a,b].
    
    # Delta for adding (c,d):
    # C4s created = number of paths c-x-y-d in the graph WITHOUT (a,b)?
    # Or in the graph BEFORE removal?
    # If we do a swap, we remove (a,b) and add (c,d).
    # The C4s created by (c,d) might use (a,b) if the path goes through a or b.
    # Specifically, if the path is c-a-...-b-d, then removing (a,b) doesn't break that path for the new C4?
    # Wait.
    # Let G be the current graph.
    # Let G' be G - (a,b) + (c,d).
    # C4s in G' = C4s in G - C4s using (a,b) + C4s using (c,d) in G' - C4s using both (a,b) and (c,d) in G'.
    # This is complex.
    # Approximation:
    # Delta ≈ - A^3_G[a,b] + A^3_G[c,d]
    # This is valid if no C4 uses both (a,b) and (c,d).
    # Given the sparsity, this is likely a good approximation for a first pass.
    # We can refine by recalculating exactly if the approximation is promising.
    
    # We will use the approximation for the search, and verify exact count for the best moves.
    
    # We need to update A and A^3 efficiently?
    # Or just recompute A^3 for candidate moves?
    # Recomputing A^3 is O(n^3) = 31^3 ~ 30k ops. Very fast.
    # So we can just recompute A^3 for each candidate swap?
    # Number of swaps: |E| * |V^2 - E| ~ 91 * (465 - 91) ~ 91 * 374 ~ 34k swaps.
    # 34k * 30k ops = 1e9 ops. Might be too slow in Python if not optimized.
    # But we can be smarter.
    
    # Let's stick to bitmask for common neighbors.
    # Delta for adding (c,d) to G is C(c,d) = common neighbors of c and d in G? No, that's for triangles.
    # For C4s, it's the number of length 3 paths.
    
    # Let's use the exact delta calculation with bitmasks?
    # Number of length 3 paths from c to d:
    # Sum_{x in N(c)} Sum_{y in N(d)} [x != y and (x,y) in E and x != d and y != c]?
    # No, path c-x-y-d.
    # x is neighbor of c. y is neighbor of d.
    # We need edge (x,y) to exist.
    # And x != y, x != d, y != c.
    
    # def count_paths_len3(u, v, bits):
    #     count = 0
    #     for x in range(n):
    #         if bits[u] & (1 << x):
    #             for y in range(n):
    #                 if bits[v] & (1 << y):
    #                     if x != y and x != v and y != u:
    #                         if bits[x] & (1 << y):
    #                             count += 1
    #     return count
    
    # This is O(n^2) per pair.
    # 34k swaps * 31^2 ~ 34k * 1000 = 34M ops. This is feasible in 120s.
    
    # Let's implement the search loop.
    
    best_edges = edges[:]
    best_c4s = current_c4s
    current_bits = adj_bits[:]
    current_A = A.copy()
    
    # We will perform a simple local search.
    # Iterate through all possible swaps.
    # Keep the best improving swap.
    # Repeat until no improvement or time limit.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap = None
        best_delta = 0 # We want to minimize C4s, so we look for negative delta
        
        # Get current list of edges and non-edges
        curr_edges = []
        curr_non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if current_A[i, j]:
                    curr_edges.append((i, j))
                else:
                    curr_non_edges.append((i, j))
        
        # Calculate A^3 for current graph to speed up removal delta?
        # Or just use the path counting function.
        # Let's use the path counting function for both add and remove deltas.
        
        def count_paths_len3(u, v, bits):
            count = 0
            # Optimization: iterate over neighbors of u and v
            neigh_u = [k for k in range(n) if bits[u] & (1 << k)]
            neigh_v = [k for k in range(n) if bits[v] & (1 << k)]
            
            for x in neigh_u:
                if x == v: continue
                for y in neigh_v:
                    if y == u or y == x: continue
                    # Check if x-y is an edge
                    if bits[x] & (1 << y):
                        count += 1
            return count

        for e1 in curr_edges:
            a, b = e1
            # Delta for removing (a,b)
            # C4s lost = number of paths a-x-y-b
            delta_remove = -count_paths_len3(a, b, current_bits)
            
            for e2 in curr_non_edges:
                c, d = e2
                # Delta for adding (c,d)
                # C4s gained = number of paths c-x-y-d
                delta_add = count_paths_len3(c, d, current_bits)
                
                # Note: This approximation ignores interactions.
                # If the path for add uses the edge being removed, it's an overestimate of gain.
                # If the path for remove uses the edge being added, it's an underestimate of loss.
                # Given the sparsity, this is a reasonable heuristic.
                
                delta = delta_remove + delta_add
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e1, e2)
                    
        if best_swap and best_delta < 0:
            e1, e2 = best_swap
            a, b = e1
            c, d = e2
            
            # Apply swap
            current_A[a, b] = 0
            current_A[b, a] = 0
            current_bits[a] &= ~(1 << b)
            current_bits[b] &= ~(1 << a)
            
            current_A[c, d] = 1
            current_A[d, c] = 1
            current_bits[c] |= (1 << d)
            current_bits[d] |= (1 << c)
            
            # Recalculate exact C4 count to be sure
            new_c4s = count_c4s(current_bits)
            # Update best if better
            if new_c4s < best_c4s:
                best_c4s = new_c4s
                best_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if current_A[i, j]:
                            best_edges.append([i, j])
            improved = True
            # We could also update current_c4s = new_c4s here for tracking, 
            # but we use best_c4s for the final result.
            # For the search, we use the approximation delta to find the next move.
            # This might be slightly inaccurate, but it's fast.
            
    # Return the best graph found
    return best_edges
