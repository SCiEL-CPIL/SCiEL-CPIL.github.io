import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: Seed graph
    squares = {x * x % 17 for x in range(1, 17)}
    seed_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    seed_result = {'n': 17, 'edges': seed_edges}
    
    best_result = seed_result
    
    # Try to find a larger graph
    # We know R(4,6) > 35 is possible (Exoo).
    # We will try n=35 down to n=20.
    for n in range(35, 19, -1):
        if time.time() > deadline - 10:
            break
        
        # Local search for a valid graph on n vertices
        result = _local_search(n, deadline)
        if result is not None:
            best_result = result
            # If we found a valid graph, we can try to increase n?
            # The loop goes downwards, so we found the largest n in this pass?
            # No, we start at 35. If we find one, it's the best so far.
            # We should break if we found 35? Yes, 35 is the known record.
            # If we find 36, we'd break. But we start at 35.
            # So if we find 35, we are done (max possible is 35 or 36).
            # Actually, if we find 35, we can try 36?
            # Let's try 36 first.
            pass
    
    # If we didn't find anything better than seed, return seed.
    # But we should try 36 first.
    
    # Revised strategy:
    # Try n=36, then 35, 34...
    # If we find a valid graph, return it.
    
    for n in [36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
        if time.time() > deadline - 10:
            break
        
        # Run local search with a time budget proportional to n
        # Larger n is harder, so give it more time?
        # Or just a fixed budget per n.
        time_budget = 8 if n <= 30 else 15
        
        result = _local_search(n, time_budget, start_time)
        if result is not None:
            return result
            
    return best_result

def _local_search(n, time_budget, global_start_time):
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initialize random graph
    # Density ~ 0.5
    A = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                A[i, j] = True
                A[j, i] = True
                
    # Calculate initial cost
    k4_count, ind6_count = _count_k4_ind6(A, n)
    cost = k4_count + 10 * ind6_count # Weight ind6 higher?
    
    if cost == 0:
        return _to_result(A, n)
        
    # Precompute edge list
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local search loop
    while time.time() < deadline:
        best_delta = 0
        best_edge = None
        best_new_k4 = k4_count
        best_new_ind6 = ind6_count
        
        # Sample a subset of edges to speed up?
        # Or iterate all if n is small.
        # n=35, edges=595.
        # Calculating delta for each edge.
        
        for u, v in edges:
            if time.time() > deadline:
                break
                
            # Calculate delta for flipping (u, v)
            # Delta K4:
            # If we add edge (u,v):
            #   New K4s formed: pairs (w,x) such that w,x in N(u) ∩ N(v) and w-x is edge?
            #   No, K4 is u,v,w,x.
            #   u,v,w,x is K4 iff w,x are common neighbors of u,v AND w-x is edge.
            #   So count edges in the subgraph induced by CommonNeighbors(u,v).
            # If we remove edge (u,v):
            #   Destroyed K4s: same count.
            
            # Delta Ind6:
            # If we add edge (u,v):
            #   Destroyed Ind6s: sets {u,v} ∪ S where S is independent set of size 4 in G \ {u,v}
            #   and S is disjoint from N(u) ∪ N(v).
            #   Let C = {w != u,v : A[u,w]==0 and A[v,w]==0}
            #   Count independent sets of size 4 in G[C].
            # If we remove edge (u,v):
            #   Created Ind6s: same count.
            
            common_neighbors = np.where(A[u] & A[v])[0]
            # Remove u, v if present (they are not neighbors of themselves)
            # A[u,u] is 0, so u is not in A[u].
            
            # Count edges in subgraph induced by common_neighbors
            # This is count of pairs (w,x) in common_neighbors such that A[w,x] is True
            if len(common_neighbors) > 0:
                sub_adj = A[np.ix_(common_neighbors, common_neighbors)]
                delta_k4 = np.sum(sub_adj) // 2
            else:
                delta_k4 = 0
                
            # Count independent sets of size 4 in G[C]
            # C = common non-neighbors
            common_non_neighbors = np.where((~A[u]) & (~A[v]) & ~np.eye(n, dtype=bool))[0]
            # Remove u, v
            common_non_neighbors = common_non_neighbors[(common_non_neighbors != u) & (common_non_neighbors != v)]
            
            # Count independent sets of size 4 in G[common_non_neighbors]
            # This is equivalent to counting K4s in the complement graph induced by common_non_neighbors
            # Or just iterating 4-sets?
            # len(common_non_neighbors) ~ 15-20.
            # 4-sets ~ 1000-3000.
            # Check if independent: 6 edges.
            # This is fast.
            
            ind4_count = 0
            cnn_list = common_non_neighbors.tolist()
            m = len(cnn_list)
            if m >= 4:
                # Iterate 4-combinations
                # Use itertools.combinations for simplicity, might be slow
                # But m is small.
                import itertools
                for comb in itertools.combinations(cnn_list, 4):
                    # Check if comb is independent in A
                    is_ind = True
                    for i in range(4):
                        for j in range(i+1, 4):
                            if A[comb[i], comb[j]]:
                                is_ind = False
                                break
                        if not is_ind:
                            break
                    if is_ind:
                        ind4_count += 1
                        
            delta_ind6 = ind4_count
            
            # Current state: A[u,v] is either 0 or 1
            if A[u, v]:
                # We are removing edge
                new_k4 = k4_count - delta_k4
                new_ind6 = ind6_count + delta_ind6
            else:
                # We are adding edge
                new_k4 = k4_count + delta_k4
                new_ind6 = ind6_count - delta_ind6
                
            new_cost = new_k4 + 10 * new_ind6
            delta_cost = new_cost - cost
            
            if delta_cost < best_delta:
                best_delta = delta_cost
                best_edge = (u, v)
                best_new_k4 = new_k4
                best_new_ind6 = new_ind6
                
        if best_edge is None:
            # No improvement, try a random move or break
            # Simulated annealing: accept worse moves with probability
            # For simplicity, break if no improvement for a while?
            # Or just pick a random edge and flip it to escape local minima.
            if random.random() < 0.1:
                u, v = random.choice(edges)
                if A[u, v]:
                    # Remove
                    # Recalculate deltas?
                    # Just flip and recalculate full cost?
                    # Expensive.
                    pass
            break
            
        # Apply best move
        u, v = best_edge
        A[u, v] = not A[u, v]
        A[v, u] = not A[u, v]
        k4_count = best_new_k4
        ind6_count = best_new_ind6
        cost = best_new_k4 + 10 * best_new_ind6
        
        if cost == 0:
            return _to_result(A, n)
            
    return None

def _count_k4_ind6(A, n):
    k4_count = 0
    for i in range(n):
        neighbors = np.where(A[i])[0]
        if len(neighbors) >= 3:
            sub_adj = A[np.ix_(neighbors, neighbors)]
            # Count triangles in sub_adj
            # Triangles = (sum of cubes of degrees - sum of degrees) / 6? No.
            # Triangles = trace(A^3) / 6
            # Or just count edges in sub_adj? No, triangles are 3-cliques.
            # For small subgraph, just count 3-cliques.
            # Or use formula:
            # t = 0
            # for x in neighbors:
            #   for y in neighbors:
            #     if x < y and sub_adj[x,y]:
            #       common = np.sum(sub_adj[x] & sub_adj[y])
            #       t += common
            # k4_count += t // 3 ? No.
            
            # Simpler:
            # Count K3 in subgraph.
            # K3 in subgraph corresponds to K4 in original containing i.
            # Each K4 containing i has 3 K3s in its neighborhood?
            # No. A K4 {i, a, b, c} has neighborhood {a, b, c} for i.
            # The subgraph on {a, b, c} is a K3.
            # So counting K3s in N(i) counts each K4 containing i exactly once?
            # No, the K3 {a,b,c} is the only K3 in N(i) for that K4.
            # So sum of K3s in N(i) over all i counts each K4 exactly 4 times.
            
            # Count K3 in sub_adj:
            # k3 = 0
            # for x in range(len(neighbors)):
            #   for y in range(x+1, len(neighbors)):
            #     if sub_adj[x, y]:
            #       k3 += np.sum(sub_adj[x] & sub_adj[y])
            # k4_count += k3 // 3 ?
            # Wait.
            # If {a,b,c} is a triangle in N(i), then {i,a,b,c} is a K4.
            # Is it counted once?
            # Yes, for vertex i, the triangle {a,b,c} is unique.
            # So sum over i of (number of triangles in N(i)) = 4 * (number of K4s).
            
            k3 = 0
            m = len(neighbors)
            for x in range(m):
                for y in range(x+1, m):
                    if sub_adj[x, y]:
                        k3 += np.sum(sub_adj[x] & sub_adj[y])
            k4_count += k3
            
    k4_count //= 4
    
    # Count Ind6
    # Ind6 in G is K6 in ~G
    # Similar logic:
    # Count K5 in non-neighborhood of each i?
    # No, Ind6 {i, a, b, c, d, e} means i is not connected to a,b,c,d,e.
    # And a,b,c,d,e form an Ind5.
    # So count Ind5 in non-N(i).
    # Sum over i of (number of Ind5s in non-N(i)) = 6 * (number of Ind6s).
    
    ind6_count = 0
    for i in range(n):
        non_neighbors = np.where(~A[i])[0]
        non_neighbors = non_neighbors[non_neighbors != i]
        if len(non_neighbors) >= 5:
            # Count Ind5 in G[non_neighbors]
            # This is K5 in ~G[non_neighbors]
            # Let B = ~A[np.ix_(non_neighbors, non_neighbors)]
            # B is adjacency of complement.
            # Count K5 in B.
            # This is expensive.
            # len(non_neighbors) ~ 17.
            # K5 count in graph of 17 vertices.
            # Iterate 5-sets? 6188 sets.
            # Check if clique in B.
            # 5-sets * 10 edges = 60k ops.
            # 35 vertices * 60k = 2M ops.
            # Acceptable.
            
            B = ~A[np.ix_(non_neighbors, non_neighbors)]
            # Remove diagonal
            np.fill_diagonal(B, False)
            
            ind5_count = 0
            nn_list = non_neighbors.tolist()
            m = len(nn_list)
            if m >= 5:
                import itertools
                for comb in itertools.combinations(nn_list, 5):
                    # Check if comb is clique in B
                    is_clique = True
                    for x in range(5):
                        for y in range(x+1, 5):
                            if not B[comb[x], comb[y]]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        ind5_count += 1
            ind6_count += ind5_count
            
    ind6_count //= 6
    
    return k4_count, ind6_count

def _to_result(A, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j]:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
