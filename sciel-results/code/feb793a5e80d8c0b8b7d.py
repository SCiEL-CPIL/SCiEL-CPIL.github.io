import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Generate cyclic orders for A and B using a simple greedy TSP heuristic
    # We treat the matrix as a distance matrix (symmetric part) for the tour.
    # QAP matrices are symmetric.
    
    def greedy_tsp_order(mat):
        # mat is n x n
        # Start at 0
        unvisited = set(range(n))
        unvisited.remove(0)
        tour = [0]
        current = 0
        while unvisited:
            # Find nearest neighbor
            # Vectorized: mat[current, :]
            dists = mat[current, :].copy()
            # Set visited to inf
            for v in tour:
                dists[v] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return tour

    order_a = greedy_tsp_order(a)
    order_b = greedy_tsp_order(b)
    
    # Map facility index to its position in order_a
    pos_in_a = np.zeros(n, dtype=np.int64)
    for idx, fac in enumerate(order_a):
        pos_in_a[fac] = idx
        
    # Map location index to its position in order_b
    pos_in_b = np.zeros(n, dtype=np.int64)
    for idx, loc in enumerate(order_b):
        pos_in_b[loc] = idx
        
    # 2. Evaluate all n shifts
    best_cost = np.inf
    best_p = None
    
    # Precompute B permuted by order_b for faster access?
    # B_perm[i, j] = b[order_b[i], order_b[j]]
    # Cost for shift s: sum a[i, j] * b[p[i], p[j]]
    # p[i] = order_b[(pos_in_a[i] + s) % n]
    
    # To speed up, we can precompute the cost for each shift efficiently?
    # C_s = sum_{i,j} a[i,j] * b[order_b[(pos_a[i]+s)%n], order_b[(pos_a[j]+s)%n]]
    # Let u = pos_a[i], v = pos_a[j].
    # a[i,j] is fixed. The term depends on (u+s)%n and (v+s)%n.
    # This is equivalent to rotating the B matrix relative to the A matrix's "index space".
    
    # Let's just loop over s and calculate cost. n=150, so 150 iterations.
    # Each iteration: O(n^2) = 22,500 ops. Total ~3.3M ops. Fast.
    
    B_full = b
    # We need b[order_b[i], order_b[j]]
    # Let B_rot = b[np.ix_(order_b, order_b)]
    B_rot = b[np.ix_(order_b, order_b)]
    
    # For a given shift s, p[i] corresponds to index (pos_a[i] + s) % n in order_b.
    # So b[p[i], p[j]] = B_rot[(pos_a[i] + s) % n, (pos_a[j] + s) % n]
    
    # We can create a matrix A_pos where A_pos[u, v] = a[i, j] for pos_a[i]=u, pos_a[j]=v
    A_pos = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            A_pos[pos_in_a[i], pos_in_a[j]] = a[i, j]
            
    # Now cost(s) = sum_{u,v} A_pos[u, v] * B_rot[(u+s)%n, (v+s)%n]
    # This is a cyclic cross-correlation-like operation.
    # We can compute this for all s efficiently?
    # Or just loop s.
    
    for s in range(n):
        # Construct the permutation p
        # p[i] = order_b[(pos_in_a[i] + s) % n]
        indices = (pos_in_a + s) % n
        p = order_b[indices]
        
        # Calculate cost
        # cost = np.sum(a * b[np.ix_(p, p)])
        # To be faster, we can use the A_pos and B_rot formulation if we precomputed B_rot shifts?
        # B_rot_shifted = np.roll(B_rot, s, axis=0) # No, we need to roll both axes?
        # Actually, B_rot[u, v] is b[order_b[u], order_b[v]].
        # We want b[order_b[(u+s)%n], order_b[(v+s)%n]]
        # This is B_rot[(u+s)%n, (v+s)%n]
        # Let M_s[u, v] = B_rot[(u+s)%n, (v+s)%n]
        # M_s is B_rot rolled by s in both dimensions?
        # np.roll(B_rot, s, axis=0) gives B_rot[(u+s)%n, v]
        # Then np.roll(that, s, axis=1) gives B_rot[(u+s)%n, (v+s)%n]
        
        B_s = np.roll(np.roll(B_rot, s, axis=0), s, axis=1)
        cost = np.sum(A_pos * B_s)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    if best_p is None:
        # Fallback
        best_p = np.arange(n)
        best_cost = np.sum(a * b)

    # 3. Local Search Refinement
    # Use 2-swap local search to improve the cyclic solution.
    
    current_p = best_p
    current_cost = best_cost
    best_p_local = current_p.copy()
    best_cost_local = current_cost
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    # Precompute for delta calculation
    # delta = sum_{k != i,j} a[i,k]b[p[i],p[k]]... + ...
    # Standard 2-swap delta:
    # D = 2 * (sum_{k != i,j} (a[i,k]-a[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])) 
    #     + (a[i,i]-a[j,i]) * (b[p[i],p[i]] - b[p[j],p[i]]) 
    #     + (a[i,j]-a[j,j]) * (b[p[i],p[j]] - b[p[j],p[j]])
    #     + (a[j,i]-a[i,j]) * (b[p[j],p[i]] - b[p[i],p[i]]) # Symmetric?
    #     + (a[j,j]-a[i,j]) * (b[p[j],p[j]] - b[p[i],p[j]])
    # Actually, simpler formula:
    # Let L_i = p[i], L_j = p[j]
    # Delta = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[L_i, p[k]] - b[L_j, p[k]])
    #         + (a[i,i] - a[j,i]) * (b[L_i, L_i] - b[L_j, L_i])
    #         + (a[i,j] - a[j,j]) * (b[L_i, L_j] - b[L_j, L_j])
    #         + (a[j,i] - a[i,j]) * (b[L_j, L_i] - b[L_i, L_i])
    #         + (a[j,j] - a[i,j]) * (b[L_j, L_j] - b[L_i, L_j])
    # Note a[i,j] = a[j,i] and b[L_i, L_j] = b[L_j, L_i] if symmetric.
    # If symmetric:
    # Delta = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[L_i, p[k]] - b[L_j, p[k]])
    #         + (a[i,i] - a[j,i]) * (b[L_i, L_i] - b[L_j, L_i])
    #         + (a[i,j] - a[j,j]) * (b[L_i, L_j] - b[L_j, L_j])
    #         + (a[j,i] - a[i,j]) * (b[L_j, L_i] - b[L_i, L_i])
    #         + (a[j,j] - a[i,j]) * (b[L_j, L_j] - b[L_i, L_j])
    # Wait, the terms for k=i and k=j in the sum are excluded.
    # Let's use the vectorized form from the previous attempt which was correct in logic but slow.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # To speed up, we can sample swaps or limit the search?
        # With n=150, full scan is ~11k pairs.
        # Let's do a full scan.
        
        for i in range(n):
            Li = current_p[i]
            ai = a[i, :]
            for j in range(i + 1, n):
                Lj = current_p[j]
                aj = a[j, :]
                
                # Vector V = b[Lj, p] - b[Li, p]
                V = b[Lj, :] - b[Li, :] # Wait, p is a permutation of indices.
                # b[Lj, p[k]] is the distance from Lj to location p[k].
                # So V[k] = b[Lj, p[k]] - b[Li, p[k]]
                V = b[Lj, current_p] - b[Li, current_p]
                
                U = ai - aj
                
                dot_prod = np.dot(U, V)
                
                # Exclude i and j from the sum
                # term_i = (a[i,i] - a[j,i]) * (b[Lj, Li] - b[Li, Li])
                # term_j = (a[i,j] - a[j,j]) * (b[Lj, Lj] - b[Li, Lj])
                
                term_i = (a[i, i] - a[j, i]) * (b[Lj, Li] - b[Li, Li])
                term_j = (a[i, j] - a[j, j]) * (b[Lj, Lj] - b[Li, Lj])
                
                sum_excl = dot_prod - term_i - term_j
                
                # Self terms
                # The formula derived earlier:
                # delta = 2 * sum_excl + (a[i,i] - a[j,i]) * (b[Li, Li] - b[Lj, Li]) 
                #        + (a[i,j] - a[j,j]) * (b[Li, Lj] - b[Lj, Lj])
                #        + (a[j,i] - a[i,j]) * (b[Lj, Li] - b[Li, Li])
                #        + (a[j,j] - a[i,j]) * (b[Lj, Lj] - b[Li, Lj])
                
                # Let's re-verify the delta formula.
                # Cost = sum_{x,y} a[x,y] b[p[x], p[y]]
                # Swap i and j.
                # New p': p'[i]=Lj, p'[j]=Li, p'[k]=p[k] for k!=i,j.
                # Delta = C(p') - C(p)
                # Terms involving i or j change.
                # 1. x=i, y=i: a[i,i](b[Lj,Lj] - b[Li,Li])
                # 2. x=j, y=j: a[j,j](b[Li,Li] - b[Lj,Lj])
                # 3. x=i, y=j: a[i,j](b[Lj,Li] - b[Li,Lj])
                # 4. x=j, y=i: a[j,i](b[Li,Lj] - b[Lj,Li])
                # 5. x=i, y=k (k!=i,j): a[i,k](b[Lj, p[k]] - b[Li, p[k]])
                # 6. x=j, y=k (k!=i,j): a[j,k](b[Li, p[k]] - b[Lj, p[k]])
                # 7. x=k, y=i (k!=i,j): a[k,i](b[p[k], Lj] - b[p[k], Li])
                # 8. x=k, y=j (k!=i,j): a[k,j](b[p[k], Li] - b[p[k], Lj])
                
                # If A and B are symmetric:
                # a[i,j]=a[j,i], b[Li,Lj]=b[Lj,Li]
                # Term 3 + 4 = a[i,j](b[Lj,Li] - b[Li,Lj]) + a[i,j](b[Li,Lj] - b[Lj,Li]) = 0
                # Term 5 + 7 = sum_{k!=i,j} (a[i,k] + a[k,i]) * (b[Lj, p[k]] - b[Li, p[k]])
                #            = 2 * sum_{k!=i,j} a[i,k] * (b[Lj, p[k]] - b[Li, p[k]])
                # Term 6 + 8 = sum_{k!=i,j} (a[j,k] + a[k,j]) * (b[Li, p[k]] - b[Lj, p[k]])
                #            = 2 * sum_{k!=i,j} a[j,k] * (b[Li, p[k]] - b[Lj, p[k]])
                #            = -2 * sum_{k!=i,j} a[j,k] * (b[Lj, p[k]] - b[Li, p[k]])
                
                # So Sum(5..8) = 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[Lj, p[k]] - b[Li, p[k]])
                
                # Term 1 + 2 = (a[i,i] - a[j,j]) * (b[Lj,Lj] - b[Li,Li])
                
                # So Delta = (a[i,i] - a[j,j]) * (b[Lj,Lj] - b[Li,Li]) 
                #           + 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[Lj, p[k]] - b[Li, p[k]])
                
                # Let's compute this.
                # V[k] = b[Lj, p[k]] - b[Li, p[k]]
                # U[k] = a[i,k] - a[j,k]
                # Sum_all = dot(U, V)
                # Sum_excl = Sum_all - U[i]*V[i] - U[j]*V[j]
                
                # U[i] = a[i,i] - a[j,i]
                # V[i] = b[Lj, Li] - b[Li, Li]
                # U[j] = a[i,j] - a[j,j]
                # V[j] = b[Lj, Lj] - b[Li, Lj]
                
                sum_excl = dot_prod - U[i]*V[i] - U[j]*V[j]
                
                self_term = (a[i,i] - a[j,j]) * (b[Lj,Lj] - b[Li,Li])
                
                delta = 2 * sum_excl + self_term
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_i != -1:
            current_p[best_i], current_p[best_j] = current_p[best_j], current_p[best_i]
            current_cost += best_delta
            improved = True
            
            if current_cost < best_cost_local:
                best_cost_local = current_cost
                best_p_local = current_p.copy()
                
    return best_p_local.tolist()
