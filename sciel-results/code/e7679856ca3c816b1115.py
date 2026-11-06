import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    def count_k4_incremental(A, u, v, adding):
        """
        Returns the change in the number of K4s if we flip edge (u,v).
        If adding=True, we are adding edge (u,v) to the graph.
        If adding=False, we are removing edge (u,v) from the graph.
        """
        # Common neighbors in current graph A
        # N(u) = {w | A[u,w] == 1}
        # We need the number of edges in the subgraph induced by common neighbors.
        # This is the number of triangles in the neighborhood intersection?
        # No, K4 is u,v,w,x. u-v is the edge. w,x must be common neighbors of u and v.
        # And w-x must be an edge.
        # So we count edges in G[N(u) intersection N(v)].
        
        # Vectorized:
        # common = np.logical_and(A[u], A[v])
        # common_indices = np.nonzero(common)[0]
        # subgraph = A[np.ix_(common_indices, common_indices)]
        # num_edges = np.sum(subgraph) // 2
        
        # This is O(n^2) per flip. For n=35, 1225 ops. Very fast.
        common = np.logical_and(A[u], A[v])
        common_indices = np.nonzero(common)[0]
        if len(common_indices) < 2:
            return 0
        subgraph = A[np.ix_(common_indices, common_indices)]
        num_edges = np.sum(subgraph) // 2
        
        if adding:
            return num_edges
        else:
            return -num_edges

    def count_k6_incremental(A, u, v, adding):
        """
        Returns the change in the number of Independent Sets of size 6 (I6)
        if we flip edge (u,v).
        I6 in G is K6 in G^c.
        Let A_c be complement.
        If we flip (u,v) in G from 0->1 (adding edge), in G^c we remove edge.
        If we flip (u,v) in G from 1->0 (removing edge), in G^c we add edge.
        
        Here 'adding' refers to adding to G.
        If adding to G, we remove from G^c.
        If removing from G, we add to G^c.
        
        We count K6s in G^c containing (u,v).
        This is equivalent to counting K4s in the subgraph of G^c induced by common neighbors of u and v in G^c.
        """
        # A_c[u] = 1 - A[u] (excluding self, but A[u,u]=0 so it's fine)
        # Common neighbors in G^c:
        # w is common neighbor in G^c if A[u,w]=0 and A[v,w]=0 (and w!=u,v)
        
        # Note: A[u,u] is 0, so A_c[u,u] is 1. We must exclude u and v from common neighbors.
        # A_c = 1 - A (with diagonal 0? No, standard adjacency has 0 diagonal.
        # If we use 1-A, diagonal becomes 1.
        # Let's just compute common non-neighbors directly.
        
        # Non-neighbors of u: ~A[u] but exclude u itself.
        # Since A is symmetric and 0 diagonal:
        # non_n_u = np.logical_not(A[u])
        # non_n_u[u] = False
        # non_n_v = np.logical_not(A[v])
        # non_n_v[v] = False
        
        # Common non-neighbors:
        common_non = np.logical_and(np.logical_not(A[u]), np.logical_not(A[v]))
        common_non[u] = False
        common_non[v] = False
        
        common_indices = np.nonzero(common_non)[0]
        if len(common_indices) < 4:
            return 0
            
        # We need to count K4s in the subgraph of G^c induced by common_indices.
        # In G^c, an edge exists if A is 0.
        # So we look at A_sub = A[np.ix_(common_indices, common_indices)]
        # Edges in G^c correspond to 0s in A_sub.
        # Number of K4s in G^c[C] is number of 4-cliques in the graph where edges are 0s in A_sub.
        
        # This is counting independent sets of size 4 in G[C].
        # Let's count 4-cliques in the complement of A_sub.
        
        # For small n, we can just iterate.
        # Or use the fact that K4 in G^c[C] is I4 in G[C].
        
        # Let's count I4 in G[C].
        # I4: w1, w2, w3, w4 in C such that no edges between them in G.
        
        # Efficient way:
        # Iterate pairs?
        # Or just use the same logic as count_k4 but on the complement matrix.
        
        A_c_sub = 1 - A[np.ix_(common_indices, common_indices)]
        # Diagonal of A_c_sub is 1. We should set it to 0 for clique counting?
        # A clique in A_c_sub of size 4 requires 6 edges.
        # If diagonal is 1, it doesn't affect the count of edges between distinct vertices.
        # np.sum(A_c_sub) // 2 counts all edges including self-loops if diagonal is 1.
        # But we want cliques, not just edges.
        
        # Wait, count_k4_incremental counted edges in the subgraph to determine K4s containing the root edge.
        # Here we need to count K4s in G^c[C].
        # A K4 in G^c[C] consists of 4 vertices.
        # This is equivalent to counting the number of 4x4 submatrices of A_c_sub that are all 1s (off-diagonal).
        
        # Since n is small (<= 35), and common_indices is likely small,
        # we can do a brute force count of I4 in G[C].
        
        # Actually, we can reuse the logic:
        # The number of I6s in G containing (u,v) when we ADD (u,v) to G (remove from G^c)
        # is the number of I4s in G[C_non].
        
        # Let's count I4 in G[C].
        # I4 in G[C] is K4 in G^c[C].
        
        # Let's just compute the number of K4s in the graph H where H[i,j] = 1 if A[c_i, c_j] == 0.
        
        # We can use a simple loop for counting K4s in a small graph.
        # Or, notice that we just need the change.
        
        # Let's implement a helper to count K4s in a small adjacency matrix (0/1, symmetric, 0 diag).
        def count_k4_in_matrix(M):
            n_m = M.shape[0]
            count = 0
            for i in range(n_m):
                for j in range(i+1, n_m):
                    if M[i,j]:
                        # Common neighbors of i, j in M
                        # This is getting slow if we do this for every flip.
                        # But common_indices is small.
                        pass
            # Let's use the incremental formula again?
            # No, we need the total count of K4s in the subgraph.
            # For n<=35, total K4s is small.
            # Let's just do O(n^4) on the subgraph?
            # Subgraph size m. m^4. If m=10, 10k. If m=20, 160k.
            # This might be too slow if done every flip.
            
            # Alternative:
            # We only need to know if the count is > 0 or the exact change.
            # The change is +/- (number of I4s in G[C_non]).
            
            # Let's optimize I4 counting.
            # I4 in G[C] is a set of 4 vertices with no edges.
            # We can iterate over pairs of non-edges?
            
            # Given the time constraint, and n=35, let's assume the subgraph is small.
            # If it's too slow, we fall back to random restarts.
            
            # Let's use a simple O(m^4) count for the subgraph.
            # m is typically < 10 for dense graphs, or larger for sparse.
            # For n=35, random graph p=0.5, m ~ 15. 15^4 = 50k.
            # 50k ops per flip * 1000 flips = 50M ops. Acceptable in 120s?
            # Python is slow. 50M simple ops might take 5-10s.
            # We need to be careful.
            
            # Let's try to count K4s in M (which is G^c[C]).
            # M is the adjacency of the complement subgraph.
            
            # We can use the formula:
            # K4s = sum over edges (i,j) of (edges in M[N(i) inter N(j)])
            # This is O(m^3) roughly.
            
            k4_count = 0
            # Precompute neighbor lists for M?
            # M is small.
            
            for i in range(n_m):
                for j in range(i+1, n_m):
                    if M[i,j]:
                        # Find common neighbors of i and j in M
                        # common = M[i] & M[j]
                        # count edges in M[common, common]
                        common = np.logical_and(M[i], M[j])
                        common_idx = np.nonzero(common)[0]
                        if len(common_idx) >= 2:
                            sub = M[np.ix_(common_idx, common_idx)]
                            k4_count += np.sum(sub) // 2
            return k4_count

        # So, count of I6s affected:
        # If adding edge to G (removing from G^c), we lose K6s in G^c.
        # Number of K6s in G^c containing (u,v) is K4s in G^c[C_non].
        # G^c[C_non] is the graph where edges are 0 in A[C_non].
        # So we count K4s in the matrix M where M[i,j] = 1 if A[c_i, c_j] == 0.
        
        A_sub = A[np.ix_(common_indices, common_indices)]
        M = 1 - A_sub
        np.fill_diagonal(M, 0) # Ensure no self loops
        
        num_k4_in_comp = count_k4_in_matrix(M)
        
        if adding:
            # We are adding edge to G, so removing from G^c.
            # We lose these K6s.
            return -num_k4_in_comp
        else:
            # We are removing edge from G, so adding to G^c.
            # We gain these K6s.
            return num_k4_in_comp

    def total_violations(A):
        # Count total K4s and I6s
        n = A.shape[0]
        # K4s
        k4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j]:
                    # Common neighbors
                    common = np.logical_and(A[i], A[j])
                    common_idx = np.nonzero(common)[0]
                    if len(common_idx) >= 2:
                        sub = A[np.ix_(common_idx, common_idx)]
                        k4_count += np.sum(sub) // 2
                        
        # I6s
        # I6 in G is K6 in G^c
        # Count K6s in G^c
        # This is expensive. O(n^6)?
        # We can count I6s directly?
        # I6: 6 vertices, no edges.
        # We can iterate over 6-tuples? 35 choose 6 is 1.6M.
        # For each, check 15 edges. 24M ops.
        # This is doable for a full check, but not for every flip.
        
        # For the main loop, we rely on incremental updates.
        # For initialization and final check, we do full count.
        
        i6_count = 0
        # Use itertools.combinations for clarity, might be slow.
        # Let's use a recursive or iterative method.
        # Or just use the K6 count in G^c.
        # K6 in G^c:
        # Iterate edges in G^c?
        # G^c has ~ n^2/2 edges.
        # For each edge (u,v) in G^c, count K4s in G^c[N(u) inter N(v)].
        # This is similar to K4 counting.
        
        A_c = 1 - A
        np.fill_diagonal(A_c, 0)
        
        for i in range(n):
            for j in range(i+1, n):
                if A_c[i,j]:
                    common = np.logical_and(A_c[i], A_c[j])
                    common_idx = np.nonzero(common)[0]
                    if len(common_idx) >= 4:
                        # Count K4s in A_c[common_idx, common_idx]
                        sub = A_c[np.ix_(common_idx, common_idx)]
                        # Count K4s in sub
                        # Use the same helper
                        k4_in_sub = count_k4_in_matrix(sub)
                        i6_count += k4_in_sub
                            
        return k4_count + i6_count

    def solve(n):
        # Initialize random graph
        # p = 0.5
        A = np.random.randint(0, 2, (n, n)).astype(float)
        np.fill_diagonal(A, 0)
        # Symmetrize
        A = (A + A.T) / 2
        A = (A > 0.5).astype(float)
        
        current_violations = total_violations(A)
        best_A = A.copy()
        best_violations = current_violations
        
        max_steps = 5000
        no_improve_steps = 0
        threshold = 500
        
        for step in range(max_steps):
            if time_left() < 5.0:
                break
                
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            # Flip
            adding = A[u,v] == 0
            
            # Calculate delta violations
            delta_k4 = count_k4_incremental(A, u, v, adding)
            delta_i6 = count_k6_incremental(A, u, v, adding)
            delta_total = delta_k4 + delta_i6
            
            # Accept if improvement, or random restart
            if delta_total < 0 or (random.random() < 0.1 and no_improve_steps > 100):
                # Flip
                A[u,v] = 1 - A[u,v]
                A[v,u] = 1 - A[v,u]
                current_violations += delta_total
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_A = A.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    
                if best_violations == 0:
                    return best_A
            else:
                no_improve_steps += 1
                
            # Restart if stuck
            if no_improve_steps > threshold:
                # Random restart
                A = np.random.randint(0, 2, (n, n)).astype(float)
                np.fill_diagonal(A, 0)
                A = (A + A.T) / 2
                A = (A > 0.5).astype(float)
                current_violations = total_violations(A)
                no_improve_steps = 0
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_A = A.copy()
                    
        return best_A

    # Try to find a valid graph for n=35
    best_graph = None
    best_n = 0
    
    # Start with n=35
    for target_n in [35, 34, 33]:
        if time_left() < 20.0:
            break
            
        # Try multiple restarts for this n
        found = False
        for restart in range(5):
            if time_left() < 10.0:
                break
            A_sol = solve(target_n)
            # Verify
            viol = total_violations(A_sol)
            if viol == 0:
                best_graph = A_sol
                best_n = target_n
                found = True
                break
                
        if found:
            break

    if best_graph is None:
        # Fallback to seed or best found in previous runs?
        # We didn't keep previous runs.
        # Let's return a valid small graph, e.g., Paley 17
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': 17, 'edges': edges}
    else:
        # Convert adjacency matrix to edge list
        edges = []
        for i in range(best_n):
            for j in range(i+1, best_n):
                if best_graph[i,j] == 1:
                    edges.append([i, j])
        return {'n': best_n, 'edges': edges}

