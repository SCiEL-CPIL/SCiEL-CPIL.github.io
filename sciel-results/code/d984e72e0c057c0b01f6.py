import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Faster verification using vectorized operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check all (i, j, k) tuples
        # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        # Sum over t: U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # We try to eliminate terms in a specific order to maximize success
        # Terms from Strassen blocks are more likely to be reducible in combination
        # But let's first try simple elimination
        
        for _ in range(5):  # Multiple passes
            if time_left() < 5:
                break
            progress = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
                # Check if term is redundant
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify(U_new, V_new, W_new, rank):
                    # Remove the column
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once by adding/subtracting them."""
        if time_left() < 10:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 10:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 10:
                    return U_list, V_list, W_list, rank
                
                # Try replacing term idx2 with a linear combination
                # Or simply check if we can merge them
                # This is computationally expensive, so we limit attempts
                
                # Simple check: if we zero out both, does it still work?
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][idx1] = 0
                    U_new[i][idx2] = 0
                for j in range(m*p):
                    V_new[j][idx1] = 0
                    V_new[j][idx2] = 0
                for k in range(p*n):
                    W_new[k][idx1] = 0
                    W_new[k][idx2] = 0
                    
                if verify(U_new, V_new, W_new, rank):
                    # Remove both columns
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 2
                    # Restart loop to find more
                    return try_pairwise_elimination(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    def build_split_middle():
        """
        Attempt the 'split-middle' decomposition.
        The idea: C = A B.
        Let A = [A_0, A_1] where A_0 is 2x5, A_1 is 2x5? No.
        Let's partition the inner dimension m=5 into m1=2, m2=3.
        A = [A_L, A_R] where A_L is 2x2, A_R is 2x3.
        B = [B_L; B_R] where B_L is 2x6, B_R is 3x6.
        C = A_L B_L + A_R B_R.
        
        Rank(A_L B_L) for 2x2 * 2x6.
        We can use Strassen for the 2x2 part of B_L?
        B_L is 2x6. Let B_L = [B_L1, B_L2] where both are 2x3?
        Or use the fact that 2x2 * 2x6 can be done in 7 * 3 = 21? No.
        For 2x2 * 2x6, we can view it as 3 independent 2x2 * 2x2 multiplications.
        Rank = 7 * 3 = 21.
        
        Rank(A_R B_R) for 2x3 * 3x6.
        This is 2x3 * 3x6.
        We can decompose 2x3 * 3x6.
        Standard rank is 2*3*6 = 36.
        Can we do better?
        2x3 * 3x6.
        Let's try to combine.
        
        Total rank = 21 + 36 = 57.
        This is worse than 54 from Strassen blocks.
        
        What if we partition m=5 into 1 and 4?
        A_L (2x1) * B_L (1x6) -> Rank 2*1*6 = 12? No, rank 2?
        A_L is 2x1 vector, B_L is 1x6 vector.
        C_L = A_L B_L is rank 1 matrix.
        Decomposition:
        For i in 0..1:
          For k in 0..5:
            Term: A_L[i,0] * B_L[0,k].
            This is 2*6 = 12 terms.
            But we can factor:
            U[i, t] = A_L[i,0]
            V[0, t] = B_L[0,k] ... wait.
            The tensor is T[i, j, k] = sum_j A[i,j] B[j,k].
            For j=0 (the 1x1 block):
            T[i, 0, k] = A[i,0] B[0,k].
            This is a rank 1 tensor in (i, k) for fixed j? No, it's a slice.
            The full tensor for j=0 is A[:,0] B[0,:].
            This is an outer product of a 2-vector and a 6-vector.
            Rank 1 in the tensor sense?
            T[i, 0, k] = u_i v_k.
            This can be represented by 1 term?
            U[i, 0, t] = u_i, V[0, 0, t] = v_k? No.
            V index is j*p+k. Here j=0. So V[0*p+k, t].
            If we use 1 term:
            U[i, 0, t] = A[i,0]
            V[0, 0, t] = 1? No, V depends on k.
            V[j*p+k, t].
            If V[0*p+k, t] = B[0,k], then V is not constant across k for a single term unless B[0,k] is constant.
            So we need 6 terms for the j=0 slice?
            Actually, the term is U[i, 0, t] * V[0, k, t] * W[k, i, t].
            If we set t=0..5:
            U[i, 0, t] = A[i, 0]
            V[0, t, t] = B[0, t] (diagonal in V for j=0)
            W[t, i, t] = 1
            Then sum_t A[i,0] B[0,t] 1 = A[i,0] B[0,t] for specific k=t.
            So yes, 6 terms for the 1x1 block.
            
            A_R (2x4) * B_R (4x6).
            Rank for 2x4 * 4x6.
            Standard 2*4*6 = 48.
            Total = 6 + 48 = 54.
            Same as before.
        """
        pass

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 20:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
            
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
