import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 8
    
    # Build the expected tensor
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_exp[a_idx, b_idx, c_idx] = 1
    
    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_exp)
    
    # Rank-44 solution: 4 blocks of rank-11 (2x3x2)
    # Each block: 4 (for j=0 part) + 7 (Strassen for j=1,2 part)
    def build_rank44():
        U = np.zeros((n*m, 44), dtype=np.int64)
        V = np.zeros((m*p, 44), dtype=np.int64)
        W = np.zeros((p*n, 44), dtype=np.int64)
        col_offset = 0
        
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            
            # Part 1: A[:,0] * B[0,k0:k1] (2x1 * 1x2) - rank 4
            # t0 = A[0,0]*B[0,k0] -> C[0,k0]
            U[0, col_offset] = 1; V[k0, col_offset] = 1; W[k0*n+0, col_offset] = 1
            col_offset += 1
            # t1 = A[0,0]*B[0,k1] -> C[0,k1]
            U[0, col_offset] = 1; V[k1, col_offset] = 1; W[k1*n+0, col_offset] = 1
            col_offset += 1
            # t2 = A[1,0]*B[0,k0] -> C[1,k0]
            U[3, col_offset] = 1; V[k0, col_offset] = 1; W[k0*n+1, col_offset] = 1
            col_offset += 1
            # t3 = A[1,0]*B[0,k1] -> C[1,k1]
            U[3, col_offset] = 1; V[k1, col_offset] = 1; W[k1*n+1, col_offset] = 1
            col_offset += 1
            
            # Part 2: A[:,1:3] * B[1:3,k0:k1] (2x2 * 2x2) - Strassen rank 7
            # A_sub: [[A[0,1], A[0,2]], [A[1,1], A[1,2]]] = indices [1,2,4,5]
            # B_sub: [[B[1,k0], B[1,k0+1]], [B[2,k0], B[2,k0+1]]] = indices [p+k0, p+k1, 2p+k0, 2p+k1]
            
            # M1 = (A[0,1]+A[1,2])*(B[1,k0]+B[2,k1])
            U[1, col_offset] = 1; U[5, col_offset] = 1
            V[p+k0, col_offset] = 1; V[2*p+k1, col_offset] = 1
            W[k0*n+0, col_offset] = 1; W[k1*n+1, col_offset] = 1
            col_offset += 1
            
            # M2 = (A[1,1]+A[1,2])*B[1,k0]
            U[4, col_offset] = 1; U[5, col_offset] = 1
            V[p+k0, col_offset] = 1
            W[k0*n+1, col_offset] = 1; W[k1*n+1, col_offset] = -1
            col_offset += 1
            
            # M3 = A[0,1]*(B[1,k1]-B[2,k1])
            U[1, col_offset] = 1
            V[p+k1, col_offset] = 1; V[2*p+k1, col_offset] = -1
            W[k1*n+0, col_offset] = 1; W[k1*n+1, col_offset] = 1
            col_offset += 1
            
            # M4 = A[1,2]*(B[2,k0]-B[1,k0])
            U[5, col_offset] = 1
            V[2*p+k0, col_offset] = 1; V[p+k0, col_offset] = -1
            W[k0*n+0, col_offset] = 1; W[k0*n+1, col_offset] = 1
            col_offset += 1
            
            # M5 = (A[0,1]+A[0,2])*B[2,k1]
            U[1, col_offset] = 1; U[2, col_offset] = 1
            V[2*p+k1, col_offset] = 1
            W[k0*n+0, col_offset] = -1; W[k1*n+0, col_offset] = 1
            col_offset += 1
            
            # M6 = (A[1,1]-A[0,1])*(B[1,k0]+B[1,k1])
            U[4, col_offset] = 1; U[1, col_offset] = -1
            V[p+k0, col_offset] = 1; V[p+k1, col_offset] = 1
            W[k1*n+1, col_offset] = 1
            col_offset += 1
            
            # M7 = (A[0,2]-A[1,2])*(B[2,k0]+B[2,k1])
            U[2, col_offset] = 1; U[5, col_offset] = -1
            V[2*p+k0, col_offset] = 1; V[2*p+k1, col_offset] = 1
            W[k0*n+0, col_offset] = 1
            col_offset += 1
        
        return U, V, W
    
    U44, V44, W44 = build_rank44()
    
    if verify(U44, V44, W44):
        best_U, best_V, best_W = U44, V44, W44
        best_rank = 44
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to improve: attempt rank-40 (10 per block for 2x3x2)
    # For a single 2x3x2 block:
    # A entries: a00=A[0,0], a01=A[0,1], a02=A[0,2], a10=A[1,0], a11=A[1,1], a12=A[1,2]
    # B entries: b00=B[0,k0], b10=B[1,k0], b20=B[2,k0], b01=B[0,k1], b11=B[1,k1], b21=B[2,k1]
    # C entries: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
    #
    # C[0,k0] = a00*b00 + a01*b10 + a02*b20
    # C[0,k1] = a00*b01 + a01*b11 + a02*b21
    # C[1,k0] = a10*b00 + a11*b10 + a12*b20
    # C[1,k1] = a10*b01 + a11*b11 + a12*b21
    #
    # Try 10 products:
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a01+a12)*(b10+b01)  -- wait, need to be careful
    # Let me try a different approach
    
    # Actually, let me try to find a rank-9 or rank-8 decomposition for 2x3x2
    # using a more systematic approach
    
    def try_find_lower_rank_2x3x2(max_rank=10, time_limit=30):
        """Try to find a rank-r decomposition for 2x3x2 matrix multiplication."""
        start = time.time()
        
        # Build the 2x3x2 tensor
        T_232 = np.zeros((6, 6, 4), dtype=np.int64)
        # A indices: 0=a00, 1=a01, 2=a02, 3=a10, 4=a11, 5=a12
        # B indices: 0=b00, 1=b10, 2=b20, 3=b01, 4=b11, 5=b21
        # C indices: 0=C[0,k0], 1=C[0,k1], 2=C[1,k0], 3=C[1,k1]
        
        # C[0,k0] = a00*b00 + a01*b10 + a02*b20
        T_232[0, 0, 0] = 1  # a00*b00
        T_232[1, 1, 0] = 1  # a01*b10
        T_232[2, 2, 0] = 1  # a02*b20
        
        # C[0,k1] = a00*b01 + a01*b11 + a02*b21
        T_232[0, 3, 1] = 1  # a00*b01
        T_232[1, 4, 1] = 1  # a01*b11
        T_232[2, 5, 1] = 1  # a02*b21
        
        # C[1,k0] = a10*b00 + a11*b10 + a12*b20
        T_232[3, 0, 2] = 1  # a10*b00
        T_232[4, 1, 2] = 1  # a11*b10
        T_232[5, 2, 2] = 1  # a12*b20
        
        # C[1,k1] = a10*b01 + a11*b11 + a12*b21
        T_232[3, 3, 3] = 1  # a10*b01
        T_232[4, 4, 3] = 1  # a11*b11
        T_232[5, 5, 3] = 1  # a12*b21
        
        def verify_232(U, V, W, rank):
            T_check = np.einsum('ad,bd,cd->abc', U, V, W)
            return np.array_equal(T_check, T_232)
        
        # Try rank from max_rank down to 7
        for rank in range(max_rank, 6, -1):
            if time.time() - start > time_limit:
                break
            
            # Try several random initializations
            for attempt in range(20):
                if time.time() - start > time_limit:
                    break
                
                # Random initialization with small integer values
                U = np.random.randint(-2, 3, (6, rank))
                V = np.random.randint(-2, 3, (6, rank))
                W = np.random.randint(-2, 3, (4, rank))
                
                # Try to refine using a simple iterative approach
                # This is a heuristic, not guaranteed to work
                for iteration in range(50):
                    if time.time() - start > time_limit:
                        break
                    
                    # Check if we're close
                    T_check = np.einsum('ad,bd,cd->abc', U, V, W)
                    if np.array_equal(T_check, T_232):
                        return U, V, W, rank
                    
                    # Simple gradient-like update (not true gradient, just heuristic)
                    # Try to adjust entries to reduce error
                    error = T_check - T_232
                    if np.all(error == 0):
                        return U, V, W, rank
                    
                    # Perturb randomly
                    idx_a = np.random.randint(0, 6)
                    idx_b = np.random.randint(0, 6)
                    idx_c = np.random.randint(0, 4)
                    idx_r = np.random.randint(0, rank)
                    
                    # Try adjusting U[idx_a, idx_r]
                    old_val = U[idx_a, idx_r]
                    for delta in [-1, 1, -2, 2]:
                        U[idx_a, idx_r] = old_val + delta
                        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
                        if np.array_equal(T_check, T_232):
                            return U, V, W, rank
                        # Check if error decreased
                        err_new = np.sum((T_check - T_232)**2)
                        err_old = np.sum(error**2)
                        if err_new < err_old:
                            error = T_check - T_232
                            break
                    else:
                        U[idx_a, idx_r] = old_val  # Revert if no improvement
                
                # Check final result
                T_check = np.einsum('ad,bd,cd->abc', U, V, W)
                if np.array_equal(T_check, T_232):
                    return U, V, W, rank
        
        return None
    
    # Try to find a lower rank for 2x3x2
    result_232 = try_find_lower_rank_2x3x2(max_rank=10, time_limit=20)
    
    if result_232 is not None:
        U_232, V_232, W_232, rank_232 = result_232
        total_rank = rank_232 * 4
        
        # Build the full rank solution
        U_full = np.zeros((n*m, total_rank), dtype=np.int64)
        V_full = np.zeros((m*p, total_rank), dtype=np.int64)
        W_full = np.zeros((p*n, total_rank), dtype=np.int64)
        
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            col_offset = block * rank_232
            
            # Map A indices: a00->0, a01->1, a02->2, a10->3, a11->4, a12->5
            # Map B indices: b00->k0, b10->p+k0, b20->2p+k0, b01->k1, b11->p+k1, b21->2p+k1
            # Map C indices: C[0,k0]->k0*n+0, C[0,k1]->k1*n+0, C[1,k0]->k0*n+1, C[1,k1]->k1*n+1
            
            # Copy U
            U_full[0:6, col_offset:col_offset+rank_232] = U_232
            
            # Copy V with remapped indices
            b_map = {0: k0, 1: p+k0, 2: 2*p+k0, 3: k1, 4: p+k1, 5: 2*p+k1}
            for b_idx in range(6):
                V_full[b_map[b_idx], col_offset:col_offset+rank_232] = V_232[b_idx]
            
            # Copy W with remapped indices
            c_map = {0: k0*n+0, 1: k1*n+0, 2: k0*n+1, 3: k1*n+1}
            for c_idx in range(4):
                W_full[c_map[c_idx], col_offset:col_offset+rank_232] = W_232[c_idx]
        
        if verify(U_full, V_full, W_full):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # If we get here, return the rank-44 solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
