def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 9
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook as fallback
    rank0 = n * m * p  # 54
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def to_list(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Best solution so far
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank0
    
    # Try block decomposition: split 3x9 into three 3x3 blocks
    # For each 3x3 block, use 2x3x3 Strassen (rank 11)
    # Total rank = 3 * 11 = 33
    
    # First, I need a 2x3x3 Strassen decomposition
    # Let me construct one numerically
    
    def find_2x3x3_strassen():
        """Find a rank-11 decomposition for 2x3x3 matrix multiplication."""
        # This is a known result. Let me try to construct it.
        # A is 2x3, B is 3x3, C = A*B is 2x3
        # We need U(6,r), V(9,r), W(6,r) such that
        # sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1 for all i,j,k
        
        # Try rank 11
        # Use random initialization and optimization
        np.random.seed(42)
        
        for r in range(11, 20):
            for attempt in range(10):
                U = np.random.randn(6, r) * 0.1
                V = np.random.randn(9, r) * 0.1
                W = np.random.randn(6, r) * 0.1
                
                # Use alternating minimization
                for it in range(200):
                    # Update U
                    for i in range(2):
                        for j in range(3):
                            # We need sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1 for all k
                            # This is a linear system in U[i*3+j,:]
                            # Let A_mat[k,c] = V[j*3+k,c]*W[k*2+i,c]
                            # Then sum_c U[i*3+j,c]*A_mat[k,c] = 1 for all k
                            # This is: A_mat @ U[i*3+j,:] = ones(3)
                            A_mat = np.outer(V[j*3:(j+1)*3], W[i:(i+1)*2])  # shape (3, r) but wrong
                            # Actually, for fixed i,j, we need for each k:
                            # sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1
                            # Let's form a matrix M where M[k,c] = V[j*3+k,c]*W[k*2+i,c]
                            M = np.zeros((3, r))
                            for k in range(3):
                                M[k,:] = V[j*3+k,:] * W[k*2+i,:]
                            # Solve M @ u = ones(3) for u = U[i*3+j,:]
                            try:
                                U[i*3+j,:] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                            except:
                                pass
                    
                    # Similarly for V and W
                    for j in range(3):
                        for k in range(3):
                            M = np.zeros((3, r))  # for i in range(3)? No, i is 0,1
                            # For fixed j,k, we need for each i in 0,1:
                            # sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1
                            # M[i,c] = U[i*3+j,c]*W[k*2+i,c]
                            M = np.zeros((2, r))
                            for i in range(2):
                                M[i,:] = U[i*3+j,:] * W[k*2+i,:]
                            try:
                                V[j*3+k,:] = np.linalg.lstsq(M, np.ones(2), rcond=None)[0]
                            except:
                                pass
                    
                    for k in range(3):
                        for i in range(2):
                            # For fixed k,i, we need for each j in 0,1,2:
                            # sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1
                            # M[j,c] = U[i*3+j,c]*V[j*3+k,c]
                            M = np.zeros((3, r))
                            for j in range(3):
                                M[j,:] = U[i*3+j,:] * V[j*3+k,:]
                            try:
                                W[k*2+i,:] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                            except:
                                pass
                
                # Check if valid
                if verify(U, V, W, tol=1e-6):
                    return U, V, W
        
        return None
    
    # Try to get 2x3x3 Strassen
    if time.time() < deadline - 60:
        s233 = find_2x3x3_strassen()
        if s233 is not None:
            U233, V233, W233 = s233
            r233 = U233.shape[1]
            
            # Build 2x3x9 by using 3 copies
            total_rank = 3 * r233
            U9 = np.zeros((6, total_rank))
            V9 = np.zeros((27, total_rank))
            W9 = np.zeros((18, total_rank))
            
            for block in range(3):
                offset = block * r233
                U9[:, offset:offset+r233] = U233
                # V: for block b, columns 3b to 3b+2 of B
                V9[block*9:(block+1)*9, offset:offset+r233] = V233
                # W: same for all blocks since W depends on k*n+i, and k is in 0..8
                # For block b, k is in 3b..3b+2, so W index is k*2+i
                for local_k in range(3):
                    global_k = block * 3 + local_k
                    W9[global_k*2:(global_k+1)*2, offset:offset+r233] = W233[local_k*2:(local_k+1)*2]
            
            if verify(U9, V9, W9):
                best_U, best_V, best_W = U9.copy(), V9.copy(), W9.copy()
                best_rank = total_rank
    
    return to_list(best_U, best_V, best_W)
