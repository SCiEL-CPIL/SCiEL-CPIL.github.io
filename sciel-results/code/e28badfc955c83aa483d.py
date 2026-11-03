import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Build 4x4x4 Strassen (Rank 49)
    U_444, V_444, W_444 = _build_444_strassen()
    
    # 2. Build 4x4x2 Kronecker (Rank 28)
    U_442, V_442, W_442 = _build_442_kronecker()
    
    # 3. Combine
    # The 4x4x4 part handles columns 0-3
    # The 4x4x2 part handles columns 4-5
    # We need to shift V and W for the second part.
    
    rank_444 = U_444.shape[1]
    rank_442 = U_442.shape[1]
    
    U = np.zeros((16, rank_444 + rank_442), dtype=int)
    V = np.zeros((24, rank_444 + rank_442), dtype=int)
    W = np.zeros((24, rank_444 + rank_442), dtype=int)
    
    # Part 1: 4x4x4
    U[:, :rank_444] = U_444
    V[:, :rank_444] = V_444
    W[:, :rank_444] = W_444
    
    # Part 2: 4x4x2 (shifted to columns 4,5)
    # V for 4x4x2 has shape (4*2, rank_442) = (8, rank_442)
    # We need to map these to the full V of shape (24, ...)
    # The 4x4x2 V indices correspond to j (0..3) and k (4,5).
    # In the full V, index is j*6 + k.
    # So for each term r in 442:
    #   V_full[:, rank_444 + r] = 0
    #   for j in 0..3:
    #     for k in 0..1:
    #       V_full[j*6 + (4+k), rank_444 + r] = V_442[j*2+k, r]
    
    for r in range(rank_442):
        for j in range(4):
            for k in range(2):
                V[j*6 + 4 + k, rank_444 + r] = V_442[j*2 + k, r]
                
    # W for 4x4x2 has shape (2*4, rank_442) = (8, rank_442)
    # W indices correspond to k (4,5) and i (0..3).
    # In the full W, index is k*4 + i.
    # So for each term r in 442:
    #   W_full[:, rank_444 + r] = 0
    #   for k in 0..1:
    #     for i in 0..3:
    #       W_full[(4+k)*4 + i, rank_444 + r] = W_442[k*4+i, r]
       
    for r in range(rank_442):
        for k in range(2):
            for i in range(4):
                W[(4+k)*4 + i, rank_444 + r] = W_442[k*4 + i, r]
                
    # U is just U_442 for the second part, same rows.
    U[:, rank_444:] = U_442
    
    # 4. Verify
    if not _verify(U, V, W, n, m, p):
         # Fallback to schoolbook if verification fails
         U, V, W = _schoolbook(n, m, p)
         return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
         
    # 5. Try to reduce rank
    # Focus on the 442 part which is likely reducible from 28 to ~24
    # We can try to reduce the whole tensor, but reducing the 442 block is safer/faster.
    
    if time.time() < deadline - 10:
        U, V, W = _reduce_rank(U, V, W, n, m, p, deadline)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_444_strassen():
    # Rank 49 decomposition for 4x4x4
    # This is the Kronecker product of two 2x2x2 Strassen (7x7=49)
    U2, V2, W2 = _build_222_strassen()
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1*7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                            if u != 0:
                                U[i1*8 + i2*4 + j1*2 + j2, c] = u
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                            if v != 0:
                                V[j1*8 + j2*4 + k1*2 + k2, c] = v
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
                            if w != 0:
                                W[k1*8 + k2*4 + i1*2 + i2, c] = w
    return U, V, W

def _build_222_strassen():
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    
    # M3
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    
    # M4
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1
    
    # M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    
    # M6
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_442_kronecker():
    # Rank 28 for 4x4x2
    # 2x2x2 Strassen (7) x 2x2x1 Schoolbook (4)
    U2, V2, W2 = _build_222_strassen()
    
    # 2x2x1 is schoolbook: rank 4
    U1 = np.zeros((4, 4), dtype=int)
    V1 = np.zeros((2, 4), dtype=int)
    W1 = np.zeros((2, 4), dtype=int)
    # i,j,k -> i=0,1 j=0,1 k=0
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0]
    # U[i, r] = A[i,j], V[j, r] = B[j,0], W[0, r] = C[i,0]
    for i in range(2):
        for j in range(2):
            r = i*2+j
            U1[i, r] = 1
            V1[j, r] = 1
            W1[0, r] = 1
            
    U = np.zeros((16, 28), dtype=int)
    V = np.zeros((8, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    for c1 in range(7):
        for c2 in range(4):
            c = c1*4 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u = U2[i1*2+j1, c1] * U1[i2*2+j2, c2]
                            if u != 0:
                                U[i1*8 + i2*4 + j1*2 + j2, c] = u
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        # k2 is always 0 for 2x2x1
                        v = V2[j1*2+k1, c1] * V1[j2, c2]
                        if v != 0:
                            V[j1*4 + j2*2 + k1, c] = v
            for k1 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        w = W2[k1*2+i1, c1] * W1[0, c2]
                        if w != 0:
                            W[k1*4 + i1*2 + i2, c] = w
    return U, V, W

def _verify(U, V, W, n, m, p):
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_exp = A @ B
    C_comp = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha = np.sum(A * U[:, r].reshape(n, m))
        beta = np.sum(B * V[:, r].reshape(m, p))
        C_comp += alpha * beta * W[:, r].reshape(p, n).T
    return np.allclose(C_comp, C_exp, atol=1e-10)

def _schoolbook(n, m, p):
    U = np.zeros((n*m, n*m*p), dtype=int)
    V = np.zeros((m*p, n*m*p), dtype=int)
    W = np.zeros((p*n, n*m*p), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _reduce_rank(U, V, W, n, m, p, deadline):
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    def _verify_fast(U_mat, V_mat, W_mat):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        C_comp = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha = np.sum(A * U_mat[:, r].reshape(n, m))
            beta = np.sum(B * V_mat[:, r].reshape(m, p))
            C_comp += alpha * beta * W_mat[:, r].reshape(p, n).T
        return np.allclose(C_comp, C_exp, atol=1e-10)

    # Greedy removal
    for seed in [1, 2, 3]:
        if time.time() > deadline - 15: break
        np.random.seed(seed)
        U_cur, V_cur, W_cur = current_U.copy(), current_V.copy(), current_W.copy()
        rank = U_cur.shape[1]
        order = list(range(rank))
        np.random.shuffle(order)
        
        for r in order:
            if time.time() > deadline - 15: break
            if U_cur.shape[1] <= 1: break
            mask = np.ones(U_cur.shape[1], dtype=bool)
            mask[r] = False
            U_test = U_cur[:, mask]
            V_test = V_cur[:, mask]
            W_test = W_cur[:, mask]
            
            if _verify_fast(U_test, V_test, W_test):
                U_cur = U_test
                V_cur = V_test
                W_cur = W_test
        
        if U_cur.shape[1] < current_U.shape[1]:
            current_U, current_V, current_W = U_cur, V_cur, W_cur
            
    return current_U, current_V, current_W
