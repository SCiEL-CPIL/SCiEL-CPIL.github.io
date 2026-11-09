import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Target rank 43
    target_rank = 43
    rank = target_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_block(ja, jb, ka, kb):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[0*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        W[ka*n+1][col] = 1
        W[kb*n+1][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0*m+ja][col] = 1
        V[ja*p+kb][col] = 1
        V[jb*p+kb][col] = -1
        W[kb*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[1*m+jb][col] = 1
        V[jb*p+ka][col] = 1
        V[ja*p+ka][col] = -1
        W[ka*n+0][col] = 1
        W[ka*n+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*m+ja][col] = 1
        U[0*m+jb][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = -1
        W[kb*n+0][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1*m+ja][col] = 1
        U[0*m+ja][col] = -1
        V[ja*p+ka][col] = 1
        V[ja*p+kb][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0*m+jb][col] = 1
        U[1*m+jb][col] = -1
        V[jb*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        col += 1

    # Block 1: J=[0,1], K=[0,1] (2x2x2) -> Rank 7
    add_strassen_block(0, 1, 0, 1)
    
    # Block 2: J=[0,1], K=[2,3,4] (2x2x3) -> Rank 11
    # Strassen on J=[0,1], K=[2,3]
    add_strassen_block(0, 1, 2, 3)
    # Border: J=[0,1], K=4
    # Terms: A[i,0]B[0,4], A[i,1]B[1,4] for i=0,1
    # 4 terms
    for i in [0, 1]:
        for j in [0, 1]:
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    # Block 3: J=[2,3,4], K=[0,1] (2x3x2) -> Rank 8
    # Strassen on J=[2,3], K=[0,1]
    add_strassen_block(2, 3, 0, 1)
    # Border: J=4, K=[0,1]
    # Term: A[i,4]B[4,k] for i=0,1, k=0,1
    # Rank 1 update?
    # U[i, r] = A[i,4], V[4,k, r] = B[4,k], W[k,i, r] = C[k,i]
    # This is 1 term?
    # No, V[4,0] and V[4,1] are different entries.
    # But they can share the same rank factor r.
    U[0*m+4][col] = 1
    U[1*m+4][col] = 1
    V[4*p+0][col] = 1
    V[4*p+1][col] = 1
    W[0*n+0][col] = 1
    W[0*n+1][col] = 1
    W[1*n+0][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # Block 4: J=[2,3,4], K=[2,3,4] (2x3x3) -> Rank 17
    # Strassen on J=[2,3], K=[2,3]
    add_strassen_block(2, 3, 2, 3)
    # Border 1: J=4, K=[2,3]
    # A[i,4]B[4,k] for k=2,3. Rank 1.
    U[0*m+4][col] = 1
    U[1*m+4][col] = 1
    V[4*p+2][col] = 1
    V[4*p+3][col] = 1
    W[2*n+0][col] = 1
    W[2*n+1][col] = 1
    W[3*n+0][col] = 1
    W[3*n+1][col] = 1
    col += 1
    
    # Border 2: J=[2,3], K=4
    # A[i,j]B[j,4] for j=2,3. 4 terms.
    for i in [0, 1]:
        for j in [2, 3]:
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    # Border 3: J=4, K=4
    # A[i,4]B[4,4]. 2 terms.
    for i in [0, 1]:
        U[i*m+4][col] = 1
        V[4*p+4][col] = 1
        W[4*n+i][col] = 1
        col += 1
        
    # Verify
    if col != rank:
        # Fallback to 46 if count mismatch (should not happen)
        pass
        
    U_arr = U
    V_arr = V
    W_arr = W
    
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    if np.allclose(T_recon, T_target, atol=1e-9, rtol=0):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: rank 46
    rank_school = 46
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    c = 0
    # Re-run the 46 logic or just use schoolbook 50?
    # Let's just return schoolbook 50 to be safe if 43 fails
    rank_50 = n * m * p
    U_50 = np.zeros((n * m, rank_50), dtype=float)
    V_50 = np.zeros((m * p, rank_50), dtype=float)
    W_50 = np.zeros((p * n, rank_50), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_50[i*m+j][c] = 1
                V_50[j*p+k][c] = 1
                W_50[k*n+i][c] = 1
                c += 1
    return {"U": U_50.tolist(), "V": V_50.tolist(), "W": W_50.tolist()}
