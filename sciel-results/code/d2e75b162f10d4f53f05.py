import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas for C = A*B where A,B,C are 2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # U[i*m+j, c] = A[i,j] coefficient for M_c
        # A = [[A11, A12], [A21, A22]]
        # Index: 0=A11, 1=A12, 2=A21, 3=A22
        # U[0,0]=1, U[0,1]=0, U[0,2]=1, U[0,3]=0, U[0,4]=1, U[0,5]=-1, U[0,6]=0
        # U[1,0]=0, U[1,1]=0, U[1,2]=0, U[1,3]=0, U[1,4]=1, U[1,5]=0, U[1,6]=1
        # U[2,0]=1, U[2,1]=1, U[2,2]=0, U[2,3]=0, U[2,4]=0, U[2,5]=-1, U[2,6]=0
        # U[3,0]=1, U[3,1]=0, U[3,2]=1, U[3,3]=1, U[3,4]=0, U[3,5]=0, U[3,6]=-1
        
        U[0,:] = [1, 0, 1, 0, 1, -1, 0]
        U[1,:] = [0, 0, 0, 0, 1, 0, 1]
        U[2,:] = [1, 1, 0, 0, 0, -1, 0]
        U[3,:] = [1, 0, 1, 1, 0, 0, -1]
        
        # V[j*p+k, c] = B[j,k] coefficient for M_c
        # B = [[B11, B12], [B21, B22]]
        # Index: 0=B11, 1=B12, 2=B21, 3=B22
        # M1: B11+B22 -> V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=1
        # M2: B11 -> V[0,1]=1, V[1,1]=0, V[2,1]=0, V[3,1]=0
        # M3: B21-B22 -> V[0,2]=0, V[1,2]=0, V[2,2]=1, V[3,2]=-1
        # M4: B12-B11 -> V[0,3]=-1, V[1,3]=1, V[2,3]=0, V[3,3]=0
        # M5: B22 -> V[0,4]=0, V[1,4]=0, V[2,4]=0, V[3,4]=1
        # M6: B11+B12 -> V[0,5]=1, V[1,5]=1, V[2,5]=0, V[3,5]=0
        # M7: B21+B22 -> V[0,6]=0, V[1,6]=0, V[2,6]=1, V[3,6]=1
        
        V[0,:] = [1, 1, 0, -1, 0, 1, 0]
        V[1,:] = [0, 0, 0, 1, 0, 1, 0]
        V[2,:] = [0, 0, 1, 0, 0, 0, 1]
        V[3,:] = [1, 0, -1, 0, 1, 0, 1]
        
        # W[k*n+i, c] = C[k,i] coefficient for M_c
        # C = [[C11, C12], [C21, C22]]
        # Index: 0=C11, 1=C12, 2=C21, 3=C22
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        W[0,:] = [1, 0, 0, 1, -1, 0, 1]
        W[1,:] = [0, 0, 1, 0, 1, 0, 0]
        W[2,:] = [0, 1, 0, 1, 0, 0, 0]
        W[3,:] = [1, 0, 1, 0, 0, 1, 0]
        
        return U, V, W
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        T_expected = np.ones((n, m, p))
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol
    
    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n*m*p
    
    # Build rank 68 decomposition using 4 Strassen 2x2x2 blocks
    # Strategy: Split 3x3x8 into blocks
    # Use the structure:
    # For 3x3x8, we can decompose using:
    # - Two 2x2x4 blocks (each rank 14 using Strassen)
    # - One 2x1x4 block (rank 8)
    # - One 1x2x4 block (rank 8)  
    # - One 1x1x4 block (rank 4)
    # Total for 3x3x4: 14+14+8+8+4 = 48? No, that's not right.
    
    # Let me think differently. The known rank 68 algorithm:
    # Split 3x3x8 as follows:
    # Think of the 3x3 as a 2x2 block plus cross terms.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 2x4, B21 is 1x4, B22 is 1x4
    # Actually, let me use a simpler approach.
    
    # Rank 68 decomposition:
    # Use 4 independent Strassen 2x2x2 blocks for the 2x2x4 part?
    # No, 2x2x4 has rank 14 (2 Strassen blocks for 2x2x2 each covering 2 columns).
    
    # Let me construct it properly:
    # The 3x3x8 multiplication can be decomposed as:
    # C = A * B where A is 3x3, B is 3x8
    # 
    # Split A into [[A11, A12], [A21, A22]] with A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # Split B into [[B11, B12], [B21, B22]] with B11 (2x4), B12 (2x4), B21 (1x4), B22 (1x4)
    # Wait, B is 3x8, so split as B11 (2x8), B21 (1x8) or B11 (2x4), B12 (2x4), B21 (1x4), B22 (1x4)
    
    # Let's use: B11 (2x4), B12 (2x4), B21 (1x4), B22 (1x4)
    # C11 = A11*B11 + A12*B21 (2x4)
    # C12 = A11*B12 + A12*B22 (2x4)
    # C21 = A21*B11 + A22*B21 (1x4)
    # C22 = A21*B12 + A22*B22 (1x4)
    
    # Rank of each:
    # A11*B11: 2x2x4 -> rank 14 (2 Strassen 2x2x2 blocks)
    # A12*B21: 2x1x4 -> rank 8 (schoolbook: 2*1*4=8)
    # A11*B12: 2x2x4 -> rank 14
    # A12*B22: 2x1x4 -> rank 8
    # A21*B11: 1x2x4 -> rank 8
    # A22*B21: 1x1x4 -> rank 4
    # A21*B12: 1x2x4 -> rank 8
    # A22*B22: 1x1x4 -> rank 4
    # Total: 14+8+14+8+8+4+8+4 = 68
    
    # Let's build this
    U_68 = np.zeros((nm, 68))
    V_68 = np.zeros((mp, 68))
    W_68 = np.zeros((pn, 68))
    
    col = 0
    # Block 1: A11 (rows 0,1; cols 0,1) * B11 (rows 0,1; cols 0,3) -> C11 (rows 0,1; cols 0,3)
    # 2x2x4 using 2 Strassen 2x2x2 blocks
    # First Strassen for cols 0,1 of B11
    sU, sV, sW = strassen_2x2x2()
    for c in range(7):
        # A11 is at U[0:4, :] (indices 0,1,2,3 in nm)
        for r in range(4):
            i, j = r // 2, r % 2
            U_68[i*m + j, col+c] = sU[r, c]
        # B11 first 2 cols: V[0:4, :] where B11[k, 0:2] -> V[j*p+k, :]
        for r in range(4):
            j, k = r // 2, r % 2
            V_68[j*p + k, col+c] = sV[r, c]
        # C11 first 2 cols: W[k*n+i, :]
        for r in range(4):
            k, i = r // 2, r % 2
            W_68[k*n + i, col+c] = sW[r, c]
    col += 7
    
    # Second Strassen for cols 2,3 of B11
    sU, sV, sW = strassen_2x2x2()
    for c in range(7):
        for r in range(4):
            i, j = r // 2, r % 2
            U_68[i*m + j, col+c] = sU[r, c]
        for r in range(4):
            j, k = r // 2, r % 2
            V_68[j*p + (k+2), col+c] = sV[r, c]
        for r in range(4):
            k, i = r // 2, r % 2
            W_68[(k+2)*n + i, col+c] = sW[r, c]
    col += 7
    
    # Block 2: A12 (rows 0,1; col 2) * B21 (row 2; cols 0,3) -> C11 (rows 0,1; cols 0,3)
    # 2x1x4 schoolbook: rank 8
    for i in range(2):
        for k in range(4):
            U_68[i*m + 2, col] = 1.0
            V_68[2*p + k, col] = 1.0
            W_68[k*n + i, col] = 1.0
            col += 1
    col += 0  # 8 terms used
    
    # Block 3: A11 (rows 0,1; cols 0,1) * B12 (rows 0,1; cols 4,7) -> C12 (rows 0,1; cols 4,7)
    # 2x2x4 using 2 Strassen 2x2x2 blocks
    sU, sV, sW = strassen_2x2x2()
    for c in range(7):
        for r in range(4):
            i, j = r // 2, r % 2
            U_68[i*m + j, col+c] = sU[r, c]
        for r in range(4):
            j, k = r // 2, r % 2
            V_68[j*p + (k+4), col+c] = sV[r, c]
        for r in range(4):
            k, i = r // 2, r % 2
            W_68[(k+4)*n + i, col+c] = sW[r, c]
    col += 7
    
    sU, sV, sW = strassen_2x2x2()
    for c in range(7):
        for r in range(4):
            i, j = r // 2, r % 2
            U_68[i*m + j, col+c] = sU[r, c]
        for r in range(4):
            j, k = r // 2, r % 2
            V_68[j*p + (k+6), col+c] = sV[r, c]
        for r in range(4):
            k, i = r // 2, r % 2
            W_68[(k+6)*n + i, col+c] = sW[r, c]
    col += 7
    
    # Block 4: A12 (rows 0,1; col 2) * B22 (row 2; cols 4,7) -> C12 (rows 0,1; cols 4,7)
    # 2x1x4 schoolbook: rank 8
    for i in range(2):
        for k in range(4):
            U_68[i*m + 2, col] = 1.0
            V_68[2*p + (k+4), col] = 1.0
            W_68[(k+4)*n + i, col] = 1.0
            col += 1
    
    # Block 5: A21 (row 2; cols 0,1) * B11 (rows 0,1; cols 0,3) -> C21 (row 2; cols 0,3)
    # 1x2x4 schoolbook: rank 8
    for j in range(2):
        for k in range(4):
            U_68[2*m + j, col] = 1.0
            V_68[j*p + k, col] = 1.0
            W_68[k*n + 2, col] = 1.0
            col += 1
    
    # Block 6: A22 (row 2; col 2) * B21 (row 2; cols 0,3) -> C21 (row 2; cols 0,3)
    # 1x1x4 schoolbook: rank 4
    for k in range(4):
        U_68[2*m + 2, col] = 1.0
        V_68[2*p + k, col] = 1.0
        W_68[k*n + 2, col] = 1.0
        col += 1
    
    # Block 7: A21 (row 2; cols 0,1) * B12 (rows 0,1; cols 4,7) -> C22 (row 2; cols 4,7)
    # 1x2x4 schoolbook: rank 8
    for j in range(2):
        for k in range(4):
            U_68[2*m + j, col] = 1.0
            V_68[j*p + (k+4), col] = 1.0
            W_68[(k+4)*n + 2, col] = 1.0
            col += 1
    
    # Block 8: A22 (row 2; col 2) * B22 (row 2; cols 4,7) -> C22 (row 2; cols 4,7)
    # 1x1x4 schoolbook: rank 4
    for k in range(4):
        U_68[2*m + 2, col] = 1.0
        V_68[2*p + (k+4), col] = 1.0
        W_68[(k+4)*n + 2, col] = 1.0
        col += 1
    
    # Verify col == 68
    assert col == 68, f"Expected 68, got {col}"
    
    # Validate the rank 68 decomposition
    if validate(U_68.tolist(), V_68.tolist(), W_68.tolist()):
        return {"U": U_68.tolist(), "V": V_68.tolist(), "W": W_68.tolist()}
    
    # If validation fails, fall back to schoolbook
    return best_result
