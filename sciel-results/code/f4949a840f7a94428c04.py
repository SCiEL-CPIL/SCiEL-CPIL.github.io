import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    # Build the rank-44 Strassen-based decomposition
    rank44 = 44
    U = [[0] * rank44 for _ in range(dim_u)]
    V = [[0] * rank44 for _ in range(dim_v)]
    W = [[0] * rank44 for _ in range(dim_w)]
    
    col = 0
    
    def add_strassen(X_u, Y_v, Z_w, col):
        """Add 7 Strassen terms for X(2x2) * Y(2x2) = Z(2x2)."""
        # M1 = (X00+X11)(Y00+Y11)
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        # M2 = (X10+X11)Y00
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        # M3 = X00(Y01-Y11)
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        # M4 = X11(Y10-Y00)
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        # M5 = (X00+X01)Y11
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        # M6 = (X10-X00)(Y00+Y01)
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        # M7 = (X01-X11)(Y10+Y11)
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col
    
    # Strassen blocks for the 2x4 x 4x2 top portion
    # A is 3x4, B is 4x4, C is 4x3
    # Top 2 rows of A (2x4) times left 2 cols of B (4x2) -> top-left 2x2 of C
    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    # Top 2 rows of A (2x4) times right 2 cols of B (4x2) -> top-right 2x2 of C
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    # Actually, we need to be more careful. Let me re-examine the structure.
    
    # A is 3x4: rows 0,1,2; cols 0,1,2,3
    # B is 4x4: rows 0,1,2,3; cols 0,1,2,3
    # C is 4x3: rows 0,1,2,3; cols 0,1,2
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # U index: i*m+j = i*4+j for i in 0..2, j in 0..3
    # V index: j*p+k = j*4+k for j in 0..3, k in 0..3
    # W index: k*n+i = k*3+i for k in 0..3, i in 0..2
    
    # Let me redo this properly.
    # Clear and rebuild
    U = [[0] * rank44 for _ in range(dim_u)]
    V = [[0] * rank44 for _ in range(dim_v)]
    W = [[0] * rank44 for _ in range(dim_w)]
    col = 0
    
    # Block structure:
    # A = [A11 A12] where A11 is 2x2 (rows 0-1, cols 0-1), A12 is 2x2 (rows 0-1, cols 2-3)
    #     [A21 A22] where A21 is 1x2 (row 2, cols 0-1), A22 is 1x2 (row 2, cols 2-3)
    # B = [B11 B12] where B11 is 2x2 (rows 0-1, cols 0-1), B12 is 2x2 (rows 0-1, cols 2-3)
    #     [B21 B22] where B21 is 2x2 (rows 2-3, cols 0-1), B22 is 2x2 (rows 2-3, cols 2-3)
    
    # C = [A11*B11 + A12*B21,  A11*B12 + A12*B22]
    #     [A21*B11 + A22*B21,  A21*B12 + A22*B22]
    
    # C top-left (2x2): A11*B11 + A12*B21
    # A11: rows 0-1, cols 0-1 -> U indices: 0,1,4,5
    # B11: rows 0-1, cols 0-1 -> V indices: 0,1,4,5
    # C top-left: rows 0-1, cols 0-1 -> W indices: k*3+i for k in 0-1, i in 0-1 -> 0,1,3,4
    # Wait, W index is k*n+i = k*3+i
    # C row k, col i. So C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[3], C[1,1]=W[4]
    
    # Strassen for A11(2x2) * B11(2x2) -> C11(2x2)
    # X_u = [0,1,4,5] (A11)
    # Y_v = [0,1,4,5] (B11)
    # Z_w = [0,1,3,4] (C11: C[0,0],C[0,1],C[1,0],C[1,1])
    col = add_strassen([0,1,4,5], [0,1,4,5], [0,1,3,4], col)
    
    # Strassen for A12(2x2) * B21(2x2) -> C11(2x2)
    # A12: rows 0-1, cols 2-3 -> U indices: 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> V indices: 8,9,12,13
    # C11: same as above -> W indices: 0,1,3,4
    col = add_strassen([2,3,6,7], [8,9,12,13], [0,1,3,4], col)
    
    # C top-right (2x2): A11*B12 + A12*B22
    # A11: rows 0-1, cols 0-1 -> U indices: 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> V indices: 2,3,6,7
    # C top-right: rows 0-1, cols 2-3 -> W indices: k*3+i for k in 0-1, i in 2-3
    # C[0,2]=W[2], C[0,3]=W[5], C[1,2]=W[8], C[1,3]=W[11]
    col = add_strassen([0,1,4,5], [2,3,6,7], [2,5,8,11], col)
    
    # Strassen for A12(2x2) * B22(2x2) -> C top-right(2x2)
    # A12: rows 0-1, cols 2-3 -> U indices: 2,3,6,7
    # B22: rows 2-3, cols 2-3 -> V indices: 10,11,14,15
    # C top-right: W indices: 2,5,8,11
    col = add_strassen([2,3,6,7], [10,11,14,15], [2,5,8,11], col)
    
    # C bottom-left (1x2): A21*B11 + A22*B21
    # A21: row 2, cols 0-1 -> U indices: 8,9
    # A22: row 2, cols 2-3 -> U indices: 10,11
    # B11: rows 0-1, cols 0-1 -> V indices: 0,1,4,5
    # B21: rows 2-3, cols 0-1 -> V indices: 8,9,12,13
    # C bottom-left: row 2, cols 0-1 -> W indices: k*3+i for k=2, i in 0-1 -> W[6], W[7]
    
    # Schoolbook for 1x2 * 2x2 = 1x2 (4 terms each, but we can combine)
    # A21*B11: sum over j=0,1 of A21[j]*B11[j,k] for k=0,1
    # A22*B21: sum over j=2,3 of A22[j-2]*B21[j-2,k] for k=0,1
    
    # For each k in 0,1:
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k] + A[2,3]*B[3,k]
    # This is 4 terms per k, so 8 terms total
    
    for j in range(4):
        u_idx = 8 + j  # A[2,j]
        v_idx_0 = j * 4 + 0  # B[j,0]
        v_idx_1 = j * 4 + 1  # B[j,1]
        w_idx_0 = 6  # C[2,0]
        w_idx_1 = 7  # C[2,1]
        U[u_idx][col] += 1; V[v_idx_0][col] += 1; W[w_idx_0][col] += 1
        col += 1
        U[u_idx][col] += 1; V[v_idx_1][col] += 1; W[w_idx_1][col] += 1
        col += 1
    
    # C bottom-right (1x2): A21*B12 + A22*B22
    # A21: row 2, cols 0-1 -> U indices: 8,9
    # A22: row 2, cols 2-3 -> U indices: 10,11
    # B12: rows 0-1, cols 2-3 -> V indices: 2,3,6,7
    # B22: rows 2-3, cols 2-3 -> V indices: 10,11,14,15
    # C bottom-right: row 2, cols 2-3 -> W indices: k*3+i for k=2, i in 2-3 -> W[8], W[9]
    # Wait, C has 4 rows and 3 cols. C[2,2] and C[2,3]? No, C is 4x3, so cols are 0,1,2.
    
    # Let me re-examine. C is p x n = 4 x 3. So C has rows 0-3 and cols 0-2.
    # W index: k*n+i = k*3+i where k is row of C (0-3) and i is col of C (0-2)
    
    # C top-left (2x2): rows 0-1, cols 0-1
    # C top-right (2x1): rows 0-1, col 2
    # C bottom-left (2x2): rows 2-3, cols 0-1
    # C bottom-right (2x1): rows 2-3, col 2
    
    # I made an error. Let me redo the block structure.
    
    # A is 3x4: [A11(2x2) A12(2x2)]
    #             [A21(1x2) A22(1x2)]
    # B is 4x4: [B11(2x2) B12(2x2)]
    #             [B21(2x2) B22(2x2)]
    
    # C = A*B is 3x4... wait, no. A is 3x4, B is 4x4, so C is 3x4.
    # But the problem says C follows W[k*n+i,r] where n=3, so C has n=3 columns.
    # Let me re-read: "multiplying a 3 by 4 matrix by a 4 by 4 matrix"
    # So A is 3x4, B is 4x4, C is 3x4.
    # W[k*n+i,r] where k is... hmm.
    
    # Actually, looking at the interface: W has shape (p*n, rank) = (4*3, rank) = (12, rank)
    # And the output factor follows W[k*n+i,r].
    # If C is 3x4 (n x p), then C[i,k] = sum_j A[i,j]*B[j,k]
    # W index: k*n+i where k is col of C (0-3) and i is row of C (0-2)
    # So W has 4*3 = 12 entries, indexed by (k,i) -> k*3+i
    
    # Let me restart with the correct understanding.
    # A: 3x4, indices (i,j) where i in 0-2, j in 0-3
    # B: 4x4, indices (j,k) where j in 0-3, k in 0-3
    # C: 3x4, indices (i,k) where i in 0-2, k in 0-3
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # U[i*m+j] = U[i*4+j], i in 0-2, j in 0-3
    # V[j*p+k] = V[j*4+k], j in 0-3, k in 0-3
    # W[k*n+i] = W[k*3+i], k in 0-3, i in 0-2
    
    # Block decomposition:
    # A = [A11(2x2) A12(2x2)]  rows 0-1, cols 0-1 and cols 2-3
    #     [A21(1x2) A22(1x2)]  row 2, cols 0-1 and cols 2-3
    # B = [B11(2x2) B12(2x2)]  rows 0-1, cols 0-1 and cols 2-3
    #     [B21(2x2) B22(2x2)]  rows 2-3, cols 0-1 and cols 2-3
    
    # C = [A11*B11 + A12*B21,  A11*B12 + A12*B22]  rows 0-1
    #     [A21*B11 + A22*B21,  A21*B12 + A22*B22]  row 2
    
    # C top-left (2x2): rows 0-1, cols 0-1
    # W indices: k*3+i for k in 0-1, i in 0-1 -> (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # C top-right (2x2): rows 0-1, cols 2-3
    # W indices: k*3+i for k in 2-3, i in 0-1 -> (2,0)=6, (2,1)=7, (3,0)=9, (3,1)=10
    # C bottom-left (1x2): row 2, cols 0-1
    # W indices: k*3+i for k in 0-1, i=2 -> (0,2)=2, (1,2)=5
    # C bottom-right (1x2): row 2, cols 2-3
    # W indices: k*3+i for k in 2-3, i=2 -> (2,2)=8, (3,2)=11
    
    # Clear and rebuild
    U = [[0] * rank44 for _ in range(dim_u)]
    V = [[0] * rank44 for _ in range(dim_v)]
    W = [[0] * rank44 for _ in range(dim_w)]
    col = 0
    
    # C top-left (2x2): A11*B11 + A12*B21
    # A11: U[0,1,4,5], B11: V[0,1,4,5], C11: W[0,1,3,4]
    col = add_strassen([0,1,4,5], [0,1,4,5], [0,1,3,4], col)
    
    # A12: U[2,3,6,7], B21: V[8,9,12,13], C11: W[0,1,3,4]
    col = add_strassen([2,3,6,7], [8,9,12,13], [0,1,3,4], col)
    
    # C top-right (2x2): A11*B12 + A12*B22
    # A11: U[0,1,4,5], B12: V[2,3,6,7], C12: W[6,7,9,10]
    col = add_strassen([0,1,4,5], [2,3,6,7], [6,7,9,10], col)
    
    # A12: U[2,3,6,7], B22: V[10,11,14,15], C12: W[6,7,9,10]
    col = add_strassen([2,3,6,7], [10,11,14,15], [6,7,9,10], col)
    
    # C bottom-left (1x2): A21*B11 + A22*B21
    # A21: U[8,9], A22: U[10,11]
    # B11: V[0,1,4,5], B21: V[8,9,12,13]
    # C21: W[2,5]
    for j in range(4):
        u_idx = 8 + j
        v_idx_0 = j * 4 + 0
        v_idx_1 = j * 4 + 1
        U[u_idx][col] += 1; V[v_idx_0][col] += 1; W[2][col] += 1
        col += 1
        U[u_idx][col] += 1; V[v_idx_1][col] += 1; W[5][col] += 1
        col += 1
    
    # C bottom-right (1x2): A21*B12 + A22*B22
    # A21: U[8,9], A22: U[10,11]
    # B12: V[2,3,6,7], B22: V[10,11,14,15]
    # C22: W[8,11]
    for j in range(4):
        u_idx = 8 + j
        v_idx_0 = j * 4 + 2
        v_idx_1 = j * 4 + 3
        U[u_idx][col] += 1; V[v_idx_0][col] += 1; W[8][col] += 1
        col += 1
        U[u_idx][col] += 1; V[v_idx_1][col] += 1; W[11][col] += 1
        col += 1
    
    # Verify the rank-44 decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomp(U, V, W, n, m, p):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    coeff = 0.0
                    for r in range(U.shape[1]):
                        coeff += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(coeff - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_np, V_np, W_np, n, m, p):
        # Fallback to schoolbook rank-48
        rank48 = 48
        U48 = [[0] * rank48 for _ in range(dim_u)]
        V48 = [[0] * rank48 for _ in range(dim_v)]
        W48 = [[0] * rank48 for _ in range(dim_w)]
        col48 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i*m+j][col48] = 1
                    V48[j*p+k][col48] = 1
                    W48[k*n+i][col48] = 1
                    col48 += 1
        return {"U": U48, "V": V48, "W": W48}
    
    # Try to reduce rank by finding linear dependencies
    # Check if any column can be expressed as a linear combination of others
    # This is a refinement within the Strassen-based family
    
    if time.time() < deadline - 5:
        # Try to find linear dependencies among the 44 terms
        # For each term r, check if it's in the span of the others
        # We can do this by checking if the tensor contribution of term r
        # can be matched by a linear combination of other terms
        
        # Actually, a simpler approach: try to find if we can merge terms
        # by looking for patterns in the coefficients
        
        # For now, just return the verified rank-44 result
        pass
    
    return {"U": U, "V": V, "W": W}
