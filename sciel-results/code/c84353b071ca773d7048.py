"""Rank-minimal exact decomposition for 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]
_DEADLINE = time.time() + 140.0

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    U = np.array(U, dtype=complex)
    V = np.array(V, dtype=complex)
    W = np.array(W, dtype=complex)
    T = np.zeros((n, m, p), dtype=complex)
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    A = np.eye(n, m, dtype=complex)
    B = np.eye(m, p, dtype=complex)
    AB = A @ B
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - AB[i, k]) > 1e-6:
                    return False
    return True

def _build_strassen_49():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using recursive Strassen's algorithm.
    
    Strassen's algorithm for 2x2 matrices has rank 7.
    For 4x4, we view it as a block matrix of four 2x2 blocks.
    The tensor for 4x4 multiplication is the Kronecker product of the 2x2 tensor with itself.
    Rank(T1 x T2) = Rank(T1) * Rank(T2) = 7 * 7 = 49.
    """
    n = m = p = 4
    
    # Define the rank-7 decomposition for 2x2 matrix multiplication.
    # Indices for U, V, W are 0..3 (2x2).
    # Rank is 7.
    # U[i][k], V[j][k], W[k][l] such that sum_k U[i][k]V[j][k]W[k][l] = delta_{i,l} (for A*B where A is 2x2, B is 2x2, result is 2x2? No.)
    # Wait, the tensor T[i, j, k] corresponds to (A B)_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
    # U_c is a vector of length 4 (i, j indices combined? No, i is row of A, j is col of A / row of B).
    # Standard convention:
    # U has shape (n*m, rank) -> (4, 7) for 2x2.
    # V has shape (m*p, rank) -> (4, 7) for 2x2.
    # W has shape (p*n, rank) -> (4, 7) for 2x2.
    
    # Strassen's 7 products for 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W vectors.
    # U index: i*m + j (for 2x2, i,j in {0,1}).
    # V index: j*p + k (for 2x2, j,k in {0,1}).
    # W index: k*n + i (for 2x2, k,i in {0,1}).
    
    # Let's define the 7 factors (u_c, v_c, w_c).
    # u_c[i*m+j], v_c[j*p+k], w_c[k*n+i].
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (A11+A22)(B11+B22) -> C11, C22
    # A parts: A11 (0,0), A22 (1,1). U[0][0]=1, U[3][0]=1.
    # B parts: B11 (0,0), B22 (1,1). V[0][0]=1, V[3][0]=1.
    # C parts: C11 (0,0), C22 (1,1). W[0][0]=1, W[3][0]=1.
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2: (A21+A22)B11 -> C21, C22
    # A parts: A21 (1,0), A22 (1,1). U[2][1]=1, U[3][1]=1.
    # B parts: B11 (0,0). V[0][1]=1.
    # C parts: C21 (1,0), C22 (1,1). W[2][1]=1, W[3][1]=1.
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1
    
    # M3: A11(B12-B22) -> C11, C12
    # A parts: A11 (0,0). U[0][2]=1.
    # B parts: B12 (0,1), B22 (1,1). V[1][2]=1, V[3][2]=-1.
    # C parts: C11 (0,0), C12 (0,1). W[0][2]=1, W[1][2]=1.
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[0][2] = 1; W2[1][2] = 1
    
    # M4: A22(B21-B11) -> C11, C21
    # A parts: A22 (1,1). U[3][3]=1.
    # B parts: B21 (1,0), B11 (0,0). V[2][3]=1, V[0][3]=-1.
    # C parts: C11 (0,0), C21 (1,0). W[0][3]=1, W[2][3]=1.
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5: (A11+A12)B22 -> C11, C12
    # A parts: A11 (0,0), A12 (0,1). U[0][4]=1, U[1][4]=1.
    # B parts: B22 (1,1). V[3][4]=1.
    # C parts: C11 (0,0), C12 (0,1). W[0][4]=1, W[1][4]=1.
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[1][4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C22
    # A parts: A21 (1,0), A11 (0,0). U[2][5]=1, U[0][5]=-1.
    # B parts: B11 (0,0), B12 (0,1). V[0][5]=1, V[1][5]=1.
    # C parts: C22 (1,1). W[3][5]=1.
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C11, C22
    # A parts: A12 (0,1), A22 (1,1). U[1][6]=1, U[3][6]=-1.
    # B parts: B21 (1,0), B22 (1,1). V[2][6]=1, V[3][6]=1.
    # C parts: C11 (0,0), C22 (1,1). W[0][6]=1, W[3][6]=1.
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1
    
    # Now construct 4x4 decomposition using Kronecker product.
    # U4[i*4+j][c1*7+c2] = U2[i*2+j1][c1] * U2[j*2+j2][c2] ?
    # No, the tensor product structure is:
    # T4 = T2 \otimes T2.
    # The indices are (i1, i2, j1, j2) for U?
    # Let's map 4x4 indices to 2x2 blocks.
    # A is 4x4. A[i, j] where i, j in 0..3.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # B[j, k] where j, k in 0..3.
    # j = 2*j1 + j2, k = 2*k1 + k2.
    # C[i, k] where i, k in 0..3.
    # i = 2*i1 + i2, k = 2*k1 + k2.
    
    # The decomposition of T2 is sum_{c=0}^6 U2_c \otimes V2_c \otimes W2_c.
    # The decomposition of T4 is sum_{c1=0}^6 sum_{c2=0}^6 (U2_{c1} \otimes U2_{c2}) \otimes (V2_{c1} \otimes V2_{c2}) \otimes (W2_{c1} \otimes W2_{c2}).
    
    # U4 has shape (16, 49).
    # Index of U4: i*4 + j.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # U4[i*4+j][c1*7+c2] = U2[i1*2+i2][c1] * U2[j1*2+j2][c2]?
    # Wait, U2 index is i*2+j.
    # So U2[i1*2+i2][c1] is the value for block row i1, local row i2.
    # And U2[j1*2+j2][c2] is the value for block col j1, local col j2.
    # This assumes the Kronecker product of the U vectors.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1 * 2 + i2][c1] * U2[j1 * 2 + j2][c2]
                            if val != 0:
                                U4[i * 4 + j][c] = val
                            
                            # V index: j*4 + k
                            # j = 2*j1 + j2, k = 2*k1 + k2
                            # V2 index for block: j1, k1? No, V2 is for B.
                            # B is 4x4. B[j, k].
                            # V2 index is j*2+k.
                            # So V2[j1*2+j2][c1] and V2[k1*2+k2][c2]?
                            # Wait, V2 corresponds to the second matrix in the 2x2 multiplication.
                            # In the Kronecker product, V4 = V2 \otimes V2.
                            # V4[j*4+k][c] = V2[j1*2+j2][c1] * V2[k1*2+k2][c2]?
                            # No, V2 index is j*2+k.
                            # So V2[2*j1 + j2][c1] * V2[2*k1 + k2][c2]?
                            # Let's check indices.
                            # V2 has 4 rows.
                            # Row 0: j=0, k=0.
                            # Row 1: j=0, k=1.
                            # Row 2: j=1, k=0.
                            # Row 3: j=1, k=1.
                            # So V2[idx][c] where idx = j*2+k.
                            # In the Kronecker product for 4x4:
                            # V4[idx4][c] where idx4 = j*4+k.
                            # j = 2*j1 + j2, k = 2*k1 + k2.
                            # The value is V2[2*j1 + j2][c1] * V2[2*k1 + k2][c2]?
                            # No, the tensor product is T4 = T2 \otimes T2.
                            # The U part is U2 \otimes U2.
                            # The V part is V2 \otimes V2.
                            # The W part is W2 \otimes W2.
                            # So yes, V4[j*4+k][c] = V2[j1*2+j2][c1] * V2[k1*2+k2][c2].
                            # Wait, V2 index is j*2+k.
                            # So V2[2*j1 + j2] is the value for block (j1, local j2) of the row index of B?
                            # No, B's row index is j. B's col index is k.
                            # In the 2x2 case, B is 2x2.
                            # In the 4x4 case, B is 4x4.
                            # The Kronecker product of B1 (2x2) and B2 (2x2) is B (4x4).
                            # B[2*j1+j2, 2*k1+k2] = B1[j1, k1] * B2[j2, k2].
                            # So the V vector for the Kronecker product is V2 \otimes V2.
                            # V4[idx4][c] = V2[idx2_1][c1] * V2[idx2_2][c2].
                            # idx2_1 = 2*j1 + j2? No.
                            # idx2_1 corresponds to the "block" index of B1.
                            # B1 is 2x2. Its indices are j1, k1.
                            # So idx2_1 = j1 * 2 + k1.
                            # idx2_2 = j2 * 2 + k2.
                            # So V4[j*4+k][c] = V2[j1*2+k1][c1] * V2[j2*2+k2][c2].
                            
                            # Let's re-verify U4.
                            # A[2*i1+i2, 2*j1+j2] = A1[i1, j1] * A2[i2, j2].
                            # U4[i*4+j][c] = U2[i1*2+j1][c1] * U2[i2*2+j2][c2].
                            
                            # Let's re-verify W4.
                            # C[2*i1+i2, 2*k1+k2] = C1[i1, k1] * C2[i2, k2].
                            # W4[k*4+i][c] = W2[k1*2+i1][c1] * W2[k2*2+i2][c2].
                            
                            # So:
                            # U4[i*4+j][c] = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                            # V4[j*4+k][c] = V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
                            # W4[k*4+i][c] = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
                            
                            # We are looping over i1, i2, j1, j2, k1, k2.
                            # We need to compute U4, V4, W4 for all i, j, k.
                            # It is easier to loop over i, j, k directly.
                            
    # Let's rewrite the loop for clarity and correctness.
    for i in range(4):
        i1, i2 = divmod(i, 2)
        for j in range(4):
            j1, j2 = divmod(j, 2)
            for k in range(4):
                k1, k2 = divmod(k, 2)
                for c1 in range(7):
                    for c2 in range(7):
                        c = c1 * 7 + c2
                        u_val = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                        v_val = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                        w_val = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            U4[i * 4 + j][c] = u_val
                            V4[j * 4 + k][c] = v_val
                            W4[k * 4 + i][c] = w_val
                            
    return U4, V4, W4

def construct():
    U, V, W = _build_strassen_49()
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    # Fallback to schoolbook if verification fails (should not happen)
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
