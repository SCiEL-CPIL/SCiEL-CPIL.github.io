"""Matrix multiplication 4x4x8 tensor decomposition via recursive Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    coeff = 0.0
                    for c in range(r):
                        coeff += W[k*n+i, c] * U[i*m+j, c] * V[j2*p+k, c]
                    expected = 1.0 if j == j2 else 0.0
                    if abs(coeff - expected) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build rank-98 decomposition:
    # 4x4x8 = two independent 4x4x4 multiplications
    # Each 4x4x4 = 7x7 = 49 Strassen products (tensor product of 2x2 Strassen)
    # Total rank = 49 + 49 = 98
    
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # 2x2 Strassen factors
    # U2x2: 4x7, rows indexed by (i,j) in row-major: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # V2x2: 4x7, rows indexed by (j,k) in row-major: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # W2x2: 4x7, rows indexed by (k,i) in row-major per W[k*n+i]: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # Strassen M factors for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # U2x2[i*m+j, c] where m=2
    U2x2 = np.zeros((4, 7), dtype=float)
    # M1: A00 (i=0,j=0) and A11 (i=1,j=1)
    U2x2[0, 0] = 1; U2x2[3, 0] = 1
    # M2: A10 (i=1,j=0) and A11 (i=1,j=1)
    U2x2[2, 1] = 1; U2x2[3, 1] = 1
    # M3: A00 (i=0,j=0)
    U2x2[0, 2] = 1
    # M4: A11 (i=1,j=1)
    U2x2[3, 3] = 1
    # M5: A00 (i=0,j=0) and A01 (i=0,j=1)
    U2x2[0, 4] = 1; U2x2[1, 4] = 1
    # M6: A10 (i=1,j=0) and A00 (i=0,j=0) with -1
    U2x2[2, 5] = 1; U2x2[0, 5] = -1
    # M7: A01 (i=0,j=1) and A11 (i=1,j=1) with -1
    U2x2[1, 6] = 1; U2x2[3, 6] = -1
    
    # V2x2[j*p+k, c] where p=2
    V2x2 = np.zeros((4, 7), dtype=float)
    # M1: B00 (j=0,k=0) and B11 (j=1,k=1)
    V2x2[0, 0] = 1; V2x2[3, 0] = 1
    # M2: B00 (j=0,k=0)
    V2x2[0, 1] = 1
    # M3: B10 (j=1,k=0) and B11 (j=1,k=1) with -1
    V2x2[2, 2] = 1; V2x2[3, 2] = -1
    # M4: B01 (j=0,k=1) and B00 (j=0,k=0) with -1
    V2x2[1, 3] = 1; V2x2[0, 3] = -1
    # M5: B11 (j=1,k=1)
    V2x2[3, 4] = 1
    # M6: B00 (j=0,k=0) and B10 (j=1,k=0)
    V2x2[0, 5] = 1; V2x2[2, 5] = 1
    # M7: B11 (j=1,k=1) and B01 (j=0,k=1)
    V2x2[3, 6] = 1; V2x2[1, 6] = 1
    
    # W2x2[k*n+i, c] where n=2
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W2x2 = np.zeros((4, 7), dtype=float)
    # C00 (k=0,i=0) -> row 0
    W2x2[0, 0] = 1; W2x2[0, 3] = 1; W2x2[0, 4] = -1; W2x2[0, 6] = -1
    # C01 (k=0,i=1) -> row 1
    W2x2[1, 2] = 1; W2x2[1, 4] = 1
    # C10 (k=1,i=0) -> row 2
    W2x2[2, 1] = 1; W2x2[2, 3] = 1
    # C11 (k=1,i=1) -> row 3
    W2x2[3, 0] = 1; W2x2[3, 1] = -1; W2x2[3, 2] = 1; W2x2[3, 5] = 1
    
    # Now build 4x4x4 rank-49 decomposition
    # 4x4x4 is 2x2 blocks of 2x2 matrices
    # Block indices: I in {0,1} (block row of A, block col of C), K in {0,1} (block row of B, block col of C)
    # A[I1, I2] is a 2x2 block, B[J1, J2] is a 2x2 block, C[K1, K2] is a 2x2 block
    # The 4x4 Strassen has 7 block-level Strassen products, each of which is a 2x2x2 Strassen (7 scalar products)
    # Total: 7 * 7 = 49 products
    
    # For the 4x4x4 case:
    # A is 4x4, split into 2x2 blocks A[I1][I2] (each 2x2)
    # B is 4x4, split into 2x2 blocks B[J1][J2] (each 2x2)
    # C is 4x4, split into 2x2 blocks C[K1][K2] (each 2x2)
    
    # Block-level Strassen for 2x2 block matrices (each entry is a 2x2 matrix):
    # M[I1,J1,K1] for the 7 Strassen products at block level
    # Each block-level product is a 2x2x2 matrix multiplication (7 scalar Strassen products)
    
    # Let me index the 49 products as (block_strassen_id, scalar_strassen_id)
    # block_strassen_id in {0,...,6}, scalar_strassen_id in {0,...,6}
    
    # Block-level U (for A blocks):
    # A is 4x4, blocks A[I1][I2] where I1 is block row, I2 is block col
    # U4x4 rows: (i,j) where i,j in {0,1,2,3} -> 16 rows
    # A block (I1, I2) contains elements A[I1*2 + i, I2*2 + j] for i,j in {0,1}
    
    # Block-level Strassen U factors (which block-level Strassen product uses which A block):
    # Block M1: (A00+A11)(B00+B11) -> uses A blocks (0,0) and (1,1)
    # Block M2: (A10+A11)B00 -> uses A blocks (1,0) and (1,1)
    # Block M3: A00(B10-B11) -> uses A block (0,0)
    # Block M4: A11(B01-B00) -> uses A block (1,1)
    # Block M5: (A00+A01)B11 -> uses A blocks (0,0) and (0,1)
    # Block M6: (A10-A00)(B00+B10) -> uses A blocks (1,0) and (0,0)
    # Block M7: (A01-A11)(B11+B01) -> uses A blocks (0,1) and (1,1)
    
    # For each block-level Strassen product s (0..6), the A block coefficients:
    # s=0 (M1): A(0,0)+A(1,1)
    # s=1 (M2): A(1,0)+A(1,1)
    # s=2 (M3): A(0,0)
    # s=3 (M4): A(1,1)
    # s=4 (M5): A(0,0)+A(0,1)
    # s=5 (M6): A(1,0)-A(0,0)
    # s=6 (M7): A(0,1)-A(1,1)
    
    block_U = np.zeros((7, 4), dtype=float)  # 7 block Strassen, 4 A blocks (I1,I2)
    block_U[0, 0] = 1; block_U[0, 3] = 1  # M1: A00 + A11 (blocks (0,0) and (1,1))
    block_U[1, 2] = 1; block_U[1, 3] = 1  # M2: A10 + A11 (blocks (1,0) and (1,1))
    block_U[2, 0] = 1                      # M3: A00 (block (0,0))
    block_U[3, 3] = 1                      # M4: A11 (block (1,1))
    block_U[4, 0] = 1; block_U[4, 1] = 1  # M5: A00 + A01 (blocks (0,0) and (0,1))
    block_U[5, 2] = 1; block_U[5, 0] = -1 # M6: A10 - A00 (blocks (1,0) and (0,0))
    block_U[6, 1] = 1; block_U[6, 3] = -1 # M7: A01 - A11 (blocks (0,1) and (1,1))
    
    # Block-level V factors (for B blocks):
    # s=0 (M1): B00 + B11 (blocks (0,0) and (1,1))
    # s=1 (M2): B00 (block (0,0))
    # s=2 (M3): B10 - B11 (blocks (1,0) and (1,1))
    # s=3 (M4): B01 - B00 (blocks (0,1) and (0,0))
    # s=4 (M5): B11 (block (1,1))
    # s=5 (M6): B00 + B10 (blocks (0,0) and (1,0))
    # s=6 (M7): B11 + B01 (blocks (1,1) and (0,1))
    
    block_V = np.zeros((7, 4), dtype=float)  # 7 block Strassen, 4 B blocks (J1,J2)
    block_V[0, 0] = 1; block_V[0, 3] = 1
    block_V[1, 0] = 1
    block_V[2, 2] = 1; block_V[2, 3] = -1
    block_V[3, 1] = 1; block_V[3, 0] = -1
    block_V[4, 3] = 1
    block_V[5, 0] = 1; block_V[5, 2] = 1
    block_V[6, 3] = 1; block_V[6, 1] = 1
    
    # Block-level W factors (for C blocks):
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    block_W = np.zeros((4, 7), dtype=float)  # 4 C blocks (K1,K2), 7 block Strassen
    block_W[0, 0] = 1; block_W[0, 3] = 1; block_W[0, 4] = -1; block_W[0, 6] = -1  # C00
    block_W[1, 2] = 1; block_W[1, 4] = 1                                           # C01
    block_W[2, 1] = 1; block_W[2, 3] = 1                                           # C10
    block_W[3, 0] = 1; block_W[3, 1] = -1; block_W[3, 2] = 1; block_W[3, 5] = 1    # C11
    
    # Now build the 4x4x4 rank-49 factors
    # U4x4: 16x49, rows (i,j) for i,j in 0..3
    # V4x4: 16x49, rows (j,k) for j,k in 0..3
    # W4x4: 16x49, rows (k,i) for k,i in 0..3 (since W[k*n+i] with n=4)
    
    U4x4 = np.zeros((16, 49), dtype=float)
    V4x4 = np.zeros((16, 49), dtype=float)
    W4x4 = np.zeros((16, 49), dtype=float)
    
    for s in range(7):  # block-level Strassen product
        for t in range(7):  # scalar-level Strassen product
            c = s * 7 + t  # column index
            
            # A coefficients: block_U[s, I1*2+I2] * U2x2[I2*2+i + j, t]
            # Wait, let me be more careful.
            # A[i,j] where i = I1*2 + ii, j = I2*2 + jj
            # The block A[I1][I2] has elements at positions (I1*2+ii, I2*2+jj) for ii,jj in {0,1}
            # U2x2[ii*2+jj, t] gives the scalar-level factor for element (ii,jj) in the block
            # So U4x4[I1*2+ii, I2*2+jj, s*7+t] = block_U[s, I1*2+I2] * U2x2[ii*2+jj, t]
            
            for I1 in range(2):
                for I2 in range(2):
                    block_coeff = block_U[s, I1*2 + I2]
                    if block_coeff == 0:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            scalar_coeff = U2x2[ii*2 + jj, t]
                            if scalar_coeff == 0:
                                continue
                            i = I1*2 + ii
                            j = I2*2 + jj
                            U4x4[i*4 + j, c] = block_coeff * scalar_coeff
            
            # B coefficients: block_V[s, J1*2+J2] * V2x2[jj*2+kk, t]
            # B[j,k] where j = J1*2 + jj, k = J2*2 + kk
            for J1 in range(2):
                for J2 in range(2):
                    block_coeff = block_V[s, J1*2 + J2]
                    if block_coeff == 0:
                        continue
                    for jj in range(2):
                        for kk in range(2):
                            scalar_coeff = V2x2[jj*2 + kk, t]
                            if scalar_coeff == 0:
                                continue
                            j = J1*2 + jj
                            k = J2*2 + kk
                            V4x4[j*4 + k, c] = block_coeff * scalar_coeff
            
            # C coefficients: block_W[K1*2+K2, s] * W2x2[kk*2+ii, t]
            # C[i,k] where i = K1*2 + ii, k = K2*2 + kk
            # W row index: k*4 + i
            for K1 in range(2):
                for K2 in range(2):
                    block_coeff = block_W[K1*2 + K2, s]
                    if block_coeff == 0:
                        continue
                    for ii in range(2):
                        for kk in range(2):
                            scalar_coeff = W2x2[kk*2 + ii, t]
                            if scalar_coeff == 0:
                                continue
                            i = K1*2 + ii
                            k = K2*2 + kk
                            W4x4[k*4 + i, c] = block_coeff * scalar_coeff
    
    # Now extend to 4x4x8:
    # 4x4x8 = two independent 4x4x4 multiplications
    # B is 4x8, split into B[:, 0:4] and B[:, 4:8]
    # C is 4x8, split into C[:, 0:4] and C[:, 4:8]
    
    # First half: use U4x4, V4x4 (for B[:, 0:4]), W4x4 (for C[:, 0:4])
    # Second half: use U4x4, V4x4 (for B[:, 4:8]), W4x4 (for C[:, 4:8])
    
    # U is the same for both halves (A is the same)
    for c in range(49):
        U[:, c] = U4x4[:, c]
        U[:, c + 49] = U4x4[:, c]
    
    # V: first half uses V4x4 for B[:, 0:4], second half uses V4x4 for B[:, 4:8]
    # V rows: (j, k) where j in 0..3, k in 0..7
    # For first half (k=0..3): V4x4[j*4+k, c]
    # For second half (k=4..7): V4x4[j*4+(k-4), c+49]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*8 + k, c] = V4x4[j*4 + k, c]
                V[j*8 + (k + 4), c + 49] = V4x4[j*4 + k, c + 49]
    
    # W: first half for C[:, 0:4], second half for C[:, 4:8]
    # W rows: (k, i) where k in 0..7, i in 0..3
    # For first half (k=0..3): W4x4[k*4+i, c]
    # For second half (k=4..7): W4x4[(k-4)*4+i, c+49]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4 + i, c] = W4x4[k*4 + i, c]
                W[(k + 4)*4 + i, c + 49] = W4x4[k*4 + i, c + 49]
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((n*m, r), dtype=float)
        V = np.zeros((m*p, r), dtype=float)
        W = np.zeros((p*n, r), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

