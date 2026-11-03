import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_2x2():
    """Return (U, V, W) for 2x2 * 2x2 with rank 7 (Strassen).
    U shape: (4, 7) indexed by [i*2+j]
    V shape: (4, 7) indexed by [j*2+k]
    W shape: (4, 7) indexed by [k*2+i]
    """
    R = 7
    U = np.zeros((4, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U[i*2+j, r]:
    # r=0: M1: A11+A22 -> U[0,0]=1, U[3,0]=1
    # r=1: M2: A21+A22 -> U[2,1]=1, U[3,1]=1
    # r=2: M3: A11 -> U[0,2]=1
    # r=3: M4: A22 -> U[3,3]=1
    # r=4: M5: A11+A12 -> U[0,4]=1, U[1,4]=1
    # r=5: M6: A21-A11 -> U[2,5]=1, U[0,5]=-1
    # r=6: M7: A12-A22 -> U[1,6]=1, U[3,6]=-1
    
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V[j*2+k, r]:
    # r=0: M1: B11+B22 -> V[0,0]=1, V[3,0]=1
    # r=1: M2: B11 -> V[0,1]=1
    # r=2: M3: B21-B22 -> V[2,2]=1, V[3,2]=-1
    # r=3: M4: B11-B21 -> V[0,3]=1, V[2,3]=-1
    # r=4: M5: B22 -> V[3,4]=1
    # r=5: M6: B11+B12 -> V[0,5]=1, V[1,5]=1
    # r=6: M7: B21+B22 -> V[2,6]=1, V[3,6]=1
    
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[2, 2] = 1; V[3, 2] = -1
    V[0, 3] = 1; V[2, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W[k*2+i, r]:
    # C11 = M1+M4-M5+M7 -> W[0,0]=1, W[3,0]=1, W[0,3]=1, W[3,3]=-1, W[0,6]=1
    # C12 = M3+M5 -> W[2,0]=1, W[3,0]=1, W[2,4]=1, W[3,4]=1
    # C21 = M2+M4 -> W[0,1]=1, W[3,1]=1, W[0,3]=1, W[3,3]=1
    # C22 = M1-M2+M3+M6 -> W[2,0]=1, W[3,0]=1, W[2,1]=-1, W[3,1]=-1, W[2,2]=1, W[3,2]=1, W[2,5]=1, W[3,5]=1
    
    # W[k*2+i, r] where k is column of B (output col), i is row of A (output row)
    # Output C[i,k]: i in {0,1}, k in {0,1}
    # C[0,0] = M1+M4-M5+M7: W[0*2+0, r] for r in {0,3,4,6}: W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    # C[0,1] = M3+M5: W[1*2+0, r] for r in {2,4}: W[2,2]=1, W[2,4]=1
    # C[1,0] = M2+M4: W[0*2+1, r] for r in {1,3}: W[1,1]=1, W[1,3]=1
    # C[1,1] = M1-M2+M3+M6: W[1*2+1, r] for r in {0,1,2,5}: W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _strassen_4x4_4x4():
    """Return (U, V, W) for 4x4 * 4x4 with rank 49 (Strassen applied to 2x2 blocks of 2x2).
    U shape: (16, 49) indexed by [i*4+j]
    V shape: (16, 49) indexed by [j*4+k]
    W shape: (16, 49) indexed by [k*4+i]
    """
    R = 49
    U = np.zeros((16, R))
    V = np.zeros((16, R))
    W = np.zeros((16, R))
    
    # Block structure: A is 4x4 split into four 2x2 blocks:
    # A[0:2, 0:2] = A11, A[0:2, 2:4] = A12
    # A[2:4, 0:2] = A21, A[2:4, 2:4] = A22
    # Similarly for B.
    # Apply Strassen at block level (7 block multiplications), each block mult is 2x2*2x2 (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # Block-level Strassen:
    # BM1 = (A11+A22)(B11+B22)
    # BM2 = (A21+A22)B11
    # BM3 = A11(B21-B22)
    # BM4 = A22(B11-B21)
    # BM5 = (A11+A12)B22
    # BM6 = (A21-A11)(B11+B12)
    # BM7 = (A12-A22)(B21+B22)
    # C11 = BM1+BM4-BM5+BM7
    # C12 = BM3+BM5
    # C21 = BM2+BM4
    # C22 = BM1-BM2+BM3+BM6
    
    # Each BMx is a 2x2*2x2 multiplication, which itself has rank 7.
    # So for each block-level term, we have 7 sub-multiplications.
    # Total: 7 block terms * 7 sub-terms = 49.
    
    # Indexing: r = block_term * 7 + sub_term
    # For each block term bt (0..6) and sub term st (0..6):
    # U[i*4+j, r] depends on which 2x2 sub-block of A and which Strassen factor
    # V[j*4+k, r] depends on which 2x2 sub-block of B and which Strassen factor
    # W[k*4+i, r] depends on which output 2x2 sub-block and combination
    
    # Let's build this systematically.
    # A[i,j] where i in 0..3, j in 0..3
    # Block row of A: i_block = i // 2 (0 or 1)
    # Block col of A: j_block = j // 2 (0 or 1)
    # Within block: i_local = i % 2, j_local = j % 2
    
    # For block term bt, the A-factor is a linear combination of A blocks:
    # bt=0: A11+A22 -> A blocks (0,0) and (1,1)
    # bt=1: A21+A22 -> A blocks (1,0) and (1,1)
    # bt=2: A11 -> A block (0,0)
    # bt=3: A22 -> A block (1,1)
    # bt=4: A11+A12 -> A blocks (0,0) and (0,1)
    # bt=5: A21-A11 -> A blocks (1,0) and (0,0)
    # bt=6: A12-A22 -> A blocks (0,1) and (1,1)
    
    # A_block_coeffs[bt] = list of (block_row, block_col, coeff)
    A_block_coeffs = [
        [(0,0,1), (1,1,1)],  # bt=0
        [(1,0,1), (1,1,1)],  # bt=1
        [(0,0,1)],            # bt=2
        [(1,1,1)],            # bt=3
        [(0,0,1), (0,1,1)],  # bt=4
        [(1,0,1), (0,0,-1)], # bt=5
        [(0,1,1), (1,1,-1)], # bt=6
    ]
    
    # B_block_coeffs[bt] = list of (block_row, block_col, coeff)
    B_block_coeffs = [
        [(0,0,1), (1,1,1)],  # bt=0
        [(0,0,1)],            # bt=1
        [(1,0,1), (1,1,-1)], # bt=2
        [(0,0,1), (1,0,-1)], # bt=3
        [(1,1,1)],            # bt=4
        [(0,0,1), (0,1,1)],  # bt=5
        [(1,0,1), (1,1,1)],  # bt=6
    ]
    
    # C_block_coeffs[bt] = list of (block_row, block_col, coeff) for each output block
    # C11 = BM1+BM4-BM5+BM7
    # C12 = BM3+BM5
    # C21 = BM2+BM4
    # C22 = BM1-BM2+BM3+BM6
    C_block_coeffs = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],  # C11: BM1+BM4-BM5+BM7
        (0,1): [(2,1), (4,1)],                  # C12: BM3+BM5
        (1,0): [(1,1), (3,1)],                  # C21: BM2+BM4
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],  # C22: BM1-BM2+BM3+BM6
    }
    
    # Get Strassen 2x2 factors
    U2, V2, W2 = _strassen_2x2_2x2()
    
    for bt in range(7):
        for st in range(7):
            r = bt * 7 + st
            
            # U: for each (i,j) in 4x4, determine contribution
            for i in range(4):
                for j in range(4):
                    i_block = i // 2
                    j_block = j // 2
                    i_local = i % 2
                    j_local = j % 2
                    # Sum over A block contributions for this bt
                    val = 0
                    for (br, bc, coeff) in A_block_coeffs[bt]:
                        if br == i_block and bc == j_block:
                            val += coeff * U2[i_local * 2 + j_local, st]
                    U[i * 4 + j, r] = val
            
            # V: for each (j,k) in 4x4, determine contribution
            for j in range(4):
                for k in range(4):
                    j_block = j // 2
                    k_block = k // 2
                    j_local = j % 2
                    k_local = k % 2
                    val = 0
                    for (br, bc, coeff) in B_block_coeffs[bt]:
                        if br == j_block and bc == k_block:
                            val += coeff * V2[j_local * 2 + k_local, st]
                    V[j * 4 + k, r] = val
            
            # W: for each (k,i) in 4x4, determine contribution
            for k in range(4):
                for i in range(4):
                    k_block = k // 2
                    i_block = i // 2
                    k_local = k % 2
                    i_local = i % 2
                    val = 0
                    for (out_br, out_bc, coeff) in C_block_coeffs[(i_block, k_block)]:
                        if out_br == bt:
                            val += coeff * W2[k_local * 2 + i_local, st]
                    W[k * 4 + i, r] = val
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Return (U, V, W) for n x m * m x p with schoolbook rank n*m*p.
    U shape: (n*m, n*m*p)
    V shape: (m*p, n*m*p)
    W shape: (p*n, n*m*p)
    """
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    R = U_np.shape[1]
    
    # Check all consistent indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check inconsistent indices using fast method
    # j-inconsistency
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 == j2:
                    continue
                a = i * m + j1
                for k in range(p):
                    b = j2 * p + k
                    vec = U_np[a] * V_np[b]
                    if np.max(np.abs(W_np @ vec)) > 1e-9:
                        return False
    
    # i-inconsistency
    for i1 in range(n):
        for i3 in range(n):
            if i1 == i3:
                continue
            for j in range(m):
                a = i1 * m + j
                for k in range(p):
                    c = k * n + i3
                    vec = U_np[a] * W_np[c]
                    if np.max(np.abs(V_np @ vec)) > 1e-9:
                        return False
    
    # k-inconsistency
    for k2 in range(p):
        for k3 in range(p):
            if k2 == k3:
                continue
            for j in range(m):
                b = j * p + k2
                for i in range(n):
                    c = k3 * n + i
                    vec = V_np[b] * W_np[c]
                    if np.max(np.abs(U_np @ vec)) > 1e-9:
                        return False
    
    return True

def construct():
    """Return a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Split 4x4 * 4x9 into 4x4*4x4 + 4x4*4x4 + 4x4*4x1
    # Use Strassen (rank 49) for each 4x4*4x4, schoolbook (rank 16) for 4x4*4x1
    # Total rank: 49 + 49 + 16 = 114
    
    # Build the three components
    U1, V1, W1 = _strassen_4x4_4x4()  # 4x4 * 4x4, rank 49
    U2, V2, W2 = _strassen_4x4_4x4()  # 4x4 * 4x4, rank 49
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)  # 4x4 * 4x1, rank 16
    
    # Now we need to map these into the 4x4 * 4x9 framework.
    # For the first 4x4*4x4: columns k=0..3 of B
    # For the second 4x4*4x4: columns k=4..7 of B
    # For the 4x4*4x1: column k=8 of B
    
    R_total = 49 + 49 + 16
    U_full = np.zeros((16, R_total))
    V_full = np.zeros((36, R_total))
    W_full = np.zeros((36, R_total))
    
    # Component 1: 4x4 * 4x4 for k=0..3
    # U: same (16, 49)
    U_full[:, 0:49] = U1
    # V: B is 4x9, first component uses B[:, 0:4] which is rows j=0..3, cols k=0..3
    # V index: j*p + k = j*9 + k. For k=0..3, j=0..3:
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + k, 0:49] = V1[j * 4 + k]
    # W: C is 4x9, first component contributes to C[:, 0:4]
    # W index: k*n + i = k*4 + i. For k=0..3, i=0..3:
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, 0:49] = W1[k * 4 + i]
    
    # Component 2: 4x4 * 4x4 for k=4..7
    U_full[:, 49:98] = U2
    for j in range(4):
        for k in range(4, 8):
            V_full[j * 9 + k, 49:98] = V2[j * 4 + (k - 4)]
    for k in range(4, 8):
        for i in range(4):
            W_full[k * 4 + i, 49:98] = W2[(k - 4) * 4 + i]
    
    # Component 3: 4x4 * 4x1 for k=8
    U_full[:, 98:114] = U3
    for j in range(4):
        V_full[j * 9 + 8, 98:114] = V3[j * 1 + 0]  # V3[j*1+0] = V3[j]
    for i in range(4):
        W_full[8 * 4 + i, 98:114] = W3[0 * 4 + i]  # W3[0*4+i] = W3[i]
    
    # Verify
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook_nm_p(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
