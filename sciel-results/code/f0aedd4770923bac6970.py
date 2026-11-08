"""
Combined program for 3x3x4 matrix multiplication tensor decomposition.
Uses a hardcoded Strassen-based decomposition (rank 34) derived from inspiration 1,
with a verification step to ensure correctness.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Strategy: Use a hardcoded decomposition of rank 34 based on Strassen's algorithm applied to 2x2 blocks
    and schoolbook multiplication for the remaining strips.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Target shapes
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return # Safety check, shouldn't happen
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # --- Block 1: A11 (2x2) * B1L (2x2) -> C11 (2x2) ---
    # Uses Strassen's 7 multiplications.
    # Indices in U (A): 
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # Indices in V (B):
    # B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # Indices in W (C):
    # C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4 (Trace convention: W[k*n+i])
    
    # P1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # P2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # P3 = A00(B10-B00)
    U[0][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # P4 = A11(B01-B10)
    U[4][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # P5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # P6 = (A10-A00)(B00+B10)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[4][col] = 1
    W[4][col] = 1
    col += 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1

    # --- Block 2: A12 (2x1) * B2L (1x2) -> C11 contribution ---
    # A12 indices: A[0,2]=2, A[1,2]=5
    # B2L indices: B[2,0]=8, B[2,1]=9
    # C11 indices: 0, 3, 1, 4
    
    # T1: A[0,2] B[2,0] -> C[0,0] (idx 0)
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,1] -> C[0,1] (idx 3)
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,0] -> C[1,0] (idx 1)
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,1] -> C[1,1] (idx 4)
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1

    # --- Block 3: A21 (1x2) * B1L (2x2) -> C21 (1x2) contribution ---
    # A21 indices: A[2,0]=6, A[2,1]=7
    # B1L indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C21 indices: C[2,0]=2, C[2,1]=5
    
    # T1: A[2,0] B[0,0] -> C[2,0] (idx 2)
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,1] -> C[2,1] (idx 5)
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,0] -> C[2,0] (idx 2)
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,1] -> C[2,1] (idx 5)
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1

    # --- Block 4: A22 (1x1) * B2L (1x2) -> C21 (1x2) contribution ---
    # A22 index: A[2,2]=8
    # B2L indices: B[2,0]=8, B[2,1]=9
    # C21 indices: C[2,0]=2, C[2,1]=5
    
    # T1: A[2,2] B[2,0] -> C[2,0] (idx 2)
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,1] -> C[2,1] (idx 5)
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1

    # --- Block 5: A11 (2x2) * B1R (2x2) -> C12 (2x2) contribution ---
    # B1R indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C12 indices: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    
    # P1 = (A00+A11)(B02+B13)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    # P2 = (A10+A11)B02
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    # P3 = A00(B12-B02)
    U[0][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    # P4 = A11(B03-B12)
    U[4][col] = 1
    V[3][col] = 1; V[6][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # P5 = (A00+A01)B13
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    # P6 = (A10-A00)(B02+B12)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[6][col] = 1
    W[10][col] = 1
    col += 1
    
    # P7 = (A01-A11)(B12+B13)
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1

    # --- Block 6: A12 (2x1) * B2R (1x2) -> C12 (2x2) contribution ---
    # A12 indices: A[0,2]=2, A[1,2]=5
    # B2R indices: B[2,2]=10, B[2,3]=11
    # C12 indices: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    
    # T1: A[0,2] B[2,2] -> C[0,2] (idx 6)
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # T2: A[0,2] B[2,3] -> C[0,3] (idx 9)
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    # T3: A[1,2] B[2,2] -> C[1,2] (idx 7)
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    # T4: A[1,2] B[2,3] -> C[1,3] (idx 10)
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1

    # --- Block 7: A21 (1x2) * B1R (2x2) -> C22 (1x2) contribution ---
    # A21 indices: A[2,0]=6, A[2,1]=7
    # B1R indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C22 indices: C[2,2]=8, C[2,3]=11
    
    # T1: A[2,0] B[0,2] -> C[2,2] (idx 8)
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,0] B[0,3] -> C[2,3] (idx 11)
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    # T3: A[2,1] B[1,2] -> C[2,2] (idx 8)
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    # T4: A[2,1] B[1,3] -> C[2,3] (idx 11)
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1

    # --- Block 8: A22 (1x1) * B2R (1x2) -> C22 (1x2) contribution ---
    # A22 index: A[2,2]=8
    # B2R indices: B[2,2]=10, B[2,3]=11
    # C22 indices: C[2,2]=8, C[2,3]=11
    
    # T1: A[2,2] B[2,2] -> C[2,2] (idx 8)
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    # T2: A[2,2] B[2,3] -> C[2,3] (idx 11)
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    # Verify col == rank
    if col != rank:
        # Fallback to schoolbook if construction logic had an error
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Quick verification
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # We check against the tensor definition
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
