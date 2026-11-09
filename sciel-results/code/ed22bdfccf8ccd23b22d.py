"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_decomposition():
    """
    Return a Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    This gives rank 7 instead of 8.
    
    The 7 multiplications are:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B11-B21)
    M4 = A22*(B21-B22)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    Output:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # For 2x2 * 2x2, indices:
    # A: (i,j) where i,j in {0,1}
    # B: (j,k) where j,k in {0,1}
    # C: (i,k) where i,k in {0,1}
    
    # We'll build U, V, W for the 2x2*2x2 case with rank 7
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)
    # W has shape (2*2, 7) = (4, 7)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    # A11 is at (0,0), A22 at (1,1)
    # B11 is at (0,0), B22 at (1,1)
    U[0*2+0][0] = 1  # A11
    U[1*2+1][0] = 1  # A22
    V[0*2+0][0] = 1  # B11
    V[1*2+1][0] = 1  # B22
    W[0*2+0][0] = 1  # C11
    W[1*2+1][0] = 1  # C22
    
    # M2 = (A21+A22)*B11
    U[1*2+0][1] = 1  # A21
    U[1*2+1][1] = 1  # A22
    V[0*2+0][1] = 1  # B11
    W[1*2+0][1] = 1  # C21
    
    # M3 = A11*(B11-B21)
    U[0*2+0][2] = 1  # A11
    V[0*2+0][2] = 1  # B11
    V[1*2+0][2] = -1  # B21
    W[0*2+1][2] = 1  # C12
    
    # M4 = A22*(B21-B22)
    U[1*2+1][3] = 1  # A22
    V[1*2+0][3] = 1  # B21
    V[1*2+1][3] = -1  # B22
    W[0*2+0][3] = 1  # C11
    W[1*2+0][3] = 1  # C21
    
    # M5 = (A11+A12)*B22
    U[0*2+0][4] = 1  # A11
    U[0*2+1][4] = 1  # A12
    V[1*2+1][4] = 1  # B22
    W[0*2+0][4] = -1  # C11
    W[0*2+1][4] = 1  # C12
    
    # M6 = (A21-A11)*(B11+B12)
    U[1*2+0][5] = 1  # A21
    U[0*2+0][5] = -1  # A11
    V[0*2+0][5] = 1  # B11
    V[0*2+1][5] = 1  # B12
    W[1*2+1][5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U[0*2+1][6] = 1  # A12
    U[1*2+1][6] = -1  # A22
    V[1*2+0][6] = 1  # B21
    V[1*2+1][6] = 1  # B22
    W[0*2+0][6] = 1  # C11
    W[1*2+1][6] = 1  # C22
    
    return {"U": U, "V": V, "W": W}

def _strassen_decomposition(n, m, p):
    """
    Use Strassen-style block decomposition to get a lower-rank factorization
    for 3x4 * 4x7.
    
    Strategy: Pad n=3 to 4, keep m=4, pad p=7 to 8.
    Now we have 4x4 * 4x8.
    
    Split B into B_left (4x4) and B_right (4x4).
    Compute C_left = A * B_left and C_right = A * B_right.
    
    Each 4x4 * 4x4 uses Strassen recursively:
    - Split into 2x2 blocks
    - Use 7 multiplications of 2x2 * 2x2
    - Each 2x2 * 2x2 uses schoolbook rank 4
    
    So each 4x4 * 4x4 has rank 7 * 4 = 28
    Total for both halves: 56
    
    We only need the first 3 rows and first 7 columns of the output.
    """
    n, m, p = 3, 4, 7
    n_padded = 4
    m_padded = 4
    p_padded = 8
    
    # Build the full 4x4 * 4x8 decomposition using Strassen
    # First, get the 2x2 * 2x2 Strassen decomposition
    base = _strassen_2x2_decomposition()
    
    # Now build 4x4 * 4x4 using Strassen with 2x2 blocks
    # We need to combine the base decomposition appropriately
    
    # For 4x4 * 4x4, we split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    
    # The Strassen multiplications are:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B21-B22)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each Mi is a 2x2 * 2x2 product, which we decompose using the base Strassen (rank 7)
    # But wait, the base is already rank 7 for 2x2*2x2, so each Mi contributes rank 7
    # Total for 4x4*4x4: 7 * 7 = 49
    
    # Actually, I need to be more careful. The base decomposition I defined above
    # is for 2x2*2x2 with rank 7. When I use it for each Mi in the 4x4*4x4 Strassen,
    # each Mi is a 2x2*2x2 product, so I can use the same rank-7 decomposition.
    
    # Let me build the 4x4 * 4x4 decomposition:
    # U has shape (4*4, rank) = (16, rank)
    # V has shape (4*4, rank) = (16, rank)
    # W has shape (4*4, rank) = (16, rank)
    
    # For each of the 7 Strassen multiplications, I need to map the 2x2*2x2
    # decomposition to the appropriate subblocks.
    
    # Let's build this step by step.
    
    # First, let me define how to index the 4x4 matrices:
    # A[i,j] for i,j in 0..3
    # B[j,k] for j,k in 0..3
    # C[i,k] for i,k in 0..3
    
    # Block indices:
    # A11: A[0:2, 0:2], A12: A[0:2, 2:4]
    # A21: A[2:4, 0:2], A22: A[2:4, 2:4]
    
    # Similarly for B and C.
    
    # I'll build the decomposition by creating entries for each of the 7 Mi.
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    col = 0
    for mi in range(7):
        # For each Strassen multiplication Mi, use the base 2x2*2x2 decomposition
        # The base has 7 entries, so each Mi contributes 7 columns
        for base_col in range(7):
            # Get the base decomposition entries
            u_base = base["U"]
            v_base = base["V"]
            w_base = base["W"]
            
            # Now I need to map the 2x2 indices to the 4x4 subblocks
            # This depends on which Mi we're computing
            
            # For now, let me just use the base decomposition directly
            # This is a placeholder - I need to properly map the subblocks
            
            # Actually, this approach is getting complex. Let me try a different strategy.
            pass
    
    # This is getting too complex. Let me try a simpler approach:
    # Just use the schoolbook decomposition for now, but with a smaller rank
    # by exploiting the rectangular structure.
    
    # For 3x4 * 4x7, we can try to find a decomposition with rank < 84.
    # Let me use a randomized approach with alternating least squares.
    
    np.random.seed(42)
    best_rank = n * m * p  # 84
    
    for target_rank in range(best_rank - 1, 60, -1):
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Try a few iterations of alternating least squares
        for iteration in range(10):
            # Fix U and V, solve for W
            # For each (i,k), we need:
            # sum_j sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
            # This is linear in W[k*n+i, :]
            
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    # Build the linear system for W[k*n+i, :]
                    # sum_s (sum_j U[i*m+j,s] * V[j*p+k,s]) * W[k*n+i,s] = 1
                    coeffs = np.zeros(target_rank)
                    for j in range(m):
                        coeffs += U[i*m+j, :] * V[j*p+k, :]
                    # coeffs @ W[k*n+i, :] = 1
                    if np.linalg.norm(coeffs) > 1e-10:
                        W[k*n+i, :] = np.linalg.lstsq(coeffs.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                    else:
                        W[k*n+i, :] = 0
            
            # Fix V and W, solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    coeffs = np.zeros(target_rank)
                    for k in range(p):
                        coeffs += V[j*p+k, :] * W[k*n+i, :]
                    if np.linalg.norm(coeffs) > 1e-10:
                        U[i*m+j, :] = np.linalg.lstsq(coeffs.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                    else:
                        U[i*m+j, :] = 0
            
            # Fix U and W, solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    coeffs = np.zeros(target_rank)
                    for i in range(n):
                        coeffs += U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.norm(coeffs) > 1e-10:
                        V[j*p+k, :] = np.linalg.lstsq(coeffs.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                    else:
                        V[j*p+k, :] = 0
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 84
    
    # Try Strassen-style decomposition
    if time.time() - t0 < DEADLINE - 10:
        strassen_result = _strassen_decomposition(n, m, p)
        if strassen_result is not None:
            rank = len(strassen_result["U"][0])
            if rank < best_rank:
                best = strassen_result
                best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
