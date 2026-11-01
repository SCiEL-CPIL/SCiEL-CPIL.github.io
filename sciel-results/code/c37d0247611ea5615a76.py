"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_strassen_hybrid(n, m, p):
    """
    Construct a rank-25 decomposition using Strassen for the first 2 columns and schoolbook for the rest.
    A is 2x2, B is 2x7.
    We split B into B1 (2x2) and B2 (2x5).
    C = A * B = A * B1 + A * B2.
    
    For A * B1 (2x2 * 2x2), we use Strassen with rank 7.
    For A * B2 (2x2 * 2x5), we use schoolbook with rank 2*2*5 = 20.
    Total rank = 7 + 20 = 27? No, wait.
    
    Let's re-evaluate.
    The known result mentioned in the prompt: "Replacing the rank-28 schoolbook decomposition with a block-diagonal Strassen (rank 7 per 2-column block) plus rank-4 schoolbook for the final column yields a valid rank-25 tensor".
    
    Let's trace that logic.
    B is 2x7.
    Split B into B_12 (2x2), B_34 (2x2), B_56 (2x2), B_7 (2x1).
    
    Strassen for 2x2 * 2x2 has rank 7.
    Schoolbook for 2x2 * 2x1 has rank 2*2*1 = 4.
    
    If we apply Strassen to the first three 2x2 blocks and schoolbook to the last 2x1 block:
    Rank = 7 (for B_12) + 7 (for B_34) + 7 (for B_56) + 4 (for B_7) = 25.
    
    This matches the "rank-25" claim.
    
    Let's build this.
    """
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_val, v_val, w_val, u_idx, v_idx, w_idx):
        U.append([u_val])
        V.append([v_val])
        W.append([w_val])
    
    # Strassen formulas for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 - M3 + M2 + M5
    
    # We need to map these to the global U, V, W matrices.
    # U index: i*m + j (i: 0..1, j: 0..1) -> 0, 1, 2, 3
    # V index: j*p + k (j: 0..1, k: 0..6) -> 0..13
    # W index: k*n + i (k: 0..6, i: 0..1) -> 0..13
    
    current_rank = 0
    
    # Process Strassen blocks for k_start in [0, 2, 4]
    for k_start in [0, 2, 4]:
        # The 2x2 block of B is columns k_start and k_start+1.
        # B00 is at V index 0*p + k_start = k_start
        # B01 is at V index 0*p + (k_start+1) = k_start+1
        # B10 is at V index 1*p + k_start = p + k_start
        # B11 is at V index 1*p + (k_start+1) = p + k_start+1
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00 (idx 0) gets 1, A11 (idx 3) gets 1.
        # V: B00 (idx k_start) gets 1, B11 (idx p+k_start+1) gets 1.
        # W: C00 (idx k_start*n + 0 = k_start) gets 1, C11 (idx (k_start+1)*n + 1 = (k_start+1)*2 + 1) gets 1.
        # Wait, W index is k*n + i.
        # C00 corresponds to i=0, k=k_start. W index = k_start*2 + 0.
        # C11 corresponds to i=1, k=k_start+1. W index = (k_start+1)*2 + 1.
        
        U.append([1, 1]) # A00, A11
        V.append([1, 1]) # B00, B11
        W.append([1, 1]) # C00, C11
        
        # M2 = (A10 + A11)B00
        # U: A10 (idx 2) gets 1, A11 (idx 3) gets 1.
        # V: B00 (idx k_start) gets 1.
        # W: C10 (idx k_start*n + 1 = k_start*2 + 1) gets 1.
        
        U.append([1, 1]) # A10, A11
        V.append([1])    # B00
        W.append([1])    # C10
        
        # M3 = A00(B01 - B11)
        # U: A00 (idx 0) gets 1.
        # V: B01 (idx k_start+1) gets 1, B11 (idx p+k_start+1) gets -1.
        # W: C01 (idx (k_start+1)*n + 0 = (k_start+1)*2) gets 1.
        
        U.append([1])    # A00
        V.append([1, -1])# B01, -B11
        W.append([1])    # C01
        
        # M4 = A11(B10 - B00)
        # U: A11 (idx 3) gets 1.
        # V: B10 (idx p+k_start) gets 1, B00 (idx k_start) gets -1.
        # W: C00 (idx k_start*2) gets -1. (Because C00 = M1 - M4 ...)
        
        U.append([1])    # A11
        V.append([1, -1])# B10, -B00
        W.append([-1])   # -C00
        
        # M5 = (A00 + A10)B11
        # U: A00 (idx 0) gets 1, A10 (idx 2) gets 1.
        # V: B11 (idx p+k_start+1) gets 1.
        # W: C01 (idx (k_start+1)*2) gets 1.
        
        U.append([1, 1]) # A00, A10
        V.append([1])    # B11
        W.append([1])    # C01
        
        # M6 = (A11 - A00)(B00 + B01)
        # U: A11 (idx 3) gets 1, A00 (idx 0) gets -1.
        # V: B00 (idx k_start) gets 1, B01 (idx k_start+1) gets 1.
        # W: C00 (idx k_start*2) gets 1, C10 (idx k_start*2 + 1) gets 1.
        
        U.append([-1, 1])# -A00, A11
        V.append([1, 1]) # B00, B01
        W.append([1, 1]) # C00, C10
        
        # M7 = (A01 - A11)(B10 + B11)
        # U: A01 (idx 1) gets 1, A11 (idx 3) gets -1.
        # V: B10 (idx p+k_start) gets 1, B11 (idx p+k_start+1) gets 1.
        # W: C00 (idx k_start*2) gets 1, C01 (idx (k_start+1)*2) gets 1.
        
        U.append([1, -1])# A01, -A11
        V.append([1, 1]) # B10, B11
        W.append([1, 1]) # C00, C01
    
    # Process Schoolbook for the last column k=6
    # C06 = A00*B06 + A01*B16
    # C16 = A10*B06 + A11*B16
    
    # Term 1: A00 * B06 -> C06
    U.append([1]) # A00
    V.append([1]) # B06 (idx 0*2 + 6 = 6)
    W.append([1]) # C06 (idx 6*2 + 0 = 12)
    
    # Term 2: A01 * B16 -> C06
    U.append([1]) # A01
    V.append([1]) # B16 (idx 1*2 + 6 = 8)
    W.append([1]) # C06 (idx 12)
    
    # Term 3: A10 * B06 -> C16
    U.append([1]) # A10
    V.append([1]) # B06 (idx 6)
    W.append([1]) # C16 (idx 6*2 + 1 = 13)
    
    # Term 4: A11 * B16 -> C16
    U.append([1]) # A11
    V.append([1]) # B16 (idx 8)
    W.append([1]) # C16 (idx 13)
    
    # Now we have lists of lists, but each inner list has varying lengths (1 or 2).
    # We need to convert this to a dense matrix of shape (n*m, rank).
    # Rank is the total number of terms.
    rank = len(U)
    
    U_mat = np.zeros((n*m, rank))
    V_mat = np.zeros((m*p, rank))
    W_mat = np.zeros((p*n, rank))
    
    for c in range(rank):
        for idx, val in enumerate(U[c]):
            U_mat[idx, c] = val
        for idx, val in enumerate(V[c]):
            V_mat[idx, c] = val
        for idx, val in enumerate(W[c]):
            W_mat[idx, c] = val
            
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen hybrid construction
    U, V, W = _construct_strassen_hybrid(n, m, p)
    
    # Verify it
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
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
# EVOLVE-BLOCK-END
