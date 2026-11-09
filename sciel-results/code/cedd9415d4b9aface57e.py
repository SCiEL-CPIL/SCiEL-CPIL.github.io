"""Constructive Strassen-based decomposition for 3x4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    rank = U.shape[1]
    result = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(result, T, atol=tol)

def _build_strassen_2x4x4():
    """
    Build Strassen decomposition for 2x4x4 matrix multiplication.
    Split A (2x4), B (4x4) into 2x2 blocks.
    A = [[A11, A12], [A21, A22]] where each block is 1x2
    B = [[B11, B12], [B21, B22]] where each block is 2x2
    
    Strassen for 2x2 blocks (each 1x2 * 2x2 = 1x2 result):
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Indexing for 2x4x4:
    # Input A: rows i in {0,1}, cols j in {0,1,2,3} -> index i*4+j
    #   A11 = A[0,0:2] (row 0, cols 0-1)
    #   A12 = A[0,2:4] (row 0, cols 2-3)
    #   A21 = A[1,0:2] (row 1, cols 0-1)
    #   A22 = A[1,2:4] (row 1, cols 2-3)
    # Input B: rows j in {0,1,2,3}, cols k in {0,1,2,3} -> index j*4+k
    #   B11 = B[0:2, 0:2] (rows 0-1, cols 0-1)
    #   B12 = B[0:2, 2:4] (rows 0-1, cols 2-3)
    #   B21 = B[2:4, 0:2] (rows 2-3, cols 0-1)
    #   B22 = B[2:4, 2:4] (rows 2-3, cols 2-3)
    # Output C: rows i in {0,1}, cols k in {0,1,2,3} -> index k*2+i
    #   C11 = C[0,0:2] (row 0, cols 0-1)
    #   C12 = C[0,2:4] (row 0, cols 2-3)
    #   C21 = C[1,0:2] (row 1, cols 0-1)
    #   C22 = C[1,2:4] (row 1, cols 2-3)
    
    # Each Strassen term is a 1x2 * 2x2 = 1x2 multiplication
    # For a 1x2 * 2x2 multiplication, schoolbook rank is 4
    # But we can use rank-3 for 1x2x2 (since min(1,2,2)=1, actually rank is 4 for 1x2x2)
    # Wait: 1x2x2 tensor has rank 4 (schoolbook is optimal for 1xnxn when n>=2? No)
    # Actually 1x2x2: C[0,k] = sum_j A[0,j]*B[j,k], that's 4 multiplications, rank 4.
    # So each of the 7 Strassen terms needs rank 4 -> total 28.
    
    # Let's build it explicitly.
    # For each Strassen term M_s = X * Y where X is 1x2, Y is 2x2:
    #   M_s[0,k] = sum_j X[0,j] * Y[j,k]
    #   This contributes to C with appropriate signs and placements.
    
    rank = 28
    U = np.zeros((8, rank))  # 2*4 = 8 input rows
    V = np.zeros((16, rank))  # 4*4 = 16 input rows
    W = np.zeros((8, rank))  # 4*2 = 8 output rows
    
    col = 0
    for s in range(7):
        # Define X (1x2) and Y (2x2) for this Strassen term
        # X is a linear combination of A blocks
        # Y is a linear combination of B blocks
        # We need to express X[j] = sum_{i,a} x_{i,a} * A[i,a] for j=0,1
        # and Y[j,k] = sum_{b,c} y_{j,k,b,c} * B[b,c]
        
        # For term s, define which A elements and B elements are used
        # A indices: (i, a) where i in {0,1}, a in {0,1,2,3}
        #   A11: i=0, a in {0,1}
        #   A12: i=0, a in {2,3}
        #   A21: i=1, a in {0,1}
        #   A22: i=1, a in {2,3}
        # B indices: (b, c) where b in {0,1,2,3}, c in {0,1,2,3}
        #   B11: b in {0,1}, c in {0,1}
        #   B12: b in {0,1}, c in {2,3}
        #   B21: b in {2,3}, c in {0,1}
        #   B22: b in {2,3}, c in {2,3}
        
        # Strassen terms:
        if s == 0:  # M1 = (A11 + A22)(B11 + B22)
            X_elems = [(0,0,1),(0,1,1),(1,2,1),(1,3,1)]  # (i, a, coeff)
            Y_elems = [(0,0,1),(0,1,1),(1,0,1),(1,1,1),(2,0,1),(2,1,1),(3,0,1),(3,1,1)]
            C_target = [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]  # (i, k, coeff) -> C11 + C22 parts
            # M1 contributes to C11 with +1 and C22 with +1
            # C11: i=0, k in {0,1}; C22: i=1, k in {2,3}
            C_target = [(0,0,1),(0,1,1),(1,2,1),(1,3,1)]
        elif s == 1:  # M2 = (A21 + A22)B11
            X_elems = [(1,0,1),(1,1,1),(1,2,1),(1,3,1)]
            Y_elems = [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]
            C_target = [(1,0,1),(1,1,1),(1,2,-1),(1,3,-1)]  # C21 +, C22 -
        elif s == 2:  # M3 = A11(B12 - B22)
            X_elems = [(0,0,1),(0,1,1)]
            Y_elems = [(0,2,1),(0,3,1),(1,2,1),(1,3,1),(2,2,-1),(2,3,-1),(3,2,-1),(3,3,-1)]
            C_target = [(0,2,1),(0,3,1),(1,2,1),(1,3,1)]  # C12 +, C22 +
        elif s == 3:  # M4 = A22(B21 - B11)
            X_elems = [(1,2,1),(1,3,1)]
            Y_elems = [(2,0,1),(2,1,1),(3,0,1),(3,1,1),(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1)]
            C_target = [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]  # C11 +, C21 +
        elif s == 4:  # M5 = (A11 + A12)B22
            X_elems = [(0,0,1),(0,1,1),(0,2,1),(0,3,1)]
            Y_elems = [(2,2,1),(2,3,1),(3,2,1),(3,3,1)]
            C_target = [(0,0,-1),(0,1,-1),(0,2,1),(0,3,1)]  # C11 -, C12 +
        elif s == 5:  # M6 = (A21 - A11)(B11 + B12)
            X_elems = [(1,0,1),(1,1,1),(0,0,-1),(0,1,-1)]
            Y_elems = [(0,0,1),(0,1,1),(1,0,1),(1,1,1),(0,2,1),(0,3,1),(1,2,1),(1,3,1)]
            C_target = [(1,2,1),(1,3,1)]  # C22 +
        elif s == 6:  # M7 = (A12 - A22)(B21 + B22)
            X_elems = [(0,2,1),(0,3,1),(1,2,-1),(1,3,-1)]
            Y_elems = [(2,0,1),(2,1,1),(3,0,1),(3,1,1),(2,2,1),(2,3,1),(3,2,1),(3,3,1)]
            C_target = [(0,0,1),(0,1,1),(1,2,1),(1,3,1)]  # C11 +, C22 +
        
        # Now for each of the 4 schoolbook multiplications in this term:
        for j in range(2):
            for k in range(2):
                # U part: X[0, j] = sum_{(i,a,coeff) in X_elems} coeff * A[i,a]
                for (i, a, coeff) in X_elems:
                    U[i * 4 + a, col] += coeff
                
                # V part: Y[j, k] = sum_{(b,c,coeff) in Y_elems} coeff * B[b,c]
                for (b, c, coeff) in Y_elems:
                    V[b * 4 + c, col] += coeff * (1 if (b, c) in [(j*2+0, k), (j*2+1, k)] else 0)
                
                # Wait, this is getting complicated. Let me redo.
                # Y is 2x2, indexed Y[row, col] where row in {0,1}, col in {0,1}
                # Y[row, col] corresponds to B block elements
                # For B11 (rows 0-1, cols 0-1): Y[0,0]=B[0,0], Y[0,1]=B[0,1], Y[1,0]=B[1,0], Y[1,1]=B[1,1]
                # For B12 (rows 0-1, cols 2-3): Y[0,0]=B[0,2], Y[0,1]=B[0,3], Y[1,0]=B[1,2], Y[1,1]=B[1,3]
                # etc.
                
                # Actually, let me think differently. 
                # The Strassen term M_s = X * Y where X is 1x2 and Y is 2x2.
                # M_s[0, k] = X[0,0]*Y[0,k] + X[0,1]*Y[1,k] for k=0,1
                # Each X[0,j] is a linear combo of A elements
                # Each Y[j,k] is a linear combo of B elements
                # The product X[0,j]*Y[j,k] is a rank-1 tensor
                # So for each j and k, we have one rank-1 term.
                # Total per Strassen term: 2*2 = 4 rank-1 terms.
                # 7 terms * 4 = 28 total.
                
                # Let me rebuild more carefully.
                pass  # This approach is getting messy, let me use a cleaner method
        
        col += 4
    
    # Let me restart with a cleaner construction
    return None

def _build_strassen_2x4x4_clean():
    """
    Clean construction of Strassen for 2x4x4.
    
    A is 2x4, B is 4x4, C is 2x4.
    Split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]  (each 1x2)
    B = [[B11, B12], [B21, B22]]  (each 2x2)
    C = [[C11, C12], [C21, C22]]  (each 1x2)
    
    Strassen:
    M1 = (A11+A22)(B11+B22)  -> 1x2 * 2x2 = 1x2
    M2 = (A21+A22)B11        -> 1x2 * 2x2 = 1x2
    M3 = A11(B12-B22)        -> 1x2 * 2x2 = 1x2
    M4 = A22(B21-B11)        -> 1x2 * 2x2 = 1x2
    M5 = (A11+A12)B22        -> 1x2 * 2x2 = 1x2
    M6 = (A21-A11)(B11+B12)  -> 1x2 * 2x2 = 1x2
    M7 = (A12-A22)(B21+B22)  -> 1x2 * 2x2 = 1x2
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For each 1x2 * 2x2 = 1x2 multiplication:
    Result[0,k] = X[0,0]*Y[0,k] + X[0,1]*Y[1,k] for k=0,1
    This is 4 rank-1 terms (j=0,1 and k=0,1).
    """
    rank = 28
    U = np.zeros((8, rank))   # A: 2x4, index i*4+j
    V = np.zeros((16, rank))  # B: 4x4, index j*4+k
    W = np.zeros((8, rank))   # C: 4x2, index k*2+i
    
    col = 0
    for s in range(7):
        # Define X (1x2) as linear combos of A elements
        # X[0,j] = sum of coeff * A[i,a]
        # Define Y (2x2) as linear combos of B elements  
        # Y[j,k] = sum of coeff * B[b,c]
        # Define output placement: for each output element C[i,k], add coeff * M_s[0, k_block]
        
        if s == 0:  # M1 = (A11+A22)(B11+B22)
            # X[0,0] = A[0,0] + A[1,2], X[0,1] = A[0,1] + A[1,3]
            X = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
            # Y[0,0]=B[0,0]+B[2,0], Y[0,1]=B[0,1]+B[2,1], Y[1,0]=B[1,0]+B[3,0], Y[1,1]=B[1,1]+B[3,1]
            Y = {(0,0): [(0,0,1),(2,0,1)], (0,1): [(0,1,1),(2,1,1)],
                 (1,0): [(1,0,1),(3,0,1)], (1,1): [(1,1,1),(3,1,1)]}
            # M1 contributes to C11 (i=0,k=0,1) with +1 and C22 (i=1,k=2,3) with +1
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 1:  # M2 = (A21+A22)B11
            X = {0: [(1,0,1),(1,2,1)], 1: [(1,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)]}
            out = {0: [(1,0,1),(1,1,1)], 1: [(1,2,-1),(1,3,-1)]}
        elif s == 2:  # M3 = A11(B12-B22)
            X = {0: [(0,0,1)], 1: [(0,1,1)]}
            Y = {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                 (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]}
            out = {0: [(0,2,1),(0,3,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 3:  # M4 = A22(B21-B11)
            X = {0: [(1,2,1)], 1: [(1,3,1)]}
            Y = {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                 (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,0,1),(1,1,1)]}
        elif s == 4:  # M5 = (A11+A12)B22
            X = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)]}
            Y = {(0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)]}
            out = {0: [(0,0,-1),(0,1,-1)], 1: [(0,2,1),(0,3,1)]}
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            X = {0: [(1,0,1),(0,0,-1)], 1: [(1,1,1),(0,1,-1)]}
            Y = {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                 (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]}
            out = {0: [], 1: [(1,2,1),(1,3,1)]}
        elif s == 6:  # M7 = (A12-A22)(B21+B22)
            X = {0: [(0,2,1),(1,2,-1)], 1: [(0,3,1),(1,3,-1)]}
            Y = {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                 (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        
        for j in range(2):
            for k in range(2):
                # U: X[0,j]
                for (i, a, coeff) in X[j]:
                    U[i * 4 + a, col] += coeff
                # V: Y[j,k]
                for (b, c, coeff) in Y[(j,k)]:
                    V[b * 4 + c, col] += coeff
                # W: output placement
                for (i, k_out, coeff) in out[k]:
                    W[k_out * 2 + i, col] += coeff
                col += 1
    
    return U, V, W

def _build_schoolbook_1x4x4():
    """
    Schoolbook for 1x4x4: A is 1x4, B is 4x4, C is 1x4.
    C[0,k] = sum_j A[0,j]*B[j,k]
    Rank = 16.
    """
    rank = 16
    U = np.zeros((4, rank))   # A: 1x4, index j (0-3)
    V = np.zeros((16, rank))  # B: 4x4, index j*4+k
    W = np.zeros((8, rank))   # C: 4x1, index k*1+0 = k... wait
    # For 1x4x4: n=1, m=4, p=4
    # U shape: n*m = 1*4 = 4
    # V shape: m*p = 4*4 = 16
    # W shape: p*n = 4*1 = 4
    # W index: k*n+i = k*1+0 = k
    W = np.zeros((4, rank))
    
    for j in range(4):
        for k in range(4):
            U[j, j*4+k] = 1
            V[j*4+k, j*4+k] = 1
            W[k, j*4+k] = 1
    
    return U, V, W

def _combine_3x4x4():
    """
    Combine 2x4x4 (Strassen, rank 28) and 1x4x4 (schoolbook, rank 16) for total rank 44.
    
    3x4x4: A is 3x4, B is 4x4, C is 3x4.
    Split A into top 2 rows (2x4) and bottom 1 row (1x4).
    B is shared (4x4).
    C is split into top 2 rows (2x4) and bottom 1 row (1x4).
    
    U shape: 3*4 = 12
    V shape: 4*4 = 16
    W shape: 4*3 = 12
    
    For the 2x4x4 part:
    U_top: rows 0-7 (i=0,1; j=0-3) -> index i*4+j
    V_shared: rows 0-15 (j=0-3; k=0-3) -> index j*4+k
    W_top: rows for i=0,1; k=0-3 -> index k*3+i
    
    For the 1x4x4 part:
    U_bot: row 8-11 (i=2; j=0-3) -> index 2*4+j = 8+j
    V_shared: same
    W_bot: rows for i=2; k=0-3 -> index k*3+2
    """
    U2, V2, W2 = _build_strassen_2x4x4_clean()
    U1, V1, W1 = _build_schoolbook_1x4x4()
    
    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # Place 2x4x4 part (columns 0-27)
    for i in range(2):
        for j in range(4):
            U[i*4+j, 0:28] = U2[i*4+j, :]
    V[0:16, 0:28] = V2[:, :]
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]
    
    # Place 1x4x4 part (columns 28-43)
    for j in range(4):
        U[8+j, 28:44] = U1[j, :]
    V[0:16, 28:44] = V1[:, :]
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]
    
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    # Try the constructive Strassen-based approach
    try:
        U, V, W = _combine_3x4x4()
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            U, V, W = _schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
