"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def _verify(U, V, W, n=4, m=5, p=5, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    
    return True

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition.
    U, V, W are 16x7 matrices.
    Indices: i,j,k in 0..3.
    """
    n = 4
    rank = 7
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Strassen formulas for 2x2x2 extended to 4x4x4?
    # Actually, standard Strassen is for 2x2.
    # For 4x4, we can use 2x2 Strassen on 2x2 blocks.
    # Or just use the known 17-term or 7-term?
    # Wait, Strassen is 7 terms for 2x2x2.
    # For 4x4x4, we can divide into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # This uses 7 multiplications.
    # We need to map this to the tensor indices.
    # The "multiplications" are the rank-1 terms.
    # Each term c corresponds to one of the M's.
    # U, V, W encode the additions/subtractions.
    
    # Let's define the 2x2 blocks.
    # i, j, k in {0,1,2,3}.
    # Block indices: ib = i//2, jb = j//2, kb = k//2.
    # Local indices: il = i%2, jl = j%2, kl = k%2.
    
    # We will construct U, V, W such that for each term c,
    # U[i,j,c] and V[j,k,c] and W[k,i,c] are set appropriately.
    
    # The 7 terms are:
    # 0: (A11+A22)(B11+B22)
    # 1: (A21+A22)B11
    # 2: A11(B12-B22)
    # 3: A22(B21-B11)
    # 4: (A11+A12)B22
    # 5: (A21-A11)(B11+B12)
    # 6: (A12-A22)(B21+B22)
    
    # We need to express A11, A22 etc as sums of A[i,j].
    # A11 is A[0,0], A[0,1], A[1,0], A[1,1].
    # This means for a term involving A11, U[i,j,c] should be 1 for i,j in A11.
    
    # Let's build U, V, W for each term.
    
    # Term 0: (A11+A22)(B11+B22)
    # U: 1 for (i,j) in A11 or A22.
    # V: 1 for (j,k) in B11 or B22.
    # W: 1 for (k,i) in C11 and C22.
    # C11 = M1 + ... -> W[k,i,0] = 1 for C11.
    # C22 = M1 + ... -> W[k,i,0] = 1 for C22.
    
    # It is easier to just iterate through the 16 indices and set values.
    
    # Helper to set blocks
    def set_block(M, r1, c1, r2, c2, val):
        M[r1:r2, c1:c2] = val

    # Indices for 2x2 blocks
    # A11: i in 0-1, j in 0-1
    # A12: i in 0-1, j in 2-3
    # A21: i in 2-3, j in 0-1
    # A22: i in 2-3, j in 2-3
    
    # Same for B, C.
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # U
    set_block(U, 0, 0, 2, 2, 1) # A11
    set_block(U, 2, 2, 4, 4, 1) # A22
    # V
    set_block(V, 0, 0, 2, 2, 1) # B11
    set_block(V, 2, 2, 4, 4, 1) # B22
    # W
    set_block(W, 0, 0, 2, 2, 1) # C11
    set_block(W, 2, 2, 4, 4, 1) # C22
    
    # Term 1: M2 = (A21+A22)B11
    # U
    set_block(U, 2, 0, 4, 2, 1) # A21
    set_block(U, 2, 2, 4, 4, 1) # A22
    # V
    set_block(V, 0, 0, 2, 2, 1) # B11
    # W
    set_block(W, 2, 0, 4, 2, 1) # C21
    set_block(W, 2, 2, 4, 4, 1) # C22? No, C22 = M1 - M2 + M3 + M6.
    # Wait, C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So W for M2 should be 1 for C21 and -1 for C22.
    set_block(W, 2, 0, 4, 2, 1) # C21
    set_block(W, 2, 2, 4, 4, -1) # C22
    
    # Term 2: M3 = A11(B12-B22)
    # U
    set_block(U, 0, 0, 2, 2, 1) # A11
    # V
    set_block(V, 0, 2, 2, 4, 1) # B12
    set_block(V, 2, 2, 4, 4, -1) # B22
    # W
    set_block(W, 0, 2, 2, 4, 1) # C12
    set_block(W, 2, 2, 4, 4, 1) # C22
    
    # Term 3: M4 = A22(B21-B11)
    # U
    set_block(U, 2, 2, 4, 4, 1) # A22
    # V
    set_block(V, 2, 0, 4, 2, 1) # B21
    set_block(V, 0, 0, 2, 2, -1) # B11
    # W
    set_block(W, 0, 0, 2, 2, 1) # C11
    set_block(W, 2, 0, 4, 2, 1) # C21
    
    # Term 4: M5 = (A11+A12)B22
    # U
    set_block(U, 0, 0, 2, 2, 1) # A11
    set_block(U, 0, 2, 2, 4, 1) # A12
    # V
    set_block(V, 2, 2, 4, 4, 1) # B22
    # W
    set_block(W, 0, 0, 2, 2, -1) # C11
    set_block(W, 0, 2, 2, 4, 1) # C12
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    # U
    set_block(U, 2, 0, 4, 2, 1) # A21
    set_block(U, 0, 0, 2, 2, -1) # A11
    # V
    set_block(V, 0, 0, 2, 2, 1) # B11
    set_block(V, 0, 2, 2, 4, 1) # B12
    # W
    set_block(W, 2, 2, 4, 4, 1) # C22
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    # U
    set_block(U, 0, 2, 2, 4, 1) # A12
    set_block(U, 2, 2, 4, 4, -1) # A22
    # V
    set_block(V, 2, 0, 4, 2, 1) # B21
    set_block(V, 2, 2, 4, 4, 1) # B22
    # W
    set_block(W, 0, 0, 2, 2, 1) # C11
    
    return U, V, W

def _construct_12_rank():
    n, m, p = 4, 5, 5
    rank = 12
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 1. Strassen 4x4x4 (Terms 0-6)
    U_s, V_s, W_s = _strassen_4x4x4()
    
    # Map Strassen indices to global
    # U_s[i*4+j, c] -> U[i*5+j, c]
    # V_s[j*4+k, c] -> V[j*5+k, c]
    # W_s[k*4+i, c] -> W[k*4+i, c]
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for c in range(7):
                    U[i*5+j, c] = U_s[i*4+j, c]
                    V[j*5+k, c] = V_s[j*4+k, c]
                    W[k*4+i, c] = W_s[k*4+i, c]
    
    # 2. Term 7: A[4,1] B[1,:]
    # U[i,4,7] = 1
    for i in range(4):
        U[i*5+4, 7] = 1
    # V[4,k,7] = 1
    for k in range(5):
        V[4*5+k, 7] = 1
    # W[k,i,7] = 1
    for k in range(5):
        for i in range(4):
            W[k*4+i, 7] = 1
            
    # 3. Terms 8-11: A[4,4] B[4,1]
    for c_local in range(4):
        c = 8 + c_local
        # U[i, c_local, c] = 1
        for i in range(4):
            U[i*5+c_local, c] = 1
        # V[c_local, 4, c] = 1
        V[c_local*5+4, c] = 1
        # W[4, i, c] = 1
        for i in range(4):
            W[4*4+i, c] = 1
            
    # Convert to lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    if _time_exceeded():
        return {"U": [], "V": [], "W": []}
    
    result = _construct_12_rank()
    
    if _verify(result['U'], result['V'], result['W']):
        return result
    else:
        # Fallback to schoolbook if verification fails
        n, m, p = 4, 5, 5
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
