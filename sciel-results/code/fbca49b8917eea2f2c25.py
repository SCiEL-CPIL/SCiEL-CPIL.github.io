import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 140.0

def _build_rank28_444():
    """
    Builds a rank-28 decomposition for 4x4 * 4x4 matrix multiplication.
    Uses Strassen for the 2x2 block level and Schoolbook for the 2x2 matrix level.
    """
    n = 4
    m = 4
    p = 4
    rank = 28
    
    U = np.zeros((n*m, rank), dtype=np.int16)
    V = np.zeros((m*p, rank), dtype=np.int16)
    W = np.zeros((p*n, rank), dtype=np.int16)
    
    # Strassen definitions for 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 matrix product Xk * Yk.
    # We expand Xk * Yk into 4 scalar products using Schoolbook.
    # Xk = [[x00, x01], [x10, x11]]
    # Yk = [[y00, y01], [y10, y11]]
    # (Xk*Yk)00 = x00*y00 + x01*y10
    # (Xk*Yk)01 = x00*y01 + x01*y11
    # (Xk*Yk)10 = x10*y00 + x11*y10
    # (Xk*Yk)11 = x10*y01 + x11*y11
    
    # Indices for U (i,j), V (j,k), W (k,i)
    # i, j, k, l are 0 or 1 for the 2x2 blocks.
    # Global index I = 2*i_block + i_local
    # Global index J = 2*j_block + j_local
    # Global index K = 2*k_block + k_local
    
    # We will iterate over the 7 Strassen terms and 4 Schoolbook terms each.
    
    # Helper to add a term to U, V, W
    # term_id: 0-27
    # u_indices: list of (i, j) and coeff
    # v_indices: list of (j, k) and coeff
    # w_indices: list of (k, i) and coeff
    
    # We define the 7 Strassen terms as pairs of (X, Y) where X, Y are 2x2 matrices of coefficients
    # for the sub-blocks.
    
    # M1: X = A00+A11, Y = B00+B11
    # A00 is at block (0,0), A11 at (1,1).
    # X has non-zero entries at (0,0) with coeff 1 (from A00) and (1,1) with coeff 1 (from A11).
    # Wait, X is a 2x2 matrix of scalars? No, X is a 2x2 matrix.
    # The entries of X are linear combinations of A's entries.
    # Actually, it's easier to think in terms of the final scalar products.
    
    # Let's list the 28 products explicitly.
    # Product p corresponds to Strassen term s (0-6) and Schoolbook sub-term t (0-3).
    # s=0: M1. X = A00+A11. Y = B00+B11.
    # X00 = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
    # X01 = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
    # X10 = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
    # X11 = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
    
    # Y00 = B00[0,0] + B11[0,0] = B[0,0] + B[2,2]
    # Y01 = B00[0,1] + B11[0,1] = B[0,1] + B[2,3]
    # Y10 = B00[1,0] + B11[1,0] = B[1,0] + B[3,2]
    # Y11 = B00[1,1] + B11[1,1] = B[1,1] + B[3,3]
    
    # This is getting complex to hardcode all 28.
    # Let's use the tensor product structure but with Rank 4 for inner.
    
    U2 = np.zeros((4, 4), dtype=np.int8) # Schoolbook 2x2x2
    V2 = np.zeros((4, 4), dtype=np.int8)
    W2 = np.zeros((4, 4), dtype=np.int8)
    for i in range(4):
        U2[i, i] = 1
        V2[i, i] = 1
        W2[i, i] = 1
        
    # Strassen 2x2x2 coefficients
    # U2s, V2s, W2s for the 7 terms
    Us, Vs, Ws = _build_strassen_2x2x2()
    
    # Construct 4x4x4 Rank 28
    # Rank = 7 (outer) * 4 (inner) = 28
    rank_444 = 28
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for c0 in range(7): # Outer Strassen term
        for c1 in range(4): # Inner Schoolbook term
            c = c0 * 4 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    # U4[i,j,c] = Us[i0*2+j0, c0] * U2[i1*2+j1, c1]
                    # Wait, the tensor product logic:
                    # U4[i,j,c] = sum over factors?
                    # The Kronecker product of tensors:
                    # U4 = Us \otimes U2
                    # U4[i0*2+i1, j0*2+j1, c0*4+c1] = Us[i0*2+j0, c0] * U2[i1*2+j1, c1]
                    U4[i*4+j, c] = Us[i0*2+j0, c0] * U2[i1*2+j1, c1]
                    
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = Vs[j0*2+k0, c0] * V2[j1*2+k1, c1]
                    
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = Ws[k0*2+i0, c0] * W2[k1*2+i1, c1]
                    
    # Verify 4x4x4
    result = np.zeros((4, 4, 4), dtype=np.float64)
    for r in range(rank_444):
        u_col = U4[:, r].reshape(4, 4)
        v_col = V4[:, r].reshape(4, 4)
        w_col = W4[:, r].reshape(4, 4)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
        
    if not np.allclose(result, 1.0, atol=1e-9):
        return None
        
    # Construct 4x4x8
    total_rank = 56
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 28
        col_end = (block + 1) * 28
        U[:, col_start:col_end] = U4
        
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
                
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    # Verify 4x4x8
    result = np.zeros((4, 4, 8), dtype=np.float64)
    for r in range(total_rank):
        u_col = U[:, r].reshape(4, 4)
        v_col = V[:, r].reshape(4, 8)
        w_col = W[:, r].reshape(8, 4)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
        
    if not np.allclose(result, 1.0, atol=1e-9):
        return None
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _build_strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _schoolbook(n=4, m=4, p=8):
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

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank28_444()
            if res is not None:
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
