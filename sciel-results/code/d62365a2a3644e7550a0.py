"""Matrix multiplication 4x4 * 4x8 -> 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 via Strassen, rank 7.
    U: (4,7) rows indexed by i*2+j
    V: (4,7) rows indexed by j*2+k
    W: (4,7) rows indexed by k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1  # C00 -= M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1  # C00 += M5, C11 -= M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1  # C11 -= M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1  # C00 -= M7, C01 += M7
    
    return U, V, W


def _build_4x4_strassen():
    """Build 4x4 * 4x4 -> 4x4 via Kronecker of Strassen, rank 49.
    U: (16, 49) rows i*4+j
    V: (16, 49) rows j*4+k
    W: (16, 49) rows k*4+i
    """
    U2, V2, W2 = _strassen_2x2()
    r = 49
    U = np.zeros((16, r))
    V = np.zeros((16, r))
    W = np.zeros((16, r))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    u_val = U2[i1 * 2 + j1, c1]
                    if u_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u_val2 = U2[i2 * 2 + j2, c2]
                            if u_val2 == 0:
                                continue
                            u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[u_idx, c] += u_val * u_val2
            
            for j1 in range(2):
                for k1 in range(2):
                    v_val = V2[j1 * 2 + k1, c1]
                    if v_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v_val2 = V2[j2 * 2 + k2, c2]
                            if v_val2 == 0:
                                continue
                            v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            V[v_idx, c] += v_val * v_val2
            
            for k1 in range(2):
                for i1 in range(2):
                    w_val = W2[k1 * 2 + i1, c1]
                    if w_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w_val2 = W2[k2 * 2 + i2, c2]
                            if w_val2 == 0:
                                continue
                            w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W[w_idx, c] += w_val * w_val2
    
    return U, V, W


def _build_decomposition():
    """Build 4x4 * 4x8 -> 4x8 decomposition using two 4x4*4x4 Strassen blocks.
    Rank: 98.
    U: (16, 98) rows i*4+j
    V: (32, 98) rows j*8+k
    W: (32, 98) rows k*4+i
    """
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4_strassen()
    r = 98
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # First 4x4 * 4x4: B columns 0-3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
    
    # Second 4x4 * 4x4: B columns 4-7
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j * 8 + k, 49:98] = V4[j * 4 + (k - 4), 0:49]
    for k in range(4, 8):
        for i in range(4):
            W[k * 4 + i, 49:98] = W4[(k - 4) * 4 + i, 0:49]
    
    return U, V, W


def _verify(U, V, W, n=4, m=4, p=8):
    """Verify tensor decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1"""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True


def _schoolbook(n=4, m=4, p=8):
    """Schoolbook decomposition, rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_direct_4x4x8():
    """Attempt a direct 4x4 * 4x8 decomposition using Strassen-like block structure.
    Split A into 2x2 blocks, B into 2x4 blocks, C into 2x4 blocks.
    Use a generalized Strassen for 2x2 * 2x4.
    
    Actually, let's try: treat 4x4 * 4x8 as computing A*B where we can use
    the 4x4*4x4 algorithm twice. That's already rank 98.
    
    Alternative: Can we do better by sharing computation?
    The key insight: the A matrix is the same for both halves of B.
    In the Kronecker construction, the U matrix is the same for both blocks.
    This is already captured in the rank-98 construction.
    
    Let me try a different approach: use a 2x2 Strassen on the row blocks,
    and handle the column blocks separately.
    """
    n, m, p = 4, 4, 8
    # Split into 2x2 row blocks and 2x4 column blocks
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] (each 2x4)
    # C = [[C00, C01], [C10, C11]] (each 2x4)
    # C_ij = sum_k A_ik * B_kj
    # C00 = A00*B00 + A01*B10 (each 2x4)
    # C01 = A00*B01 + A01*B11 (each 2x4)
    # C10 = A10*B00 + A11*B10 (each 2x4)
    # C11 = A10*B01 + A11*B11 (each 2x4)
    
    # We can use Strassen-like formulas:
    # M1 = (A00+A11)*(B00+B10) -> 2x4 product, rank 7
    # M2 = (A10+A11)*B00 -> 2x4 product, rank 7
    # M3 = A00*(B01-B11) -> 2x4 product, rank 7
    # M4 = A11*(B10-B00) -> 2x4 product, rank 7
    # M5 = (A00+A01)*B11 -> 2x4 product, rank 7
    # M6 = (A10-A00)*(B00+B01) -> 2x4 product, rank 7
    # M7 = (A01-A11)*(B10+B11) -> 2x4 product, rank 7
    
    # But 2x4 * 2x4 is not the same as 2x2 * 2x2.
    # 2x4 matrix multiplication: A(2x4) * B(4x4) -> C(2x4)? No.
    # We need 2x2 * 2x4 -> 2x4.
    # This is just 4 independent 2x2 * 2x1 products? No, it's 2x2 * 2x4.
    # A(2x2) * B(2x4) = C(2x4).
    # This is equivalent to 4 independent 2x1 * 1x1? No.
    # A(2x2) * B(2x4): for each column of B, we compute A * b_col (2x1).
    # So it's 4 independent 2x2 * 2x1 -> 2x1 products.
    # Each 2x2 * 2x1 has rank 8 (schoolbook) or can we do better?
    # 2x2 * 2x1: C = A*B where A is 2x2, B is 2x1, C is 2x1.
    # This is just 4 scalar multiplications: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1.
    # Rank 4 is achievable (it's just a linear map from 4D to 2D, but we need the tensor).
    # Actually, 2x2 * 2x1 tensor: M[i,j,k] where i in 0..1, j in 0..1, k in 0..1.
    # M[i,j,k] = 1 if j==k (since C_i = sum_j A_ij B_jk, and B is 2x1 so k is fixed... wait)
    # B is 2x1, so B[j, 0] for j in 0..1. C[i, 0] = sum_j A[i,j] * B[j, 0].
    # So the tensor is M[i, j, 0] = A[i,j] * B[j,0] * C[i,0]... 
    # The multiplication tensor for 2x2 * 2x1: 
    # C_i = sum_j A_ij B_j
    # Tensor M[i,j,k] where k is the output index (only k=0).
    # M[i,j,0] = 1 if the (i,j) entry of A multiplies the j-th entry of B to contribute to C_i.
    # So M[i,j,0] = 1 for all i,j. That's 4 terms, rank 4.
    
    # So 2x2 * 2x4 is 4 independent 2x2 * 2x1, each rank 4, total rank 16.
    # But we can do better by sharing.
    # Actually, 2x2 * 2x4: A(2x2) * B(2x4) = C(2x4).
    # C[i, k] = sum_j A[i,j] * B[j,k] for i in 0..1, k in 0..3.
    # This is 8 scalar products, but they share A.
    # We can use the 4x4*4x4 approach on a 2x2 subproblem... 
    # This is getting complicated. Let's stick with rank 98.
    
    return None  # Don't use this for now


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s limit
    
    try:
        U, V, W = _build_decomposition()
        if time.time() > deadline:
            # Timeout during construction, use schoolbook
            U, V, W = _schoolbook(n, m, p)
        elif _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Verification failed, use schoolbook
            U, V, W = _schoolbook(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
