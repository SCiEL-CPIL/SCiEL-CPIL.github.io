"""Matrix multiplication 3x4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_2x2_strassen():
    """Return U, V, W for 2x2 * 2x2 multiplication (rank 7).
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    A is 2x2, B is 2x2, C = A*B is 2x2.
    C[i,k] = sum_j A[i,j]*B[j,k]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (k,i) -> k*2+i
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # C00 = M1+M4-M5+M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C01 = M3+M5
    W[1, 2] = 1; W[1, 4] = 1
    # C10 = M2+M4
    W[2, 1] = 1; W[2, 3] = 1
    # C11 = M1-M2+M3+M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_3x2x4():
    """Return U, V, W for 3x2 * 2x4 multiplication (rank 24).
    This is just 3 independent 1x2*2x4 problems, each rank 4.
    n=3, m=2, p=4, rank=3*2*4=24
    """
    n, m, p = 3, 2, 4
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
    return U, V, W


def _tensor_product_decompose(n1, m1, p1, U1, V1, W1, n2, m2, p2, U2, V2, W2):
    """
    Combine two matrix multiplication decompositions via tensor product.
    
    Problem 1: (n1, m1, p1) with rank r1
    Problem 2: (n2, m2, p2) with rank r2
    
    Combined: (n1*n2, m1*m2, p1*p2) with rank r1*r2
    
    U_combined[i1*n2*m2 + i2*m2 + j1*m1 + j2, c1*r2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    Wait, need to be careful with index ordering.
    
    A has shape (n1*n2, m1*m2). Index a = (a1, a2) where a1 in [0,n1), a2 in [0,m1).
    Actually, let's think of A as A[i1,i2, j1,j2] where i1 in [0,n1), i2 in [0,n2), j1 in [0,m1), j2 in [0,m2).
    Flat index for A: i1*(n2*m1*m2) + i2*(m1*m2) + j1*m2 + j2? 
    Hmm, the flat index convention is: for matrix of shape (rows, cols), flat index = row*cols + col.
    So A flat: row = i1*n2 + i2, col = j1*m2 + j2.
    Flat index = (i1*n2 + i2)*(m1*m2) + j1*m2 + j2 = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2.
    
    For U1: flat index in A1 is i1*m1 + j1 (shape n1 x m1).
    For U2: flat index in A2 is i2*m2 + j2 (shape n2 x m2).
    
    The combined U should have rows indexed by the combined A flat index and columns by (c1, c2).
    U_combined[flat_A, c1*r2 + c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    where flat_A = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2.
    
    Similarly for V: B has shape (m1*m2, p1*p2).
    B flat: row = j1*m2 + j2, col = k1*p2 + k2.
    Flat index = (j1*m2+j2)*(p1*p2) + k1*p2 + k2 = j1*m2*p1*p2 + j2*p1*p2 + k1*p2 + k2.
    V1[j1*p1+k1, c1], V2[j2*p2+k2, c2].
    V_combined[flat_B, c1*r2+c2] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
    
    For W: C has shape (p1*p2, n1*n2) in the W convention (W[k*n+i, c]).
    W flat: row = k1*p2 + k2, col = i1*n2 + i2.
    Flat index = (k1*p2+k2)*(n1*n2) + i1*n2 + i2 = k1*p2*n1*n2 + k2*n1*n2 + i1*n2 + i2.
    W1[k1*n1+i1, c1], W2[k2*n2+i2, c2].
    W_combined[flat_W, c1*r2+c2] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1, c1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2, c2]
                            if u2_val == 0:
                                continue
                            flat_A = i1 * n2 * m1 * m2 + i2 * m1 * m2 + j1 * m2 + j2
                            U[flat_A, c] = u1_val * u2_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1, c1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2, c2]
                            if v2_val == 0:
                                continue
                            flat_B = j1 * m2 * p1 * p2 + j2 * p1 * p2 + k1 * p2 + k2
                            V[flat_B, c] = v1_val * v2_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1, c1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2, c2]
                            if w2_val == 0:
                                continue
                            flat_W = k1 * p2 * n1 * n2 + k2 * n1 * n2 + i1 * n2 + i2
                            W[flat_W, c] = w1_val * w2_val
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, seed=42):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(seed)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(U.shape[1]):
        u_reshaped = U[:, c].reshape(n, m)
        v_reshaped = V[:, c].reshape(m, p)
        w_reshaped = W[:, c].reshape(p, n).T  # shape (n, p)
        term = (u_reshaped * A) @ (v_reshaped * B)
        C_computed += w_reshaped * term
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True


def construct():
    """Construct tensor decomposition for 3x4 * 4x8 multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strategy: Decompose 3x4 * 4x8 as tensor product of (3x2 * 2x4) and (1x2 * 2x2)
    # n1=3, m1=2, p1=4 (rank 24 schoolbook)
    # n2=1, m2=2, p2=2 (rank 4 schoolbook)
    # Combined: (3*1, 2*2, 4*2) = (3, 4, 8) with rank 24*4 = 96
    
    # This gives the same rank as schoolbook, but let's construct it properly.
    # The tensor product structure is:
    # A[i1,i2, j1,j2] with i1 in [0,3), i2 in [0,1), j1 in [0,2), j2 in [0,2)
    # B[j1,j2, k1,k2] with j1 in [0,2), j2 in [0,2), k1 in [0,4), k2 in [0,2)
    # C[i1,i2, k1,k2] = sum_{j1,j2} A[i1,i2,j1,j2] * B[j1,j2,k1,k2]
    
    # Problem 1: (3, 2, 4) - A1[i1, j1] * B1[j1, k1]
    # Problem 2: (1, 2, 2) - A2[i2, j2] * B2[j2, k2]
    
    U1, V1, W1 = _schoolbook(3, 2, 4)
    U2, V2, W2 = _schoolbook(1, 2, 2)
    
    U, V, W = _tensor_product_decompose(3, 2, 4, U1, V1, W1, 1, 2, 2, U2, V2, W2)
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        if not _verify(U, V, W, n, m, p):
            # Fall back to schoolbook
            U, V, W = _schoolbook(n, m, p)
            if not _verify(U, V, W, n, m, p):
                # Should not happen
                pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return whatever we have
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
