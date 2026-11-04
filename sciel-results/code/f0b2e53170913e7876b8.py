"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_3x3_schoolbook():
    """Return (U, V, W) for 3x3 matrix multiplication, rank 27.
    U: (9, 27), V: (9, 27), W: (9, 27)
    Convention: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    n = 3
    rank = n * n * n
    U = np.zeros((n * n, rank), dtype=np.float64)
    V = np.zeros((n * n, rank), dtype=np.float64)
    W = np.zeros((n * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j, col] = 1.0
                V[j * n + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _build_2x2_strassen():
    """Return (U, V, W) for 2x2 matrix multiplication via Strassen, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n = 2
    rank = 7
    U = np.zeros((n * n, rank), dtype=np.float64)
    V = np.zeros((n * n, rank), dtype=np.float64)
    W = np.zeros((n * n, rank), dtype=np.float64)
    
    # U rows: A00=0, A01=1, A10=2, A11=3
    # V rows: B00=0, B01=1, B10=2, B11=3
    # W rows: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W


def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """Combine two matrix multiplication decompositions via Kronecker product.
    
    If tensor1 has rank r1 and tensor2 has rank r2, the combined tensor has rank r1*r2.
    U_combined[i1*m1+j1, i2*m2+j2, c1*r2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    Similarly for V and W.
    
    For 6x6 = 3x3 ⊗ 2x2:
    U shape: (9*4, r1*r2) = (36, r1*r2)
    V shape: (9*4, r1*r2) = (36, r1*r2)
    W shape: (9*4, r1*r2) = (36, r1*r2)
    
    Index convention:
    For the 6x6 case, A is 6x6, B is 6x6, C is 6x6.
    We view A as a 3x3 block matrix where each block is 2x2.
    A[i1*2+i2, j1*2+j2] where i1,j1 in {0,1,2} (3x3 blocks) and i2,j2 in {0,1} (within block).
    
    U row index: (i1*m1 + j1) * (m2) + (j2) ... wait, let me think about this more carefully.
    
    Actually, the Kronecker product structure for matrix multiplication:
    If we have two matrix multiplications of sizes (n1,m1,p1) and (n2,m2,p2),
    then the tensor for (n1*n2, m1*m2, p1*p2) matrix multiplication is the Kronecker product.
    
    The indices:
    For the combined matrix multiplication:
    A has shape (n1*n2, m1*m2), B has shape (m1*m2, p1*p2), C has shape (p1*p2, n1*n2).
    
    We can write A[i1*n2+i2, j1*m2+j2] = A1[i1,j1] ⊗ A2[i2,j2] (in the tensor sense)
    
    The U matrix for the combined problem:
    U_combined[(i1*n2+i2)*(m1*m2) + (j1*m2+j2), c1*r2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    
    Wait, I need to be more careful. Let me use the standard Kronecker tensor decomposition.
    
    For matrix multiplication T_{n,m,p} with indices (i,j,k) where i in [n], j in [m], k in [p]:
    T[i,j,k] = delta_{i,k} (this is the standard matrix multiplication tensor where C[i,k] = sum_j A[i,j]*B[j,k])
    
    Actually, the tensor for matrix multiplication has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
    The standard form: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,k}... no.
    
    Let me re-read the convention. The task says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    So the tensor is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k], the tensor T[i,j,k] = 1 for all (i,j,k) in the appropriate range... no, that's not right either.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 if and only if... actually, the tensor is defined such that:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    In tensor form: sum_{j} A[i,j] B[j,k] = C[i,k]
    
    The bilinear form: <A, T[:, :, k] > = C[:, k]
    
    The standard tensor decomposition: T[i,j,k] is the tensor where T[i,j,k] = 1 for all valid (i,j,k). No, that's the identity.
    
    Let me think again. The matrix multiplication tensor M_{n,m,p} is defined by:
    M[i,j,k] = 1 if i, j, k are in valid ranges, and the tensor satisfies:
    sum_{j} A[i,j] * B[j,k] = C[i,k]
    
    In terms of a rank-1 decomposition:
    M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    For the schoolbook algorithm, each (i,j,k) triple gets its own rank-1 term:
    U[i,j,c_{ijk}] = 1, V[j,k,c_{ijk}] = 1, W[k,i,c_{ijk}] = 1
    
    This gives: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for each (i,j,k)
    
    So the tensor T[i,j,k] = 1 for all valid (i,j,k). This is indeed the matrix multiplication tensor.
    
    Now for the Kronecker product: if T1 is the tensor for (n1,m1,p1) and T2 is for (n2,m2,p2),
    then T1 ⊗ T2 is the tensor for (n1*n2, m1*m2, p1*p2).
    
    (T1 ⊗ T2)[(i1,i2),(j1,j2),(k1,k2)] = T1[i1,j1,k1] * T2[i2,j2,k2]
    
    Since T1[i1,j1,k1] = 1 and T2[i2,j2,k2] = 1, the combined tensor is also all 1s, which is correct.
    
    For the rank-1 decomposition:
    If T1 = sum_{c1} U1[i1,j1,c1] V1[j1,k1,c1] W1[k1,i1,c1]
    and T2 = sum_{c2} U2[i2,j2,c2] V2[j2,k2,c2] W2[k2,i2,c2]
    
    Then T1 ⊗ T2 = sum_{c1,c2} (U1[i1,j1,c1]*U2[i2,j2,c2]) * (V1[j1,k1,c1]*V2[j2,k2,c2]) * (W1[k1,i1,c1]*W2[k2,i2,c2])
    
    So the combined decomposition has rank r1*r2.
    
    Now for the indexing in the 6x6 case:
    n1=3, m1=3, p1=3 (first factor)
    n2=2, m2=2, p2=2 (second factor)
    
    Combined: n=6, m=6, p=6
    
    U_combined row index: i*m + j where i in [0,6), j in [0,6)
    i = i1*2 + i2, j = j1*2 + j2
    So U_combined[i1*2+i2, j1*2+j2, c1*7+c2] = U1[i1*3+j1, c1] * U2[i2*2+j2, c2]
    
    In the flat index: U_combined[(i1*2+i2)*6 + (j1*2+j2), c1*7+c2]
    
    Similarly for V and W.
    """
    # Determine dimensions
    n1, m1, p1 = U1.shape[0] // m1 if False else (U1.shape[0], U1.shape[0], U1.shape[0])
    # Actually, let me just pass dimensions explicitly
    pass


def _kron_decomposition_explicit(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Combine two matrix multiplication decompositions via Kronecker product.
    
    U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
    U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
    
    Returns U, V, W for (n1*n2, m1*m2, p1*p2) matrix multiplication.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # U[i*m + j, c1*r2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    # where i = i1*n2 + i2, j = j1*m2 + j2
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u1_val = U1[i1 * m1 + j1, c1]
                if u1_val == 0:
                    continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            u2_val = U2[i2 * m2 + j2, c2]
                            if u2_val == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            c = c1 * r2 + c2
                            U[i * m + j, c] = u1_val * u2_val
    
    # V[j*p + k, c1*r2+c2] = V1[j1*m2... wait
    # V1[j1*p1 + k1, c1], V2[j2*p2 + k2, c2]
    # j = j1*m2 + j2, k = k1*p2 + k2
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                v1_val = V1[j1 * p1 + k1, c1]
                if v1_val == 0:
                    continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            v2_val = V2[j2 * p2 + k2, c2]
                            if v2_val == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            c = c1 * r2 + c2
                            V[j * p + k, c] = v1_val * v2_val
    
    # W[k*n + i, c1*r2+c2] = W1[k1*n1 + i1, c1] * W2[k2*n2 + i2, c2]
    # k = k1*p2... wait, k is the p-index
    # W1[k1*n1 + i1, c1], W2[k2*n2 + i2, c2]
    # k = k1*p2 + k2, i = i1*n2 + i2
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                w1_val = W1[k1 * n1 + i1, c1]
                if w1_val == 0:
                    continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            w2_val = W2[k2 * n2 + i2, c2]
                            if w2_val == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            c = c1 * r2 + c2
                            W[k * n + i, c] = w1_val * w2_val
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)."""
    rank = U.shape[1]
    # Check a sample of entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7)
    U1, V1, W1 = _build_3x3_schoolbook()
    U2, V2, W2 = _build_2x2_strassen()
    
    # Combine via Kronecker product: rank 27 * 7 = 189
    U, V, W = _kron_decomposition_explicit(U1, V1, W1, U2, V2, W2, 3, 3, 3, 2, 2, 2)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
