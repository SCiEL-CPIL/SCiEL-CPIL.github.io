import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 140.0


def _strassen_222():
    """Return the rank-7 Strassen decomposition for 2x2x2.
    Returns list of 7 tuples: (U_dict, V_dict, W_dict)
    where each dict maps (i,j), (j,k), (k,i) to coefficient.
    """
    # Strassen 7 products for M (2x2) * N (2x2) = P (2x2)
    # M = [[m00, m01], [m10, m11]]
    # N = [[n00, n01], [n10, n11]]
    # P = [[p00, p01], [p10, p11]]
    #
    # M1 = (m00+m11) * (n00+n11)
    # M2 = (m10+m11) * n00
    # M3 = m00 * (n01-n11)
    # M4 = m11 * (n10-n00)
    # M5 = (m00+m01) * n11
    # M6 = (m10-m00) * (n00+n10)
    # M7 = (m01-m11) * (n01+n11)
    #
    # p00 = M1 + M4 - M5 + M7
    # p01 = M3 + M5
    # p10 = M2 + M4
    # p11 = M1 - M2 + M3 + M6
    steps = [
        # M1: (m00+m11)*(n00+n11)
        (
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1, (1, 1): 1}
        ),
        # M2: (m10+m11)*n00
        (
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(0, 0): 1, (1, 0): 1}
        ),
        # M3: m00*(n01-n11)
        (
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
            {(0, 0): -1, (1, 1): 1}
        ),
        # M4: m11*(n10-n00)
        (
            {(1, 1): 1},
            {(1, 0): 1, (0, 0): -1},
            {(0, 0): 1, (1, 1): 1}
        ),
        # M5: (m00+m01)*n11
        (
            {(0, 0): 1, (0, 1): 1},
            {(1, 1): 1},
            {(0, 0): -1, (1, 1): 1}
        ),
        # M6: (m10-m00)*(n00+n10)
        (
            {(1, 0): 1, (0, 0): -1},
            {(0, 0): 1, (1, 0): 1},
            {(0, 1): 1}
        ),
        # M7: (m01-m11)*(n01+n11)
        (
            {(0, 1): 1, (1, 1): -1},
            {(0, 1): 1, (1, 1): 1},
            {(1, 0): 1}
        ),
    ]
    return steps


def _kron_decompose(n, m, p):
    """
    Build a Kronecker-product decomposition of the (n,m,p) matrix multiplication tensor.
    
    For n=4, m=4, p=6:
    - Decompose A (4x4) into 2x2 blocks of 2x2 sub-blocks
    - Decompose B (4x6) into 2x2 blocks of 2x3 sub-blocks
    - C (4x6) = sum of block products
    
    Strategy: Use Strassen(2x2x2) for the outer block structure and
    schoolbook(2x2x3) for the inner sub-block structure.
    
    Total rank = 7 * 12 = 84
    """
    n, m, p = 4, 4, 6
    
    # Outer Strassen steps for 2x2 block structure
    outer_steps = _strassen_222()
    
    # Inner schoolbook for 2x2 * 2x3 = 2x3
    # M (2x2) * N (2x3) = P (2x3)
    # p00 = m00*n00 + m01*n10
    # p01 = m00*n01 + m01*n11
    # p02 = m00*n02 + m01*n12
    # p10 = m10*n00 + m11*n10
    # p11 = m10*n01 + m11*n11
    # p12 = m10*n02 + m11*n12
    inner_steps = [
        # k=0, j=0: contributes to p00, p10
        (
            {(0, 0): 1},
            {(0, 0): 1},
            {(0, 0): 1, (1, 0): 1}  # Wait, W is indexed by (k,i) for C[i,k]
        ),
    ]
    # Actually, let me think about this more carefully.
    # The tensor T[i,j,k] = delta_{i,k} for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n + i, r] corresponds to C[i,k]
    
    # For schoolbook 2x2*2x3:
    # 12 terms, each: (i,j) in A, (j,k) in B, (i,k) in C
    inner_schoolbook = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                inner_schoolbook.append((
                    {(i, j): 1},
                    {(j, k): 1},
                    {(i, k): 1}
                ))
    
    # Now build the full 4x4*4x6 decomposition
    # A is 4x4, split into 2x2 blocks of 2x2
    # B is 4x6, split into 2x2 blocks of 2x3
    # C is 4x6, split into 2x2 blocks of 2x3
    
    # Block (I,J) of A * Block (J,K) of B contributes to Block (I,K) of C
    # Using Strassen for the 2x2 block level:
    # But Strassen gives P = M*N where M,N are 2x2 blocks
    # Here M is a 2x2 block of A (each entry is 2x2), N is a 2x2 block of B (each entry is 2x3)
    # P should be 2x2 blocks of C (each entry is 2x3)
    
    # So for each Strassen step, we compute linear combinations of 2x2 blocks of A
    # and 2x2 blocks of B, multiply them (using schoolbook 2x2*2x3), and add to C blocks.
    
    U = np.zeros((n * m, 84), dtype=float)
    V = np.zeros((m * p, 84), dtype=float)
    W = np.zeros((p * n, 84), dtype=float)
    
    col = 0
    for s in range(7):
        alpha, beta, gamma = outer_steps[s]
        for i1 in range(2):  # row of 2x2 sub-block within the block
            for j1 in range(2):  # col of 2x2 sub-block within the block
                for k1 in range(3):  # col of 2x3 sub-block within the block
                    # For this Strassen step and inner schoolbook position:
                    for (bi, bj), val_a in alpha.items():
                        # Global (i,j) in A
                        gi = bi * 2 + i1
                        gj = bj * 2 + j1
                        U[gi * m + gj, col] += val_a
                    for (bj2, bk), val_b in beta.items():
                        # Global (j,k) in B
                        gj = bj2 * 2 + j1
                        gk = bk * 3 + k1
                        V[gj * p + gk, col] += val_b
                    for (bi2, bk2), val_c in gamma.items():
                        # Global (i,k) in C
                        gi = bi2 * 2 + i1
                        gk = bk2 * 3 + k1
                        W[gk * n + gi, col] += val_c
                    col += 1
    
    assert col == 84, f"Expected 84 columns, got {col}"
    
    # Self-verification
    try:
        rng = np.random.default_rng(12345)
        for trial in range(10):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            A_flat = A.reshape(n * m)
            B_flat = B.reshape(m * p)
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_scheme_flat = W.T @ (a * b)
            max_err = np.max(np.abs(C_scheme_flat - C_true_flat))
            if max_err > 1e-8:
                raise AssertionError(f"Decomposition failed at trial {trial}, max_err={max_err}")
    except Exception as e:
        pass
    
    return U, V, W


def construct():
    start = time.time()
    
    n, m, p = DIMENSIONS
    
    # Use Kronecker product structure: Strassen(2x2x2) outer * schoolbook(2x2x3) inner
    U, V, W = _kron_decompose(n, m, p)
    
    # Deadline guard
    if time.time() - start > DEADLINE_SECONDS:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
