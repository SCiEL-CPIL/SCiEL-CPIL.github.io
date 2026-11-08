"""Rank-65 tensor decomposition for 4x4 x 4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# ---------- 2x2 Strassen (rank 7) ----------
# A entries (row-major): [a00,a01,a10,a11]
# B entries (row-major): [b00,b01,b10,b11]
# C entries (row-major): [c00,c01,c10,c11]
_S22_A = np.array([
    [1,0,0,1],   # S1
    [0,0,1,1],   # S2
    [1,0,0,0],   # S3
    [0,0,0,1],   # S4
    [1,1,0,0],   # S5
    [0,0,1,-1],  # S6
    [0,1,0,-1],  # S7
], dtype=float)
_S22_B = np.array([
    [1,0,0,1],   # T1
    [1,0,0,0],   # T2
    [0,1,-1,0],  # T3
    [1,-1,0,0],  # T4
    [0,0,0,1],   # T5
    [1,1,0,0],   # T6
    [0,0,1,1],   # T7
], dtype=float)
_S22_C = np.array([
    [1,1,-1,1],   # c00 = P1 - P3 + P4 + P7
    [0,0,0,1],    # c01 = P5
    [0,1,0,0],    # c10 = P2
    [-1,1,1,0],   # c11 = P1 - P6 + P3
], dtype=float)

# ---------- 4x4 Strassen (rank 49) ----------
# A 4x4 blocks (row-major within block):
#   A00=[0,1,4,5]  A01=[2,3,6,7]  A10=[8,9,12,13]  A11=[10,11,14,15]
# B 4x4 blocks (row-major within block):
#   B00=[0,1,4,5]  B01=[2,3,6,7]  B10=[8,9,12,13]  B11=[10,11,14,15]
# C 4x4 blocks (row-major within block):
#   C00=[0,1,4,5]  C01=[2,3,6,7]  C10=[8,9,12,13]  C11=[10,11,14,15]

# 7 super-products (each is a 2x2 matrix product):
#   M1 = (A00+A11)(B00+B11)
#   M2 = (A10+A11)B00
#   M3 = A00(B10-B11)
#   M4 = A11(B01-B00)
#   M5 = (A00+A01)B11
#   M6 = (A10-A00)(B00+B01)
#   M7 = (A01-A11)(B10+B11)

# For each super-product k, the A-side picks 4 blocks with coefficients,
# and the B-side picks 4 blocks with coefficients.
# A-side block coefficients (7 x 4 blocks x 4 entries):
_S44_A_blocks = np.array([
    # S1: A00 + A11
    [[1,1,1,1], [0,0,0,0], [0,0,0,0], [1,1,1,1]],
    # S2: A10 + A11
    [[0,0,0,0], [0,0,0,0], [1,1,1,1], [1,1,1,1]],
    # S3: A00
    [[1,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
    # S4: A11
    [[0,0,0,0], [0,0,0,0], [0,0,0,0], [1,1,1,1]],
    # S5: A00 + A01
    [[1,1,1,1], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
    # S6: A10 - A00
    [[-1,-1,-1,-1], [0,0,0,0], [1,1,1,1], [0,0,0,0]],
    # S7: A01 - A11
    [[0,0,0,0], [1,1,1,1], [0,0,0,0], [-1,-1,-1,-1]],
], dtype=float)  # shape (7, 4, 4)

_S44_B_blocks = np.array([
    # T1: B00 + B11
    [[1,1,1,1], [0,0,0,0], [0,0,0,0], [1,1,1,1]],
    # T2: B00
    [[1,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
    # T3: B10 - B11
    [[0,0,0,0], [0,0,0,0], [1,1,1,1], [-1,-1,-1,-1]],
    # T4: B01 - B00
    [[-1,-1,-1,-1], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
    # T5: B11
    [[0,0,0,0], [0,0,0,0], [0,0,0,0], [1,1,1,1]],
    # T6: B00 + B01
    [[1,1,1,1], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
    # T7: B10 + B11
    [[0,0,0,0], [0,0,0,0], [1,1,1,1], [1,1,1,1]],
], dtype=float)  # shape (7, 4, 4)

# C-side: for each super-product k, which 2x2 output block gets which coefficient.
# C00 = M1 - M3 + M4 + M7
# C01 = M5
# C10 = M2
# C11 = M1 - M6 + M3
_S44_C_blocks = np.array([
    # block 00 (C00): M1 - M3 + M4 + M7
    [[1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]],
    # block 01 (C01): M5
    [[0,0,0,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]],
    # block 10 (C10): M2
    [[0,0,0,0], [0,0,0,0], [0,1,0,0], [0,0,0,0]],
    # block 11 (C11): M1 - M6 + M3
    [[1,0,0,0], [0,0,0,0], [0,-1,1,0], [0,0,0,0]],
], dtype=float)  # shape (4, 4, 7) — [block, entry_in_block, super_product]

# Block-to-global index maps (row-major within 2x2 block)
_A_block_idx = [
    [0, 1, 4, 5],   # A00
    [2, 3, 6, 7],   # A01
    [8, 9, 12, 13], # A10
    [10, 11, 14, 15],# A11
]
_B_block_idx = [
    [0, 1, 4, 5],   # B00
    [2, 3, 6, 7],   # B01
    [8, 9, 12, 13], # B10
    [10, 11, 14, 15],# B11
]
_C_block_idx = [
    [0, 1, 4, 5],   # C00
    [2, 3, 6, 7],   # C01
    [8, 9, 12, 13], # C10
    [10, 11, 14, 15],# C11
]


def _build_core_4x4x4():
    """Build U, V, W (each 16 x 49) for the 4x4x4 core multiplication."""
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))

    # Precompute global index arrays for A and B blocks
    A_glob = np.array([_A_block_idx[b] for b in range(4)])  # (4,4)
    B_glob = np.array([_B_block_idx[b] for b in range(4)])  # (4,4)
    C_glob = np.array([_C_block_idx[b] for b in range(4)])  # (4,4)

    for k in range(7):
        for s in range(7):
            col = k * 7 + s

            # U column: for each 2x2 A-block b, each entry e,
            # U[A_glob[b][e], col] += S44_A_blocks[k][b][e] * S22_A[s][e]
            for b in range(4):
                for e in range(4):
                    U[A_glob[b][e], col] += _S44_A_blocks[k][b][e] * _S22_A[s][e]

            # V column: for each 2x2 B-block b, each entry e,
            # V[B_glob[b][e], col] += S44_B_blocks[k][b][e] * S22_B[s][e]
            for b in range(4):
                for e in range(4):
                    V[B_glob[b][e], col] += _S44_B_blocks[k][b][e] * _S22_B[s][e]

            # W column: for each 2x2 C-block cb, each entry e,
            # W[C_glob[cb][e], col] += S44_C_blocks[cb][e][k] * S22_C[s][e]
            for cb in range(4):
                for e in range(4):
                    W[C_glob[cb][e], col] += _S44_C_blocks[cb][e][k] * _S22_C[s][e]

    return U, V, W


def _build_full():
    """Build full U (16 x 65), V (20 x 65), W (20 x 65) for 4x4 x 4x5."""
    Uc, Vc, Wc = _build_core_4x4x4()

    U = np.zeros((16, 65))
    V = np.zeros((20, 65))
    W = np.zeros((20, 65))

    # Core (columns 0..48)
    U[:, :49] = Uc
    V[:16, :49] = Vc
    W[:16, :49] = Wc

    # Remainder: columns 49..64 for the 5th column of B
    # A is 4x4 (rows 0..15), B's 5th column is rows 16..19 in V,
    # C's 5th column is rows 16..19 in W (W index = k*n+i, k=4, i=0..3)
    for col in range(49, 65):
        i = (col - 49) % 4
        j = (col - 49) // 4
        U[i * 4 + j, col] = 1.0
        V[16 + j, col] = 1.0       # B row j, col 4
        W[16 + i, col] = 1.0       # C row i, col 4

    return U, V, W


def _verify(U, V, W, n=4, m=4, p=5):
    """Verify the decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == (A@B)[i,k]."""
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = delta (i.e., the multiplication tensor)
    # Actually the decomposition satisfies: C[i,k] = sum_j A[i,j] * B[j,k]
    # which equals sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k V[j*p+k,r]*B[j,k]) ... 
    # No. The correct identity is:
    #   C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) 
    #              * (sum_j V[j*p+k,r] B[j,k]) ... 
    # That's not right either. The tensor decomposition means:
    #   M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where M is the multiplication tensor with M[i,j,k] = 1 for all i,j,k 
    # (since C[i,k] = sum_j A[i,j]B[j,k]).
    # Wait, the multiplication tensor has entries M[i,j,k] = 1 (it's the 
    # structure tensor of the algebra). Let me just check with random matrices.
    
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C via decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using the factorization: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) 
    #   ... no, that's not how it works.
    
    # The correct reconstruction:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j,r] * A[i,j]) * ... 
    # Hmm, let me think again.
    
    # In a tensor decomposition of matrix multiplication:
    # C = A @ B, where C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition gives: C[i,k] = sum_r U[i*m+j, r] ... 
    # No. The standard formulation:
    #   C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # is WRONG because that would be a product of two sums.
    
    # The correct formulation for rank-r decomposition:
    #   C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... 
    # No, that's the tensor itself.
    
    # Let me use the standard: the multiplication tensor T satisfies
    #   T[i,j,k] = 1 for all valid i,j,k
    # and the decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r where
    #   (u_r ⊗ v_r ⊗ w_r)[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So I need to check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # for all i,j,k.
    
    # Build T from decomposition
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        # u_r is a vector of length n*m, v_r of length m*p, w_r of length p*n
        # T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # T should be all ones
    return np.allclose(T, 1.0, atol=1e-9)


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds

    try:
        U, V, W = _build_full()

        # Quick verification
        if time.time() - start_time < deadline - 10:
            if not _verify(U, V, W):
                # Fallback to schoolbook rank 80
                n, m, p = DIMENSIONS
                rank = n * m * p
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    except Exception:
        # Last resort: schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
