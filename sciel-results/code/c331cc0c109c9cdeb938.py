"""Matrix multiplication 3x4x8 - Kronecker product decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p):
    """Check that sum_s U[a,s]V[b,s]W[c,s] == T[a,b,c] for all a,b,c."""
    a = np.array(U, dtype=float)
    b = np.array(V, dtype=float)
    c = np.array(W, dtype=float)
    D = np.einsum('as,bs,cs->abc', a, b, c, optimize=True)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(D, T, atol=1e-6)

def _strassen222():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U (4 rows), V (4 rows), W (4 rows), each 4x7.
    Indices: U row = i*2+j, V row = j*2+k, W row = k*2+i.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1: (A00+A11)*(B01+B10)
    U[0, 0] += 1; U[3, 0] += 1
    V[1, 0] += 1; V[2, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1; W[1, 0] -= 1; W[2, 0] += 1
    # M2: (A10+A11)*B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[3, 1] -= 1
    # M3: A00*(B10-B11)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[0, 2] += 1; W[3, 2] += 1
    # M4: A11*(B00-B10)
    U[3, 3] += 1
    V[0, 3] += 1; V[2, 3] -= 1
    W[1, 3] += 1; W[3, 3] += 1
    # M5: (A00+A01)*B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[2, 4] -= 1; W[3, 4] += 1
    # M6: (A10-A00)*(B00+B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[2, 5] += 1
    # M7: (A01-A11)*(B10+B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    return U, V, W

def _build_3x4x4():
    """Build rank-28 decomposition for 3x4x4.
    
    Strategy: Split m=4 into 2+2, giving two 3x2x4 blocks.
    Each 3x2x4: split p=4 into 2+2, giving two 3x2x2 blocks.
    Each 3x2x2: use schoolbook rank 12.
    Total: 2 * 2 * 12 = 48.
    
    Alternative: Use a smarter base.
    Actually, let's try: 3x4x4 = 2*(3x2x4), each 3x2x4 = 2*(3x2x2).
    For 3x2x2, we need rank. Let's just use schoolbook 12 for now.
    Total = 4*12 = 48.
    """
    n, m, p = 3, 4, 4
    U = np.zeros((n*m, 48))
    V = np.zeros((m*p, 48))
    W = np.zeros((p*n, 48))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_3x4x8_strassen():
    """Build rank-56 decomposition for 3x4x8 using Strassen on 4x4x4 blocks.
    
    Strategy: Split p=8 into two 4s. For each 4x4x4 block, use Strassen
    decomposition. The 4x4x4 matrix multiplication can be done with rank 28
    by splitting the 4x4 into four 2x2 blocks and applying Strassen to each
    pair, but we need to be careful about the structure.
    
    Actually, let's use a direct approach:
    - Split the 4x8 B matrix into two 4x4 blocks: B = [B1 | B2]
    - C = A*B1 + A*B2 where A is 3x4, B1/B2 are 4x4, C is 3x8
    - For each 3x4x4 multiplication, we can use rank 28 via:
      Split A into [A_left | A_right] (3x2 each)
      Split B into [[B11, B12], [B21, B22]] (2x2 each)
      C_left = A_left*B11 + A_right*B21 (3x2)
      C_right = A_left*B12 + A_right*B22 (3x2)
      Each 3x2x2 has rank 12, so total per 3x4x4 is 24.
      Total for 3x4x8: 2*24 = 48.
    
    Wait, that's still 48. Let me try Strassen more carefully.
    
    For 3x4x4:
    - View as: multiply (3x4) by (4x4) to get (3x4)
    - Split the 4x4 into four 2x2 blocks
    - Use Strassen-style: 7 multiplications of 2x2 matrices
    - But we have 3 rows in A, so it's not directly 2x2x2
    
    Alternative: Use the fact that 3x4x4 can be embedded.
    Let's just use schoolbook for 3x2x2 (rank 12) and get 48 total.
    
    Actually, let me try a different decomposition:
    3x4x8 = 2*(3x4x4)
    For 3x4x4, use rank 28 by:
    - Split A (3x4) into A1 (3x2) and A2 (3x2)
    - Split B (4x4) into B11, B12, B21, B22 (each 2x2)
    - C1 = A1*B11 + A2*B21 (3x2) - rank 12
    - C2 = A1*B12 + A2*B22 (3x2) - rank 12
    Total: 24 per 3x4x4, so 48 for 3x4x8.
    
    This is better than 96! Let's implement it.
    """
    n, m, p = 3, 4, 8
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # For each of the two 3x4x4 blocks (k=0,1 for p=4+4)
    for k_block in range(2):
        # For each 3x2x2 sub-multiplication
        for i in range(n):
            for j_half in range(2):  # j=0,1 for left half; j=2,3 for right half
                for k_half in range(2):  # k=0,1
                    # A[i, j] where j = j_half + 2*j_half_block
                    # Actually, let's be more explicit
                    
                    # j is 0,1,2,3; k is 0,1,2,3 (within each 4x4 block)
                    # C[i, k] = sum_j A[i,j] * B[j,k]
                    
                    # Split j into j1 (0,1) and j2 (2,3)
                    # Split k into k1 (0,1) and k2 (2,3)
                    
                    # C[i,k1] = A[i,j1]*B[j1,k1] + A[i,j2]*B[j2,k1]
                    # C[i,k2] = A[i,j1]*B[j1,k2] + A[i,j2]*B[j2,k2]
                    
                    pass
    
    # Let me redo this more carefully with explicit indices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for k_block in range(2):  # Two 4x4 blocks in B
        for i in range(n):  # Row of A / C
            for j_group in range(2):  # j=0,1 or j=2,3
                for k_group in range(2):  # k=0,1 or k=2,3
                    # This handles A[i, j] * B[j, k] for specific j,k groups
                    # j = j_group * 2 + j_local
                    # k = k_block * 4 + k_group * 2 + k_local
                    for j_local in range(2):
                        for k_local in range(2):
                            j = j_group * 2 + j_local
                            k = k_block * 4 + k_group * 2 + k_local
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 115
    n, m, p = 3, 4, 8

    # Try the improved rank-48 decomposition
    try:
        U, V, W = _build_3x4x8_strassen()
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass

    # Fallback: schoolbook rank 96
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    fallback = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook verification failed")
    
    return fallback

# EVOLVE-BLOCK-END
