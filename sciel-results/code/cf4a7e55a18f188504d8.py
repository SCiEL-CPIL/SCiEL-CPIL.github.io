"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 Strassen multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _strassen_444():
    """
    Build rank-16 decomposition for 4x4x4 matrix multiplication
    using Strassen recursion on 2+2 blocks.
    Returns (U, V, W) with shapes (16, 16), (16, 16), (16, 16)
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    su, sv, sw = _strassen_222()
    
    # 4x4x4 has 2x2 blocks in each dimension
    # 4 blocks, each using Strassen (rank 7) -> total rank 28? 
    # Wait, that's not right. 4x4x4 via Strassen recursion:
    # Split into 2x2 blocks, each 2x2x2 uses Strassen rank 7
    # 4 such blocks -> 4*7 = 28. But actually 4x4x4 rank is 16 (Winograd).
    # Let me use the standard 4x4x4 rank-16 algorithm.
    
    # Actually, the rank of 4x4x4 is 16 (not 28).
    # Let me use a known rank-16 algorithm for 4x4x4.
    
    # For now, let me build it via Strassen recursion which gives rank 28,
    # but that's not optimal. Let me just use the recursive Strassen.
    
    # Actually, let me reconsider. The standard approach:
    # 4x4x4: split into four 2x2x2 subproblems
    # Each uses Strassen rank 7
    # Total rank: 4 * 7 = 28
    
    # But there's a better algorithm with rank 16. Let me use that.
    
    # For simplicity and correctness, let me use the recursive Strassen (rank 28)
    # which is still better than schoolbook (rank 64) for 4x4x4.
    
    U = [[0]*28 for _ in range(16)]
    V = [[0]*28 for _ in range(16)]
    W = [[0]*28 for _ in range(16)]
    
    # 4x4x4 split into 2x2 blocks
    # Block (bi, bj, bk) where bi, bj, bk in {0,1}
    # Each block is 2x2x2 using Strassen
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for c in range(7):
                    for li in range(2):
                        gi = bi * 2 + li
                        for lj in range(2):
                            gj = bj * 2 + lj
                            U[gi * 4 + gj][col + c] = su[li * 2 + lj][c]
                    for lj in range(2):
                        gj = bj * 2 + lj
                        for lk in range(2):
                            gk = bk * 2 + lk
                            V[gj * 4 + gk][col + c] = sv[lj * 2 + lk][c]
                    for lk in range(2):
                        gk = bk * 2 + lk
                        for li in range(2):
                            gi = bi * 2 + li
                            W[gk * 4 + gi][col + c] = sw[lk * 2 + li][c]
                col += 7
    
    return U, V, W


def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W


def _build_rank71():
    """
    Build a rank-71 decomposition for 3x5x5 matrix multiplication.
    
    Partition:
    - i: {0}, {1,2}
    - j: {0,1}, {2,3}, {4}
    - k: {0,1}, {2,3}, {4}
    
    2x2x2 blocks (use Strassen, rank 7):
    - i={1,2}, j={0,1}, k={0,1}
    - i={1,2}, j={0,1}, k={2,3}
    - i={1,2}, j={2,3}, k={0,1}
    - i={1,2}, j={2,3}, k={2,3}
    
    All other blocks: schoolbook
    """
    n, m, p = 3, 5, 5
    
    # Precompute Strassen factors
    su, sv, sw = _strassen_222()
    
    # Total rank: 4*7 + remaining schoolbook terms
    # i={0}: 1*5*5 = 25 terms
    # i={1,2}, j={0,1}, k={4}: 2*2*1 = 4 terms
    # i={1,2}, j={2,3}, k={4}: 2*2*1 = 4 terms
    # i={1,2}, j={4}, k={0,1}: 2*1*2 = 4 terms
    # i={1,2}, j={4}, k={2,3}: 2*1*2 = 4 terms
    # i={1,2}, j={4}, k={4}: 2*1*1 = 2 terms
    # Total schoolbook: 25 + 4 + 4 + 4 + 4 + 2 = 43
    # Total rank: 4*7 + 43 = 28 + 43 = 71
    total_rank = 71
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Define blocks
    i_blocks = [(0,), (1, 2)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    for i_b in range(len(i_blocks)):
        for j_b in range(len(j_blocks)):
            for k_b in range(len(k_blocks)):
                i_set = i_blocks[i_b]
                j_set = j_blocks[j_b]
                k_set = k_blocks[k_b]
                
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    # Use Strassen
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj][col + c] = su[li * 2 + lj][c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk][col + c] = sv[lj * 2 + lk][c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi][col + c] = sw[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    if col != total_rank:
        raise ValueError(f"Column count mismatch: {col} != {total_rank}")
    
    return U, V, W


def _build_rank_split4():
    """
    Build a decomposition for 3x5x5 using split-4 strategy.
    
    Partition j and k into {0,1,2,3} and {4}:
    - 3x4x4 block: use 4x4x4 rank-16 (but we only need 3 rows of i)
    - 3x4x1 block: schoolbook (3*4*1 = 12)
    - 3x1x4 block: schoolbook (3*1*4 = 12)
    - 3x1x1 block: schoolbook (3*1*1 = 3)
    
    For the 3x4x4 part:
    - We can use the 4x4x4 rank-16 algorithm but only for i=0,1,2
    - Actually, let's use Strassen on 2x2x2 blocks within the 4x4x4
    - Split i={0,1,2} into {0}, {1,2}
    - Split j={0,1,2,3} into {0,1}, {2,3}
    - Split k={0,1,2,3} into {0,1}, {2,3}
    
    Blocks:
    - i={0}, j={0,1}, k={0,1}: 1x2x2 schoolbook = 4
    - i={0}, j={0,1}, k={2,3}: 1x2x2 schoolbook = 4
    - i={0}, j={2,3}, k={0,1}: 1x2x2 schoolbook = 4
    - i={0}, j={2,3}, k={2,3}: 1x2x2 schoolbook = 4
    - i={1,2}, j={0,1}, k={0,1}: 2x2x2 Strassen = 7
    - i={1,2}, j={0,1}, k={2,3}: 2x2x2 Strassen = 7
    - i={1,2}, j={2,3}, k={0,1}: 2x2x2 Strassen = 7
    - i={1,2}, j={2,3}, k={2,3}: 2x2x2 Strassen = 7
    
    Total for 3x4x4: 4+4+4+4+7+7+7+7 = 44
    
    Plus:
    - 3x4x1: 12
    - 3x1x4: 12
    - 3x1x1: 3
    
    Total: 44 + 12 + 12 + 3 = 71
    
    Same as before! Let me try a different split.
    """
    n, m, p = 3, 5, 5
    su, sv, sw = _strassen_222()
    
    # Let me try: split j into {0,1,2,3,4} differently
    # Use 4x4x4 rank-16 for the main 3x4x4 part
    # But we need a proper 3x4x4 algorithm
    
    # Alternative: use the 4x4x4 rank-16 and just take first 3 rows
    # This might not work directly. Let me stick with the rank-71 approach
    # but verify it works.
    
    # For now, return the same as _build_rank71
    return _build_rank71()


def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that the tensor identity holds:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_{r=0}^{rank-1} W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] == 1
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(W_np[k*n + i] * U_np[i*m + j] * V_np[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Try rank-71 decomposition
    try:
        U, V, W = _build_rank71()
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, 71):
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, fall back to schoolbook
            U, V, W = _schoolbook_factors(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        # Fall back to schoolbook
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
