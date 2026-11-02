"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_222():
    """
    Returns Strassen decomposition for 2x2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrices
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U maps A[i,j] -> coefficients
    U = np.zeros((4, 7), dtype=float)
    # V maps B[j,k] -> coefficients
    V = np.zeros((4, 7), dtype=float)
    # W maps C[i,k] -> coefficients (W[k*2+i, r])
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)*(B00+B11) -> C00 += M1, C01 -= M3, C10 -= M4, C11 += M1
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[1, 0] = 1  # B11
    W[0, 0] = 1  # C00 (k=0, i=0)
    W[1, 0] = 1  # C01 (k=1, i=0)
    W[2, 0] = 1  # C10 (k=0, i=1)
    W[3, 0] = 1  # C11 (k=1, i=1)
    
    # M2: (A10+A11)*B00 -> C10 += M2
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 (k=0, i=1)
    
    # M3: A00*(B01-B11) -> C01 -= M3
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # B11
    W[1, 2] = -1  # C01 (k=1, i=0)
    
    # M4: A11*(B10-B00) -> C10 -= M4
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # B00
    W[2, 3] = -1  # C10 (k=0, i=1)
    
    # M5: (A00+A01)*B11 -> C11 += M5
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[3, 4] = 1  # C11 (k=1, i=1)
    
    # M6: (A10-A00)*(B00+B01) -> C01 += M6, C11 += M6
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[1, 5] = 1  # C01 (k=1, i=0)
    W[3, 5] = 1  # C11 (k=1, i=1)
    
    # M7: (A01-A11)*(B10+B11) -> C00 += M7, C10 += M7
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00 (k=0, i=0)
    W[2, 6] = 1  # C10 (k=0, i=1)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Test with random matrices
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _snap_to_half_integer(mat):
    """Snap values to nearest half-integer if close."""
    mat = np.array(mat, dtype=float)
    # Try snapping to half-integers
    snapped = np.round(mat * 2) / 2
    if np.allclose(mat, snapped, atol=1e-9):
        return snapped
    return mat

def _build_strassen_based(n, m, p):
    """
    Build a decomposition using Strassen for 2x2 blocks where possible.
    For 2x5x6, we can use Strassen on 2x2 sub-blocks.
    """
    # For 2x5x6, we have n=2, m=5, p=6
    # We can partition m into 2+2+1 and p into 2+2+2
    # Use Strassen for the 2x2x2 blocks and schoolbook for the rest
    
    # This is complex; let's use a simpler approach:
    # Use Strassen where we have 2x2x2 blocks
    
    # Actually, let's construct it more carefully.
    # The key insight: for each pair of rows in A and columns in B,
    # we can use Strassen if both dimensions allow 2x2 blocks.
    
    # For n=2, m=5, p=6:
    # Partition m into [2, 2, 1] and p into [2, 2, 2]
    # For blocks where m_block=2 and p_block=2, use Strassen (rank 7 instead of 8)
    # For other blocks, use schoolbook
    
    # Let's count the blocks:
    # m partitions: [2, 2, 1]
    # p partitions: [2, 2, 2]
    
    # Blocks (m_idx, p_idx):
    # (0,0): m=2, p=2 -> Strassen, rank 7
    # (0,1): m=2, p=2 -> Strassen, rank 7
    # (0,2): m=2, p=2 -> Strassen, rank 7
    # (1,0): m=2, p=2 -> Strassen, rank 7
    # (1,1): m=2, p=2 -> Strassen, rank 7
    # (1,2): m=2, p=2 -> Strassen, rank 7
    # (2,0): m=1, p=2 -> schoolbook, rank 2
    # (2,1): m=1, p=2 -> schoolbook, rank 2
    # (2,2): m=1, p=2 -> schoolbook, rank 2
    
    # Total rank: 6*7 + 3*2 = 42 + 6 = 48
    
    # Wait, that's not right. Let me reconsider.
    # For each (i, k) output, we sum over j.
    # The decomposition is per (i, j, k) triple in schoolbook.
    # With Strassen, we group 2x2 blocks of (j, k) for fixed i.
    
    # Actually, the standard approach is:
    # For each i (row of A), we compute row i of C.
    # C[i, :] = A[i, :] @ B
    # This is a (1 x m) @ (m x p) multiplication.
    # We can partition m into blocks and use Strassen where possible.
    
    # But Strassen requires square matrices. For 1xm @ mxp, we can't directly use Strassen.
    
    # Let me think differently. The tensor decomposition for matrix multiplication
    # M_{n,m,p} has entries indexed by (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    # Schoolbook: rank = n*m*p, with one term per (i,j,k).
    # Strassen for 2x2x2: rank 7 instead of 8.
    
    # For 2x5x6, we can try to find a decomposition that uses Strassen-like tricks.
    # One approach: partition the j and k indices into blocks.
    
    # Let's try: partition m=5 into [2, 2, 1] and p=6 into [2, 2, 2].
    # For each i in {0, 1}, and each block pair (m_block, p_block):
    # If m_block=2 and p_block=2, use Strassen (7 terms)
    # Otherwise, use schoolbook (m_block * p_block terms)
    
    # But this doesn't quite work because Strassen combines terms across i as well.
    
    # Let me just implement the schoolbook version first and verify it works,
    # then try to improve.
    
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

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_strassen_based(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to pure schoolbook
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
    
    # Snap to half-integers
    U = _snap_to_half_integer(U)
    V = _snap_to_half_integer(V)
    W = _snap_to_half_integer(W)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
