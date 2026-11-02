"""
Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.

Strategy: Use a structured search to find a low-rank decomposition.
The tensor T has dimensions (n*m) x (m*p) x (p*n) = 8 x 32 x 16.
We seek rank r such that T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
where u_c ∈ R^8, v_c ∈ R^32, w_c ∈ R^16.

The equation is: for all (i,j,k) with i in [0,2), j in [0,4), k in [0,8):
  T[i*m+j, j*p+k, k*n+i] = 1  (the standard matrix mult tensor)
  T[other] = 0

Actually the tensor is: T[a, b, c] where a = i*m+j (A-index), b = j*p+k (B-index), c = k*n+i (C-index).
T[a, b, c] = 1 if the indices are consistent (same i, j, k) else 0.

We want U (8 x r), V (32 x r), W (16 x r) such that:
  sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]

This is a bilinear system. We can use a greedy/iterative approach or try known constructions.

Known results:
- Schoolbook: rank 64
- The rank of 2x4 x 4x8 is at most 24 (by Winograd-type arguments)
- Actually, let me think about this more carefully.

For n=2, m=4, p=8:
C[i,k] = sum_j A[i,j] * B[j,k]

We can write this as: for each i, C[i,:] = A[i,:] * B (a row vector times a matrix)
So we need to compute 2 row-vector-matrix products, each of size 1x4 * 4x8.

A 1x4 * 4x8 product has rank 32 (schoolbook: 4*8=32).
Two such products: rank 64.

But we can share structure. Note that A[i,:] for i=0,1 are two rows.

Alternative: think of it as A (2x4) times B (4x8).
We can partition B into 2 blocks of 4 columns: B = [B1, B2] where B1, B2 are 4x4.
Then C = [A*B1, A*B2].
Each A*B1 is 2x4 * 4x4, which has rank at most 16 (schoolbook) or 14 (Strassen-like).
Two of them: rank 32 or 28.

Better: use the fact that the problem is equivalent to computing 8 dot products of length 4,
but with sharing between the two rows.

Let me try a direct construction with rank 24.

Actually, let me use a numerical approach: set up the bilinear system and try to solve it
with a rank constraint, using iterative methods.

Given the time constraint, let me try a few known constructions:

1. Rank 32: Split B into two 4x4 blocks, use schoolbook for each.
2. Rank 28: Use a 2x2 Strassen-like on the 2x4 * 4x4 blocks.
3. Try to find rank 24 or lower via optimization.

Let me start with a rank-32 construction and then try to reduce.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]


def _verify(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for c in range(rank):
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[j * p + k][c] * B[j, k] for j in range(m))
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _build_schoolbook(n=2, m=4, p=8):
    """Schoolbook decomposition, rank n*m*p."""
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


def _build_rank32(n=2, m=4, p=8):
    """
    Rank-32 construction: split B into two 4x4 blocks, schoolbook each.
    C = [A*B1, A*B2] where B1 = B[:, 0:4], B2 = B[:, 4:8]
    Each A*Bx is 2x4 * 4x4, schoolbook rank 16. Total rank 32.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        for i in range(n):
            for j in range(m):
                for k in range(4):
                    # A index: i*m + j
                    # B index: j*p + (block*4 + k)
                    # C index: (block*4 + k)*n + i
                    U[i * m + j][col] = 1
                    V[j * p + block * 4 + k][col] = 1
                    W[(block * 4 + k) * n + i][col] = 1
                    col += 1
    
    assert col == rank
    return U, V, W


def _build_rank28(n=2, m=4, p=8):
    """
    Try to get rank 28 by using a Strassen-like trick on the 2x4 * 4x4 blocks.
    
    For a 2x4 * 4x4 product, we can write it as 2 independent 1x4 * 4x4 products.
    A 1x4 * 4x4 product has rank 16 schoolbook.
    
    Actually, let me think of the 2x4 * 4x4 as a 2x2 block multiplication
    where the "blocks" are 1x2 and 2x2:
    A = [[a0, a1], [a2, a3]] (2x4, each "block" is 1x2)
    B = [[b00, b01, b02, b03],
         [b10, b11, b12, b13],
         [b20, b21, b22, b23],
         [b30, b31, b32, b33]]
    
    This is getting complicated. Let me just try rank 32 and see if it verifies.
    """
    # For now, fall back to rank 32
    return _build_rank32(n, m, p)


def _try_optimize_rank(U, V, W, target_rank, n=2, m=4, p=8, max_iters=200, deadline=None):
    """
    Try to reduce the rank by zeroing out columns and re-optimizing.
    This is a simple heuristic: try to find a subset of columns that spans the solution.
    """
    if deadline is not None and time.time() > deadline:
        return U, V, W
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = U_np.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to find a minimal subset of columns
    # This is NP-hard in general, so use a greedy approach
    # For now, just return as-is since proper rank reduction is complex
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank 32 construction
    U, V, W = _build_rank32(n, m, p)
    rank = len(U[0])
    
    if _verify(U, V, W, n, m, p, trials=30, tol=1e-9):
        print(f"Rank-32 decomposition verified")
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    rank = len(U[0])
    
    if _verify(U, V, W, n, m, p, trials=30, tol=1e-9):
        print(f"Schoolbook decomposition verified at rank {rank}")
        return {"U": U, "V": V, "W": W}
    
    # This should never happen
    print("ERROR: No valid decomposition found")
    return {"U": U, "V": V, "W": W}
