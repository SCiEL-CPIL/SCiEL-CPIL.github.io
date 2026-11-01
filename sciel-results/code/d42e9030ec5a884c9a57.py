"""
Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x9 matrix.

Key insight: M_{2,2,9} = M_{2,2,2} ⊗ M_{2,1,9} (tensor/Kronecker product structure).
- M_{2,2,2} has rank 7 (Strassen)
- M_{2,1,9} has rank 9 (schoolbook: 2 inputs × 9 outputs, each output needs both inputs)

Wait, let me reconsider. M_{2,1,9}: multiply 2x1 by 1x9 = 2x9. Rank is 2*9=18? No.

Actually, M_{n,m,p}: A is n×m, B is m×p, C is n×p.
M_{2,1,9}: A is 2×1, B is 1×9, C is 2×9.
Each C[i,k] = A[i,0]*B[0,k]. That's 2*9=18 terms schoolbook.
But rank of M_{2,1,9} = 2*1*9? No, the tensor has entries T[i*1+0, 0*9+k, k*2+i] = 1.
That's 18 non-zero entries, each in a different "fiber", so rank = 18? No, rank can be less.

Actually for M_{2,1,9}: T has T[a, b, c] = 1 when a=i, b=k, c=2k+i for i in {0,1}, k in {0..8}.
These are 18 independent rank-1 tensors (each has a unique triple), so rank = 18.

Hmm, that gives 7*18 = 126 which is worse. Let me reconsider the decomposition.

Better approach: M_{2,2,9} can be seen as multiplying a 2x2 matrix A by a 2x9 matrix B.
C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..8}, j in {0,1}.

Alternative: think of it as two rows of A times B.
Row 0 of C: A[0,0]*B[0,:] + A[0,1]*B[1,:]
Row 1 of C: A[1,0]*B[0,:] + A[1,1]*B[1,:]

This is like: for each row i of C, we compute A[i,0]*B[0,:] + A[i,1]*B[1,:].
That's 2 rows × (2 multiplications of 1×9 vectors) = 4 vector multiplications.
Each 1×9 × scalar is rank 9? No, scalar times vector is just scaling.

Let me think differently. The tensor T_{i,j,k} = 1 if the index matches.
T is in R^{(2*2)} × R^{(2*9)} × R^{(9*2)} = R^4 × R^18 × R^18.

T[i*2+j, j*9+k, k*2+i] = 1 for i∈{0,1}, j∈{0,1}, k∈{0..8}.

Total 2*2*9 = 36 non-zero entries.

Strassen approach for 4 blocks of 2x2x2 + 1 block of 2x2x1 gives 4*7+4=32.

Can we do better? Let's try rank 28 = 7*4.
Idea: Group the 9 columns into 4 groups: {0,1}, {2,3}, {4,5}, {6,7,8}?
No, that's uneven.

Actually, the tensor product approach: if we can write M_{2,2,9} as a sum of rank-1 tensors
that factor through a lower-rank intermediate, we can exploit structure.

Let me try: treat B as a 2×9 matrix. Write B = [B_0 | B_1] where B_0 is 2×9.
C = A * B where A is 2×2.

C[:, k] = A * B[:, k] for each column k.
So C is determined by 9 independent 2×2×1 multiplications? No, they share the same A.

Key: C[:,k] = A * b_k where b_k is column k of B (a 2-vector).
So we need to compute A times 9 different 2-vectors.

Using Strassen: A * x for a 2-vector x can be done with... well Strassen is for 2×2 × 2×2.
For 2×2 × 2×1, schoolbook is 4 mults, and that's optimal (rank 4).

But we have 9 such multiplications sharing A. Can we batch them?
The full problem is: given A (2×2) and B (2×9), compute C = AB (2×9).

This is equivalent to: given 4 scalars (A) and 18 scalars (B), produce 18 products.
The tensor is T[a_idx, b_idx, c_idx] where a_idx = i*2+j, b_idx = j*9+k, c_idx = k*2+i.

Let me try a different factorization. Note that:
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

Define for each k: C[:,k] = [A[i,0]*B[0,k] + A[i,1]*B[1,k]]_{i=0,1}

This is a linear combination. For fixed k, it's a 2×2 × 2×1 matmul.

Total rank if independent: 9 * 4 = 36 (schoolbook).
With Strassen on 2×2×2 blocks: 4 blocks × 7 + 1 block × 4 = 32.

Can we do 28? Let's think about M_{2,2,9} more carefully.

Actually, I recall that M_{2,2,n} has been studied. For n=9, the best known might be around 28-32.

Let me try a Kronecker product approach:
M_{2,2,9} ⊇ M_{2,2,4} ⊕ M_{2,2,5} (split into 4+5 columns)
rank(M_{2,2,4}) ≤ 4*7 = 28 (Strassen on 4 blocks)
rank(M_{2,2,5}) ≤ ? 

Or: M_{2,2,9} as M_{2,2,2} ⊗ M_{2,1,9/2}? Doesn't divide evenly.

Let me just try to verify rank 28 is achievable with a different structure.

Actually, here's a cleaner idea: 
- Use Strassen (rank 7) for the 2×2×2 "core" 
- The 9 columns can be handled by noting that B is 2×9, and we need A*B.
- If we write the computation as: for each of 9 columns, do a 2×2×1 multiply.
- But Strassen computes 2×2×2 with 7 mults. If we pack 2 columns together: 2×2×2 with 7 mults handles 2 output columns.
- 9 columns = 4 pairs (8 cols) + 1 single.
- 4 × 7 + 4 = 32. Same as before.

For rank 28, we'd need 28 = 7×4. This suggests 4 "Strassen units" each handling the full 2×2×2, but we have 9 columns not 8.

Unless we can overlap: use 7 mults per 2-column group, and for the 9th column, share some of the Strassen intermediates.

Let me just implement the rank-32 solution cleanly and also try to find rank 28 or 31 via search.
"""

import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    
    Indexing convention:
    - U: shape (n*m, rank) = (4, rank). U[i*m + j, r]
    - V: shape (m*p, rank) = (18, rank). V[j*p + k, r]
    - W: shape (p*n, rank) = (18, rank). W[k*n + i, r]
    
    Tensor T[i*m+j, j*p+k, k*n+i] = 1
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]      # 4 x 32
    V = [[0] * rank for _ in range(m * p)]      # 18 x 32
    W = [[0] * rank for _ in range(p * n)]      # 18 x 32

    # Strassen for 2x2x2: 
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = A*B
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # U index: i*m+j -> i*2+j: a00=0, a01=1, a10=2, a11=3
    # V index: j*p+k -> for a 2x2 block at columns (2b, 2b+1):
    #   j=0,k=0 -> V[0*2+0] = V[0] within block -> global V[2b+0]
    #   j=0,k=1 -> V[0*2+1] = V[1] within block -> global V[2b+1]
    #   j=1,k=0 -> V[1*2+0] = V[2] within block -> global V[2b+2]
    #   j=1,k=1 -> V[1*2+1] = V[3] within block -> global V[2b+3]
    # W index: k*n+i -> for a 2x2 block at columns (2b, 2b+1):
    #   k=0,i=0 -> W[0*2+0] = W[0] within block -> global W[2b+0]
    #   k=0,i=1 -> W[0*2+1] = W[1] within block -> global W[2b+1]
    #   k=1,i=0 -> W[1*2+0] = W[2] within block -> global W[2b+2]
    #   k=1,i=1 -> W[1*2+1] = W[3] within block -> global W[2b+3]
    #
    # Strassen:
    # P0 = (a00+a11)*(b00+b11) -> contributes to C00+=P0, C01+=P0, C10+=P0, C11-=P0
    # P1 = (a10+a11)*b00       -> contributes to C10+=P1, C11-=P1
    # P2 = a00*(b01-b11)       -> contributes to C01+=P2, C11+=P2
    # P3 = a11*(b10-b00)       -> contributes to C00+=P3, C10-=P3
    # P4 = (a00+a01)*b11       -> contributes to C01+=P4, C11+=P4
    # P5 = (a10-a00)*(b00+b01) -> contributes to C00-=P5, C10+=P5
    # P6 = (a01-a11)*(b10+b11) -> contributes to C00+=P6, C01+=P6

    # For each Strassen term, specify:
    #   u: list of (U_local_idx, coeff) where U_local_idx in {0,1,2,3} for a00,a01,a10,a11
    #   v: list of (V_local_idx, coeff) where V_local_idx in {0,1,2,3} for b00,b01,b10,b11
    #   w: list of (W_local_idx, coeff) where W_local_idx in {0,1,2,3} for C00,C01,C10,C11
    #     W_local: k*2+i: C00=0, C01=1, C10=2, C11=3
    
    strassen_terms = [
        # P0 = (a00+a11)*(b00+b11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (1, 1), (2, 1), (3, -1) ]),
        # P1 = (a10+a11)*b00
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # P2 = a00*(b01-b11)
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # P3 = a11*(b10-b00)
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        # P4 = (a00+a01)*b11
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        # P5 = (a10-a00)*(b00+b01)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        # P6 = (a01-a11)*(b10+b11)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]

    # Fill in 4 Strassen blocks for columns 0-7
    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]

            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff

            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx][col] += coeff

            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx][col] += coeff

    # Schoolbook terms for column 8 (2x2x1 block)
    # For k=8: 
    #   V[j*9+8]: j=0 -> V[8], j=1 -> V[17]
    #   W[8*2+i]: i=0 -> W[16], i=1 -> W[17]
    # C[0,8] = a00*b[0,8] + a01*b[1,8]
    # C[1,8] = a10*b[0,8] + a11*b[1,8]
    schoolbook_terms = [
        ([ (0, 1) ], [ (8, 1) ],  [ (16, 1) ]),   # a00 * b08 -> C08
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),   # a01 * b18 -> C08
        ([ (2, 1) ], [ (8, 1) ],  [ (17, 1) ]),   # a10 * b08 -> C18
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),   # a11 * b18 -> C18
    ]

    col_start = 4 * 7  # column 28
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]

        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff

        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff

        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff

    return U, V, W


def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _build_rank_28_tensor_product(n, m, p):
    """
    Try to build a rank-28 decomposition using the tensor product structure.
    
    M_{2,2,9} can potentially be decomposed by grouping the 9 output columns.
    
    Approach: Use the fact that M_{2,2,4} can be done with rank 4*7=28 (Strassen on 4 blocks)
    and M_{2,2,5} separately, but that gives more than 28.
    
    Alternative: Try a different grouping. Actually, let's try to see if we can
    achieve rank 28 by a more clever arrangement.
    
    For now, return None if we can't find one.
    """
    return None


def _try_rank_31(n, m, p):
    """
    Try to find a rank-31 decomposition by checking if any single term
    in the rank-32 solution is redundant.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    U32, V32, W32 = _build_rank_32(n, m, p)
    Ua = np.array(U32, dtype=float)
    Va = np.array(V32, dtype=float)
    Wa = np.array(W32, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    for r in range(32):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        mask = [i for i in range(32) if i != r]
        U = Ua[:, mask]
        V = Va[:, mask]
        W = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T, T_recon, atol=1e-10):
            return (
                [[U[i][j] for j in range(31)] for i in range(n * m)],
                [[V[i][j] for j in range(31)] for i in range(m * p)],
                [[W[i][j] for j in range(31)] for i in range(p * n)],
            )
    return None


def _try_rank_28_search(n, m, p):
    """
    Attempt to find a rank-28 decomposition using a structured search.
    
    Strategy: The tensor M_{2,2,9} has a natural decomposition. We try to find
    28 rank-1 tensors that sum to the multiplication tensor.
    
    Use the structure: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    For rank 28 = 7 * 4, we hypothesize a tensor product structure where
    7 Strassen-like terms handle the 2x2 aspect and 4 terms handle grouping
    of the 9 columns.
    
    This is a heuristic search - try random perturbations of the rank-32 solution
    and see if we can reduce rank.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 20:
        return None

    U32, V32, W32 = _build_rank_32(n, m, p)
    Ua = np.array(U32, dtype=float)
    Va = np.array(V32, dtype=float)
    Wa = np.array(W32, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try removing pairs of terms and checking if the remaining 30 can be
    # adjusted (via SVD-like projection) to still span the tensor.
    # This is a heuristic - check if any 28-column subset of the rank-32
    # solution can be completed to exactly represent T.
    
    # For each pair of columns to remove, check if T is in the span of
    # the remaining 30 rank-1 tensors (it won't be, but check if we can
    # find 28 that work).
    
    # Actually, a simpler check: for each subset of 28 columns out of 32,
    # check if the tensor can be represented. This is C(32,28) = C(32,4) = 35960.
    # Too many. Instead, try removing 4 columns at a time from specific patterns.
    
    # Try: remove 1 column from each of 4 Strassen blocks (7->6 each) + keep schoolbook
    # That's 6*4 + 4 = 28.
    if time.time() - _START_TIME > _TIMEOUT - 10:
        return None
    
    # For each combination of which Strassen term to drop from each block
    # 7^4 = 2401 combinations
    import itertools
    
    for drops in itertools.product(range(7), repeat=4):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        
        # Build mask: keep all columns except the dropped ones
        mask = []
        for block in range(4):
            col_start = block * 7
            for t in range(7):
                if t != drops[block]:
                    mask.append(col_start + t)
        # Add schoolbook columns (28-31)
        mask.extend(range(28, 32))
        
        U = Ua[:, mask]
        V = Va[:, mask]
        W = Wa[:, mask]
        
        # Check if T can be represented by these 28 rank-1 tensors
        # T_recon = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We need T_recon == T exactly.
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T, T_recon, atol=1e-10):
            return (
                [[U[i][j] for j in range(28)] for i in range(n * m)],
                [[V[i][j] for j in range(28)] for i in range(m * p)],
                [[W[i][j] for j in range(28)] for i in range(p * n)],
            )
    
    return None


def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME

    # Try rank 28 first (tensor product structured search)
    if elapsed < _TIMEOUT - 25:
        result = _try_rank_28_search(n, m, p)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Try rank 31
    if elapsed < _TIMEOUT - 10:
        result = _try_rank_31(n, m, p)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Fall back to rank-32 Strassen-based decomposition
    U, V, W = _build_rank_32(n, m, p)
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Final fallback: schoolbook rank 36
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
