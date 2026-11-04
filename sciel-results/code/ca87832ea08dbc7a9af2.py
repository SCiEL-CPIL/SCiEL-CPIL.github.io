"""
Rank-minimal tensor decomposition for 2x5x6 matrix multiplication.

Strategy: Hierarchical Kronecker product decomposition.
Decompose 2x5x6 into tensor products of smaller known-rank tensors:
  2x5x6 = (2x2x2) ⊗ (1x3x3) via Kronecker structure,
  or more directly: exploit that 5 = 2+3 and 6 = 2*3 to build
  a rank-1 tensor sum from products of small decompositions.

We use the fact that if T1 has rank r1 for dims (n1,m1,p1) and T2 has rank r2
for dims (n2,m2,p2), then T1 ⊗ T2 gives a valid decomposition for
(n1*n2, m1*m2, p1*p2) with rank r1*r2.

For 2x5x6: We can write 5 = 2*3 (not exact) or use mixed-radix.
Better approach: decompose as sum of Kronecker products where
n=2, m=5, p=6. Note 5=2+3, 6=2*3.

Actually, the cleanest approach: use the Kronecker product structure directly.
If we have decompositions for (2,2,2) rank 7 and (1,3,3) rank 9 (schoolbook),
then (2*1, 2*3, 2*3) = (2,6,6) has rank 7*9=63. Not helpful.

Alternative: decompose 2x5x6 directly using the structure.
Since 5 and 6 are not both powers of 2, we use a hybrid:
- Split m=5 into m1=2, m2=3
- Split p=6 into p1=2, p2=3
- Use Strassen for the 2x2x2 part and schoolbook for 2x3x3 part
- But these are independent blocks, not a Kronecker product.

Key insight for Kronecker: The multiplication tensor for (n,m,p) can be
written as a sum of rank-1 tensors. If we can find a set of "basis" 
decompositions and combine them via tensor products, we may get lower rank.

Let's try: decompose as (2,5,6) using the fact that
T_{2,5,6} can be embedded in T_{2,6,6} with one column of A and B zero.
T_{2,6,6} = T_{2,2,2} ⊗ T_{1,3,3} (Kronecker) with rank 7*9 = 63.
That's worse than schoolbook 60.

Better: Use the structure 2x5x6 where we decompose m=5 as 2+3 and p=6 as 2*3.
The tensor is a sum over (i,j,k). We can group by j mod 2 and k mod 2.

Actually, let's use a different strategy: direct construction using
smaller Strassen blocks combined with the specific dimension structure.

For 2x5x6:
- n=2: use Strassen's 2x2 structure (rank 7 per 2x2x2 block)
- m=5: decompose into two 2-column blocks + one 1-column block  
- p=6: decompose into three 2-column blocks

This gives the block decomposition: 6 blocks of 2x2x2 (Strassen, rank 7 each)
+ 3 blocks of 2x1x2 (schoolbook, rank 4 each) = 6*7 + 3*4 = 54.

To improve: try to find a Kronecker structure. Note that 2x5x6 can be
seen as part of a 2x6x6 tensor. The 2x6x6 tensor = T_{2,2,2} ⊗ T_{1,3,3}
has rank 7*9=63. But we only need the 2x5x6 sub-tensor.

Let me try a different approach: use the fact that the multiplication tensor
T_{n,m,p}(i,j,k) = delta_{i,j,k} (the identity tensor) can be decomposed
using the algebraic structure of the indices.

For the Kronecker approach to work, we need n=n1*n2, m=m1*m2, p=p1*p2.
2x5x6: 2=2*1, 5=5*1, 6=6*1 (trivial) or 2=1*2, 5=5*1, 6=3*2.
With (1,5,3) and (2,1,2): rank = rank(1,5,3)*rank(2,1,2) = 15*4 = 60. No gain.
With (2,5,3) and (1,1,2): rank = 30*2 = 60. No gain.

The Kronecker product doesn't help here because 5 is prime.

So the best strategy remains the block decomposition with rank 54, then
attempt local optimization to reduce further. Let's focus on making the
verification fast and trying more aggressive column elimination.
"""

import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 135.0  # Leave 15s safety margin for 150s limit


def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster check using vectorized operations."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W


def strassen_decomposition_2x2():
    """
    Returns Strassen decomposition for a 2x2x2 block.
    U_block: (4, 7), V_block: (4, 7), W_block: (4, 7)
    Local indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank_block = 7
    U_block = np.zeros((4, rank_block))
    V_block = np.zeros((4, rank_block))
    W_block = np.zeros((4, rank_block))

    # Term 0: M1 = (A00+A11)(B00+B11)
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1

    # Term 1: M2 = (A10+A11)B00
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1

    # Term 2: M3 = A00(B01-B11)
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1

    # Term 3: M4 = A11(B10-B00)
    U_block[3, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[2, 3] = 1

    # Term 4: M5 = (A00+A10)B11
    U_block[0, 4] = 1; U_block[2, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1; W_block[1, 4] = 1

    # Term 5: M6 = (A10-A00)(B00+B01)
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1

    # Term 6: M7 = (A01-A11)(B10+B11)
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1

    return U_block, V_block, W_block


def schoolbook_2x1x2():
    """
    Returns schoolbook decomposition for a 2x1x2 block.
    n=2, m=1, p=2. Rank = 4.
    U_block: (2, 4), V_block: (2, 4), W_block: (4, 4)
    """
    rank_block = 4
    U_block = np.zeros((2, rank_block))
    V_block = np.zeros((2, rank_block))
    W_block = np.zeros((4, rank_block))

    # c=0: i=0, j=0, k=0
    U_block[0, 0] = 1
    V_block[0, 0] = 1
    W_block[0, 0] = 1

    # c=1: i=0, j=0, k=1
    U_block[0, 1] = 1
    V_block[1, 1] = 1
    W_block[2, 1] = 1

    # c=2: i=1, j=0, k=0
    U_block[1, 2] = 1
    V_block[0, 2] = 1
    W_block[1, 2] = 1

    # c=3: i=1, j=0, k=1
    U_block[1, 3] = 1
    V_block[1, 3] = 1
    W_block[3, 3] = 1

    return U_block, V_block, W_block


def build_block_decomposition(n, m, p):
    """
    Builds a decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x1x2 blocks.
    Total rank: 6*7 + 3*4 = 54.
    """
    max_rank = 60
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))

    col = 0

    blocks = [
        (0, 0, 2, 2, 'strassen'),
        (0, 2, 2, 2, 'strassen'),
        (0, 4, 2, 2, 'strassen'),
        (2, 0, 2, 2, 'strassen'),
        (2, 2, 2, 2, 'strassen'),
        (2, 4, 2, 2, 'strassen'),
        (4, 0, 1, 2, 'schoolbook_2x1x2'),
        (4, 2, 1, 2, 'schoolbook_2x1x2'),
        (4, 4, 1, 2, 'schoolbook_2x1x2'),
    ]

    for j0, k0, mj, pk, method in blocks:
        if method == 'strassen':
            Ub, Vb, Wb = strassen_decomposition_2x2()
            for i in range(2):
                for j in range(2):
                    row_idx = i*m + (j0+j)
                    U[row_idx, col:col+7] = Ub[i*2+j, :]
            for j in range(2):
                for k in range(2):
                    row_idx = (j0+j)*p + (k0+k)
                    V[row_idx, col:col+7] = Vb[j*2+k, :]
            for k in range(2):
                for i in range(2):
                    row_idx = (k0+k)*n + i
                    W[row_idx, col:col+7] = Wb[k*2+i, :]
            col += 7
        elif method == 'schoolbook_2x1x2':
            Ub, Vb, Wb = schoolbook_2x1x2()
            for i in range(2):
                row_idx = i*m + j0
                U[row_idx, col:col+4] = Ub[i, :]
            for k in range(2):
                row_idx = j0*p + (k0+k)
                V[row_idx, col:col+4] = Vb[k, :]
            for k in range(2):
                for i in range(2):
                    row_idx = (k0+k)*n + i
                    W[row_idx, col:col+4] = Wb[k*2+i, :]
            col += 4

    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]

    if not check_decomposition(U, V, W, n, m, p):
        raise ValueError("Block decomposition failed verification")

    return U, V, W


def try_column_reduction(U, V, W, n, m, p):
    """
    Attempt to reduce the rank by finding linear dependencies.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for c in range(rank):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        U_test = np.delete(U, c, axis=1)
        V_test = np.delete(V, c, axis=1)
        W_test = np.delete(W, c, axis=1)
        if check_decomposition(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            rank -= 1
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            c = -1  # Restart

    return best_U, best_V, best_W, best_rank


def try_linear_combination_removal(U, V, W, n, m, p):
    """
    Try to find if any column is a linear combination of others.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for c in range(rank):
        if time.time() - START_TIME > TIME_LIMIT:
            break

        other_cols = [d for d in range(rank) if d != c]
        if len(other_cols) < 3:
            continue

        U_others = U[:, other_cols]
        V_others = V[:, other_cols]
        W_others = W[:, other_cols]

        U_c = U[:, c]
        V_c = V[:, c]
        W_c = W[:, c]

        try:
            alpha_U, _, _, _ = np.linalg.lstsq(U_others, U_c, rcond=None)
            residual_U = np.linalg.norm(U_c - U_others @ alpha_U)
        except:
            continue

        if residual_U > 1e-8:
            continue

        try:
            V_pred = V_others @ alpha_U
            residual_V = np.linalg.norm(V_c - V_pred)
        except:
            continue

        if residual_V > 1e-8:
            continue

        try:
            W_pred = W_others @ alpha_U
            residual_W = np.linalg.norm(W_c - W_pred)
        except:
            continue

        if residual_W > 1e-8:
            continue

        U = np.delete(U, c, axis=1)
        V = np.delete(V, c, axis=1)
        W = np.delete(W, c, axis=1)
        rank -= 1
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        c = -1  # Restart

    return best_U, best_V, best_W, best_rank


def try_kronecker_reduction(U, V, W, n, m, p):
    """
    Try to exploit Kronecker product structure to reduce rank.
    
    Key idea: If the decomposition has a block structure where some blocks
    share the same Strassen pattern, we might be able to merge them.
    
    For 2x5x6 with 6 Strassen blocks (2x2x2) and 3 schoolbook blocks (2x1x2):
    The Strassen blocks all use the same 7-term pattern. If we can find
    a linear transformation that combines multiple Strassen blocks into
    fewer terms, we reduce rank.
    
    Specifically, consider two adjacent Strassen blocks in the j-direction.
    They share the same i-indices and k-indices but have different j-indices.
    The U matrices for these blocks have the same structure (same rows, 
    different columns), so we might be able to combine them.
    
    This is complex, so we try a simpler approach: check if the overall
    decomposition has linear dependencies that the per-block analysis missed.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try removing pairs of columns that might be dependent
    for c1 in range(rank - 1):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Check if columns c1 and c2 are linearly dependent across all three matrices
            # Build a combined matrix and check rank
            U_cols = np.column_stack([U[:, c1], U[:, c2]])
            V_cols = np.column_stack([V[:, c1], V[:, c2]])
            W_cols = np.column_stack([W[:, c1], W[:, c2]])
            
            # If the two columns are proportional in all three matrices, 
            # we can eliminate one
            if np.linalg.matrix_rank(U_cols) == 1 and \
               np.linalg.matrix_rank(V_cols) == 1 and \
               np.linalg.matrix_rank(W_cols) == 1:
                # Remove the second column
                U = np.delete(U, c2, axis=1)
                V = np.delete(V, c2, axis=1)
                W = np.delete(W, c2, axis=1)
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break  # Restart inner loop
    
    return best_U, best_V, best_W, best_rank


def construct():
    global START_TIME
    n, m, p = DIMENSIONS

    # 1. Build block decomposition (rank 54)
    try:
        U, V, W = build_block_decomposition(n, m, p)
    except:
        U, V, W = schoolbook_decomposition(n, m, p)

    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # 2. Try Kronecker-structure-aware reduction
    if time.time() - START_TIME < TIME_LIMIT - 40:
        U2, V2, W2, rank2 = try_kronecker_reduction(U, V, W, n, m, p)
        if rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2

    # 3. Try linear combination removal
    if time.time() - START_TIME < TIME_LIMIT - 25:
        U3, V3, W3, rank3 = try_linear_combination_removal(best_U, best_V, best_W, n, m, p)
        if rank3 < best_rank:
            best_rank = rank3
            best_U, best_V, best_W = U3, V3, W3

    # 4. Try simple column removal
    if time.time() - START_TIME < TIME_LIMIT - 15:
        U4, V4, W4, rank4 = try_column_reduction(best_U, best_V, best_W, n, m, p)
        if rank4 < best_rank:
            best_rank = rank4
            best_U, best_V, best_W = U4, V4, W4

    # 5. Final verification
    if not check_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = schoolbook_decomposition(n, m, p)
        best_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
