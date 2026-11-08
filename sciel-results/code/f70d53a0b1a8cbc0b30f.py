"""
Tensor decomposition for 3x3 by 3x4 matrix multiplication.

Strategy: use a block structure
  C = [[C11, C12], [C21, C22]]  with C11, C12 = 2x2, C21, C22 = 1x2
  A = [[A11, A12], [A21, A22]]  with A11 = 2x2, A12 = 2x1, A21 = 1x2, A22 = 1x1
  B = [[B11, B12], [B21, B22]]  with B11, B12 = 2x2, B21, B22 = 1x2

We use the rank-7 Winograd/Strassen scheme for the 2x2 blocks and
direct rank-2 terms for the remaining 1x2/2x1/1x1 interactions.
This gives rank 34.

We also attempt to reduce rank via a limited local search: we start
from the rank-34 tensor and try to eliminate one column by finding
a linear combination of existing columns that cancels a single column,
or by checking whether any column is in the span of the others.
"""

import time
import numpy as np

def _build_rank34():
    """
    Construct the rank-34 decomposition.

    Indexing:
      U: 9 rows (A indices i*n + k, n=3, k=0..2)
      V: 12 rows (B indices j*m + l, m=3, l=0..2)
      W: 12 rows (C indices k*p + i, p=4, i=0..2)  -- W[k*p + i]

    A[i,k] index in U: i*3 + k
    B[j,l] index in V: j*3 + l
    C[k,i] index in W: k*4 + i

    Blocks:
      A11 = A[0:2, 0:2] -> U indices 0,1,3,4
      A12 = A[0:2, 2]   -> U indices 2,5
      A21 = A[2, 0:2]   -> U indices 6,7
      A22 = A[2, 2]     -> U index 8

      B11 = B[0:2, 0:2] -> V indices 0,1,3,4
      B12 = B[0:2, 2:4] -> V indices 2,3,6,7   (B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7)
      B21 = B[2, 0:2]   -> V indices 8,9
      B22 = B[2, 2:4]   -> V indices 10,11

      C11 = C[0:2, 0:2] -> W indices 0,1,4,5   (C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5)
      C12 = C[0:2, 2:4] -> W indices 2,3,6,7   (C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7)
      C21 = C[2, 0:2]   -> W indices 8,9       (C[2,0]=8, C[2,1]=9)
      C22 = C[2, 2:4]   -> W indices 10,11     (C[2,2]=10, C[2,3]=11)
    """
    rank = 34
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((12, rank), dtype=float)
    W = np.zeros((12, rank), dtype=float)

    col = 0

    # --- 2x2 * 2x2 block product C11 = A11 * B11 ---
    # A11 indices: 0,1,3,4  (A[0,0], A[0,1], A[1,0], A[1,1])
    # B11 indices: 0,1,3,4  (B[0,0], B[0,1], B[1,0], B[1,1])
    # C11 indices: 0,1,4,5  (C[0,0], C[0,1], C[1,0], C[1,1])

    # P1: (A00 + A11) * (B00 + B11) -> C00, C11
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[4, col] = 1
    W[0, col] = 1; W[5, col] = 1
    col += 1

    # P2: (A10 + A11) * B00 -> C10, C11
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[4, col] = 1; W[5, col] = -1
    col += 1

    # P3: A00 * (B01 - B11) -> C01, C11
    U[0, col] = 1
    V[1, col] = 1; V[4, col] = -1
    W[1, col] = 1; W[5, col] = 1
    col += 1

    # P4: A11 * (B10 - B00) -> C00, C10
    U[4, col] = 1
    V[3, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[4, col] = 1
    col += 1

    # P5: (A00 + A01) * B11 -> C00, C01
    U[0, col] = 1; U[1, col] = 1
    V[4, col] = 1
    W[0, col] = -1; W[1, col] = 1
    col += 1

    # P6: (A10 - A00) * (B00 + B01) -> C11
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[5, col] = 1
    col += 1

    # P7: (A01 - A11) * (B10 + B11) -> C00
    U[1, col] = 1; U[4, col] = -1
    V[3, col] = 1; V[4, col] = 1
    W[0, col] = 1
    col += 1

    # --- A12 * B11: 2x1 * 2x2 -> C11 contribution ---
    # A12 indices: 2,5 (A[0,2], A[1,2])
    # B11 indices: 0,1,3,4
    # C11 contribution: C[0,0]+=A02*B00, C[0,1]+=A02*B01, C[1,0]+=A12*B10, C[1,1]+=A12*B11

    # T1: A02 * B00 -> C00
    U[2, col] = 1
    V[0, col] = 1
    W[0, col] = 1
    col += 1

    # T2: A02 * B01 -> C01
    U[2, col] = 1
    V[1, col] = 1
    W[1, col] = 1
    col += 1

    # T3: A12 * B10 -> C10
    U[5, col] = 1
    V[3, col] = 1
    W[4, col] = 1
    col += 1

    # T4: A12 * B11 -> C11
    U[5, col] = 1
    V[4, col] = 1
    W[5, col] = 1
    col += 1

    # --- A21 * B11: 1x2 * 2x2 -> C21 contribution ---
    # A21 indices: 6,7 (A[2,0], A[2,1])
    # B11 indices: 0,1,3,4
    # C21: C[2,0]+=A20*B00+A21*B10, C[2,1]+=A20*B01+A21*B11

    # T1: A20 * B00 -> C20
    U[6, col] = 1
    V[0, col] = 1
    W[8, col] = 1
    col += 1

    # T2: A20 * B01 -> C21
    U[6, col] = 1
    V[1, col] = 1
    W[9, col] = 1
    col += 1

    # T3: A21 * B10 -> C20
    U[7, col] = 1
    V[3, col] = 1
    W[8, col] = 1
    col += 1

    # T4: A21 * B11 -> C21
    U[7, col] = 1
    V[4, col] = 1
    W[9, col] = 1
    col += 1

    # --- A22 * B11: 1x1 * 2x2 -> C21 contribution ---
    # A22 index: 8
    # B11 indices: 0,1,3,4
    # C21: C[2,0]+=A22*B00, C[2,1]+=A22*B01

    # T1: A22 * B00 -> C20
    U[8, col] = 1
    V[0, col] = 1
    W[8, col] = 1
    col += 1

    # T2: A22 * B01 -> C21
    U[8, col] = 1
    V[1, col] = 1
    W[9, col] = 1
    col += 1

    # --- 2x2 * 2x2 block product C12 = A11 * B12 ---
    # A11 indices: 0,1,3,4
    # B12 indices: 2,3,6,7 (B[0,2], B[0,3], B[1,2], B[1,3])
    # C12 indices: 2,3,6,7 (C[0,2], C[0,3], C[1,2], C[1,3])

    # P1: (A00 + A11) * (B02 + B12) -> C02, C13
    U[0, col] = 1; U[4, col] = 1
    V[2, col] = 1; V[7, col] = 1
    W[2, col] = 1; W[7, col] = 1
    col += 1

    # P2: (A10 + A11) * B02 -> C12, C13
    U[3, col] = 1; U[4, col] = 1
    V[2, col] = 1
    W[6, col] = 1; W[7, col] = -1
    col += 1

    # P3: A00 * (B03 - B13) -> C03, C13
    U[0, col] = 1
    V[3, col] = 1; V[7, col] = -1
    W[3, col] = 1; W[7, col] = 1
    col += 1

    # P4: A11 * (B12 - B02) -> C02, C12
    U[4, col] = 1
    V[6, col] = 1; V[2, col] = -1
    W[2, col] = 1; W[6, col] = 1
    col += 1

    # P5: (A00 + A01) * B13 -> C02, C03
    U[0, col] = 1; U[1, col] = 1
    V[7, col] = 1
    W[2, col] = -1; W[3, col] = 1
    col += 1

    # P6: (A10 - A00) * (B02 + B03) -> C13
    U[3, col] = 1; U[0, col] = -1
    V[2, col] = 1; V[3, col] = 1
    W[7, col] = 1
    col += 1

    # P7: (A01 - A11) * (B12 + B13) -> C02
    U[1, col] = 1; U[4, col] = -1
    V[6, col] = 1; V[7, col] = 1
    W[2, col] = 1
    col += 1

    # --- A12 * B21: 2x1 * 1x2 -> C12 contribution ---
    # A12 indices: 2,5
    # B21 indices: 8,9
    # C12: C[0,2]+=A02*B20, C[0,3]+=A02*B21, C[1,2]+=A12*B20, C[1,3]+=A12*B21

    # T1: A02 * B20 -> C02
    U[2, col] = 1
    V[8, col] = 1
    W[2, col] = 1
    col += 1

    # T2: A02 * B21 -> C03
    U[2, col] = 1
    V[9, col] = 1
    W[3, col] = 1
    col += 1

    # T3: A12 * B20 -> C12
    U[5, col] = 1
    V[8, col] = 1
    W[6, col] = 1
    col += 1

    # T4: A12 * B21 -> C13
    U[5, col] = 1
    V[9, col] = 1
    W[7, col] = 1
    col += 1

    # --- A21 * B12: 1x2 * 2x2 -> C22 contribution ---
    # A21 indices: 6,7
    # B12 indices: 2,3,6,7
    # C22: C[2,2]+=A20*B02+A21*B12, C[2,3]+=A20*B03+A21*B13

    # T1: A20 * B02 -> C22
    U[6, col] = 1
    V[2, col] = 1
    W[10, col] = 1
    col += 1

    # T2: A20 * B03 -> C23
    U[6, col] = 1
    V[3, col] = 1
    W[11, col] = 1
    col += 1

    # T3: A21 * B12 -> C22
    U[7, col] = 1
    V[6, col] = 1
    W[10, col] = 1
    col += 1

    # T4: A21 * B13 -> C23
    U[7, col] = 1
    V[7, col] = 1
    W[11, col] = 1
    col += 1

    # --- A22 * B21: 1x1 * 1x2 -> C22 contribution ---
    # A22 index: 8
    # B21 indices: 8,9
    # C22: C[2,2]+=A22*B20, C[2,3]+=A22*B21

    # T1: A22 * B20 -> C22
    U[8, col] = 1
    V[8, col] = 1
    W[10, col] = 1
    col += 1

    # T2: A22 * B21 -> C23
    U[8, col] = 1
    V[9, col] = 1
    W[11, col] = 1
    col += 1

    assert col == rank
    return U, V, W


def _verify_decomposition(U, V, W):
    """
    Verify that the tensor decomposition is exact.
    The matmul tensor T[i,k,j,l] = 1 if the output position (k,l) 
    is (i,k) * (k,j) in the matrix product, but here we use:
    T[a_idx, b_idx, c_idx] where a_idx = i*3+k, b_idx = j*3+l, c_idx = l*4+i
    and T = 1 iff c_idx corresponds to (i,k)*(j,l) contributing to (i,l).
    
    Actually, the matmul tensor: C[i,j] = sum_k A[i,k] B[k,j]
    So the tensor is T[i,k,j,l] = delta_{k, k} = 1 for all valid i,j,k,l 
    where the middle index matches. In our indexing:
    A index a = i*3 + k
    B index b = j*3 + l  (here j is the row of B = column of A, so j=k)
    C index c = l*4 + i

    The tensor T has T[a,b,c] = 1 iff b's row (j) equals a's column (k) and c's row (l) equals b's column (l).
    So T[a,b,c] = 1 iff k_a == j_b and l_b == i_c, where:
      a = i_a * 3 + k_a
      b = j_b * 3 + l_b
      c = l_c * 4 + i_c
    And the condition is k_a == j_b (the shared index).
    
    The decomposition: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    """
    n, m, p = 3, 3, 4
    # Build the full tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(m):
                for l in range(p):
                    a = i * m + k
                    b = j * p + l
                    c = l * n + i
                    if k == j:
                        T[a, b, c] = 1.0

    # Compute from decomposition
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-10)


def _try_reduce_rank(U, V, W, max_time=100):
    """
    Attempt to reduce the rank by checking if any column is redundant.
    Returns (U, V, W) with potentially reduced rank.
    """
    start = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Try to eliminate each column by expressing it as a combination of others
    for c in range(best_rank):
        if time.time() - start > max_time:
            break
        # Try removing column c
        remaining = [r for r in range(best_rank) if r != c]
        if len(remaining) < 1:
            continue

        # Check if column c is in the span of the other columns
        # For a column to be eliminable, we need U[:,c] = sum_{r in remaining} alpha_r U[:,r]
        # and similarly for V and W. This is a system of linear equations.
        
        # Build the system: for each row in U, V, W
        # U[:,c] = U[:,remaining] @ alpha
        # V[:,c] = V[:,remaining] @ alpha
        # W[:,c] = W[:,remaining] @ alpha
        
        U_rem = best_U[:, remaining]
        V_rem = best_V[:, remaining]
        W_rem = best_W[:, remaining]
        
        # Stack all constraints
        A_mat = np.vstack([U_rem, V_rem, W_rem])  # (9+12+12) x (rank-1)
        b_vec = np.concatenate([best_U[:, c], best_V[:, c], best_W[:, c]])
        
        # Solve least squares
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        # Check if the fit is exact
        A_alpha = A_mat @ alpha
        if np.allclose(A_alpha, b_vec, atol=1e-10):
            # We can eliminate column c
            new_U = best_U[:, remaining]
            new_V = best_V[:, remaining]
            new_W = best_W[:, remaining]
            best_rank -= 1
            best_U, best_V, best_W = new_U, new_V, new_W
            print(f"Eliminated column {c}, new rank: {best_rank}")

    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 110

    U, V, W = _build_rank34()

    # Verify
    if not _verify_decomposition(U, V, W):
        raise ValueError("Decomposition is not valid")

    # Try to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, max_time=min(100, deadline - time.time()))

    if time.time() > deadline - 10:
        pass  # Return what we have

    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                # Round to nearest half-integer
                v = round(val * 2) / 2
                if abs(v - round(v)) < 1e-10:
                    v = int(round(v))
                r.append(v)
            result.append(r)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
