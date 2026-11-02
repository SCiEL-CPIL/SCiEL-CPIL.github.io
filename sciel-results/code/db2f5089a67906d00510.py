import numpy as np
import time

# EVOLVE-BLOCK-START

# Deadline guard: the evaluator kills after 150s wall-clock.
# We use a safety margin of 20s, so we stop searching after 130s.
_DEADLINE = time.time() + 130.0

def _time_left():
    return _DEADLINE - time.time()

# Base 2x2 Strassen combinations
# Each entry is a list of (row, col, sign) tuples
_A_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (0, 0, -1)],
    [(0, 1, 1), (1, 1, -1)]
]

_B_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(0, 1, 1), (1, 1, -1)],
    [(1, 0, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (1, 1, 1)]
]

_C_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 1, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1)]
]

def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 Strassen decomposition.
    Rank = 49.
    """
    n = 4
    rank = 49
    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)
    
    for t_outer in range(7):
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            # A (U) part: U[i*n + k, r] where i is row, k is col of A
            for (br, bc, sign) in _A_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _A_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    u_idx = i_global * n + k_global
                    U[u_idx, r_idx] += sign * sign_inner
            
            # B (V) part: V[j*n + l, r] where j is row, l is col of B
            for (br, bc, sign) in _B_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _B_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    v_idx = i_global * n + k_global
                    V[v_idx, r_idx] += sign * sign_inner
            
            # C (W) part: W[k*n + i, r] where k is col, i is row of C
            # The trace convention is W[k*n+i, r]
            for (br, bc, sign) in _C_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _C_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    w_idx = k_global * n + i_global
                    W[w_idx, r_idx] += sign * sign_inner

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    This should equal 1 if (i,j,k) corresponds to a valid multiplication term,
    but actually for the full tensor, T[i,j,k] = delta_{i,k} (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    Wait, the tensor for matrix multiplication C = A * B is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T[i,j,k] = 1 if we are looking at the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1 for all i,j,k.
    No, that's not right. The tensor is T_{i,j,k} = delta_{i,k} ... no.
    
    The bilinear form is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    The tensor of structure constants is T_{i,j,k} = 1 for all i,j,k? No.
    The tensor is T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    Since C_{i,k} = sum_j A_{i,j} B_{j,k}, the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    So T_{i,j,k} = 1 for all i,j,k? That would mean the tensor is all ones, which is not right.
    
    Let's re-read the task description: "the output factor follows the official trace convention W[k*n+i,r]"
    And the verification in the seed program:
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))
    
    So T[i,j,k] should be 1 for all i,j,k? That doesn't make sense for matrix multiplication.
    
    Wait, let's think about this differently. The matrix multiplication tensor M is defined by:
    M[i,j,k] = 1 if and only if the product A[i,j] * B[j,k] contributes to C[i,k].
    But actually, in the standard tensor decomposition of matrix multiplication, the tensor is:
    M[i,j,k] = delta_{i,k} ... no, that's not right either.
    
    Actually, the matrix multiplication tensor is:
    M[i,j,k] = 1 for all i,j,k in the range, because the term A[i,j]*B[j,k] appears in C[i,k] with coefficient 1.
    But that's only if we're looking at a specific (i,j,k) triple.
    
    Let me reconsider. The tensor for n x n matrix multiplication is a 3-tensor of size n^3.
    The entry M[i,j,k] is 1 if the product of the (i,j)-th entry of A and the (j,k)-th entry of B
    contributes to the (i,k)-th entry of C. Since C[i,k] = sum_j A[i,j] * B[j,k], the coefficient
    of A[i,j] * B[j,k] in C[i,k] is 1. So M[i,j,k] = 1 for all i,j,k?
    
    No, that's not right. The tensor M is indexed by (i, j, k) where i is the row of C, j is the
    summation index, and k is the column of C. The entry M[i,j,k] is 1 if A[i,j] * B[j,k] appears
    in C[i,k]. Since C[i,k] = sum_j' A[i,j'] * B[j',k], the term A[i,j] * B[j,k] appears in C[i,k]
    only when j = j'. So M[i,j,k] = 1 for all i,j,k.
    
    Wait, that can't be right either, because then the tensor would be all ones, and its rank
    would be much smaller.
    
    Let me look at this more carefully. The matrix multiplication tensor is:
    M = sum_{i,j,k} M[i,j,k] * e_i^A * e_j^B * e_k^C
    where M[i,j,k] = 1 if the (i,j)-th entry of A and the (j,k)-th entry of B multiply to give
    a contribution to the (i,k)-th entry of C.
    
    Since C[i,k] = sum_j A[i,j] * B[j,k], the term A[i,j] * B[j,k] contributes to C[i,k] with
    coefficient 1. So M[i,j,k] = 1 for all i,j,k.
    
    But that means the tensor is the all-ones tensor, which has rank 1! That's clearly wrong.
    
    I think I'm confusing myself. Let me look at the seed program's verification again:
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))
    
    So T[i,j,k] should be 1 for all i,j,k. This means the tensor is indeed the all-ones tensor.
    But that's not the matrix multiplication tensor...
    
    Oh wait, I think I see. The tensor for matrix multiplication is not the all-ones tensor.
    The tensor is M[i,j,k] = 1 if and only if the indices are "compatible" in the sense that
    the j-th column of A (indexed by i,j) and the j-th row of B (indexed by j,k) multiply to
    give the (i,k)-th entry of C.
    
    But since C[i,k] = sum_j A[i,j] * B[j,k], every triple (i,j,k) is valid. So M[i,j,k] = 1
    for all i,j,k.
    
    I'm going in circles. Let me just trust the seed program's verification and proceed.
    """
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))

def _schoolbook_decomposition(n, m, p):
    """
    Fallback schoolbook decomposition with rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 4, 4, 4
    
    # Try the rank-49 Strassen decomposition
    U, V, W = _generate_strassen_4x4()
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
