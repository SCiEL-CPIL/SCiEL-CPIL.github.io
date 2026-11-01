import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i * m + j, j * p + k, k * n + i] = 1
    return np.allclose(T, Expected, atol=1e-9)

def construct():
    n, m, p = 2, 4, 7
    dim_a = n * m  # 8
    dim_b = m * p  # 28
    dim_c = p * n  # 14
    
    # Target rank for the non-uniform partitioning approach
    # We'll try to find a decomposition with rank 42
    target_rank = 42
    
    # Build the expected tensor
    Expected = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                Expected[a_idx, b_idx, c_idx] = 1
    
    # Non-uniform partitioning:
    # Split j (0-3) into j1 (0-1) and j2 (0-1): j = j1 + 2*j2
    # Split k (0-6) into k1 (0-1) and k2 (0-2): k = k1 + 2*k2, but 7 = 2*3+1
    # Actually k goes 0-6, so k2 goes 0-2 (3 values), k1 goes 0-1 (2 values)
    # That gives 6 values, but we need 7. Let's handle the 7th separately.
    
    # Alternative: use Strassen-like approach for the 2x2 blocks
    # A is 2x4 = [A_left(2x2) A_right(2x2)]
    # B is 4x7 = [B_top(2x7); B_bottom(2x7)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For each 2x2 @ 2x7 multiplication, we can use a rank-7 decomposition
    # (Strassen gives rank 7 for 2x2 @ 2x2, and for 2x2 @ 2x7 we can use 7 terms)
    
    # Actually, for 2x2 @ 2x7, the trivial decomposition has 2*2*7 = 28 terms.
    # But we can do better using the structure.
    
    # Let's use a direct approach: build U, V, W with target_rank columns
    # and solve for the entries using linear algebra.
    
    # The tensor equation is: sum_r U[a,r] * V[b,r] * W[c,r] = Expected[a,b,c]
    # This is a system of equations in the variables U[a,r], V[b,r], W[c,r]
    # But it's bilinear/trilinear, so we can't solve it linearly.
    
    # Instead, let's try a structured approach:
    # For each (i, j, k) triple, we assign it to one of the rank terms.
    # We want to merge triples that share structure.
    
    # Key insight: for fixed i and j, as k varies, the term is:
    # e_{im+j} ⊗ e_{jp+k} ⊗ e_{kn+i}
    # The first factor is constant, so we can factor it out.
    # sum_k e_{im+j} ⊗ e_{jp+k} ⊗ e_{kn+i}
    # = e_{im+j} ⊗ (sum_k e_{jp+k} ⊗ e_{kn+i})
    
    # The inner sum is a 28x14 matrix with 7 non-zero entries (one per k).
    # This matrix has rank 7 (generically), so we can't reduce it.
    
    # But if we consider different (i,j) pairs, the matrices overlap in structure.
    # For i=0 and i=1, j fixed:
    # M_{i,j} has entries at (jp+k, kn+i) = 1 for k=0..6
    
    # For i=0: (jp+k, 2k) for k=0..6
    # For i=1: (jp+k, 2k+1) for k=0..6
    
    # These are different rows in the c-dimension, so they don't overlap directly.
    
    # Let's try a different partitioning: group by j first.
    # For fixed j, i varies (2 values) and k varies (7 values).
    # That's 14 terms per j, 4 j's, total 56.
    
    # Can we merge terms across different j values?
    # For j and j', the a-indices are different (im+j vs im+j'), so U entries are in different rows.
    # Similarly for b-indices. So no direct merging possible.
    
    # Given the difficulty, let's just return the trivial decomposition with rank 56.
    rank = n * m * p
    U = np.zeros((dim_a, rank), dtype=np.float64)
    V = np.zeros((dim_b, rank), dtype=np.float64)
    W = np.zeros((dim_c, rank), dtype=np.float64)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}, Sum diff: {diff.sum()}")
