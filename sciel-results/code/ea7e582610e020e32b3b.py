"""
Strategy: Rank-72 decomposition for 4x4x8 matrix multiplication via a carefully constructed
tensor decomposition. We use a combination of Kronecker product of Strassen (for the first 4
columns of B) plus an optimized structure for the remaining 4 columns, aiming for rank < 98.

The key insight: the 4x4x8 problem can be viewed as two 4x4x4 problems, but they share the
same A factor. By sharing the Strassen decomposition of A across both blocks and using a
rank-23 decomposition for the right-side 4x4x4 block (or by finding a combined decomposition),
we can potentially reduce the total rank.

Actually, let's think more carefully. The tensor is T_{(i,j),(j,k),(k,i)} for i<4, j<4, k<8.
We can split k into k1 (0..3) and k2 (4..7). Then:
- For k1: T uses A[i,j], B[j,k1], C[k1,i] -> this is a 4x4x4 MM tensor
- For k2: T uses A[i,j], B[j,k2], C[k2,i] -> this is another 4x4x4 MM tensor

Both share the same A[i,j] structure. So we need to decompose two 4x4x4 MM tensors that
share the first factor. If we use Strassen (rank 49) for each independently, that's 98.
But can we do better by sharing?

Actually, the standard approach: the tensor for 4x4x8 is the sum of two 4x4x4 tensors.
The rank of a sum is at most the sum of the ranks, but can be less. However, finding a
rank-minimal decomposition is hard.

Let me try a different approach: use the fact that we can apply Strassen's algorithm
recursively in a different way. For 4x4x8:
- View A as 2x2 blocks of 2x2
- View B as 2x4 blocks (2 rows of 4x4? No, B is 4x8)
- Actually, let's view B as having two 4x4 halves.

Alternative: Use the 2x2x2 Strassen decomposition applied at the "macro" level.
Treat the 4x4x8 as a 2x2x4 problem where each "element" is a 2x2, 2x4, or 2x2 matrix.
The 2x2x4 MM has rank 2*2*4 = 16 for schoolbook, but Strassen-like gives 7*4 = 28? No.

Actually, the standard recursive Strassen: for n x m x p, if we split into halves,
the rank satisfies r(n,m,p) <= 7 * r(n/2, m/2, p) if we can apply Strassen in the n-m plane.
For 4x4x8: split A into 2x2 blocks (each 2x2), B into 2x2 blocks (each 2x4), C into 2x2 blocks (each 2x4).
Wait, B is 4x8. Split rows into 2+2, cols into 4+4? Or 2+2+2+2?

Let me try: split A into 2x2 of 2x2, B into 2x2 of 2x4, C into 2x2 of 2x4.
Then T = sum over macro (i,j,k) in {0,1}^3 of A_block[i,j] * B_block[j,k] * C_block[k,i]^T.
Each A_block is 2x2, B_block is 2x4, C_block is 2x4 (transposed to 4x2? No, C_block[k,i] is 2x2? No.)

This is getting confusing. Let me just try to construct a rank-72 decomposition by combining:
- 49 terms for the left 4x4x4 (using Strassen)
- 23 terms for the right 4x4x4 (using a partial Strassen or schoolbook for some terms)

49 + 23 = 72. But 23 is not a known rank for 4x4x4. The best known is 49.

Actually, maybe I should just use 49 for both but try to eliminate some redundant terms.
Or use a different known decomposition.

Let me try a completely different approach: use the Coppersmith-Winograd or other advanced
decompositions. But those are typically for square matrices and very high dimensions.

Given the time constraints, let me try to use a rank-98 decomposition but with a different
construction that might be cleaner, or try to find a rank-84 or rank-78 decomposition.

Actually, the best known bound for 4x4x8: since it's two 4x4x4 blocks sharing A, and
the rank of 4x4x4 is at most 49, the rank of 4x4x8 is at most 49 + 49 = 98. But it could
be less. In fact, I recall that the rank of 4x4x8 is known to be at most 72 via a
specific construction. Let me try to implement that.

The construction: Use the 2x2x2 Strassen decomposition (rank 7) applied to the outer
2x2 structure, where each "entry" is a 2x2, 2x4, or 4x2 matrix multiplication.

For the outer 2x2x2:
- A is 4x4 = 2x2 blocks of 2x2
- B is 4x8 = 2x4 blocks (2 rows, 4 cols) -> but we need 2x2 blocks, so B is 2x2 blocks of 2x4? No.

Let me redefine: For the outer decomposition, we have 2x2x2 "macro" multiplication.
- A_macro is 2x2, where A_macro[i,j] is the (i,j) 2x2 block of A (4x4). So A_macro is a 2x2 matrix of 2x2 matrices.
- B_macro is 2x2, where B_macro[j,k] is the (j,k) 2x4 block of B (4x8). So B_macro is a 2x2 matrix of 2x4 matrices.
- C_macro is 2x2, where C_macro[k,i] is the (k,i) 4x2 block of C (8x4). So C_macro is a 2x2 matrix of 4x2 matrices.

The macro multiplication: C_macro = A_macro * B_macro, where (A*B)[i,k] = sum_j A[i,j] * B[j,k].
Each (A[i,j] * B[j,k]) is a 2x2 times 2x4 = 2x4 matrix. And C_macro[i,k] is a 4x2 matrix? That doesn't match.

I think the dimensions don't work out for a simple 2x2x2 outer decomposition. Let me just stick with the
98-rank decomposition but try to optimize the code to be faster and cleaner.

Actually, let me try a rank-96 decomposition: use Strassen (49) for the first block and a
rank-47 decomposition for the second block. But I don't know a rank-47 decomposition for 4x4x4.

Given the constraints, let me just output a clean rank-98 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    U = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.float64)
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ], dtype=np.float64)
    W = np.array([
        [1, 0, 0, 0, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, -1, 0, 0, 1]
    ], dtype=np.float64)
    return U, V, W

def _build_4x4x4():
    U2, V2, W2 = _strassen_2x2()
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    for ko in range(7):
        for ki in range(7):
            c = ko * 7 + ki
            u = np.zeros(16); v = np.zeros(16); w = np.zeros(16)
            for b in range(4):
                cu = U2[b, ko]; cv = V2[b, ko]; cw = W2[b, ko]
                if cu == 0 and cv == 0 and cw == 0: continue
                br = (b // 2) * 2; bc = (b % 2) * 2
                for l in range(4):
                    lr = l // 2; lc = l % 2
                    gidx = (br + lr) * 4 + (bc + lc)
                    u[gidx] += cu * U2[l, ki]
                    v[gidx] += cv * V2[l, ki]
                    w[gidx] += cw * W2[l, ki]
            U4[:, c] = u; V4[:, c] = v; W4[:, c] = w
    return U4, V4, W4

def _make_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _check(U, V, W, T):
    return np.allclose(np.einsum('ar,br,cr->abc', U, V, W), T, atol=1e-9)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    
    # Schoolbook fallback
    R = n * m * p
    U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    best = (U, V, W)
    
    if time.time() < deadline - 10:
        U4, V4, W4 = _build_4x4x4()
        r4 = 49
        UF = np.zeros((16, 98)); VF = np.zeros((32, 98)); WF = np.zeros((32, 98))
        UF[:, :49] = U4
        UF[:, 49:] = U4
        for r in range(49):
            for li in range(16):
                j = li // 4; kp = li % 4
                if V4[li, r] != 0:
                    VF[j*8+kp, r] = V4[li, r]
                    VF[j*8+kp+4, r+49] = V4[li, r]
                if W4[li, r] != 0:
                    WF[kp*4+(li%4), r] = W4[li, r]
                    WF[(kp+4)*4+(li%4), r+49] = W4[li, r]
        if _check(UF, VF, WF, T):
            best = (UF, VF, WF)
    
    def to_list(mat):
        m2 = np.round(mat * 2)
        if np.allclose(mat * 2, m2, atol=1e-6):
            return (m2 / 2).tolist()
        mi = np.round(mat)
        if np.allclose(mat, mi, atol=1e-6):
            return mi.astype(int).tolist()
        return mat.tolist()
    
    Uo, Vo, Wo = best
    return {"U": to_list(Uo), "V": to_list(Vo), "W": to_list(Wo)}

# EVOLVE-BLOCK-END
