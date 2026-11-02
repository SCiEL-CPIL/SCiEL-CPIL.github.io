"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Combined approach:
1. Constructive combinatorial design:
   - Two 2x2x2 Strassen blocks (rank 7 each) for column pairs (0,1) and (2,3)
   - A 2x2x1 schoolbook block (rank 4) for column 4
   - Total rank 18 baseline
2. Attempt to reduce rank by sharing/absorbing terms between blocks
3. Numerical ALS + exact rational polish as fallback
4. Schoolbook fallback (rank 20)

The output factor W follows the official trace convention W[k*n+i, r].
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _check_decomposition(U, V, W, T, tol=1e-8):
    """Verify decomposition matches target tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    R = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = 0.0
                for r in range(R):
                    total += U_arr[u, r] * V_arr[v, r] * W_arr[w, r]
                if abs(total - T[u, v, w]) > tol:
                    return False
    return True


def _check_exact(U, V, W, T):
    """Exact integer/half-integer verification using Fraction."""
    def to_frac(x):
        hx = round(x * 2)
        if abs(x - hx / 2.0) < 1e-7:
            return Fraction(int(hx), 2)
        f = Fraction(x).limit_denominator(64)
        if abs(x - float(f)) < 1e-7:
            return f
        return Fraction(x).limit_denominator(64)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = len(U[0])
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _build_strassen_block(col0, col1, base, U, V, W, n, m, p):
    """Add 7 Strassen terms for 2x2 x 2x2 block with B columns col0, col1."""
    # M1: f=A00+A11, g=B[0,col0]+B[1,col1]
    r = base
    U[0][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1; W[col1*n+1][r] += 1
    
    # M2: f=A10+A11, g=B[0,col0]
    r = base + 1
    U[2][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1
    W[col0*n+1][r] += 1; W[col1*n+1][r] += -1
    
    # M3: f=A00, g=B[0,col1]-B[1,col1]
    r = base + 2
    U[0][r] += 1
    V[0*p+col1][r] += 1; V[1*p+col1][r] += -1
    W[col1*n+0][r] += 1; W[col1*n+1][r] += 1
    
    # M4: f=A11, g=B[1,col0]-B[0,col0]
    r = base + 3
    U[3][r] += 1
    V[1*p+col0][r] += 1; V[0*p+col0][r] += -1
    W[col0*n+0][r] += 1; W[col0*n+1][r] += 1
    
    # M5: f=A00+A01, g=B[1,col1]
    r = base + 4
    U[0][r] += 1; U[1][r] += 1
    V[1*p+col1][r] += 1
    W[col0*n+0][r] += -1; W[col1*n+0][r] += 1
    
    # M6: f=A10-A00, g=B[0,col0]+B[0,col1]
    r = base + 5
    U[2][r] += 1; U[0][r] += -1
    V[0*p+col0][r] += 1; V[0*p+col1][r] += 1
    W[col1*n+1][r] += 1
    
    # M7: f=A01-A11, g=B[1,col0]+B[1,col1]
    r = base + 6
    U[1][r] += 1; U[3][r] += -1
    V[1*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1


def _build_rank18():
    """Construct the rank-18 decomposition: 2 Strassen blocks + schoolbook for col 4."""
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen block 1: columns 0, 1
    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)
    
    # Strassen block 2: columns 2, 3
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W


def _build_cyclic_decomposition():
    """
    Attempt a cyclic group algebra (C5) based decomposition.
    
    The 2x2 x 2x5 multiplication tensor can be viewed as two 2x2 x 2x2 blocks
    plus one 2x2 x 2x1 block. The C5 structure suggests using periodic patterns
    across the 5 columns to share rank terms.
    
    Key insight: For matrix multiplication A(2x2) * B(2x5), we can exploit
    the fact that the output C(2x5) has structure C[i,k] = sum_j A[i,j]*B[j,k].
    
    Using DFT over Z_5, we can decompose the 5-column dimension into 5 frequency
    components. Each frequency component gives a 2x2 x 2x1 problem (rank 4 each),
    but with shared U factors across frequencies.
    
    Actually, let's think more carefully. The tensor is:
    T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    else 0
    
    For the C5 approach, we use the fact that multiplication by a circulant
    matrix can be diagonalized by DFT. But here B is 2x5, not square.
    
    Alternative: Use the structure that A is 2x2 (small) and B is 2x5.
    We can write B as [B_0 | B_1 | B_2 | B_3 | B_4] where B_j is 2x1.
    Then C[:,j] = A * B_j for each j.
    
    The rank of the full tensor is at most the sum of ranks of individual
    2x2 x 2x1 tensors, but we can share the A-side factors.
    
    For a 2x2 x 2x1 tensor (A*B where B is 2x1), the rank is 4 (schoolbook).
    But if we share the U factors across all 5 columns, we need:
    U factors: 4 (for the 4 entries of A)
    V factors: 5*2 = 10 (for each entry of B)  
    W factors: 5*2 = 10 (for each entry of C)
    
    Total rank if fully shared: max(4, 10, 10) = 10? No, that's not how it works.
    
    Actually, the tensor decomposition rank is the minimum number of rank-1
    tensors needed. For the 2x2 x 2x5 case, let's try a systematic approach.
    
    Let me try: Use 2 Strassen blocks (rank 14) + 1 schoolbook column (rank 4) = 18.
    Can we do better by overlapping?
    
    Try: 3 overlapping Strassen-like blocks covering columns (0,1), (2,3), and
    a modified block for column 4 that shares with column 3.
    """
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Try a rank-15 construction based on cyclic symmetry
    # Use the idea that columns 0-4 form a cycle, and we can use
    # 5 "frequency" terms plus additional terms for the 2x2 structure.
    
    # Approach: For each of the 5 columns, we need 4 multiplications (2x2).
    # Total schoolbook: 20. Strassen on pairs: 2*7 + 4 = 18.
    # Can we get to 15?
    
    # Let's try a direct search for rank 15 using the cyclic structure.
    # The key is that the U matrix (4xR) has only 4 rows, so the rank
    # is bounded below by 4 (need at least 4 independent U vectors).
    # Similarly V has 10 rows, W has 10 rows.
    
    # Lower bound: The tensor has 20 nonzero entries. Each rank-1 term
    # can cover at most 1 nonzero entry in this structured tensor.
    # So rank >= 20? No, that's not right. A rank-1 term U_r * V_r * W_r
    # can have multiple nonzero entries if U_r, V_r, W_r have multiple nonzeros.
    
    # For the matrix multiplication tensor, the rank is known to be:
    # 2x2 x 2x2: rank 7 (Strassen)
    # 2x2 x 2x3: rank ? 
    # 2x2 x 2x4: rank ?
    # 2x2 x 2x5: rank ?
    
    # Known results: R(2,2,3) = 8, R(2,2,4) = 11, R(2,2,5) = ?
    # Actually, I recall that R(2,2,k) for k>=3 might follow a pattern.
    
    # Let me try to construct a rank-15 solution explicitly.
    # Use 5 columns, each needing 2x2 multiplication.
    
    # Strategy: Use 5 "column-specific" rank-1 terms for the diagonal,
    # plus 10 additional terms for the off-diagonal, sharing U factors.
    
    # Actually, let's just try ALS with more seeds and a longer time budget
    # for rank 15, 14, 13.
    
    return None


def _try_reduce_rank18_to_17():
    """Attempt to find a rank-17 or lower decomposition using ALS with more seeds."""
    n, m, p = DIMENSIONS
    T = _build_target()
    deadline = time.time() + 100  # Leave time for other checks
    
    # Try ranks 17, 16, 15, 14 with multiple seeds
    for R in [17, 16, 15, 14]:
        if time.time() > deadline:
            break
        result = _try_als_rank(R, T, deadline, n_seeds=20, max_iters=80)
        if result is not None:
            U_l, V_l, W_l = result
            if _check_decomposition(U_l, V_l, W_l, T, tol=1e-7):
                return result
    return None


def _try_als_rank(R, T, deadline, n_seeds=5, max_iters=100):
    """Try ALS for a given rank with rational polish."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n

    seeds = [42, 123, 7, 999, 2024]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)

        inits = []
        inits.append((np.random.randn(U_rows, R) * 0.3,
                      np.random.randn(V_rows, R) * 0.3,
                      np.random.randn(W_rows, R) * 0.3))
        inits.append((np.random.randn(U_rows, R) * 1.0,
                      np.random.randn(V_rows, R) * 1.0,
                      np.random.randn(W_rows, R) * 1.0))

        for U0, V0, W0 in inits:
            if time.time() > deadline:
                break
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Update U
                A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
                for u in range(U_rows):
                    b = T[u].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[u] = sol
                
                # Update V
                A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
                for v in range(V_rows):
                    b = T[:, v, :].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[v] = sol
                
                # Update W
                A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
                for w in range(W_rows):
                    b = T[:, :, w].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[w] = sol
                
                result = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
                for r in range(R):
                    result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(result - T.astype(np.float64))
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

            if _check_decomposition(U, V, W, T, tol=1e-7):
                # Try to snap to rationals
                for denom in [1, 2]:
                    U_s = np.round(U * denom) / denom
                    V_s = np.round(V * denom) / denom
                    W_s = np.round(W * denom) / denom
                    if _check_exact(U_s, V_s, W_s, T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                return U.tolist(), V.tolist(), W.tolist()
    return None


def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    T = _build_target()

    # First, try the constructive rank-18 solution
    U18, V18, W18 = _build_rank18()
    if _check_decomposition(U18, V18, W18, T, tol=1e-8):
        # Try to reduce to rank 17
        result17 = _try_reduce_rank18_to_17()
        if result17 is not None:
            U17, V17, W17 = result17
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}
        
        # Try ALS for rank 17 and 16 with a time limit
        for R in [17, 16]:
            if time.time() > deadline - 10: break
            result = _try_als_rank(R, T, deadline - 5, n_seeds=10, max_iters=50)
            if result is not None:
                U_l, V_l, W_l = result
                if _check_decomposition(U_l, V_l, W_l, T, tol=1e-7):
                    return {"U": U_l, "V": V_l, "W": W_l}

        # Return rank-18 solution
        return {"U": U18, "V": V18, "W": W18}

    # Fallback: try ALS for lower ranks
    for R in range(7, 19):
        if time.time() > deadline:
            break
        result = _try_als_rank(R, T, deadline, n_seeds=3, max_iters=50)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}

    # Final fallback: schoolbook
    U_l, V_l, W_l = _schoolbook_decomposition()
    return {"U": U_l, "V": V_l, "W": W_l}
