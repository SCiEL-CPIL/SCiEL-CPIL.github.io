"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition.

Strategy: Block partitioning with Strassen for 2x2 sub-blocks and structured
random search for low-rank decompositions of rectangular sub-problems.

Partition:
A (4x5) -> rows split 2+2, cols split 2+3
B (5x5) -> rows split 2+3, cols split 2+3
C (4x5) -> rows split 2+2, cols split 2+3

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

Use Strassen (rank 7) for 2x2*2x2, and search for low-rank algorithms
for the rectangular products.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return Strassen rank-7 decomposition for 2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)*(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)*b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11*(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a22*(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)*b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0 if j == k else 0.0
                if abs(s - expected) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """Combine multiple sub-decompositions into one for the full problem."""
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for U_sub, V_sub, W_sub, a_r, a_c, b_r, b_c, c_r, c_c in decomps:
        sub_rank = U_sub.shape[1]
        
        for ri in range(U_sub.shape[0]):
            local_i = ri // a_c
            local_j = ri % a_c
            gi = a_r + local_i
            gj = a_c + local_j
            for r in range(sub_rank):
                U[gi*m + gj, col_offset + r] = U_sub[ri, r]
        
        for ri in range(V_sub.shape[0]):
            local_j = ri // b_c
            local_k = ri % b_c
            gj = b_r + local_j
            gk = b_c + local_k
            for r in range(sub_rank):
                V[gj*p + gk, col_offset + r] = V_sub[ri, r]
        
        for ri in range(W_sub.shape[0]):
            local_k = ri // c_r
            local_i = ri % c_r
            gk = c_r + local_k
            gi = c_c + local_i
            for r in range(sub_rank):
                W[gk*n + gi, col_offset + r] = W_sub[ri, r]
        
        col_offset += sub_rank
    
    return U, V, W

def _search_rectangular(n_sub, m_sub, p_sub, max_rank, time_budget, rng):
    """Search for low-rank decomposition of n_sub x m_sub * m_sub x p_sub.
    
    Returns (U, V, W) or None if not found within time budget.
    """
    nm = n_sub * m_sub
    mp = m_sub * p_sub
    pn = p_sub * n_sub
    
    # Try ranks from max_rank down to some minimum
    for rank in range(max_rank, 0, -1):
        if time.time() > time_budget:
            break
        
        # Multiple trials per rank
        num_trials = min(100, max(10, int((time_budget - time.time()) * 50)))
        
        for trial in range(num_trials):
            if time.time() > time_budget:
                break
            
            # Random initialization with small integer/half-integer values
            U = rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(nm, rank))
            V = rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(mp, rank))
            W = rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(pn, rank))
            
            # Verify
            valid = True
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m_sub+j, r] * V[j*p_sub+k, r] * W[k*n_sub+i, r]
                        expected = 1.0 if j == k else 0.0
                        if abs(s - expected) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
    
    return None

def _build_block_decomposition():
    """Build decomposition using Strassen for 2x2 blocks and schoolbook for rest."""
    n, m, p = 4, 5, 5
    
    # Get Strassen decomposition for 2x2
    S_U, S_V, S_W = _strassen_2x2()
    
    decomps = []
    
    # C11 = A11*B11 + A12*B21
    # Term 1: A11(2x2) * B11(2x2) -> C11(2x2), use Strassen
    decomps.append((S_U, S_V, S_W, 0, 0, 0, 0, 0, 0))
    
    # Term 2: A12(2x3) * B21(3x2) -> C11(2x2), schoolbook rank 12
    U12_B21 = np.zeros((6, 12))
    V12_B21 = np.zeros((6, 12))
    W12_B21 = np.zeros((4, 12))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U12_B21[i*3+j, col] = 1
                V12_B21[j*2+k, col] = 1
                W12_B21[k*2+i, col] = 1
                col += 1
    decomps.append((U12_B21, V12_B21, W12_B21, 0, 2, 2, 2, 0, 0))
    
    # C12 = A11*B12 + A12*B22
    # Term 1: A11(2x2) * B12(2x3) -> C12(2x3), schoolbook rank 12
    U11_B12 = np.zeros((4, 12))
    V11_B12 = np.zeros((6, 12))
    W11_B12 = np.zeros((6, 12))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U11_B12[i*2+j, col] = 1
                V11_B12[j*3+k, col] = 1
                W11_B12[k*2+i, col] = 1
                col += 1
    decomps.append((U11_B12, V11_B12, W11_B12, 0, 0, 0, 2, 0, 2))
    
    # Term 2: A12(2x3) * B22(3x3) -> C12(2x3), schoolbook rank 18
    U12_B22 = np.zeros((6, 18))
    V12_B22 = np.zeros((9, 18))
    W12_B22 = np.zeros((6, 18))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U12_B22[i*3+j, col] = 1
                V12_B22[j*3+k, col] = 1
                W12_B22[k*2+i, col] = 1
                col += 1
    decomps.append((U12_B22, V12_B22, W12_B22, 0, 2, 2, 2, 0, 2))
    
    # C21 = A21*B11 + A22*B21
    # Term 1: A21(2x2) * B11(2x2) -> C21(2x2), use Strassen
    decomps.append((S_U, S_V, S_W, 2, 0, 0, 0, 2, 0))
    
    # Term 2: A22(2x3) * B21(3x2) -> C21(2x2), schoolbook rank 12
    U22_B21 = np.zeros((6, 12))
    V22_B21 = np.zeros((6, 12))
    W22_B21 = np.zeros((4, 12))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U22_B21[i*3+j, col] = 1
                V22_B21[j*2+k, col] = 1
                W22_B21[k*2+i, col] = 1
                col += 1
    decomps.append((U22_B21, V22_B21, W22_B21, 2, 2, 2, 2, 2, 0))
    
    # C22 = A21*B12 + A22*B22
    # Term 1: A21(2x2) * B12(2x3) -> C22(2x3), schoolbook rank 12
    U21_B12 = np.zeros((4, 12))
    V21_B12 = np.zeros((6, 12))
    W21_B12 = np.zeros((6, 12))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U21_B12[i*2+j, col] = 1
                V21_B12[j*3+k, col] = 1
                W21_B12[k*2+i, col] = 1
                col += 1
    decomps.append((U21_B12, V21_B12, W21_B12, 2, 0, 0, 2, 2, 2))
    
    # Term 2: A22(2x3) * B22(3x3) -> C22(2x3), schoolbook rank 18
    U22_B22 = np.zeros((6, 18))
    V22_B22 = np.zeros((9, 18))
    W22_B22 = np.zeros((6, 18))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U22_B22[i*3+j, col] = 1
                V22_B22[j*3+k, col] = 1
                W22_B22[k*2+i, col] = 1
                col += 1
    decomps.append((U22_B22, V22_B22, W22_B22, 2, 2, 2, 2, 2, 2))
    
    # Total rank: 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18 = 98
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Build the base block decomposition with Strassen + schoolbook
    U_base, V_base, W_base = _build_block_decomposition()
    base_rank = U_base.shape[1]  # 98
    
    # Verify base decomposition
    if not _verify_decomposition(U_base, V_base, W_base, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to improve by searching for better decompositions of rectangular blocks
    # The rectangular blocks are: 2x3*3x2 (rank 12), 2x2*2x3 (rank 12), 2x3*3x3 (rank 18)
    
    rng = np.random.RandomState(42)
    
    # Try to find better decompositions for 2x3 * 3x2 -> 2x2
    # Schoolbook rank is 12, try to find lower
    best_2x3_3x2 = None
    time_budget_1 = start_time + 40
    for rank in range(11, 5, -1):
        if time.time() > time_budget_1:
            break
        result = _search_rectangular(2, 3, 2, rank, time_budget_1, rng)
        if result is not None:
            best_2x3_3x2 = result
            break
    
    # Try to find better decompositions for 2x2 * 2x3 -> 2x3
    # Schoolbook rank is 12, try to find lower
    best_2x2_2x3 = None
    time_budget_2 = start_time + 70
    for rank in range(11, 5, -1):
        if time.time() > time_budget_2:
            break
        result = _search_rectangular(2, 2, 3, rank, time_budget_2, rng)
        if result is not None:
            best_2x2_2x3 = result
            break
    
    # Try to find better decompositions for 2x3 * 3x3 -> 2x3
    # Schoolbook rank is 18, try to find lower
    best_2x3_3x3 = None
    time_budget_3 = start_time + 100
    for rank in range(17, 10, -1):
        if time.time() > time_budget_3:
            break
        result = _search_rectangular(2, 3, 3, rank, time_budget_3, rng)
        if result is not None:
            best_2x3_3x3 = result
            break
    
    # Build improved decomposition if we found better sub-decompositions
    S_U, S_V, S_W = _strassen_2x2()
    decomps = []
    
    # C11 = A11*B11 + A12*B21
    decomps.append((S_U, S_V, S_W, 0, 0, 0, 0, 0, 0))
    
    if best_2x3_3x2 is not None:
        U12_B21, V12_B21, W12_B21 = best_2x3_3x2
    else:
        U12_B21 = np.zeros((6, 12))
        V12_B21 = np.zeros((6, 12))
        W12_B21 = np.zeros((4, 12))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U12_B21[i*3+j, col] = 1
                    V12_B21[j*2+k, col] = 1
                    W12_B21[k*2+i, col] = 1
                    col += 1
    decomps.append((U12_B21, V12_B21, W12_B21, 0, 2, 2, 2, 0, 0))
    
    # C12 = A11*B12 + A12*B22
    if best_2x2_2x3 is not None:
        U11_B12, V11_B12, W11_B12 = best_2x2_2x3
    else:
        U11_B12 = np.zeros((4, 12))
        V11_B12 = np.zeros((6, 12))
        W11_B12 = np.zeros((6, 12))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U11_B12[i*2+j, col] = 1
                    V11_B12[j*3+k, col] = 1
                    W11_B12[k*2+i, col] = 1
                    col += 1
    decomps.append((U11_B12, V11_B12, W11_B12, 0, 0, 0, 2, 0, 2))
    
    if best_2x3_3x3 is not None:
        U12_B22, V12_B22, W12_B22 = best_2x3_3x3
    else:
        U12_B22 = np.zeros((6, 18))
        V12_B22 = np.zeros((9, 18))
        W12_B22 = np.zeros((6, 18))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U12_B22[i*3+j, col] = 1
                    V12_B22[j*3+k, col] = 1
                    W12_B22[k*2+i, col] = 1
                    col += 1
    decomps.append((U12_B22, V12_B22, W12_B22, 0, 2, 2, 2, 0, 2))
    
    # C21 = A21*B11 + A22*B21
    decomps.append((S_U, S_V, S_W, 2, 0, 0, 0, 2, 0))
    
    if best_2x3_3x2 is not None:
        U22_B21, V22_B21, W22_B21 = best_2x3_3x2
    else:
        U22_B21 = np.zeros((6, 12))
        V22_B21 = np.zeros((6, 12))
        W22_B21 = np.zeros((4, 12))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U22_B21[i*3+j, col] = 1
                    V22_B21[j*2+k, col] = 1
                    W22_B21[k*2+i, col] = 1
                    col += 1
    decomps.append((U22_B21, V22_B21, W22_B21, 2, 2, 2, 2, 2, 0))
    
    # C22 = A21*B12 + A22*B22
    if best_2x2_2x3 is not None:
        U21_B12, V21_B12, W21_B12 = best_2x2_2x3
    else:
        U21_B12 = np.zeros((4, 12))
        V21_B12 = np.zeros((6, 12))
        W21_B12 = np.zeros((6, 12))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U21_B12[i*2+j, col] = 1
                    V21_B12[j*3+k, col] = 1
                    W21_B12[k*2+i, col] = 1
                    col += 1
    decomps.append((U21_B12, V21_B12, W21_B12, 2, 0, 0, 2, 2, 2))
    
    if best_2x3_3x3 is not None:
        U22_B22, V22_B22, W22_B22 = best_2x3_3x3
    else:
        U22_B22 = np.zeros((6, 18))
        V22_B22 = np.zeros((9, 18))
        W22_B22 = np.zeros((6, 18))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U22_B22[i*3+j, col] = 1
                    V22_B22[j*3+k, col] = 1
                    W22_B22[k*2+i, col] = 1
                    col += 1
    decomps.append((U22_B22, V22_B22, W22_B22, 2, 2, 2, 2, 2, 2))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Verify the combined decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to base decomposition
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
# EVOLVE-BLOCK-END
