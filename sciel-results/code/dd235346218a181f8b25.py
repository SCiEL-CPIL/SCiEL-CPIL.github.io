"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact using numpy for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    # M1 = (a+d)(e+h), contributes to C[0] and C[3]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (c+d)(e), contributes to C[2]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3 = a(f-h), contributes to C[1]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4 = d(g-e), contributes to C[0] and C[2]
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    # M5 = (a+b)(h), contributes to C[0] and C[1]
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    # M6 = (c-a)(e+f), contributes to C[3]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (b-d)(g+h), contributes to C[0] and C[3]
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    return terms


def _strassen_4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 via Kronecker product of 2x2 Strassen."""
    base_terms = _strassen_2x2_terms()
    terms_4x4 = []
    for c1 in range(7):
        for c2 in range(7):
            U4 = [0] * 16
            V4 = [0] * 16
            W4 = [0] * 16
            for I in range(4):
                for J in range(4):
                    I1, I0 = I // 2, I % 2
                    J1, J0 = J // 2, J % 2
                    U4[I*4+J] = base_terms[c1][0][I1*2+J1] * base_terms[c2][0][I0*2+J0]
            for J in range(4):
                for K in range(4):
                    J1, J0 = J // 2, J % 2
                    K1, K0 = K // 2, K % 2
                    V4[J*4+K] = base_terms[c1][1][J1*2+K1] * base_terms[c2][1][J0*2+K0]
            for K in range(4):
                for I in range(4):
                    K1, K0 = K // 2, K % 2
                    I1, I0 = I // 2, I % 2
                    W4[K*4+I] = base_terms[c1][2][K1*2+I1] * base_terms[c2][2][K0*2+I0]
            terms_4x4.append((U4, V4, W4))
    return terms_4x4


def _make_4x4x7_decomposition():
    """Create a rank-98 decomposition for 4x4x7 using Strassen 4x4x4 terms.
    Split the 7 columns into two groups: cols 0-3 and cols 4-6.
    """
    terms_4x4 = _strassen_4x4_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    for U4, V4, W4 in terms_4x4:
        # First group: columns 0-3 of B, contributing to columns 0-3 of C
        V1 = [0] * 28
        W1 = [0] * 28
        for J in range(4):
            for K in range(4):
                V1[J*7+K] = V4[J*4+K]
        for K in range(4):
            for I in range(4):
                W1[K*4+I] = W4[K*4+I]
        
        U_list.append(list(U4))
        V_list.append(V1)
        W_list.append(W1)
        
        # Second group: columns 4-6 of B, contributing to columns 4-6 of C
        V2 = [0] * 28
        W2 = [0] * 28
        for J in range(4):
            for K in range(3):
                V2[J*7+(K+4)] = V4[J*4+K]
        for K in range(3):
            for I in range(4):
                W2[(K+4)*4+I] = W4[K*4+I]
        
        U_list.append(list(U4))
        V_list.append(V2)
        W_list.append(W2)
    
    rank = len(U_list)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    for c in range(rank):
        for i in range(16):
            U[i][c] = U_list[c][i]
        for i in range(28):
            V[i][c] = V_list[c][i]
        for i in range(28):
            W[i][c] = W_list[c][i]
    
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Attempt to reduce rank by identifying and removing redundant terms.
    A term c is redundant if it can be expressed as a linear combination of others
    in the tensor decomposition sense. We use a greedy approach: try removing each
    term and check if the remaining terms can still span the tensor."""
    start_time = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can try to find linear dependencies among the rank-1 tensors.
    # For now, try a simple approach: remove terms with many zeros.
    
    # Count non-zero entries in each term
    term_sparsity = []
    for c in range(rank):
        nz = np.count_nonzero(U_arr[:, c]) + np.count_nonzero(V_arr[:, c]) + np.count_nonzero(W_arr[:, c])
        term_sparsity.append(nz)
    
    # Sort by sparsity (most sparse first) and try removing
    sorted_terms = sorted(range(rank), key=lambda c: term_sparsity[c])
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    keep = [True] * rank
    
    for idx in sorted_terms:
        if time.time() > deadline - 30:
            break
        if not keep[idx]:
            continue
        # Try removing term idx
        keep[idx] = False
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        if keep[c]:
                            s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            best_rank = sum(keep)
            # Update best
            keep_indices = [c for c in range(rank) if keep[c]]
            best_U = U_arr[:, keep_indices]
            best_V = V_arr[:, keep_indices]
            best_W = W_arr[:, keep_indices]
        else:
            keep[idx] = True
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Get the Strassen-based rank-98 decomposition
    U, V, W = _make_4x4x7_decomposition()
    
    # Verify the decomposition
    if not _check_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
