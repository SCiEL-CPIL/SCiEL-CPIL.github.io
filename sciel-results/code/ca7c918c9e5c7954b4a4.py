"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses block decomposition with Strassen on all 2x2x2 sub-blocks.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices (row-major indexing within the 2x2 block).
    U[local_i*2+local_j, c], V[local_j*2+local_k, c], W[local_k*2+local_i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a+d)(e+h)
    # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # C[0,0] += M1, C[1,1] += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (c+d)e
    # c=A[1,0], d=A[1,1], e=B[0,0]
    # C[1,0] += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3 = a(f-h)
    # a=A[0,0], f=B[0,1], h=B[1,1]
    # C[0,1] += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = d(g-e)
    # d=A[1,1], g=B[1,0], e=B[0,0]
    # C[1,0] += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Term 5: M5 = (a+b)h
    # a=A[0,0], b=A[0,1], h=B[1,1]
    # C[0,0] -= M5, C[0,1] += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: M6 = (c-d)(f+g)
    # c=A[1,0], d=A[1,1], f=B[0,1], g=B[1,0]
    # C[1,1] += M6
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (b-d)(e+f)
    # b=A[0,1], d=A[1,1], e=B[0,0], f=B[0,1]
    # C[0,0] += M7
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify_exact(n, m, p, U, V, W):
    """
    Verify exact decomposition using Fraction arithmetic.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    uc = u_row[c] if isinstance(u_row[c], Fraction) else Fraction(u_row[c])
                    vc = v_row[c] if isinstance(v_row[c], Fraction) else Fraction(v_row[c])
                    wc = w_row[c] if isinstance(w_row[c], Fraction) else Fraction(w_row[c])
                    s += uc * vc * wc
                if s != 1:
                    return False
    return True


def _construct_5x5x5():
    """
    Construct a 5x5x5 matrix multiplication decomposition.
    
    Strategy: Partition 5 = 2+2+1 in each dimension.
    Use Strassen (rank 7) for each 2x2x2 sub-block where all three
    dimensions are 2x2. There are 2*2*2 = 8 such blocks.
    Use schoolbook (rank 1) for all remaining triples.
    
    Total rank = 8*7 + (125 - 8*4) = 56 + 93 = 149... 
    Wait, that's wrong. Let me recount.
    
    Total triples: 125
    Triples handled by Strassen: 8 blocks * 4 triples each = 32 triples
    Triples handled by schoolbook: 125 - 32 = 93
    Total rank = 8*7 + 93 = 56 + 93 = 149
    
    Hmm, that's worse than 125. The issue is that Strassen uses 7 terms
    for 4 output entries, so it's 7/4 = 1.75 per output vs schoolbook 1 per output.
    
    Wait, I'm confusing myself. Let me reconsider.
    
    In the original approach, we had ONE 2x2x2 block handled by Strassen (7 terms)
    instead of 4 schoolbook terms. That saved 4-7 = -3... no that's worse.
    
    Actually wait: schoolbook for a 2x2x2 block uses 4*4*4 = 64 terms? No.
    
    Let me re-read the problem. The tensor T[i,j,k] = 1 for all i,j,k in [0,5).
    The decomposition is T = sum_c U_c * V_c * W_c where:
    U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    Schoolbook: for each (i,j,k), one term with U[i,j]=1, V[j,k]=1, W[k,i]=1.
    That's 125 terms.
    
    Strassen on a 2x2x2 sub-block: For i,j,k in a specific 2x2x2 sub-block,
    instead of 8 schoolbook terms (one for each i,j,k pair), we use 7 Strassen terms.
    Wait no, for a 2x2x2 block, there are 2*2*2 = 8 triples (i,j,k).
    Schoolbook would use 8 terms. Strassen uses 7 terms. Saves 1.
    
    So for 8 such blocks: saves 8*1 = 8. Total rank = 125 - 8 = 117.
    
    But wait, in the original code, it only handled ONE 2x2x2 block (i,j,k all in {0,1}),
    which has 8 triples, and used 7 Strassen terms instead of 8 schoolbook terms.
    That saves 1, giving 124.
    
    Now I need to handle ALL 8 possible 2x2x2 sub-blocks.
    The 2x2 sub-blocks start at offsets 0 and 2 in each dimension.
    So the 8 blocks are at (i0,j0,k0) where i0,j0,k0 in {0,2}.
    
    For each such block, the 8 triples are:
    (i0+di, j0+dj, k0+dk) for di,dj,dk in {0,1}
    
    Each block uses 7 Strassen terms instead of 8 schoolbook terms, saving 1.
    Total: 125 - 8 = 117.
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Get Strassen decomposition for 2x2x2 (local coordinates)
    U_s, V_s, W_s = _strassen_2x2()
    
    # Determine which triples are handled by Strassen
    strassen_triples = set()
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                for di in [0, 1]:
                    for dj in [0, 1]:
                        for dk in [0, 1]:
                            i, j, k = i0 + di, j0 + dj, k0 + dk
                            strassen_triples.add((i, j, k))
    
    # Total rank: 8 Strassen blocks * 7 terms + (125 - 32) schoolbook terms
    n_strassen_blocks = 8
    n_strassen_terms = n_strassen_blocks * 7
    n_schoolbook_terms = 125 - len(strassen_triples)
    rank = n_strassen_terms + n_schoolbook_terms
    assert rank == 117, f"Expected rank 117, got {rank}"
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    # Place Strassen terms for each 2x2x2 block
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            # U index: global row = i0+li, global col = j0+lj
                            u_idx = (i0 + li) * m + (j0 + lj)
                            # Local U index in 2x2: li*2 + lj
                            local_u = li * 2 + lj
                            val = U_s[local_u][c]
                            if val != 0:
                                U[u_idx][col] += val
                    
                    for lj in range(2):
                        for lk in range(2):
                            # V index: global row = j0+lj, global col = k0+lk
                            v_idx = (j0 + lj) * p + (k0 + lk)
                            # Local V index in 2x2: lj*2 + lk
                            local_v = lj * 2 + lk
                            val = V_s[local_v][c]
                            if val != 0:
                                V[v_idx][col] += val
                    
                    for lk in range(2):
                        for li in range(2):
                            # W index: global row = k0+lk, global col = i0+li
                            w_idx = (k0 + lk) * n + (i0 + li)
                            # Local W index in 2x2: lk*2 + li
                            local_w = lk * 2 + li
                            val = W_s[local_w][c]
                            if val != 0:
                                W[w_idx][col] += val
                    
                    col += 1
    
    # Add schoolbook terms for all other triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_triples:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}, rank


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110  # Safety margin before 120s limit
    
    # Construct the 117-rank decomposition
    result, rank = _construct_5x5x5()
    
    # Verify it's correct
    if time.time() < deadline and _verify_exact(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook (rank 125) if verification fails or times out
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
