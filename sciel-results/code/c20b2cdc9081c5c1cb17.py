"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def _build_subtensor(n, m, p):
    """Build the (n,m,p) matrix multiplication tensor with shapes (n*m, m*p, p*n)."""
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _strassen_2x2_block(n, m, p, i0, j0, k0):
    """
    Generate Strassen terms for a 2x2 block multiplication:
    A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
    
    Strassen's 7 multiplications:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    Results:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # Each term is a list of (u_idx, coeff), (v_idx, coeff), (w_idx, coeff)
    # u_idx = i*m + j, v_idx = j*p + k, w_idx = k*n + i
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # A00 = A[i0, j0], A11 = A[i0+1, j0+1]
    # B00 = B[j0, k0], B11 = B[j0+1, k0+1]
    m1_u = [(i0*m+j0, 1.0), ((i0+1)*m+(j0+1), 1.0)]
    m1_v = [(j0*p+k0, 1.0), ((j0+1)*p+(k0+1), 1.0)]
    # C00 += M1, C11 += M1
    m1_w = [(k0*n+i0, 1.0), ((k0+1)*n+(i0+1), 1.0)]
    terms.append((m1_u, m1_v, m1_w))
    
    # M2 = (A10+A11)B00
    # A10 = A[i0+1, j0], A11 = A[i0+1, j0+1]
    # B00 = B[j0, k0]
    m2_u = [((i0+1)*m+j0, 1.0), ((i0+1)*m+(j0+1), 1.0)]
    m2_v = [(j0*p+k0, 1.0)]
    # C10 += M2, C11 -= M2
    m2_w = [((k0)*n+(i0+1), 1.0), ((k0+1)*n+(i0+1), -1.0)]
    terms.append((m2_u, m2_v, m2_w))
    
    # M3 = A00(B10-B11)
    # A00 = A[i0, j0]
    # B10 = B[j0+1, k0], B11 = B[j0+1, k0+1]
    m3_u = [(i0*m+j0, 1.0)]
    m3_v = [((j0+1)*p+k0, 1.0), ((j0+1)*p+(k0+1), -1.0)]
    # C01 += M3, C11 += M3
    m3_w = [(k0*n+(i0+1), 1.0), ((k0+1)*n+(i0+1), 1.0)]
    terms.append((m3_u, m3_v, m3_w))
    
    # M4 = A11(B01-B00)
    # A11 = A[i0+1, j0+1]
    # B01 = B[j0, k0+1], B00 = B[j0, k0]
    m4_u = [((i0+1)*m+(j0+1), 1.0)]
    m4_v = [(j0*p+(k0+1), 1.0), (j0*p+k0, -1.0)]
    # C00 += M4, C10 += M4
    m4_w = [(k0*n+i0, 1.0), (k0*n+(i0+1), 1.0)]
    terms.append((m4_u, m4_v, m4_w))
    
    # M5 = (A00+A01)B11
    # A00 = A[i0, j0], A01 = A[i0, j0+1]
    # B11 = B[j0+1, k0+1]
    m5_u = [(i0*m+j0, 1.0), (i0*m+(j0+1), 1.0)]
    m5_v = [((j0+1)*p+(k0+1), 1.0)]
    # C00 -= M5, C01 += M5
    m5_w = [(k0*n+i0, -1.0), (k0*n+(i0+1), 1.0)]
    terms.append((m5_u, m5_v, m5_w))
    
    # M6 = (A10-A00)(B00+B10)
    # A10 = A[i0+1, j0], A00 = A[i0, j0]
    # B00 = B[j0, k0], B10 = B[j0+1, k0]
    m6_u = [((i0+1)*m+j0, 1.0), (i0*m+j0, -1.0)]
    m6_v = [(j0*p+k0, 1.0), ((j0+1)*p+k0, 1.0)]
    # C11 += M6
    m6_w = [((k0+1)*n+(i0+1), 1.0)]
    terms.append((m6_u, m6_v, m6_w))
    
    # M7 = (A01-A11)(B01+B11)
    # A01 = A[i0, j0+1], A11 = A[i0+1, j0+1]
    # B01 = B[j0, k0+1], B11 = B[j0+1, k0+1]
    m7_u = [(i0*m+(j0+1), 1.0), ((i0+1)*m+(j0+1), -1.0)]
    m7_v = [(j0*p+(k0+1), 1.0), ((j0+1)*p+(k0+1), 1.0)]
    # C00 += M7
    m7_w = [(k0*n+i0, 1.0)]
    terms.append((m7_u, m7_v, m7_w))
    
    return terms

def _strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for n x m * m x p matrix multiplication.
    Apply Strassen to all possible 2x2 blocks where the A-block is 2x2 and B-block is 2x2.
    For 3x3x8: apply Strassen to A[0:2,0:2] with B blocks at various k positions.
    
    Strategy: Use Strassen for the top-left 2x2 A block with each 2x2 B block,
    and schoolbook for the remaining elements.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    # Collect all terms as (u_indices, v_indices, w_indices)
    all_terms = []
    
    # Track which (i,j,k) triples have been covered by Strassen
    covered = set()
    
    # Apply Strassen to A[0:2, 0:2] with B blocks at k=0:2, k=2:4, k=4:6, k=6:8
    for k0 in range(0, p-1, 2):
        if k0 + 2 <= p:
            terms = _strassen_2x2_block(n, m, p, i0=0, j0=0, k0=k0)
            all_terms.extend(terms)
            # Mark covered triples
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        covered.add((i, j, k0+k))
    
    # Apply Strassen to A[1:3, 0:2] with B blocks
    for k0 in range(0, p-1, 2):
        if k0 + 2 <= p:
            terms = _strassen_2x2_block(n, m, p, i0=1, j0=0, k0=k0)
            all_terms.extend(terms)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        covered.add((i+1, j, k0+k))
    
    # Apply Strassen to A[0:2, 1:3] with B blocks
    for k0 in range(0, p-1, 2):
        if k0 + 2 <= p:
            terms = _strassen_2x2_block(n, m, p, i0=0, j0=1, k0=k0)
            all_terms.extend(terms)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        covered.add((i, j+1, k0+k))
    
    # Apply Strassen to A[1:3, 1:3] with B blocks
    for k0 in range(0, p-1, 2):
        if k0 + 2 <= p:
            terms = _strassen_2x2_block(n, m, p, i0=1, j0=1, k0=k0)
            all_terms.extend(terms)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        covered.add((i+1, j+1, k0+k))
    
    # Schoolbook for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    all_terms.append(
                        ([(i*m+j, 1.0)], [(j*p+k, 1.0)], [(k*n+i, 1.0)])
                    )
    
    rank = len(all_terms)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for c, (u_list, v_list, w_list) in enumerate(all_terms):
        for (idx, coeff) in u_list:
            U[idx, c] += coeff
        for (idx, coeff) in v_list:
            V[idx, c] += coeff
        for (idx, coeff) in w_list:
            W[idx, c] += coeff
    
    return U, V, W, rank

def construct():
    """Find a low-rank exact tensor decomposition for (3,3,8) matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    U shape: (n*m, rank) indexed by i*m+j
    V shape: (m*p, rank) indexed by j*p+k
    W shape: (p*n, rank) indexed by k*n+i
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build target tensor
    T = _build_subtensor(n, m, p)
    
    # Try Strassen-based decomposition
    U_str, V_str, W_str, rank_str = _strassen_decomposition(n, m, p)
    
    # Verify
    recon = np.zeros((nm, mp, pn))
    for c in range(rank_str):
        recon += np.einsum('a,b,c->abc', U_str[:,c], V_str[:,c], W_str[:,c])
    
    if np.abs(recon - T).max() < 1e-8:
        U_out, V_out, W_out = U_str, V_str, W_str
    else:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_out = np.zeros((nm, schoolbook_rank))
        V_out = np.zeros((mp, schoolbook_rank))
        W_out = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_out[i*m+j, col] = 1.0
                    V_out[j*p+k, col] = 1.0
                    W_out[k*n+i, col] = 1.0
                    col += 1
    
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
