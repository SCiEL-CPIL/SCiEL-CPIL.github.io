import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2 x 2x2 matrix mult.
    Returns (alpha, beta, recon) where:
    alpha[s] = list of (block_idx, coeff) for A-side
    beta[s] = list of (block_idx, coeff) for B-side
    recon[s] = list of (out_row, out_col, coeff) for C-side
    Block indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    alpha = [
        [(0, 1), (3, 1)],       # s0: A00+A11
        [(2, 1), (3, 1)],       # s1: A10+A11
        [(0, 1)],               # s2: A00
        [(3, 1)],               # s3: A11
        [(0, 1), (1, 1)],       # s4: A00+A01
        [(2, 1), (0, -1)],      # s5: A10-A00
        [(1, 1), (3, -1)],      # s6: A01-A11
    ]
    beta = [
        [(0, 1), (3, 1)],       # s0: B00+B11
        [(0, 1)],               # s1: B00
        [(1, 1), (3, -1)],      # s2: B01-B11
        [(2, 1), (0, -1)],      # s3: B10-B00
        [(3, 1)],               # s4: B11
        [(0, 1), (1, 1)],       # s5: B00+B01
        [(2, 1), (3, 1)],       # s6: B10+B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]
    return alpha, beta, recon

def _rank12_223():
    """Return rank-12 schoolbook decomposition for 2x2 x 2x3 matrix mult.
    A is 2x2, B is 2x3, C is 2x3.
    """
    alpha = [
        [(0, 1)], [(0, 1)], [(0, 1)],
        [(1, 1)], [(1, 1)], [(1, 1)],
        [(2, 1)], [(2, 1)], [(2, 1)],
        [(3, 1)], [(3, 1)], [(3, 1)],
    ]
    beta = [
        [(0, 1)], [(1, 1)], [(2, 1)],
        [(3, 1)], [(4, 1)], [(5, 1)],
        [(0, 1)], [(1, 1)], [(2, 1)],
        [(3, 1)], [(4, 1)], [(5, 1)],
    ]
    recon = [
        [(0, 0, 1)], [(0, 1, 1)], [(0, 2, 1)],
        [(0, 0, 1)], [(0, 1, 1)], [(0, 2, 1)],
        [(1, 0, 1)], [(1, 1, 1)], [(1, 2, 1)],
        [(1, 0, 1)], [(1, 1, 1)], [(1, 2, 1)],
    ]
    return alpha, beta, recon

def _build():
    """Build Kronecker product decomposition: M_{2,2,3} (outer) x M_{2,2,2} (inner).
    Outer: 2x2 grid of A blocks, 2x3 grid of B blocks, 2x3 grid of C blocks.
    Inner: each block is 2x2, so inner multiplication is 2x2 x 2x2 (Strassen rank 7).
    Total rank = 12 * 7 = 84.
    """
    n, m, p = 4, 4, 6
    outer_alpha, outer_beta, outer_recon = _rank12_223()
    inner_alpha, inner_beta, inner_recon = _strassen_222()
    
    R_outer = len(outer_alpha)
    R_inner = len(inner_alpha)
    R = R_outer * R_inner
    
    Umat = np.zeros((16, R))
    Vmat = np.zeros((24, R))
    Wmat = np.zeros((24, R))
    
    for so in range(R_outer):
        for si in range(R_inner):
            col = so * R_inner + si
            
            # U: A[i,k] where i = outer_row*2 + inner_row, k = outer_col*2 + inner_col
            for (block_o, coeff_o) in outer_alpha[so]:
                ar, ac = block_o // 2, block_o % 2
                for (block_i, coeff_i) in inner_alpha[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gi = ar * 2 + ir
                    gk = ac * 2 + ic
                    Umat[gi * 4 + gk, col] += coeff_o * coeff_i
            
            # V: B[k,j] where k = outer_row*2 + inner_row, j = outer_col*2 + inner_col
            for (block_o, coeff_o) in outer_beta[so]:
                br, bc = block_o // 3, block_o % 3
                for (block_i, coeff_i) in inner_beta[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gk = br * 2 + ir
                    gj = bc * 2 + ic
                    Vmat[gk * 6 + gj, col] += coeff_o * coeff_i
            
            # W: C[j,i] in trace convention, W row = j*n+i = j*4+i
            for (out_r, out_c, coeff_o) in outer_recon[so]:
                for (in_r, in_c, coeff_i) in inner_recon[si]:
                    gi = out_r * 2 + in_r
                    gj = out_c * 2 + in_c
                    Wmat[gj * 4 + gi, col] += coeff_o * coeff_i
    
    return Umat, Vmat, Wmat

def _verify(U, V, W):
    n, m, p = 4, 4, 6
    R = U.shape[1]
    # Vectorized verification using tensor reconstruction
    # T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # We check that T is all ones
    
    # Reshape to 3D arrays for efficient computation
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # Compute tensor contraction: T[i,k,j] = sum_r U3[i,k,r] * V3[k,j,r] * W3[j,i,r]
    # This is a 3-index tensor contraction
    # We can do this with einsum
    T = np.einsum('ikr,kjr,jir->ikj', U3, V3, W3, optimize=True)
    
    # Check if all entries are 1.0
    return np.allclose(T, 1.0, atol=1e-6)

def _build_fallback():
    """Fallback: schoolbook decomposition (rank = n*m*p = 96)"""
    n, m, p = 4, 4, 6
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m + k, idx] = 1
                V[k*p + j, idx] = 1
                W[j*n + i, idx] = 1
                idx += 1
    return U, V, W

def construct():
    deadline = time.time() + 110.0
    
    # Try the Kronecker product construction
    U, V, W = _build()
    
    if not _verify(U, V, W):
        # Fallback: schoolbook decomposition
        U, V, W = _build_fallback()
    
    # Round to integers where possible
    def to_list(M):
        out = []
        for row in M:
            r = []
            for x in row:
                rx = round(x)
                r.append(int(rx) if abs(x - rx) < 1e-9 else float(x))
            out.append(r)
        return out
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
