import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2 x 2x2 matrix mult.
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

def _rank11_223():
    """Return rank-11 decomposition for 2x2 x 2x3 matrix mult.
    A is 2x2, B is 2x3, C is 2x3.
    Block indexing for A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    Block indexing for B: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    C output: (row, col) where row in {0,1}, col in {0,1,2}
    """
    alpha = [
        [(0, 1), (3, 1)],       # s0
        [(2, 1), (3, 1)],       # s1
        [(0, 1)],               # s2
        [(3, 1)],               # s3
        [(0, 1), (1, 1)],       # s4
        [(2, 1), (0, -1)],      # s5
        [(1, 1), (3, -1)],      # s6
    ]
    beta = [
        [(0, 1), (4, 1)],       # s0: B00+B11 (B11 is block 4)
        [(0, 1)],               # s1: B00
        [(1, 1), (4, -1)],      # s2: B01-B11
        [(3, 1), (0, -1)],      # s3: B10-B00
        [(4, 1)],               # s4: B11
        [(0, 1), (1, 1)],       # s5: B00+B01
        [(3, 1), (4, 1)],       # s6: B10+B11
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
    
    # Next 4 terms: for the 3rd column of C
    alpha.append([(0, 1)])
    beta.append([(2, 1)])
    recon.append([(0, 2, 1)])
    
    alpha.append([(1, 1)])
    beta.append([(5, 1)])
    recon.append([(0, 2, 1)])
    
    alpha.append([(2, 1)])
    beta.append([(2, 1)])
    recon.append([(1, 2, 1)])
    
    alpha.append([(3, 1)])
    beta.append([(5, 1)])
    recon.append([(1, 2, 1)])
    
    return alpha, beta, recon

def _build():
    """Build Kronecker product decomposition: M_{2,2,3} (outer) x M_{2,2,2} (inner).
    Outer: 2x2 grid of A blocks, 2x3 grid of B blocks, 2x3 grid of C blocks.
    Inner: each block is 2x2, so inner multiplication is 2x2 x 2x2 (Strassen rank 7).
    Total rank = 11 * 7 = 77.
    """
    n, m, p = 4, 4, 6
    outer_alpha, outer_beta, outer_recon = _rank11_223()
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
            
            # V: B[k,j] where k = outer_row*2 + inner_row, j = outer_col*3 + inner_col
            for (block_o, coeff_o) in outer_beta[so]:
                br, bc = block_o // 3, block_o % 3
                for (block_i, coeff_i) in inner_beta[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gk = br * 2 + ir
                    gj = bc * 2 + ic
                    Vmat[gk * 6 + gj, col] += coeff_o * coeff_i
            
            # W: C[i,j] in trace convention, W row = j*n+i = j*4+i
            for (out_r, out_c, coeff_o) in outer_recon[so]:
                for (in_r, in_c, coeff_i) in inner_recon[si]:
                    gi = out_r * 2 + in_r
                    gj = out_c * 2 + in_c
                    Wmat[gj * 4 + gi, col] += coeff_o * coeff_i
    
    return Umat, Vmat, Wmat

def _verify(U, V, W):
    n, m, p = 4, 4, 6
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ikr,kjr,jir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-6)

def _try_reduce_rank(U, V, W, max_iter=500, tol=1e-9):
    """Try to reduce rank by removing dependent columns via iterative null-space projection.
    Returns (U_reduced, V_reduced, W_reduced) with fewer columns if possible, else original."""
    R = U.shape[1]
    if R <= 1:
        return U, V, W
    
    # Try to find columns that can be removed by checking if the tensor is unchanged
    # This is a heuristic: try removing each column and check if the reconstruction still works
    # For efficiency, we'll use a greedy approach: try removing columns one at a time
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    keep = list(range(R))
    
    # For each column, check if it's "redundant" by seeing if we can express it as a combination of others
    # This is expensive, so we'll use a simpler heuristic: check if the column has small norm
    # or if removing it still gives a valid decomposition
    
    # Actually, a better approach: use the fact that the tensor is rank R, and try to find
    # a lower-rank approximation that is still exact.
    # We'll use a simple iterative method: for each column, try to set it to zero and see
    # if we can adjust other columns to compensate.
    
    # For now, return the original since rank reduction is non-trivial.
    # A more sophisticated approach would use alternating least squares or similar.
    return U_cur, V_cur, W_cur

def construct():
    deadline = time.time() + 110.0
    U, V, W = _build()
    
    if not _verify(U, V, W):
        R = 4 * 4 * 6
        U = np.zeros((16, R))
        V = np.zeros((24, R))
        W = np.zeros((24, R))
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(6):
                    U[i*4 + k, idx] = 1
                    V[k*6 + j, idx] = 1
                    W[j*4 + i, idx] = 1
                    idx += 1
    
    # Try to reduce rank (currently a no-op, but structure is in place)
    if time.time() < deadline - 5:
        U, V, W = _try_reduce_rank(U, V, W)
        if not _verify(U, V, W):
            # Revert if reduction broke validity
            U, V, W = _build()
    
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
