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

def _schoolbook_221():
    """Return rank-4 schoolbook decomposition for 2x2 x 2x1 matrix mult.
    A is 2x2, B is 2x1, C is 2x1.
    Block indexing for A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    Block indexing for B: 0=(0,0), 1=(1,0)
    C output: (row, col) where row in {0,1}, col in {0}
    """
    alpha = [
        [(0, 1)],   # s0: a00
        [(1, 1)],   # s1: a01
        [(2, 1)],   # s2: a10
        [(3, 1)],   # s3: a11
    ]
    beta = [
        [(0, 1)],   # s0: b00
        [(1, 1)],   # s1: b10
        [(0, 1)],   # s2: b00
        [(1, 1)],   # s3: b10
    ]
    recon = [
        [(0, 0, 1)],   # s0: C[0,0]
        [(0, 0, 1)],   # s1: C[0,0]
        [(1, 0, 1)],   # s2: C[1,0]
        [(1, 0, 1)],   # s3: C[1,0]
    ]
    return alpha, beta, recon

def _build():
    """Build decomposition: M_{2,2,2} (Strassen) x M_{2,2,2} (Strassen) + M_{2,2,1} (Schoolbook) x M_{2,2,2} (Strassen).
    Total rank = (7 + 4) * 7 = 77.
    """
    n, m, p = 4, 4, 6
    outer_alpha_strassen, outer_beta_strassen, outer_recon_strassen = _strassen_222()
    outer_alpha_school, outer_beta_school, outer_recon_school = _schoolbook_221()
    inner_alpha, inner_beta, inner_recon = _strassen_222()
    
    R_outer_strassen = len(outer_alpha_strassen)
    R_outer_school = len(outer_alpha_school)
    R_inner = len(inner_alpha)
    R = (R_outer_strassen + R_outer_school) * R_inner
    
    Umat = np.zeros((16, R))
    Vmat = np.zeros((24, R))
    Wmat = np.zeros((24, R))
    
    # Strassen part
    for so in range(R_outer_strassen):
        for si in range(R_inner):
            col = so * R_inner + si
            
            # U
            for (block_o, coeff_o) in outer_alpha_strassen[so]:
                ar, ac = block_o // 2, block_o % 2
                for (block_i, coeff_i) in inner_alpha[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gi = ar * 2 + ir
                    gk = ac * 2 + ic
                    Umat[gi * 4 + gk, col] += coeff_o * coeff_i
            
            # V
            for (block_o, coeff_o) in outer_beta_strassen[so]:
                br, bc = block_o // 2, block_o % 2
                for (block_i, coeff_i) in inner_beta[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gk = br * 2 + ir
                    gj = bc * 2 + ic
                    Vmat[gk * 6 + gj, col] += coeff_o * coeff_i
            
            # W
            for (out_r, out_c, coeff_o) in outer_recon_strassen[so]:
                for (in_r, in_c, coeff_i) in inner_recon[si]:
                    gi = out_r * 2 + in_r
                    gj = out_c * 2 + in_c
                    Wmat[gj * 4 + gi, col] += coeff_o * coeff_i
    
    # Schoolbook part
    offset = R_outer_strassen * R_inner
    for so in range(R_outer_school):
        for si in range(R_inner):
            col = offset + so * R_inner + si
            
            # U
            for (block_o, coeff_o) in outer_alpha_school[so]:
                ar, ac = block_o // 2, block_o % 2
                for (block_i, coeff_i) in inner_alpha[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gi = ar * 2 + ir
                    gk = ac * 2 + ic
                    Umat[gi * 4 + gk, col] += coeff_o * coeff_i
            
            # V
            for (block_o, coeff_o) in outer_beta_school[so]:
                br, bc = block_o // 2, block_o % 2
                for (block_i, coeff_i) in inner_beta[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gk = br * 2 + ir
                    gj = bc * 2 + ic
                    # Schoolbook for 2x2 x 2x1, so bc is always 0 (last column)
                    # But we want the last column of B, which is column 2 of the 2x3 block
                    # So gj = 2 * 2 + ic = 4 + ic
                    Vmat[gk * 6 + (2 * 2 + ic), col] += coeff_o * coeff_i
            
            # W
            for (out_r, out_c, coeff_o) in outer_recon_school[so]:
                for (in_r, in_c, coeff_i) in inner_recon[si]:
                    gi = out_r * 2 + in_r
                    gj = out_c * 2 + in_c
                    # Schoolbook for 2x2 x 2x1, so out_c is always 0 (last column)
                    # But we want the last column of C, which is column 2 of the 2x3 block
                    # So gj = 2 * 2 + in_c = 4 + in_c
                    Wmat[(2 * 2 + in_c) * 4 + gi, col] += coeff_o * coeff_i
    
    return Umat, Vmat, Wmat

def _verify(U, V, W):
    n, m, p = 4, 4, 6
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ikr,kjr,jir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-6)

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
