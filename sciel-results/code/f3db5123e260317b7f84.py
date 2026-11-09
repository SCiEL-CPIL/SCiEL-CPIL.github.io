import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _try_rank24_fft():
    """Try to construct a rank-24 decomposition via 4-point FFT on contraction index k.
    If it exists, return (U, V, W). Otherwise return None.
    """
    n, m, p = 4, 4, 6
    # 4-point DFT matrix
    omega = np.exp(2j * np.pi / 4)
    F = np.array([[omega**(k*l) for l in range(4)] for k in range(4)], dtype=complex)
    F_inv = np.conj(F.T) / 4.0
    
    # For each DFT frequency k in {0,1,2,3}, we need a 4x6 matrix mult
    # A_k is 4x4, B_k is 4x6, C_k = A_k @ B_k is 4x6
    # We need rank decomposition of each 4x4 x 4x6 multiplication
    
    # The known result: 4x4 x 4x4 has rank 49 (Strassen).
    # For 4x4 x 4x6, the rank is unknown but likely > 6 per frequency.
    # 4 frequencies * 6 = 24 only works if each frequency has rank 6,
    # which is impossible since 4x4x4x4 already needs rank 49.
    # So rank-24 via FFT is NOT achievable.
    return None

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

def _rank11_223():
    """Return rank-11 decomposition for 2x2 x 2x3 matrix mult.
    A is 2x2, B is 2x3, C is 2x3.
    Block indexing for A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    Block indexing for B: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    C output: (row, col) where row in {0,1}, col in {0,1,2}
    """
    # First 7 terms: Strassen for the 2x2 x 2x2 part (cols 0,1 of B and C)
    # A blocks: 0,1,2,3
    # B blocks used: 0,1,3,4 (cols 0,1)
    # C blocks used: 0,1,3,4 (cols 0,1)
    
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
        [(1, 1), (4, -1)],      # s2: B01-B11 (B01 is block 1, B11 is block 4)
        [(3, 1), (0, -1)],      # s3: B10-B00 (B10 is block 3, B00 is block 0)
        [(4, 1)],               # s4: B11
        [(0, 1), (1, 1)],       # s5: B00+B01
        [(3, 1), (4, 1)],       # s6: B10+B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],   # C00, C11
        [(1, 0, 1), (1, 1, -1)],  # C10, C11
        [(0, 1, 1), (1, 1, 1)],   # C01, C11
        [(0, 0, 1), (1, 0, 1)],   # C00, C10
        [(0, 0, -1), (0, 1, 1)],  # C00, C01
        [(1, 1, 1)],              # C11
        [(0, 0, 1)],              # C00
    ]
    
    # Next 4 terms: for the 3rd column of C
    # C02 = a00*b02 + a01*b12
    # C12 = a10*b02 + a11*b12
    
    # s7: a00 * b02 -> C02
    alpha.append([(0, 1)])
    beta.append([(2, 1)])
    recon.append([(0, 2, 1)])
    
    # s8: a01 * b12 -> C02
    alpha.append([(1, 1)])
    beta.append([(5, 1)])
    recon.append([(0, 2, 1)])
    
    # s9: a10 * b02 -> C12
    alpha.append([(2, 1)])
    beta.append([(2, 1)])
    recon.append([(1, 2, 1)])
    
    # s10: a11 * b12 -> C12
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
    
    # First try rank-24 FFT approach
    result = _try_rank24_fft()
    if result is not None:
        return result
    
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
            
            for (block_o, coeff_o) in outer_alpha[so]:
                ar, ac = block_o // 2, block_o % 2
                for (block_i, coeff_i) in inner_alpha[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gi = ar * 2 + ir
                    gk = ac * 2 + ic
                    Umat[gi * 4 + gk, col] += coeff_o * coeff_i
            
            for (block_o, coeff_o) in outer_beta[so]:
                br, bc = block_o // 3, block_o % 3
                for (block_i, coeff_i) in inner_beta[si]:
                    ir, ic = block_i // 2, block_i % 2
                    gk = br * 2 + ir
                    gj = bc * 2 + ic
                    Vmat[gk * 6 + gj, col] += coeff_o * coeff_i
            
            for (out_r, out_c, coeff_o) in outer_recon[so]:
                for (in_r, in_c, coeff_i) in inner_recon[si]:
                    gi = out_r * 2 + in_r
                    gj = out_c * 2 + in_c
                    Wmat[gj * 4 + gi, col] += coeff_o * coeff_i
    
    return Umat, Vmat, Wmat

def _verify(U, V, W):
    n, m, p = 4, 4, 6
    R = U.shape[1]
    # Use numpy for speed
    # T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    # T[i,k,j] = sum_r U3[i,k,r] * V3[k,j,r] * W3[j,i,r]
    # = einsum('ikr,kjr,jir->ijk', U3, V3, W3)
    T = np.einsum('ikr,kjr,jir->ijk', U3, V3, W3)
    # Should be all 1s
    return np.allclose(T, 1.0, atol=1e-6)

def construct():
    deadline = time.time() + 110.0
    U, V, W = _build()
    if not _verify(U, V, W):
        # Fallback: schoolbook decomposition (rank = n*m*p = 96)
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
