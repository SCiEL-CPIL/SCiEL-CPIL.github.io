import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_222():
    """Rank-7 Strassen for 2x2 x 2x2.
    Returns (alpha, beta, recon) with block indices 0..3 for 2x2 blocks.
    """
    alpha = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    beta = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
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


def _build():
    """Build rank-63 decomposition for 4x4 x 4x6 matrix multiplication.
    
    Structure: 2x2 outer blocks of 2x2 inner blocks.
    A: 2x2 grid of 2x2 blocks (4x4 total)
    B: 2x3 grid of 2x2 blocks (4x6 total)
    C: 2x3 grid of 2x2 blocks (4x6 total)
    
    Outer Strassen (7 factors) on the 2x2 x 2x3 block structure.
    For each outer factor, inner multiplication:
      - Columns 0,1: 2x2 x 2x2 Strassen (7 factors)
      - Column 2: 2x2 x 2x1 schoolbook (2 factors)
    Total rank = 7 * (7 + 2) = 63
    """
    n, m, p = 4, 4, 6
    alpha, beta, recon = _strassen_222()
    inner_alpha, inner_beta, inner_recon = _strassen_222()
    
    R = 7 * 9  # 63
    Umat = np.zeros((16, R))
    Vmat = np.zeros((24, R))
    Wmat = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for B columns 0,1 (7 factors)
        for si in range(7):
            c = col
            # U: A[i,k]
            for (a, ci) in inner_alpha[si]:
                ir, ic = a // 2, a % 2
                for (r, co) in alpha[s]:
                    ar, ac = r // 2, r % 2
                    gi = ar * 2 + ir
                    gk = ac * 2 + ic
                    Umat[gi * 4 + gk, c] += co * ci
            # V: B[k,j] for j in {0,1}
            for (b, ci) in inner_beta[si]:
                ir, ic = b // 2, b % 2
                for (r, co) in beta[s]:
                    ar, ac = r // 2, r % 2
                    gk = ar * 2 + ir
                    gj = ac * 2 + ic  # inner col maps to global col ac*2+ic
                    Vmat[gk * 6 + gj, c] += co * ci
            # W: C[j,i] trace convention, j*n+i
            for (cr, cc, go) in recon[s]:
                for (ir, ic, gi_) in inner_recon[si]:
                    gi = cr * 2 + ir
                    gj = cc * 2 + ic
                    Wmat[gj * 4 + gi, c] += go * gi_
            col += 1
        
        # Part 2: Schoolbook for B column 2 (2 factors)
        for v in range(2):
            c = col
            # U: A[i, v] for the relevant outer factor
            for (r, co) in alpha[s]:
                ar, ac = r // 2, r % 2
                gi = ar * 2
                gk = ac * 2 + v
                Umat[gi * 4 + gk, c] += co
            # V: B[k, 2] for the relevant outer factor
            for (r, co) in beta[s]:
                ar, ac = r // 2, r % 2
                gk = ar * 2 + v
                gj = ac * 2 + 2
                Vmat[gk * 6 + gj, c] += co
            # W: C[i, 2] for the relevant outer factor
            for (cr, cc, go) in recon[s]:
                gi = cr * 2
                gj = cc * 2 + 2
                Wmat[gj * 4 + gi, c] += go
            col += 1
    
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
        # Fallback: schoolbook decomposition (rank = 96)
        R = 4 * 4 * 6
        U = np.zeros((16, R))
        V = np.zeros((24, R))
        W = np.zeros((24, R))
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(6):
                    U[i * 4 + k, idx] = 1
                    V[k * 6 + j, idx] = 1
                    W[j * 4 + i, idx] = 1
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
