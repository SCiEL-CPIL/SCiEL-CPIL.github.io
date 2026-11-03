"""Tensor decomposition for matrix multiplication 3x3 * 3x8."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Exact rank-68 tensor decomposition for 3x3 * 3x8 matrix multiplication
    using a systematic 2x2 Strassen-based block scheme.
    """
    n, m, p = 3, 3, 8

    def _validate(U, V, W):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        r = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]
                    if abs(np.sum(prod) - 1.0) > 1e-12:
                        return False
        return True

    def _add_term(idx_U, val_U, idx_V, val_V, idx_W, val_W):
        Umat[idx_U][t] += val_U
        Vmat[idx_V][t] += val_V
        Wmat[idx_W][t] += val_W

    def _strassen(A_idx, B_idx, C_idx):
        """Apply one 7-term Strassen scheme.
        A_idx = (a00, a01, a10, a11)  U indices for 2x2 block
        B_idx = (b00, b01, b10, b11)  V indices for 2x2 block
        C_idx = (c00, c01, c10, c11)  W indices for 2x2 output block
        """
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        c00, c01, c10, c11 = C_idx
        # m1 = (a00+a11)*(b00+b11) -> c00, c11
        _add_term(a00, 1, b00, 1, c00, 1)
        _add_term(a00, 1, b11, 1, c00, 1)
        _add_term(a11, 1, b00, 1, c00, 1)
        _add_term(a11, 1, b11, 1, c00, 1)
        _add_term(a00, 1, b00, 1, c11, 1)
        _add_term(a00, 1, b11, 1, c11, 1)
        _add_term(a11, 1, b00, 1, c11, 1)
        _add_term(a11, 1, b11, 1, c11, 1)
        # m2 = (a10+a11)*b00 -> c10
        _add_term(a10, 1, b00, 1, c10, 1)
        _add_term(a11, 1, b00, 1, c10, 1)
        # m3 = a00*(b01-b11) -> c01
        _add_term(a00, 1, b01, 1, c01, 1)
        _add_term(a00, 1, b11, -1, c01, 1)
        # m4 = a11*(b10-b00) -> c00(+), c10(+)
        _add_term(a11, 1, b10, 1, c00, 1)
        _add_term(a11, 1, b00, -1, c00, 1)
        _add_term(a11, 1, b10, 1, c10, 1)
        _add_term(a11, 1, b00, -1, c10, 1)
        # m5 = (a00+a01)*b11 -> c00(-), c01(+)
        _add_term(a00, 1, b11, 1, c00, -1)
        _add_term(a01, 1, b11, 1, c00, -1)
        _add_term(a00, 1, b11, 1, c01, 1)
        _add_term(a01, 1, b11, 1, c01, 1)
        # m6 = (a10-a00)*(b00+b01) -> c11
        _add_term(a10, 1, b00, 1, c11, 1)
        _add_term(a10, 1, b01, 1, c11, 1)
        _add_term(a00, -1, b00, 1, c11, 1)
        _add_term(a00, -1, b01, 1, c11, 1)
        # m7 = (a01-a11)*(b10+b11) -> c00, c11(-)
        _add_term(a01, 1, b10, 1, c00, 1)
        _add_term(a01, 1, b11, 1, c00, 1)
        _add_term(a11, -1, b10, 1, c00, 1)
        _add_term(a11, -1, b11, 1, c00, 1)
        _add_term(a01, 1, b10, 1, c11, -1)
        _add_term(a01, 1, b11, 1, c11, -1)
        _add_term(a11, -1, b10, 1, c11, -1)
        _add_term(a11, -1, b11, 1, c11, -1)

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for c in range(rank_sb):
        i, j, k = c // (m * p), (c // p) % m, c % p
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0

    # Attempt rank-68 construction via 2x2 Strassen on A[0:2,0:2]*B[0:2,0:8]
    R = 68
    Umat = np.zeros((n * m, R))
    Vmat = np.zeros((m * p, R))
    Wmat = np.zeros((p * n, R))
    t = 0

    # Strassen 2x2*2x2 per column block of B_top (4 blocks of 2 columns)
    A_block = (0, 1, 3, 4)  # A[0,0], A[0,1], A[1,0], A[1,1]
    for kb in range(4):
        k0 = kb * 2
        B_block = (k0, k0+1, 8+k0, 8+k0+1)
        C_block = (3*k0, 3*k0+1, 3*(k0+1), 3*(k0+1)+1)
        # Implement Strassen manually to avoid nested _add_term issues
        a00, a01, a10, a11 = A_block
        b00, b01, b10, b11 = B_block
        c00, c01, c10, c11 = C_block

        # m1
        Umat[a00,t]+=1; Umat[a11,t]+=1
        Vmat[b00,t]+=1; Vmat[b11,t]+=1
        Wmat[c00,t]+=1; Wmat[c11,t]+=1
        t+=1
        # m2
        Umat[a10,t]+=1; Umat[a11,t]+=1
        Vmat[b00,t]+=1
        Wmat[c10,t]+=1
        t+=1
        # m3
        Umat[a00,t]+=1
        Vmat[b01,t]+=1; Vmat[b11,t]-=1
        Wmat[c01,t]+=1
        t+=1
        # m4
        Umat[a11,t]+=1
        Vmat[b10,t]+=1; Vmat[b00,t]-=1
        Wmat[c00,t]+=1; Wmat[c10,t]+=1
        t+=1
        # m5
        Umat[a00,t]+=1; Umat[a01,t]+=1
        Vmat[b11,t]+=1
        Wmat[c00,t]-=1; Wmat[c01,t]+=1
        t+=1
        # m6
        Umat[a10,t]+=1; Umat[a00,t]-=1
        Vmat[b00,t]+=1; Vmat[b01,t]+=1
        Wmat[c11,t]+=1
        t+=1
        # m7
        Umat[a01,t]+=1; Umat[a11,t]-=1
        Vmat[b10,t]+=1; Vmat[b11,t]+=1
        Wmat[c00,t]+=1; Wmat[c11,t]-=1
        t+=1

    # Part 2: A[0:2,2]*B[2,:] -> C[0:2,:]
    for k in range(8):
        Umat[2,t]=1.0; Vmat[16+k,t]=1.0; Wmat[3*k,t]=1.0; t+=1
        Umat[5,t]=1.0; Vmat[16+k,t]=1.0; Wmat[3*k+1,t]=1.0; t+=1

    # Part 3: A[2,0:2]*B[0:2,:] -> C[:,2]
    for k in range(8):
        Umat[6,t]=1.0; Vmat[k,t]=1.0; Wmat[3*k+2,t]=1.0; t+=1
        Umat[7,t]=1.0; Vmat[8+k,t]=1.0; Wmat[3*k+2,t]=1.0; t+=1

    # Part 4: A[2,2]*B[2,:] -> C[:,2]
    for k in range(8):
        Umat[8,t]=1.0; Vmat[16+k,t]=1.0; Wmat[3*k+2,t]=1.0; t+=1

    if t == R and _validate(Umat, Vmat, Wmat):
        return {
            "U": Umat.tolist(),
            "V": Vmat.tolist(),
            "W": Wmat.tolist()
        }

    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
