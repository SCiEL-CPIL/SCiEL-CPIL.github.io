"""Optimized tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 8]

def _strassen_factors_2x2():
    """Return (U, V, W) for 2x2 * 2x2 Strassen decomposition, rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4: a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i,j,k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    return np.allclose(tensor, 1.0, atol=1e-10)

def _schoolbook():
    """Return schoolbook decomposition, rank n*m*p."""
    n, m, p = DIMENSIONS
    r = n*m*p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying 4x4 by 4x8 matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Build 4x4*4x4 Strassen factors by tiling 2x2 Strassen
    def _strassen_4x4():
        n2, m2, p2 = 4, 4, 4
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        U2, V2, W2 = _strassen_factors_2x2()
        
        a_blocks = {
            0: [(0,0), (1,1)],
            1: [(1,0), (1,1)],
            2: [(0,0)],
            3: [(1,1)],
            4: [(0,0), (0,1)],
            5: [(1,0), (0,0)],
            6: [(0,1), (1,1)],
        }
        a_coeffs = {
            0: {0: 1, 1: 1},
            1: {0: 1, 1: 1},
            2: {0: 1},
            3: {0: 1},
            4: {0: 1, 1: 1},
            5: {0: 1, 1: -1},
            6: {0: 1, 1: -1},
        }
        b_blocks = {
            0: [(0,0), (1,1)],
            1: [(0,0)],
            2: [(0,1), (1,1)],
            3: [(1,0), (0,0)],
            4: [(1,1)],
            5: [(0,0), (0,1)],
            6: [(1,0), (1,1)],
        }
        b_coeffs = {
            0: {0: 1, 1: 1},
            1: {0: 1},
            2: {0: 1, 1: -1},
            3: {0: 1, 1: -1},
            4: {0: 1},
            5: {0: 1, 1: 1},
            6: {0: 1, 1: 1},
        }
        c_blocks = {
            0: [(0,0), (1,1)],
            1: [(1,0)],
            2: [(0,1)],
            3: [(0,0), (1,0)],
            4: [(0,0), (0,1)],
            5: [(1,1)],
            6: [(0,0)],
        }
        c_coeffs = {
            0: {0: 1, 1: 1},
            1: {0: 1},
            2: {0: 1},
            3: {0: 1, 1: 1},
            4: {0: -1, 1: 1},
            5: {0: 1},
            6: {0: 1},
        }
        
        for x in range(7):
            for y in range(7):
                col = x * 7 + y
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            u_coeff = U2[ii*2+jj, y]
                            v_coeff = V2[jj*2+kk, y]
                            w_coeff = W2[kk*2+ii, y]
                            if u_coeff != 0 and v_coeff != 0 and w_coeff != 0:
                                for idx, (bi, bj) in enumerate(a_blocks[x]):
                                    a_coeff = a_coeffs[x][idx]
                                    gi = 2*bi + ii
                                    gj = 2*bj + jj
                                    U4[gi*4+gj, col] += a_coeff * u_coeff
                                for idx, (bj, bk) in enumerate(b_blocks[x]):
                                    b_coeff = b_coeffs[x][idx]
                                    gj = 2*bj + jj
                                    gk = 2*bk + kk
                                    V4[gj*4+gk, col] += b_coeff * v_coeff
                                for idx, (bi, bk) in enumerate(c_blocks[x]):
                                    c_coeff = c_coeffs[x][idx]
                                    gi = 2*bi + ii
                                    gk = 2*bk + kk
                                    W4[gk*4+gi, col] += c_coeff * w_coeff
        return U4, V4, W4
    
    U4, V4, W4 = _strassen_4x4()
    
    # Verify 4x4*4x4
    def _verify_4x4(U, V, W):
        n2, m2, p2 = 4, 4, 4
        rank = U.shape[1]
        tensor = np.zeros((n2, m2, p2))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        tensor[i,j,k] += uc[i*m2+j] * vc[j*p2+k] * wc[k*n2+i]
        return np.allclose(tensor, 1.0, atol=1e-10)
    
    if _verify_4x4(U4, V4, W4):
        print("4x4*4x4 Strassen verified, rank 49")
    else:
        print("4x4*4x4 Strassen verification failed, using schoolbook")
        r4 = 64
        U4 = np.zeros((16, r4))
        V4 = np.zeros((16, r4))
        W4 = np.zeros((16, r4))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i*4+j, col] = 1
                    V4[j*4+k, col] = 1
                    W4[k*4+i, col] = 1
                    col += 1
    
    r4 = U4.shape[1]
    total_rank = 2 * r4
    
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((32, total_rank))
    W_full = np.zeros((32, total_rank))
    
    # Left half (k=0..3)
    U_full[:, :r4] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*8+k, :r4] = V4[j*4+k]
    for i in range(4):
        for k in range(4):
            W_full[k*4+i, :r4] = W4[k*4+i]
    
    # Right half (k=4..7)
    U_full[:, r4:] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*8+(k+4), r4:] = V4[j*4+k]
    for i in range(4):
        for k in range(4):
            W_full[(k+4)*4+i, r4:] = W4[k*4+i]
    
    if _verify(U_full, V_full, W_full):
        print(f"4x4*4x8 decomposition verified, rank {total_rank}")
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        print("Verification failed, falling back to schoolbook")
        U, V, W = _schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
