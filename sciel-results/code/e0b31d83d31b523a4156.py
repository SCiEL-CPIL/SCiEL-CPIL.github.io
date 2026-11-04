import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 130.0

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition of 2x2 * 2x2.
    U,V,W shapes (4,7). Convention U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # M2 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    # M3 = A00(B10-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    # M4 = (A10-A00)B11
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    # M5 = (A00+A01)B10
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    # M6 = (A01-A10)(B00+B10)
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # M7 = (A11-A01)(B01+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _build_base():
    """Build rank-98 base via 2 halves x 7x7 Strassen block structure."""
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()

    # Specs for the 7 Strassen products P_k
    # A_specs[kp]: list of (coeff, block_idx) for A_eff
    # B_specs[kp]: list of (coeff, block_idx) for B_eff
    # C_specs[kp]: list of (coeff, block_idx) for C_eff
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    A_specs = [
        [(1, 0), (1, 3)], # P0: A00 + A11
        [(1, 2), (1, 3)], # P1: A10 + A11
        [(1, 0)],         # P2: A00
        [(1, 3)],         # P3: A11
        [(1, 0), (1, 1)], # P4: A00 + A01
        [(1, 2), (-1, 0)],# P5: A10 - A00
        [(1, 1), (-1, 3)],# P6: A01 - A11
    ]
    B_specs = [
        [(1, 0), (1, 3)], # P0: B00 + B11
        [(1, 0)],         # P1: B00
        [(1, 1), (-1, 3)],# P2: B01 - B11
        [(1, 2), (-1, 0)],# P3: B10 - B00
        [(1, 3)],         # P4: B11
        [(1, 0), (1, 1)], # P5: B00 + B01
        [(1, 2), (1, 3)], # P6: B10 + B11
    ]
    C_specs = [
        [(1, 0), (1, 3)], # P0: C00 + C11
        [(1, 2), (-1, 3)],# P1: C10 - C11
        [(1, 1), (1, 3)], # P2: C01 + C11
        [(1, 0), (1, 2)], # P3: C00 + C10
        [(-1, 0), (1, 1)],# P4: -C00 + C01
        [(1, 3)],         # P5: C11
        [(1, 0)],         # P6: C00
    ]

    for half in range(2):
        # half 0: B cols 0-3, C cols 0-3
        # half 1: B cols 4-7, C cols 4-7
        b_offset = half * 4
        c_offset = half * 4
        
        for kp in range(7):
            for ms in range(7):
                col = half * 49 + kp * 7 + ms
                
                # U (Input A)
                # A is 4x4, no offset.
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2 + r
                            gj = bcj*2 + c
                            U[gi*4 + gj, col] += ca * U2[r*2 + c, ms]
                            
                # V (Input B)
                # B is 4x8. Offset by b_offset for the half.
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2 + r
                            gj = bcj*2 + c + b_offset
                            V[gi*8 + gj, col] += cb * V2[r*2 + c, ms]
                            
                # W (Output C)
                # C is 8x4 (p*n). Index W[k*n + i, r].
                # k is column of C (0..7), i is row of C (0..3).
                # C_specs refers to C blocks in the 4x4 sub-problem.
                # C_block (bi, bcj) corresponds to rows bi*2+r, cols bcj*2+c.
                # Global col k = bcj*2 + c + c_offset.
                # Global row i = bi*2 + r.
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2 + r # Row i
                            gj = bcj*2 + c + c_offset # Col k
                            W[gj*4 + gi, col] += cc * W2[r*2 + c, ms]
                            
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    # Vectorized check
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 8
    U, V, W = _build_base()
    
    if not _verify(U, V, W, n, m, p):
        # Fallback: schoolbook (guaranteed valid)
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
