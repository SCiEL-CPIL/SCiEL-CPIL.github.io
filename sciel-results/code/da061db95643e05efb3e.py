"""Strassen-based exact tensor decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110

    strassen_2x2 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    strassen_4x4_outer = [
        ({0:1,3:1}, {0:1,3:1}),
        ({2:1,3:1}, {0:1}),
        ({0:1}, {1:1,3:-1}),
        ({3:1}, {2:1,0:-1}),
        ({0:1,1:1}, {3:1}),
        ({2:1,0:-1}, {0:1,1:1}),
        ({1:1,3:-1}, {2:1,3:1}),
    ]
    c_blocks_4x4 = [
        {(0,0):1.0,(1,1):1.0},
        {(1,0):1.0,(1,1):-1.0},
        {(0,1):1.0,(1,1):1.0},
        {(0,0):1.0,(1,0):1.0},
        {(0,0):-1.0,(0,1):1.0},
        {(1,1):1.0},
        {(0,0):1.0},
    ]

    def build_base():
        R = 93
        U = [[0.0]*R for _ in range(16)]
        V = [[0.0]*R for _ in range(28)]
        W = [[0.0]*R for _ in range(28)]
        # 4x4x4 Strassen (cols 0..48)
        r = 0
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                for (li,lj),cx in x_sub.items():
                    for b,cb in x_def.items():
                        bi,bj = b//2, b%2
                        i = 2*bi+li; j = 2*bj+lj
                        U[i*m + j][r] += cx*cb
                for (li,lj),cy in y_sub.items():
                    for b,cb in y_def.items():
                        bi,bj = b//2, b%2
                        i = 2*bi+li; j = 2*bj+lj
                        V[i*p + j][r] += cy*cb
                for (li,lj),cc in c_sub.items():
                    if cc == 0: continue
                    for (br,bc),cb_c in c_block.items():
                        i_final = 2*br+li; k_final = 2*bc+lj
                        W[k_final*n + i_final][r] += cc*cb_c
                r += 1
        # 4x4x2 (cols 49..76)
        def add_2x2x2(Ar, Ac, Br, Bc, Cr, Cc, r0):
            for t in range(7):
                x_sub,y_sub,c_sub = strassen_2x2[t]
                rr = r0 + t
                for (li,lj),cx in x_sub.items():
                    U[Ar[li]*m + Ac[lj]][rr] += cx
                for (li,lj),cy in y_sub.items():
                    V[Br[li]*p + Bc[lj]][rr] += cy
                for (li,lj),cc in c_sub.items():
                    if cc==0: continue
                    W[Cc[lj]*n + Cr[li]][rr] += cc
        A00r=[0,1];A00c=[0,1]; A01r=[0,1];A01c=[2,3]; A10r=[2,3];A10c=[0,1]; A11r=[2,3];A11c=[2,3]
        Btr=[0,1];Btc=[4,5]; Bbr=[2,3];Bbc=[4,5]
        Ctr=[0,1];Ctc=[4,5]; Cbr=[2,3];Cbc=[4,5]
        add_2x2x2(A00r,A00c,Btr,Btc,Ctr,Ctc,49)
        add_2x2x2(A01r,A01c,Bbr,Bbc,Ctr,Ctc,56)
        add_2x2x2(A10r,A10c,Btr,Btc,Cbr,Cbc,63)
        add_2x2x2(A11r,A11c,Bbr,Bbc,Cbr,Cbc,70)
        # 4x4x1 (cols 77..92)
        for i in range(4):
            for j in range(4):
                rr = 77 + i*4 + j
                U[i*m+j][rr] = 1.0
                V[j*p+6][rr] = 1.0
                W[6*n+i][rr] = 1.0
        return U, V, W, R

    def verify_exact(Ul, Vl, Wl, rank):
        Uf = [[Fraction(x) for x in row] for row in Ul]
        Vf = [[Fraction(x) for x in row] for row in Vl]
        Wf = [[Fraction(x) for x in row] for row in Wl]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                    if s != 1:
                        return False
        return True

    U, V, W, R = build_base()
    if not verify_exact(U, V, W, R):
        R0 = n*m*p
        U0=[[0]*R0 for _ in range(n*m)]; V0=[[0]*R0 for _ in range(m*p)]; W0=[[0]*R0 for _ in range(p*n)]
        c=0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j][c]=1; V0[j*p+k][c]=1; W0[k*n+i][c]=1; c+=1
        return {"U":U0,"V":V0,"W":W0}

    # Reduction: find linearly dependent term columns
    def try_reduce(Ul, Vl, Wl, rank):
        if time.time() > deadline - 30:
            return None
        Ua = np.array(Ul, dtype=float)
        Va = np.array(Vl, dtype=float)
        Wa = np.array(Wl, dtype=float)
        N = n*m*p
        T = np.zeros((N, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[idx] = Ua[i*m+j,:] * Va[j*p+k,:] * Wa[k*n+i,:]
                    idx += 1
        used = []
        for r in range(rank):
            if time.time() > deadline - 25:
                break
            if not used:
                used.append(r); continue
            A = T[:, used]
            try:
                coeff, *_ = np.linalg.lstsq(A, T[:, r], rcond=None)
            except Exception:
                used.append(r); continue
            resid = T[:, r] - A @ coeff
            if np.linalg.norm(resid) < 1e-9 * (1.0 + np.linalg.norm(T[:, r])):
                continue
            used.append(r)
        if len(used) < rank:
            return Ua[:, used].tolist(), Va[:, used].tolist(), Wa[:, used].tolist(), len(used)
        return None

    red = try_reduce(U, V, W, R)
    if red is not None:
        U2, V2, W2, R2 = red
        if verify_exact(U2, V2, W2, R2):
            U, V, W, R = U2, V2, W2, R2

    return {"U": U, "V": V, "W": W}
