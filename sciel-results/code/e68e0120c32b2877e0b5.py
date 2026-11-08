"""Matrix multiplication 3x4x8 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen 2x2x2 factors (M1..M7)
    # sU[t][li*2+lj], sV[t][lj*2+lk], sW[t][li*2+lk]
    sU = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    sV = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    sW = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 1],   # M3: C01+=1, C11+=1
        [1, 0, 1, 0],   # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 0, 1],   # M6: C11+=1
        [1, 0, 0, 0],   # M7: C00+=1
    ]

    # Part 1: i=0, schoolbook 1x4x8 -> 32 rank-1 terms (minimal for single row)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 -> 2x4x8 subproblem.
    # Split k into 4 blocks of 2; each 2x4x2 = two 2x2x2 Strassen (j blocks [0,1],[2,3]).
    for jb in range(2):          # j block index (j0 = 2*jb)
        j0 = jb * 2
        for kb in range(4):      # k block index (k0 = 2*kb)
            k0 = kb * 2
            for t in range(7):   # Strassen term
                # U: local (li, lj) -> global (i=1+li, j=j0+lj)
                for li in range(2):
                    for lj in range(2):
                        val = sU[t][li * 2 + lj]
                        if val != 0:
                            U[(1 + li) * m + (j0 + lj)][col] += val
                # V: local (lj, lk) -> global (j=j0+lj, k=k0+lk)
                for lj in range(2):
                    for lk in range(2):
                        val = sV[t][lj * 2 + lk]
                        if val != 0:
                            V[(j0 + lj) * p + (k0 + lk)][col] += val
                # W: local (li, lk) -> global (k=k0+lk, i=1+li)
                for li in range(2):
                    for lk in range(2):
                        val = sW[t][li * 2 + lk]
                        if val != 0:
                            W[(k0 + lk) * n + (1 + li)][col] += val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import numpy as np
    f = construct()
    U = np.array(f["U"], dtype=float)
    V = np.array(f["V"], dtype=float)
    W = np.array(f["W"], dtype=float)
    n, m, p = 3, 4, 8
    rank = U.shape[1]
    # Build tensor T[i*m+j, j*p+k, k*n+i] and check U@V^T * W contraction
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T = np.einsum("ar,br,cr->abc", U, V, W)
    # True multiplication tensor
    Ttrue = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Ttrue[i * m + j, j * p + k, k * n + i] = 1.0
    err = np.max(np.abs(T - Ttrue))
    print("rank =", rank, "max_err =", err)
    assert err < 1e-9, "decomposition incorrect"
    print("OK")
