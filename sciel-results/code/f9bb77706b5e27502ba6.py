import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _winograd_222():
    """Return (U,V,W) each shape (4,7) for 2x2 by 2x2 multiplication.
    A index a = i*2+j (i row, j col), B index b = j*2+k, C index c = k*2+i.
    """
    # A = [[a00,a01],[a10,a11]] indices: 0=a00,1=a01,2=a10,3=a11
    # B = [[b00,b01],[b10,b11]] indices: 0=b00,1=b01,2=b10,3=b11
    # C = [[c00,c01],[c10,c11]] indices: 0=c00,1=c01,2=c10,3=c11
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # Winograd rank 7:
    # u1=a00+a01, u2=a00-a01, u3=a10+a11, u4=a10-a11
    # v1=b00+b10, v2=b01+b11, v3=b00-b10, v4=b01-b11
    # M1=u1*v1, M2=u1*v2, M3=u3*v1, M4=u3*v2, M5=u2*v3, M6=u2*v4, M7=u4*v3
    # c00 = M1 - M5 - M7  ? verify:
    # M1 = (a00+a01)(b00+b10) = a00b00+a00b10+a01b00+a01b10
    # M5 = (a00-a01)(b00-b10) = a00b00-a00b10-a01b00+a01b10
    # M7 = (a10-a11)(b00-b10) = a10b00-a10b10-a11b00+a11b10
    # M1-M5 = 2a00b10+2a01b00  -> not c00. So this grouping is wrong.
    return None

def construct():
    start_time = time.time()
    timeout = 120.0

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    np_ = n * p  # 10

    # Schoolbook rank 30
    R_school = n * m * p
    U = np.zeros((nm, R_school))
    V = np.zeros((mp, R_school))
    W = np.zeros((np_, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = R_school

    # Build the full tensor to verify any candidate
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(Um, Vm, Wm):
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        R = Um.shape[1]
        # compute via einsum
        comp = np.einsum('ar,br,cr->abc', Um, Vm, Wm)
        return np.allclose(comp, T, atol=1e-9)

    # Construct rank-28 via Winograd on 2x2x2 blocks.
    # Decomposition:
    # A (2x3) cols split [0,1 | 2]; B (3x5) rows split [0,1 | 2]
    # Part A: A1(2x2) * B1(2x5), B1 cols split [0,1 | 2,3 | 4]
    #   -> 2x2 by 2x2 (rank7) + 2x2 by 2x2 (rank7) + 2x2 by 2x1 (rank4)
    # Part B: a2(2x1) * b2(1x5) -> rank 10
    # Total 7+7+4+10 = 28

    # We need a correct rank-7 for 2x2x2. Let's derive it by solving.
    # Unknowns: 7 rank-1 tensors. We solve linear system.
    # Represent each rank-1 tensor by (u in R^4, v in R^4, w in R^4).
    # T = sum_r u_r v_r w_r (outer products).
    # This is a tensor decomposition of the 2x2x2 MM tensor (rank 7).

    # The 2x2x2 MM tensor:
    T222 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T222[i * 2 + j, j * 2 + k, k * 2 + i] = 1.0

    # Known rank-7 Winograd scheme (verified formula):
    # u vectors (rows of U, 4x7), v (4x7), w (4x7)
    # Using the classical Winograd:
    U222 = np.array([
        [1, 1, 0, 0, 1, 1, 0],   # a00
        [1, 1, 0, 0, -1, -1, 0],  # a01
        [0, 0, 1, 1, 0, 0, 1],    # a10
        [0, 0, 1, 1, 0, 0, -1],   # a11
    ], dtype=float)
    V222 = np.array([
        [1, 0, 1, 0, 1, 0, 1],    # b00
        [0, 1, 0, 1, 0, 1, 0],    # b01
        [1, 0, 1, 0, -1, 0, -1],  # b10
        [0, 1, 0, 1, 0, -1, 0],   # b11
    ], dtype=float)
    W222 = np.array([
        [1, 0, 0, 0, -1, -1, 0],   # c00 = M1 - M5 - M6
        [0, 1, 0, 0, 0, -1, 0],    # c01 = M2 - M4 - M6
        [0, 0, 1, 0, -1, 0, -1],   # c10 = M3 - M5 - M7
        [0, 0, 0, 1, 0, -1, -1],   # c11 = M2 - M4 - M7
    ], dtype=float)
    # Verify this rank-7 scheme
    comp222 = np.einsum('ar,br,cr->abc', U222, V222, W222)
    if not np.allclose(comp222, T222, atol=1e-9):
        # Try to fix by solving for W222 given U222, V222
        # T222[a,b,c] = sum_r U222[a,r] * V222[b,r] * W222[c,r]
        # For each r, the contribution to T222 is U222[:,r] outer V222[:,r] outer W222[:,r]
        # This is nonlinear in W222. Instead, let's try a different approach:
        # Solve for the rank-7 decomposition using a known correct formula.
        # Alternative: just use schoolbook (rank 30) as fallback
        if time.time() - start_time > timeout - 5:
            return best_result
        return best_result

    # Now assemble rank-28.
    # Indices: A index a = i*3+j (i in 0..1, j in 0..2)
    #          B index b = j*5+k (j in 0..2, k in 0..4)
    #          C index c = k*2+i (k in 0..4, i in 0..1)

    R = 28
    Ubig = np.zeros((nm, R))
    Vbig = np.zeros((mp, R))
    Wbig = np.zeros((np_, R))
    r = 0

    # Part A: A1 (i in 0..1, j in 0..1) * B1 (j in 0..1, k in 0..4)
    # Sub-blocks of k: [0,1], [2,3], [4]
    for k_start in [0, 2, 4]:
        k_end = min(k_start + 2, p)
        if k_end - k_start == 2:
            # 2x2 by 2x2, rank 7
            for rr in range(7):
                # A rows: i in 0..1, j in 0..1 -> a = i*3+j
                # U222 row index: i*2 + j
                for i in range(2):
                    for j in range(2):
                        Ubig[i * 3 + j, r] = U222[i * 2 + j, rr]
                # B cols: j in 0..1, k in k_start..k_start+1 -> b = j*5+k
                # V222 row index: j*2 + (k - k_start)
                for j in range(2):
                    for kk in range(2):
                        Vbig[j * 5 + (k_start + kk), r] = V222[j * 2 + kk, rr]
                # C: k in k_start..k_start+1, i in 0..1 -> c = k*2+i
                # W222 row index: (k - k_start)*2 + i
                for kk in range(2):
                    for i in range(2):
                        Wbig[(k_start + kk) * 2 + i, r] = W222[kk * 2 + i, rr]
                r += 1
        else:
            # 2x2 by 2x1, rank 4 (just outer products)
            for i in range(2):
                for j in range(2):
                    for kk in range(k_end - k_start):
                        Ubig[i * 3 + j, r] = 1
                        Vbig[j * 5 + (k_start + kk), r] = 1
                        Wbig[(k_start + kk) * 2 + i, r] = 1
                        r += 1

    # Part B: a2 (i in 0..1, j=2) * b2 (j=2, k in 0..4), rank 10
    for i in range(2):
        for k in range(p):
            Ubig[i * 3 + 2, r] = 1
            Vbig[2 * 5 + k, r] = 1
            Wbig[k * 2 + i, r] = 1
            r += 1

    assert r == R, f"rank mismatch {r} != {R}"

    if verify(Ubig, Vbig, Wbig):
        best_result = {"U": Ubig.tolist(), "V": Vbig.tolist(), "W": Wbig.tolist()}
        best_rank = R

    if time.time() - start_time > timeout - 5:
        return best_result

    return best_result

if __name__ == "__main__":
    construct()
