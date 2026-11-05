import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 118.0

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Index convention:
      U[i*2+j, r]   (A has shape 2x2, row-major)
      V[j*2+k, r]   (B has shape 2x2)
      W[k*2+i, r]   (C has shape 2x2, trace convention)
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _kronecker_strassen_4x4x4():
    """
    Build 4x4x4 matrix multiplication tensors from the 2x2x2 Strassen
    decomposition via Kronecker (tensor) product. Rank = 7*7 = 49.

    Index conventions:
      U4[(i0,i1),(j0,j1), r=(r0,r1)] = U2[(i0,j0),r0] * U2[(i1,j1),r1]
      V4[(j0,j1),(k0,k1), r=(r0,r1)] = V2[(j0,k0),r0] * V2[(j1,k1),r1]
      W4[(k0,k1),(i0,i1), r=(r0,r1)] = W2[(k0,i0),r0] * W2[(k1,i1),r1]
    Flattened:  i = 2*i0+i1, j = 2*j0+j1, k = 2*k0+k1
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank444 = 49
    U4 = np.zeros((16, rank444), dtype=np.int16)
    V4 = np.zeros((16, rank444), dtype=np.int16)
    W4 = np.zeros((16, rank444), dtype=np.int16)

    for r0 in range(7):
        for r1 in range(7):
            r = r0 * 7 + r1
            for i0 in range(2):
                for i1 in range(2):
                    for j0 in range(2):
                        for j1 in range(2):
                            i = 2 * i0 + i1
                            j = 2 * j0 + j1
                            U4[i * 4 + j, r] = U2[i0 * 2 + j0, r0] * U2[i1 * 2 + j1, r1]
            for j0 in range(2):
                for j1 in range(2):
                    for k0 in range(2):
                        for k1 in range(2):
                            j = 2 * j0 + j1
                            k = 2 * k0 + k1
                            V4[j * 4 + k, r] = V2[j0 * 2 + k0, r0] * V2[j1 * 2 + k1, r1]
            for k0 in range(2):
                for k1 in range(2):
                    for i0 in range(2):
                        for i1 in range(2):
                            k = 2 * k0 + k1
                            i = 2 * i0 + i1
                            W4[k * 4 + i, r] = W2[k0 * 2 + i0, r0] * W2[k1 * 2 + i1, r1]

    return U4, V4, W4

def _build_rank98():
    """
    Rank-98 decomposition for 4x4 * 4x8.
    Split B into two 4x4 column blocks. Each block uses the same
    Strassen-Kronecker 4x4x4 decomposition. Total rank = 49 + 49 = 98.
    """
    U4, V4, W4 = _kronecker_strassen_4x4x4()

    n, m, p = 4, 4, 8
    total_rank = 98
    U = np.zeros((n * m, total_rank), dtype=np.int16)
    V = np.zeros((m * p, total_rank), dtype=np.int16)
    W = np.zeros((p * n, total_rank), dtype=np.int16)

    for block in range(2):
        cs = block * 49
        ce = cs + 49

        U[:, cs:ce] = U4

        for j in range(4):
            for kl in range(4):
                kg = block * 4 + kl
                V[j * 8 + kg, cs:ce] = V4[j * 4 + kl, :]

        for kl in range(4):
            kg = block * 4 + kl
            for i in range(4):
                W[kg * 4 + i, cs:ce] = W4[kl * 4 + i, :]

    # Verification
    idx = np.indices((n, m, p))
    i_f = idx[0].ravel()
    j_f = idx[1].ravel()
    k_f = idx[2].ravel()
    u_rows = i_f * m + j_f
    v_rows = j_f * p + k_f
    w_rows = k_f * n + i_f
    total = (U[u_rows, :] * V[v_rows, :] * W[w_rows, :]).sum(axis=1)

    if not np.allclose(total, 1.0, atol=1e-9):
        return None

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n=4, m=4, p=8):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
