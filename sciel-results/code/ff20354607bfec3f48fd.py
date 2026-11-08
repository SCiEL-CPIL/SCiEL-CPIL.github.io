"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces C = A*B.
    C[i,k] = sum_{j=0}^{m-1} A[i,j]*B[j,k].
    Tensor entry T[i*m+j, j*p+k, k*n+i] must equal 1 for all i,j,k, 0 otherwise.
    We check by reconstructing the tensor T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    and verifying all required entries equal 1 and no spurious non-zero elsewhere.
    """
    # Build the full tensor as a 3D array
    # T shape: (n*m, m*p, p*n)
    T = [[0]* (p*n) for _ in range(m*p)]
    # Actually we need 3D; use nested lists: T[a][b][c]
    T = [[[0]*(p*n) for _ in range(m*p)] for _ in range(n*m)]
    for r in range(rank):
        # collect non-zero entries in each factor for this r
        u_nz = [(i, U[i][r]) for i in range(n*m) if U[i][r] != 0]
        v_nz = [(j, V[j][r]) for j in range(m*p) if V[j][r] != 0]
        w_nz = [(k, W[k][r]) for k in range(p*n) if W[k][r] != 0]
        for (a, ua) in u_nz:
            for (b, vb) in v_nz:
                for (c, wc) in w_nz:
                    T[a][b][c] += ua * vb * wc
    # Check required entries: for each i in [0,n), j in [0,m), k in [0,p):
    #   a = i*m + j, b = j*p + k, c = k*n + i  -> T[a][b][c] must be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if T[a][b][c] != 1:
                    return False
    # Check no spurious non-zeros: for any (a,b,c) not of the form above, T must be 0
    # Build set of required (a,b,c)
    required = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                required.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            row = T[a][b]
            for c in range(p*n):
                if row[c] != 0 and (a,b,c) not in required:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank55(n, m, p):
    """Block-based rank-55 decomposition for 2x3 x 3x10.
    Groups p=10 columns into 5 pairs; per pair uses 7 shared + 4 direct = 11 terms.
    """
    rank = 55
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    col = 0
    for t in range(5):
        k0 = 2*t
        k1 = 2*t + 1

        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0*3+0][col] += 1
        U[1*3+1][col] += 1
        V[0*10+k0][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # M2: (A10 + A11) * B0,k0
        U[1*3+0][col] += 1
        U[1*3+1][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1
        W[k1*2+1][col] -= 1
        col += 1

        # M3: A00 * (B0,k1 - B1,k1)
        U[0*3+0][col] += 1
        V[0*10+k1][col] += 1
        V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # M4: A11 * (B1,k0 - B0,k0)
        U[1*3+1][col] += 1
        V[1*10+k0][col] += 1
        V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1
        W[k0*2+1][col] += 1
        col += 1

        # M5: (A00 + A01) * B1,k1
        U[0*3+0][col] += 1
        U[0*3+1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1
        W[k1*2+0][col] += 1
        col += 1

        # M6: (A10 - A00) * (B0,k0 + B0,k1) -> contributes to C[0,k0]
        U[1*3+0][col] += 1
        U[0*3+0][col] -= 1
        V[0*10+k0][col] += 1
        V[0*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1

        # M7: (A01 - A11) * (B1,k0 + B1,k1) -> contributes to C[1,k1]
        U[0*3+1][col] += 1
        U[1*3+1][col] -= 1
        V[1*10+k0][col] += 1
        V[1*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # Direct terms for row 2 (A02, A12) with B2
        U[0*3+2][col] += 1
        V[2*10+k0][col] += 1
        W[k0*2+0][col] += 1
        col += 1

        U[0*3+2][col] += 1
        V[2*10+k1][col] += 1
        W[k1*2+0][col] += 1
        col += 1

        U[1*3+2][col] += 1
        V[2*10+k0][col] += 1
        W[k0*2+1][col] += 1
        col += 1

        U[1*3+2][col] += 1
        V[2*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120

    # Try rank-55 decomposition with verification
    try:
        res55 = _rank55(n, m, p)
        if time.time() < deadline - 5:
            if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                return res55
    except Exception:
        pass

    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
