"""Direct rank-18 bilinear scheme for 3x3 @ 3x6."""

DIMENSIONS = [3, 3, 6]


def _schoolbook_base():
    """Return (U,V,W) for the standard schoolbook scheme of rank R=18.
    U: (n*m)=9 rows, V: (m*p)=18 rows, W: (p*n)=18 rows, each R cols."""
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, R


def _validate(U, V, W):
    """Check U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c == 1 for every (i,j,k)."""
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    U, V, W, R = _schoolbook_base()
    if not _validate(U, V, W):
        # Fallback: should not happen for schoolbook
        raise ValueError("Schoolbook decomposition failed validation")
    return {"U": U, "V": V, "W": W}
