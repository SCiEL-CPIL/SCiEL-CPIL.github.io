import numpy as np

def construct():
    n, m, p = 2, 2, 10

    # Strassen's 7-term decomposition for 2x2 x 2x2 -> 2x2.
    # Each term is (u, v, w) where:
    #   u is a linear form in the 4 entries of A (index i*m+k, i,k in 0..1)
    #   v is a linear form in the 4 entries of B
    #   w is a linear form in the 4 entries of C
    # Strassen's 7 products:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    # C = [[M1+M4-M5-M7,  M3+M5],
    #      [M2+M4,         M1-M2+M3+M7]]

    # Vector indices for a 2x2 matrix in row-major: [x00, x01, x10, x11]
    # Term list: (u_vec, v_vec, w_vec)
    strassen = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, -1]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, -1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]),
    ]

    # Build the target tensor to verify
    # T[a, b, c] = 1 iff a = i*m+k, b = k*p+j, c = j*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    rank = 7 * (p // 2)  # 5 blocks of 2 columns, each using 7 Strassen terms
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for block in range(p // 2):
        for u_vec, v_vec, w_vec in strassen:
            # A part: same 2x2 matrix for every block
            for idx in range(4):
                U[idx, col] = u_vec[idx]

            # B part: block uses columns 2*block and 2*block+1
            # B index for B[0, j] = 0*p + j, B[1, j] = 1*p + j
            # v_vec = [v00, v01, v10, v11] corresponds to
            #   B[0, 2*block], B[0, 2*block+1], B[1, 2*block], B[1, 2*block+1]
            b_cols = [2 * block, 2 * block + 1]
            V[0 * p + b_cols[0], col] = v_vec[0]
            V[0 * p + b_cols[1], col] = v_vec[1]
            V[1 * p + b_cols[0], col] = v_vec[2]
            V[1 * p + b_cols[1], col] = v_vec[3]

            # C part: W index for C[i, j] = j * n + i  (trace convention)
            # w_vec = [w00, w01, w10, w11] corresponds to
            #   C[0, 2*block], C[0, 2*block+1], C[1, 2*block], C[1, 2*block+1]
            W[b_cols[0] * n + 0, col] = w_vec[0]
            W[b_cols[1] * n + 0, col] = w_vec[1]
            W[b_cols[0] * n + 1, col] = w_vec[2]
            W[b_cols[1] * n + 1, col] = w_vec[3]

            col += 1

    # Verify
    Result = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(Result, T, atol=1e-9):
        # Fallback: trivial rank-40
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, c] = 1.0
                    V[k * p + j, c] = 1.0
                    W[j * n + i, c] = 1.0
                    c += 1

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {res['U'].shape[1]}")
