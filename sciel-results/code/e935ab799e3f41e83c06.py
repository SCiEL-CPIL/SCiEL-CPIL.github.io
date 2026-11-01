import time
import random

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = 35
    nU, nV, nW = n * m, m * p, p * n

    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]

    # Alternative rank-7 scheme for 2x2 matrix multiplication (different from standard Strassen)
    # This uses a symmetric variant with different linear combinations
    # A entries: [a00, a01, a10, a11]  B entries: [b00, b01, b10, b11]
    # C entries: [c00, c10, c01, c11] (row-major within 2x2 block: C[i][k*n+j])
    terms = [
        # Term 0: (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # Term 1: (A10+A01)(B10-B00)
        ([0, 1, 1, 0], [1, -1, 0, 0], [-1, 0, 1, 0]),
        # Term 2: (A00-A11)(B11-B10)
        ([1, 0, 0, -1], [0, 0, -1, 1], [0, 1, -1, 1]),
        # Term 3: (A00-A11)(B00+B10)
        ([1, 0, 0, -1], [1, 0, 1, 0], [1, 0, 0, -1]),
        # Term 4: (A01+A10)(B00-B01)
        ([0, 1, 1, 0], [1, -1, 0, 0], [1, 1, 0, 0]),
        # Term 5: (A11+A00)(B01-B10) -- wait, let me re-derive
        # Let me use a verified rank-7 scheme
        # Actually let me use the standard one but with different ordering/signs
    ]

    # Let me use a verified alternative: rank-7 with different basis
    # Using: M1=(a0+a3)(b0+b3), M2=(a2+a3)(b0), M3=(a0)(b1-b3),
    #        M4=(a3)(b2+b1), M5=(a0+a1)(b3), M6=(a2-a0)(b1+b2), M7=(a1-a2)(b0+b2)
    # C00 = M1+M4-M5-M7, C01 = M3+M5, C10 = M2+M6, C11 = M1-M2+M3+M7

    terms = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)(B00)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3 = A00*(B01-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4 = A11*(B10+B01)
        ([0, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]),
        # M5 = (A00+A01)*(B11)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6 = (A10-A00)*(B01+B10)
        ([-1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]),
        # M7 = (A01-A10)*(B00+B10)
        ([0, 1, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]

    # Verify the decomposition is correct
    # C = sum of W_k * (U_k dot A) * (V_k dot B)
    # Let's check: C[i,j] = sum_k U[i,k] * W[j,k] * V[?,k]
    # Actually the tensor: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r] where
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i*m + j, ?] -- need to be careful about indexing

    # Let me verify numerically
    import numpy as np
    U_arr = np.zeros((nU, rank), dtype=float)
    V_arr = np.zeros((nV, rank), dtype=float)
    W_arr = np.zeros((nW, rank), dtype=float)

    col = 0
    for block in range(p // 2):
        for a_vec, b_vec, c_vec in terms:
            # Fill U for this term (same for all blocks)
            for idx in range(nU):
                U_arr[idx, col] = a_vec[idx]

            # Fill V: B is 2x10, columns 2*block and 2*block+1
            # V[row, col] corresponds to B[row, col_in_block]
            # B index: [j * p + k] for row j, column k
            V_arr[0 * p + 2 * block, col] = b_vec[0]
            V_arr[0 * p + 2 * block + 1, col] = b_vec[1]
            V_arr[1 * p + 2 * block, col] = b_vec[2]
            V_arr[1 * p + 2 * block + 1, col] = b_vec[3]

            # Fill W: C is 2x10, convention W[k*n + i] where k=col, i=row
            # C[0, 2*block]   -> W[2*block * n + 0]
            # C[1, 2*block]   -> W[2*block * n + 1]
            # C[0, 2*block+1] -> W[(2*block+1) * n + 0]
            # C[1, 2*block+1] -> W[(2*block+1) * n + 1]
            W_arr[2 * block * n + 0, col] = c_vec[0]
            W_arr[2 * block * n + 1, col] = c_vec[1]
            W_arr[(2 * block + 1) * n + 0, col] = c_vec[2]
            W_arr[(2 * block + 1) * n + 1, col] = c_vec[3]

            col += 1

    # Verify: the multiplication tensor T[i,j,k] where C = A @ B
    # T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For random test matrices:
    max_err = 0.0
    num_tests = 200
    rng = random.Random(42)
    for t in range(num_tests):
        A_test = np.array([[rng.uniform(-1,1) for _ in range(m)] for _ in range(n)], dtype=float)
        B_test = np.array([[rng.uniform(-1,1) for _ in range(p)] for _ in range(m)], dtype=float)
        C_true = A_test @ B_test  # shape (n, p)

        # Compute via decomposition
        C_decomp = np.zeros((n, p), dtype=float)
        for r in range(rank):
            a_proj = np.dot(A_test.flatten(), U_arr[:, r])  # U has shape (n*m, rank)
            b_proj = np.dot(B_test.flatten(), V_arr[:, r])  # V has shape (m*p, rank)
            c_coeff = W_arr[:, r]  # shape (p*n,)
            # c_coeff indexed as [k*n + i] for C[i,k]
            for kk in range(p):
                for ii in range(n):
                    C_decomp[ii, kk] += a_proj * b_proj * c_coeff[kk * n + ii]

        err = np.max(np.abs(C_true - C_decomp))
        if err > max_err:
            max_err = err

    if max_err > 1e-10:
        # Fallback to known-correct Strassen terms
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        col = 0
        for block in range(p // 2):
            for a_vec, b_vec, c_vec in terms:
                for idx in range(nU):
                    U_arr[idx, col] = a_vec[idx]
                V_arr[0 * p + 2 * block, col] = b_vec[0]
                V_arr[0 * p + 2 * block + 1, col] = b_vec[1]
                V_arr[1 * p + 2 * block, col] = b_vec[2]
                V_arr[1 * p + 2 * block + 1, col] = b_vec[3]
                W_arr[2 * block * n + 0, col] = c_vec[0]
                W_arr[2 * block * n + 1, col] = c_vec[1]
                W_arr[(2 * block + 1) * n + 0, col] = c_vec[2]
                W_arr[(2 * block + 1) * n + 1, col] = c_vec[3]
                col += 1

    # Convert back to list of lists
    U = U_arr.tolist()
    V = V_arr.tolist()
    W = W_arr.tolist()

    return {"U": U, "V": V, "W": W}
