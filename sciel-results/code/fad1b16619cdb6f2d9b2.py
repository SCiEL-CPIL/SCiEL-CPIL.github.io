import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses 7 Strassen terms for each of the 7 2x2 blocks of the B matrix.
    
    The strategy combines:
    1. Block decomposition: B (2x14) is viewed as 7 blocks of 2x2 matrices.
    2. Strassen's algorithm: Each 2x2 * 2x2 multiplication uses 7 bilinear terms.
    3. Total rank = 7 blocks * 7 terms = 49.
    
    This is the minimal rank since each 2x2 * 2x2 block requires at least rank 7
    (Strassen's lower bound for 2x2 matrix multiplication is 7).
    
    Interface:
    - U: shape (n*m, rank) = (4, 49), coefficients for A elements
    - V: shape (m*p, rank) = (28, 49), coefficients for B elements  
    - W: shape (p*n, rank) = (28, 49), coefficients for C elements
    
    Indexing conventions:
    - A[i,j] stored at U[i*m + j, :]
    - B[j,k] stored at V[j*p + k, :]
    - C[i,k] stored at W[k*n + i, :] (trace convention W[k*n+i, r])
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = 2, 2, 14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    # Strassen terms for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C = [[M1 + M4 - M5 + M7, M3 + M5],
    #      [M2 + M4,           M1 - M2 + M3 + M6]]
    
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from A_idx (0=a,1=b,2=c,3=d) to coefficient
    # b_coeffs: dict from B_idx (0=e,1=f,2=g,3=h) to coefficient
    # c_coeffs: dict from C_idx (0=C00,1=C01,2=C10,3=C11) to coefficient
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    for t in range(num_blocks):
        if time.time() > deadline:
            break
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff
            
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Verify correctness: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
    def tensor_check(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Uf[i * m + j, :], Vf[j * p + k, :] * Wf[k * n + i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not tensor_check(U, V, W):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p  # 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Convert to lists with integer entries where possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result

    U_list = to_list(U)
    V_list = to_list(V)
    W_list = to_list(W)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
