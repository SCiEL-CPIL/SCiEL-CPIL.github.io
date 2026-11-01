"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy:
    - Use Strassen's algorithm on 2x2 blocks of the 2x14 matrix.
    - The 2x14 matrix can be viewed as 7 blocks of 2x2.
    - Each 2x2 * 2x2 multiplication uses Strassen with rank 7.
    - Total rank = 7 * 7 = 49.
    
    The multiplication is: C = A * B where A is 2x2 and B is 2x14.
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,...,13}
    
    We can write B as [B_0 B_1 ... B_6] where each B_t is 2x2.
    Then C = A * B = [A*B_0, A*B_1, ..., A*B_6]
    
    For each block t, compute A*B_t using Strassen's 7-term formula.
    Each term contributes to all 4 output positions of the 2x2 result block.
    
    Strassen's algorithm for 2x2 * 2x2:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For the full 2x2 * 2x14 multiplication, we have 7 such blocks.
    Total rank = 7 (Strassen terms) * 7 (blocks) = 49.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    # Number of 2x2 blocks in the 2x14 matrix
    num_blocks = p // 2  # 7

    # Strassen coefficients: for each of 7 terms, the coefficients for
    # A elements (indices 0-3 for A00,A01,A10,A11) and B elements
    # (indices 0-3 for B00,B01,B10,B11) and C elements (indices 0-3 for C00,C01,C10,C11)
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # Strassen terms: (a_coeffs, b_coeffs, c_coeffs)
    # where a_coeffs is dict from A_idx to coeff, etc.
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # C00 += M1, C11 += M1
        # M2 = (A10 + A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),  # C10 += M2, C11 -= M2
        # M3 = A00(B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),  # C01 += M3, C11 += M3
        # M4 = A11(B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),  # C00 += M4, C10 += M4
        # M5 = (A00 + A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),  # C00 -= M5, C01 += M5
        # M6 = (A10 - A00)(B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),  # C11 += M6
        # M7 = (A01 - A11)(B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),  # C00 += M7
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            # U: shape (n*m, rank) = (4, rank)
            # U[i*m + j, col] = coefficient of A[i,j] in term s
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m  # 0 or 1
                a_j = a_idx % m   # 0 or 1
                U[a_i * m + a_j, col] = coeff
            
            # V: shape (m*p, rank) = (28, rank)
            # V[j*p + k, col] = coefficient of B[j,k] in term s for block t
            # Block t covers columns 2*t and 2*t+1 of B
            # B[j, 2*t + b_col] where b_col is 0 or 1
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2   # 0 or 1
                b_col = b_idx % 2  # 0 or 1
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff
            
            # W: shape (p*n, rank) = (28, rank)
            # W[k*n + i, col] = coefficient of C[i,k] in term s for block t
            # Block t covers columns 2*t and 2*t+1 of C
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2   # 0 or 1
                c_col = c_idx % 2  # 0 or 1
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Verify correctness
    def tensor_check(Uf, Vf, Wf):
        """Check that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not tensor_check(U, V, W):
        # Fallback to schoolbook if Strassen construction is incorrect
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

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Convert to integer if possible
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
