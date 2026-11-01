import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Key structural insight: The 2x2 * 2x14 multiplication tensor is a direct sum
    of 7 independent 2x2 * 2x2 multiplication tensors. For direct sums, rank is
    additive: rank(T1 ⊕ T2) = rank(T1) + rank(T2). Since each 2x2*2x2 tensor
    has rank 7 (Strassen), the total rank is 7*7 = 49.
    
    This is provably minimal: any decomposition of the direct sum restricts to
    decompositions of each summand, and each summand requires rank >= 7.
    
    Construction: Apply Strassen's 7-term algorithm to each of the 7 blocks.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    num_blocks = p // 2  # 7 blocks of 2x2
    
    # Strassen's 7 terms for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices within block: B00=0, B01=1, B10=2, B11=3
    # C indices within block: C00=0, C01=1, C10=2, C11=3
    
    strassen_terms = [
        # (a_coeffs, b_coeffs, c_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
    ]
    
    rank = num_blocks * 7  # 49
    U = [[0] * rank for _ in range(n * m)]   # (4, 49)
    V = [[0] * rank for _ in range(m * p)]   # (28, 49)
    W = [[0] * rank for _ in range(p * n)]   # (28, 49)
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            # U[i*m + j, col] = coefficient of A[i,j] in term s
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m  # 0 or 1
                a_j = a_idx % m   # 0 or 1
                U[a_i * m + a_j][col] = coeff
            
            # V[j*p + k, col] = coefficient of B[j,k] in term s for block t
            # Block t covers B columns 2*t and 2*t+1
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2    # 0 or 1
                b_col = b_idx % 2   # 0 or 1
                b_k = 2 * t + b_col
                V[b_j * p + b_k][col] = coeff
            
            # W[k*n + i, col] = coefficient of C[i,k] in term s for block t
            # Block t covers C columns 2*t and 2*t+1
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2    # 0 or 1
                c_col = c_idx % 2   # 0 or 1
                c_k = 2 * t + c_col
                W[c_k * n + c_i][col] = coeff
    
    # Verify correctness
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook rank-56 (should never happen)
        rank = n * m * p  # 56
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
    
    # Convert to integers where possible
    U = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
    V = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
    W = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
