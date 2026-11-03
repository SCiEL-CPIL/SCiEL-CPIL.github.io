"""Matrix multiplication 6x6x6 rank-minimal decomposition via Strassen block recursion."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_decomposition(n, m, p):
    """
    Recursively build a tensor decomposition for multiplying an n x m matrix by an m x p matrix.
    Returns (U, V, W) as lists of lists.
    """
    # Base case: schoolbook
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if n == 1 or m == 1 or p == 1:
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
        return U, V, W
    
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        n2, m2, p2 = n // 2, m // 2, p // 2
        U_sub, V_sub, W_sub = _build_decomposition(n2, m2, p2)
        r = len(U_sub[0])
        rank_total = 7 * r
        
        U_np = np.zeros((n * m, rank_total), dtype=float)
        V_np = np.zeros((m * p, rank_total), dtype=float)
        W_np = np.zeros((p * n, rank_total), dtype=float)
        
        U_sub_np = np.array(U_sub, dtype=float)
        V_sub_np = np.array(V_sub, dtype=float)
        W_sub_np = np.array(W_sub, dtype=float)
        
        A_offsets = {'A11': (0, 0), 'A12': (0, m2), 'A21': (n2, 0), 'A22': (n2, m2)}
        B_offsets = {'B11': (0, 0), 'B12': (0, p2), 'B21': (m2, 0), 'B22': (m2, p2)}
        C_offsets = {'C11': (0, 0), 'C12': (0, p2), 'C21': (n2, 0), 'C22': (n2, p2)}
        
        terms = [
            ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
            ({'A21': 1, 'A22': 1}, {'B11': 1}, {'C21': 1, 'C22': -1}),
            ({'A11': 1}, {'B21': 1, 'B22': -1}, {'C12': 1, 'C22': 1}),
            ({'A22': 1}, {'B12': 1, 'B11': -1}, {'C11': 1, 'C21': 1}),
            ({'A11': 1, 'A12': 1}, {'B22': 1}, {'C11': -1, 'C12': 1}),
            ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}, {'C22': 1}),
            ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}, {'C11': 1}),
        ]
        
        for t, (a_coeffs, b_coeffs, c_contrib) in enumerate(terms):
            base_col = t * r
            for c in range(r):
                for block_name, coeff in a_coeffs.items():
                    ro, co = A_offsets[block_name]
                    for i in range(n2):
                        for j in range(m2):
                            U_np[(i + ro) * m + (j + co), base_col + c] += coeff * U_sub_np[i * m2 + j, c]
                for block_name, coeff in b_coeffs.items():
                    ro, co = B_offsets[block_name]
                    for j in range(m2):
                        for k in range(p2):
                            V_np[(j + ro) * p + (k + co), base_col + c] += coeff * V_sub_np[j * p2 + k, c]
                for block_name, coeff in c_contrib.items():
                    ro, co = C_offsets[block_name]
                    for i in range(n2):
                        for k in range(p2):
                            W_np[(k + co) * n + (i + ro), base_col + c] += coeff * W_sub_np[k * n2 + i, c]
        
        U = U_np.tolist()
        V = V_np.tolist()
        W = W_np.tolist()
        for mat in [U, V, W]:
            for row in mat:
                for idx in range(len(row)):
                    if abs(row[idx] - round(row[idx])) < 1e-9:
                        row[idx] = int(round(row[idx]))
        return U, V, W
    
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
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    U, V, W = _build_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
