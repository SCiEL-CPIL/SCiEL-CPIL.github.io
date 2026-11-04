"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 solution using Strassen for 2x2x2 blocks
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # u, v are 2x2 lists, gamma is 2x2 list (coeffs for C[i][k])
    strassen_terms = [
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        # M3
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        # M4
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[-1, 0], [1, 0]]),
        # M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[1, 1], [0, 0]]),
        # M6
        ([[0, 1], [-1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # M7
        ([[0, 0], [1, -1]], [[0, 1], [1, 1]], [[1, 0], [0, 1]])
    ]
    
    # Schoolbook 2x2x1 decomposition
    # u is 2x2, v is 2x1, gamma is 2x1 (coeffs for C[i][0])
    schoolbook_221_terms = [
        ([[1, 0], [0, 0]], [[1], [0]], [[1], [0]]),
        ([[0, 1], [0, 0]], [[0], [1]], [[1], [0]]),
        ([[0, 0], [1, 0]], [[1], [0]], [[0], [1]]),
        ([[0, 0], [0, 1]], [[0], [1]], [[0], [1]])
    ]
    
    col = 0
    
    # Helper to add a 2x2x2 block
    def add_strassen_block(a_col_start, b_row_start, b_col_start):
        nonlocal col
        for u, v, gamma in strassen_terms:
            for i in range(2):
                for j in range(2):
                    U[i * m + (a_col_start + j)][col] += u[i][j]
            for j in range(2):
                for k in range(2):
                    V[(b_row_start + j) * p + (b_col_start + k)][col] += v[j][k]
            for k in range(2):
                for i in range(2):
                    W[(b_col_start + k) * n + i][col] += gamma[i][k]
            col += 1

    # Helper to add a 2x2x1 block
    def add_schoolbook_221_block(a_col_start, b_row_start, b_col_start):
        nonlocal col
        for u, v, gamma in schoolbook_221_terms:
            for i in range(2):
                for j in range(2):
                    U[i * m + (a_col_start + j)][col] += u[i][j]
            for j in range(2):
                # k is 0
                V[(b_row_start + j) * p + b_col_start][col] += v[j][0]
            for i in range(2):
                # k is 0
                W[b_col_start * n + i][col] += gamma[i][0]
            col += 1

    # Sub-problem 1: A cols 0,1 ; B rows 0,1
    # B cols split: 0,1 (Strassen), 2,3 (Strassen), 4 (Schoolbook)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_schoolbook_221_block(0, 0, 4)
    
    # Sub-problem 2: A cols 2,3 ; B rows 2,3
    # B cols split: 0,1 (Strassen), 2,3 (Strassen), 4 (Schoolbook)
    add_strassen_block(2, 2, 0)
    add_strassen_block(2, 2, 2)
    add_schoolbook_221_block(2, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
