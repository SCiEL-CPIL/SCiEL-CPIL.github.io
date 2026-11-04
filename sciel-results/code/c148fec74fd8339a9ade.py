"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term where dicts map (row, col) -> coefficient."""
        nonlocal col
        for (i, j), c in u_dict.items():
            if c != 0:
                U[i * m + j][col] += c
        for (j, k), c in v_dict.items():
            if c != 0:
                V[j * p + k][col] += c
        for (i, k), c in w_dict.items():
            if c != 0:
                W[k * n + i][col] += c
        col += 1

    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]."""
        # M1 = (A00+A11)(B00+B11)
        add_term(
            {(i0, j0): 1, (i0+1, j0+1): 1},
            {(j0, k0): 1, (j0+1, k0+1): 1},
            {(i0, k0): 1, (i0+1, k0+1): 1, (i0, k0+1): -1, (i0+1, k0): -1}
        )
        # M2 = (A10+A11)B00
        add_term(
            {(i0+1, j0): 1, (i0+1, j0+1): 1},
            {(j0, k0): 1},
            {(i0+1, k0): 1}
        )
        # M3 = A00(B10-B11)
        add_term(
            {(i0, j0): 1},
            {(j0, k0+1): 1, (j0+1, k0+1): -1},
            {(i0, k0+1): 1}
        )
        # M4 = A11(B00-B01)
        add_term(
            {(i0+1, j0+1): 1},
            {(j0, k0): 1, (j0+1, k0): -1},
            {(i0+1, k0): 1}
        )
        # M5 = (A00+A01)B11
        add_term(
            {(i0, j0): 1, (i0, j0+1): 1},
            {(j0+1, k0+1): 1},
            {(i0, k0+1): 1}
        )
        # M6 = (A10-A00)(B00+B01)
        add_term(
            {(i0+1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0+1, k0): 1},
            {(i0+1, k0): 1, (i0+1, k0+1): 1}
        )
        # M7 = (A01-A11)(B10+B11)
        add_term(
            {(i0, j0+1): 1, (i0+1, j0+1): -1},
            {(j0+1, k0): 1, (j0+1, k0+1): 1},
            {(i0, k0): 1}
        )

    def add_schoolbook(n_sub, m_sub, p_sub, i0, j0, k0):
        """Add schoolbook for A[i0:i0+n_sub, j0:j0+m_sub] * B[j0:j0+m_sub, k0:k0+p_sub]."""
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    add_term(
                        {(i0+i, j0+j): 1},
                        {(j0+j, k0+k): 1},
                        {(i0+i, k0+k): 1}
                    )

    # Strategy: split A into rows [0:2] and [2:4], B into rows [0:2], [2:4], [4:5]
    # For each 2x5 block of A times 5x5 block of B, use Strassen where possible.
    
    # A rows 0-1 (2x5) times B:
    #   A[0:2, 0:2] (2x2) * B[0:2, :] (2x5): split B cols into [0:2],[2:4],[4:5]
    #     2x2x2 Strassen for B cols [0:2]
    #     2x2x2 Strassen for B cols [2:4]  
    #     2x2x1 schoolbook for B col [4:5]
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    add_schoolbook(2, 2, 1, 0, 0, 4)
    
    #   A[0:2, 2:4] (2x2) * B[2:4, :] (2x5): split B cols into [0:2],[2:4],[4:5]
    add_strassen_2x2x2(0, 2, 0)
    add_strassen_2x2x2(0, 2, 2)
    add_schoolbook(2, 2, 1, 0, 2, 4)
    
    #   A[0:2, 4:5] (2x1) * B[4:5, :] (1x5): schoolbook 2x1x5
    add_schoolbook(2, 1, 5, 0, 4, 0)
    
    # A rows 2-3 (2x5) times B:
    #   A[2:4, 0:2] (2x2) * B[0:2, :] (2x5): split B cols
    add_strassen_2x2x2(2, 0, 0)
    add_strassen_2x2x2(2, 0, 2)
    add_schoolbook(2, 2, 1, 2, 0, 4)
    
    #   A[2:4, 2:4] (2x2) * B[2:4, :] (2x5): split B cols
    add_strassen_2x2x2(2, 2, 0)
    add_strassen_2x2x2(2, 2, 2)
    add_schoolbook(2, 2, 1, 2, 2, 4)
    
    #   A[2:4, 4:5] (2x1) * B[4:5, :] (1x5): schoolbook 2x1x5
    add_schoolbook(2, 1, 5, 2, 4, 0)
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
