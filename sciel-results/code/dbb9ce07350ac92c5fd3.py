"""Tensor decomposition for 3x3 x 3x6 matrix multiplication."""
import time

N_ROWS_A = 3
N_COLS_A = 3
N_COLS_B = 6


def construct():
    """Return factor matrices U, V, W for exact matrix-multiplication decomposition.

    Shapes: U is (N_ROWS_A * N_COLS_A, rank), V is (N_COLS_A * N_COLS_B, rank),
            W is (N_COLS_B * N_ROWS_A, rank).
    Indexing convention: U[a_row * N_COLS_A + a_col, t],
                         V[a_col * N_COLS_B + b_col, t],
                         W[b_col * N_ROWS_A + a_row, t].
    """
    a_dim = N_ROWS_A   # rows of A
    b_dim = N_COLS_A   # cols of A = rows of B
    c_dim = N_COLS_B   # cols of B
    total_rank = a_dim * b_dim * c_dim

    U = [[0] * total_rank for _ in range(a_dim * b_dim)]
    V = [[0] * total_rank for _ in range(b_dim * c_dim)]
    W = [[0] * total_rank for _ in range(c_dim * a_dim)]

    slot = 0
    for row_a in range(a_dim):
        for col_ab in range(b_dim):
            for col_b in range(c_dim):
                U[row_a * b_dim + col_ab][slot] = 1
                V[col_ab * c_dim + col_b][slot] = 1
                W[col_b * a_dim + row_a][slot] = 1
                slot += 1

    return {"U": U, "V": V, "W": W}
