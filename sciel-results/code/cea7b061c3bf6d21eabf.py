def _column_redundant(M, r):
    """Check if column r of matrix M is in the span of other columns."""
    if M.shape[1] <= 1:
        return False
    cols = [M[:, i] for i in range(M.shape[1]) if i != r]
    col_r = M[:, r]
    A = np.column_stack(cols)
    # Check if col_r is in column space of A
    rank_A = np.linalg.matrix_rank(A)
    rank_Ar = np.linalg.matrix_rank(np.column_stack([A, col_r]))
    return rank_A == rank_Ar
