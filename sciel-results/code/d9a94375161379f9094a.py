import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a column
    def add_column(u_entries, v_entries, w_entries):
        nonlocal col_idx
        if col_idx >= rank:
            return
        for (u_idx, val) in u_entries:
            U[u_idx][col_idx] += val
        for (v_idx, val) in v_entries:
            V[v_idx][col_idx] += val
        for (w_idx, val) in w_entries:
            W[w_idx][col_idx] += val
        col_idx += 1

    # Helper to get global indices
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    # Recursive decomposition
    # i_start, j_start, k_start are global offsets
    def decompose(n_loc, m_loc, p_loc, i_start, j_start, k_start):
        # Base cases
        if n_loc == 1 and m_loc == 1 and p_loc == 1:
            add_column(
                [(u_idx(i_start, j_start), 1)],
                [(v_idx(j_start, k_start), 1)],
                [(w_idx(k_start, i_start), 1)]
            )
            return

        if n_loc == 1 and m_loc == 2 and p_loc == 2:
            # Schoolbook for 1x2x2
            # i=0
            # j=0, k=0
            add_column(
                [(u_idx(i_start, j_start), 1)],
                [(v_idx(j_start, k_start), 1)],
                [(w_idx(k_start, i_start), 1)]
            )
            # j=1, k=0
            add_column(
                [(u_idx(i_start, j_start+1), 1)],
                [(v_idx(j_start+1, k_start), 1)],
                [(w_idx(k_start, i_start), 1)]
            )
            # j=0, k=1
            add_column(
                [(u_idx(i_start, j_start), 1)],
                [(v_idx(j_start, k_start+1), 1)],
                [(w_idx(k_start+1, i_start), 1)]
            )
            # j=1, k=1
            add_column(
                [(u_idx(i_start, j_start+1), 1)],
                [(v_idx(j_start+1, k_start+1), 1)],
                [(w_idx(k_start+1, i_start), 1)]
            )
            return

        if n_loc == 2 and m_loc == 2 and p_loc == 2:
            # Strassen for 2x2x2
            # M1 = (A00+A11)(B00+B11) -> C00, C11
            add_column(
                [(u_idx(i_start, j_start), 1), (u_idx(i_start+1, j_start+1), 1)],
                [(v_idx(j_start, k_start), 1), (v_idx(j_start+1, k_start+1), 1)],
                [(w_idx(k_start, i_start), 1), (w_idx(k_start+1, i_start+1), 1)]
            )
            # M2 = (A10+A11)B00 -> C10, C11(-)
            add_column(
                [(u_idx(i_start+1, j_start), 1), (u_idx(i_start+1, j_start+1), 1)],
                [(v_idx(j_start, k_start), 1)],
                [(w_idx(k_start, i_start+1), 1), (w_idx(k_start+1, i_start+1), -1)]
            )
            # M3 = A00(B01-B11) -> C01, C11
            add_column(
                [(u_idx(i_start, j_start), 1)],
                [(v_idx(j_start, k_start+1), 1), (v_idx(j_start+1, k_start+1), -1)],
                [(w_idx(k_start+1, i_start), 1), (w_idx(k_start+1, i_start+1), 1)]
            )
            # M4 = A11(B10-B00) -> C00, C10
            add_column(
                [(u_idx(i_start+1, j_start+1), 1)],
                [(v_idx(j_start+1, k_start), 1), (v_idx(j_start, k_start), -1)],
                [(w_idx(k_start, i_start), 1), (w_idx(k_start, i_start+1), 1)]
            )
            # M5 = (A00+A01)B11 -> C00(-), C01
            add_column(
                [(u_idx(i_start, j_start), 1), (u_idx(i_start, j_start+1), 1)],
                [(v_idx(j_start+1, k_start+1), 1)],
                [(w_idx(k_start, i_start), -1), (w_idx(k_start+1, i_start), 1)]
            )
            # M6 = (A10-A00)(B00+B01) -> C11
            add_column(
                [(u_idx(i_start+1, j_start), 1), (u_idx(i_start, j_start), -1)],
                [(v_idx(j_start, k_start), 1), (v_idx(j_start, k_start+1), 1)],
                [(w_idx(k_start+1, i_start+1), 1)]
            )
            # M7 = (A01-A11)(B10+B11) -> C00
            add_column(
                [(u_idx(i_start, j_start+1), 1), (u_idx(i_start+1, j_start+1), -1)],
                [(v_idx(j_start+1, k_start), 1), (v_idx(j_start+1, k_start+1), 1)],
                [(w_idx(k_start, i_start), 1)]
            )
            return

        # Recursive steps
        if n_loc == 3 and m_loc == 2 and p_loc == 2:
            # Split n into 1, 2
            decompose(1, 2, 2, i_start, j_start, k_start)
            decompose(2, 2, 2, i_start+1, j_start, k_start)
            return

        if n_loc == 3 and m_loc == 2 and p_loc == 6:
            # Split p into 2, 2, 2
            decompose(3, 2, 2, i_start, j_start, k_start)
            decompose(3, 2, 2, i_start, j_start, k_start+2)
            decompose(3, 2, 2, i_start, j_start, k_start+4)
            return

        if n_loc == 3 and m_loc == 4 and p_loc == 6:
            # Split m into 2, 2
            decompose(3, 2, 6, i_start, j_start, k_start)
            decompose(3, 2, 6, i_start, j_start+2, k_start)
            return

        # Fallback: Schoolbook
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    add_column(
                        [(u_idx(i_start+i, j_start+j), 1)],
                        [(v_idx(j_start+j, k_start+k), 1)],
                        [(w_idx(k_start+k, i_start+i), 1)]
                    )

    decompose(n, m, p, 0, 0, 0)
    
    return {"U": U, "V": V, "W": W}
