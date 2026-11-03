"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6

    U_entries = []
    V_entries = []
    W_entries = []

    def add_term(u_dict, v_dict, w_dict):
        U_entries.append(u_dict)
        V_entries.append(v_dict)
        W_entries.append(w_dict)

    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # p1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1]) -> C[i0,k0]+C[i1,k1]
        add_term(
            {i0*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: 1, k1*n+i1: 1}
        )
        # p2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0] -> C[i1,k0]
        add_term(
            {i1*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1},
            {k0*n+i1: 1}
        )
        # p3 = A[i0,j0] * (B[j0,k1]-B[j1,k1]) -> C[i0,k1]
        add_term(
            {i0*m+j0: 1},
            {j0*p+k1: 1, j1*p+k1: -1},
            {k1*n+i0: 1}
        )
        # p4 = A[i1,j1] * (B[j1,k0]-B[j0,k0]) -> C[i0,k0]+C[i1,k0]
        add_term(
            {i1*m+j1: 1},
            {j1*p+k0: 1, j0*p+k0: -1},
            {k0*n+i0: 1, k0*n+i1: 1}
        )
        # p5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1] -> -C[i0,k0]+C[i0,k1]
        add_term(
            {i0*m+j0: 1, i0*m+j1: 1},
            {j1*p+k1: 1},
            {k0*n+i0: -1, k1*n+i0: 1}
        )
        # p6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1]) -> C[i1,k1]
        add_term(
            {i1*m+j0: 1, i0*m+j0: -1},
            {j0*p+k0: 1, j0*p+k1: 1},
            {k1*n+i1: 1}
        )
        # p7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1]) -> C[i0,k0]-C[i1,k1]
        add_term(
            {i0*m+j1: 1, i1*m+j1: -1},
            {j1*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: 1, k1*n+i1: -1}
        )

    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term(
                        {i*m+j: 1},
                        {j*p+k: 1},
                        {k*n+i: 1}
                    )

    i_parts = [[0], [1, 2]]
    j_parts = [[0, 1], [2, 3]]
    k_parts = [[0, 1], [2, 3], [4, 5]]

    for i_list in i_parts:
        for j_list in j_parts:
            for k_list in k_parts:
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    i0, i1 = i_list
                    j0, j1 = j_list
                    k0, k1 = k_list
                    add_strassen_block(i0, i1, j0, j1, k0, k1)
                else:
                    add_schoolbook_block(i_list, j_list, k_list)

    rank = len(U_entries)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col in range(rank):
        for row, val in U_entries[col].items():
            U[row][col] = val
        for row, val in V_entries[col].items():
            V[row][col] = val
        for row, val in W_entries[col].items():
            W[row][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
