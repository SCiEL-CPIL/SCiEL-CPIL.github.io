import numpy as np

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    """
    n, m, p = 4, 4, 9
    u_dim = n * m  # 16
    v_dim = m * p  # 36
    w_dim = p * n  # 36

    U_list = []
    V_list = []
    W_list = []

    # Strassen 2x2x2 terms: (A_mask, B_mask, C_mask)
    # Each mask is a dict {(r,c): coeff} for r,c in {0,1}
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6

    s2_terms = [
        # M1: (A00+A11)(B00+B11) -> C00+, C11+
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)B00 -> C10+, C11-
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: A00(B01-B11) -> C01+, C11+
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: A11(B10-B00) -> C00+, C10+
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (A00+A01)B11 -> C00-, C01+
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A10-A00)(B00+B01) -> C11+
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A01-A11)(B10+B11) -> C00+
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def add_strassen_4x4(b_col_off):
        """
        Add 49 terms for A(4x4) * B_block(4x4) -> C_block(4x4).
        b_col_off: starting column index in B (0 or 4).
        """
        for oa, ob, oc in s2_terms:
            for ia, ib, ic in s2_terms:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36

                # U vector: A indices
                for (oor, ooc), c_ao in oa.items():
                    for (ior, ioc), c_ia in ia.items():
                        r = oor * 2 + ior
                        c = ooc * 2 + ioc
                        u[r * 4 + c] += c_ao * c_ia

                # V vector: B indices
                for (oor, ooc), c_bo in ob.items():
                    for (ior, ioc), c_ib in ib.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        v[r * 9 + c] += c_bo * c_ib

                # W vector: C indices
                # C output index: W[k*n+i] where k is column, i is row
                for (oor, ooc), c_co in oc.items():
                    for (ior, ioc), c_ic in ic.items():
                        r = oor * 2 + ior  # global row
                        c = b_col_off + ooc * 2 + ioc  # global col
                        w[c * 4 + r] += c_co * c_ic

                add_term(u, v, w)

    def add_schoolbook_4x1(b_col):
        """
        Add 16 terms for A(4x4) * B_col(4x1) -> C_col(4x1).
        b_col: column index in B (8).
        """
        for i in range(4):
            for j in range(4):
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36

                u[i * 4 + j] = 1.0
                v[j * 9 + b_col] = 1.0
                w[b_col * 4 + i] = 1.0

                add_term(u, v, w)

    # Add Strassen for B cols 0-3
    add_strassen_4x4(0)

    # Add Strassen for B cols 4-7
    add_strassen_4x4(4)

    # Add Schoolbook for B col 8
    add_schoolbook_4x1(8)

    rank = len(U_list)

    # Build output matrices
    U = np.array(U_list).T  # shape (16, rank)
    V = np.array(V_list).T  # shape (36, rank)
    W = np.array(W_list).T  # shape (36, rank)

    # Convert to lists for the required output format
    U_list_out = U.tolist()
    V_list_out = V.tolist()
    W_list_out = W.tolist()

    return {"U": U_list_out, "V": V_list_out, "W": W_list_out}
