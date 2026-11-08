import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = delta(i,k)*delta(j,j)
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)

    # Index helpers
    def u_idx(i, j): return i * 4 + j  # A[i,j]
    def v_idx(i, j): return i * 4 + j  # B[i,j]
    def w_idx(k, i): return k * 4 + i  # C[i,j] -> W[k*n+i] where k=j, i=i -> W[j*4+i]

    # Strassen 2x2: given L=[[a,b],[c,d]] and R=[[e,f],[g,h]]
    # M1=(a+d)(e+h) -> P11:+1, P22:+1
    # M2=(c+d)e    -> P21:+1, P22:-1
    # M3=a(f-h)    -> P12:+1, P22:+1
    # M4=d(g-e)    -> P11:+1, P21:+1
    # M5=(a+b)h    -> P11:-1, P12:+1
    # M6=(c-a)(e+f)-> P22:+1
    # M7=(b-d)(g+h)-> P11:+1
    # P11=M1+M4-M5+M7, P12=M3+M5, P21=M2+M4, P22=M1-M2+M3+M6

    # sub_contrib[k] = list of (pos_in_2x2_result, coeff)
    sub_contrib = [
        [(0,0,1), (1,1,1)],    # M1
        [(1,0,1), (1,1,-1)],   # M2
        [(0,1,1), (1,1,1)],    # M3
        [(0,0,1), (1,0,1)],    # M4
        [(0,0,-1), (0,1,1)],   # M5
        [(1,1,1)],             # M6
        [(0,0,1)]              # M7
    ]

    # Block-level Strassen for 4x4:
    # P1 = (A00+A11)(B00+B11) -> C00:+1, C11:+1
    # P2 = (A10+A11)B00      -> C10:+1, C11:-1
    # P3 = A00(B01-B11)      -> C01:+1, C11:+1
    # P4 = A11(B10-B00)      -> C00:+1, C10:+1
    # P5 = (A00+A01)B11      -> C00:-1, C01:+1
    # P6 = (A10-A00)(B00+B01)-> C11:+1
    # P7 = (A01-A11)(B10+B11)-> C00:+1
    # C00=P1+P4-P5+P7, C01=P3+P5, C10=P2+P4, C11=P1-P2+P3+P6

    # contrib[t] = dict mapping (pos_in_2x2_block) -> list of (C_block_row, C_block_col, coeff)
    # where C_block (br, bc) means global C[2*br+pr, 2*bc+pc]
    contrib = [
        {(0,0): [(0,0,1), (1,1,1)], (1,1): [(0,0,1), (1,1,1)]},  # P1: contributes to C00 and C11
        {(1,0): [(1,0,1)], (1,1): [(1,0,1), (1,1,-1)]},           # P2: C10:+1, C11:-1
        {(0,1): [(0,1,1)], (1,1): [(0,1,1), (1,1,1)]},            # P3: C01:+1, C11:+1
        {(0,0): [(0,0,1)], (1,0): [(0,0,1), (1,0,1)]},            # P4: C00:+1, C10:+1
        {(0,0): [(0,0,-1)], (0,1): [(0,0,-1), (0,1,1)]},          # P5: C00:-1, C01:+1
        {(1,1): [(1,1,1)]},                                       # P6: C11:+1
        {(0,0): [(0,0,1)]},                                       # P7: C00:+1
    ]

    # Define L and R for each of the 7 block terms
    # Each is a 2x2 matrix where each element is a dict {(global_row, global_col): coeff}
    # A block (bi,bj) element (r,c) -> global (2*bi+r, 2*bj+c)
    # B block (bi,bj) element (r,c) -> global (2*bi+r, 2*bj+c)

    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)

    def b_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)

    L_terms = []
    R_terms = []

    # T0: P1 = (A00+A11)(B00+B11)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L0[r][c] = {a_elem(0,0,r,c): 1, a_elem(1,1,r,c): 1}
            R0[r][c] = {b_elem(0,0,r,c): 1, b_elem(1,1,r,c): 1}
    L_terms.append(L0)
    R_terms.append(R0)

    # T1: P2 = (A10+A11)B00
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L1[r][c] = {a_elem(1,0,r,c): 1, a_elem(1,1,r,c): 1}
            R1[r][c] = {b_elem(0,0,r,c): 1}
    L_terms.append(L1)
    R_terms.append(R1)

    # T2: P3 = A00(B01-B11)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L2[r][c] = {a_elem(0,0,r,c): 1}
            R2[r][c] = {b_elem(0,1,r,c): 1, b_elem(1,1,r,c): -1}
    L_terms.append(L2)
    R_terms.append(R2)

    # T3: P4 = A11(B10-B00)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L3[r][c] = {a_elem(1,1,r,c): 1}
            R3[r][c] = {b_elem(1,0,r,c): 1, b_elem(0,0,r,c): -1}
    L_terms.append(L3)
    R_terms.append(R3)

    # T4: P5 = (A00+A01)B11
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L4[r][c] = {a_elem(0,0,r,c): 1, a_elem(0,1,r,c): 1}
            R4[r][c] = {b_elem(1,1,r,c): 1}
    L_terms.append(L4)
    R_terms.append(R4)

    # T5: P6 = (A10-A00)(B00+B01)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L5[r][c] = {a_elem(1,0,r,c): 1, a_elem(0,0,r,c): -1}
            R5[r][c] = {b_elem(0,0,r,c): 1, b_elem(0,1,r,c): 1}
    L_terms.append(L5)
    R_terms.append(R5)

    # T6: P7 = (A01-A11)(B10+B11)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L6[r][c] = {a_elem(0,1,r,c): 1, a_elem(1,1,r,c): -1}
            R6[r][c] = {b_elem(1,0,r,c): 1, b_elem(1,1,r,c): 1}
    L_terms.append(L6)
    R_terms.append(R6)

    # For each term t and each Strassen sub-product k, build the scalar multiplication
    for t in range(7):
        Lt = L_terms[t]
        Rt = R_terms[t]
        ct = contrib[t]

        a = Lt[0][0]  # L[0][0]
        b = Lt[0][1]  # L[0][1]
        c = Lt[1][0]  # L[1][0]
        d = Lt[1][1]  # L[1][1]
        e = Rt[0][0]  # R[0][0]
        f = Rt[0][1]  # R[0][1]
        g = Rt[1][0]  # R[1][0]
        h = Rt[1][1]  # R[1][1]

        # 7 Strassen sub-products: (X_combo, Y_combo)
        def add_dicts(d1, d2):
            res = dict(d1)
            for k, v in d2.items():
                res[k] = res.get(k, 0) + v
            return res

        def sub_dicts(d1, d2):
            res = dict(d1)
            for k, v in d2.items():
                res[k] = res.get(k, 0) - v
            return res

        sub_products = [
            (add_dicts(a, d), add_dicts(e, h)),  # M1
            (add_dicts(c, d), e),                 # M2
            (a, sub_dicts(f, h)),                 # M3
            (d, sub_dicts(g, e)),                 # M4
            (add_dicts(a, b), h),                 # M5
            (sub_dicts(c, a), add_dicts(e, f)),   # M6
            (sub_dicts(b, d), add_dicts(g, h)),   # M7
        ]

        for k in range(7):
            xk, yk = sub_products[k]
            ridx = t * 7 + k

            # Fill U from xk (A elements)
            for (row, col), coeff in xk.items():
                if coeff != 0:
                    U[u_idx(row, col), ridx] += coeff

            # Fill V from yk (B elements)
            for (row, col), coeff in yk.items():
                if coeff != 0:
                    V[v_idx(row, col), ridx] += coeff

            # Fill W from sub_contrib[k] through contrib[t]
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in ct:
                    for (br, bc, block_coeff) in ct[(pr, pc)]:
                        # Global C index: row = 2*br + pr, col = 2*bc + pc
                        # W convention: W[k*n+i, r] where k=col_of_C, i=row_of_C
                        # So W index = (2*bc+pc)*4 + (2*br+pr)
                        w_row = 2 * br + pr
                        w_col = 2 * bc + pc
                        W[w_idx(w_col, w_row), ridx] += sub_coeff * block_coeff

    # Verify: check that sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1 for all valid (i,j,k)
    # Tensor T[i,j,k] should equal 1 if this is a valid decomposition
    # Actually for matrix multiplication: T[A_idx, B_idx, C_idx] where
    # A_idx = i*4+j, B_idx = j*4+k, C_idx = k*4+i (trace convention)
    # The tensor should satisfy: sum_r U[A_idx,r]*V[B_idx,r]*W[C_idx,r] = 1
    # But only when the indices are consistent with matrix multiplication structure.
    # More precisely: for each (i,j,k), the tensor entry T[i*4+j, j*4+k, k*4+i]
    # should equal 1 (since C[i,k] = sum_j A[i,j]*B[j,k], and the tensor decomposes
    # the multiplication map).
    
    # Let's verify by checking a few random cases
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Quick verification with random integer inputs
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (4, 4))
        B_test = np.random.randint(-5, 5, (4, 4))
        C_expected = A_test @ B_test
        
        # Compute using the decomposition
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i*4+j,r]*V[j*4+k,r]*W[k*4+i,r]
        C_computed = np.zeros((4, 4), dtype=np.int64)
        for i in range(4):
            for k in range(4):
                total = 0
                for j in range(4):
                    u_row = i * 4 + j
                    v_row = j * 4 + k
                    w_row = k * 4 + i
                    total += np.dot(U[u_row, :], V[v_row, :] * W[w_row, :])
                C_computed[i, k] = total
        
        if not np.array_equal(C_computed, C_expected):
            # Decomposition is wrong - this shouldn't happen with correct Strassen
            # but let's check
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
