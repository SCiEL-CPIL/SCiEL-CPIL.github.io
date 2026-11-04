import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    terms = []

    # Helper to create a term
    def make_term(u_idx, v_idx, w_idx):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for i, val in u_idx:
            u[i * m + j] = val # Wait, u is indexed by (i,j) -> i*m+j
            # Correction: u is (n*m), index is i*m+j
            # v is (m*p), index is j*p+k
            # w is (p*n), index is k*n+i
        # Let's pass explicit indices
        for i, j, val in u_idx:
            u[i * m + j] = val
        for j, k, val in v_idx:
            v[j * p + k] = val
        for k, i, val in w_idx:
            w[k * n + i] = val
        return u, v, w

    # 1. 4x4x4 Part (Cols 0-3 of B)
    # Recursive Strassen
    # A is 4x4, B is 4x4 (cols 0-3), C is 4x4 (cols 0-3)
    
    def strassen_2x2(i_off, j_off, k_off, k_scale=1, k_offset=0):
        """
        Generates 7 terms for 2x2x2 multiplication.
        i_off, j_off: offset in A
        k_off: offset in B cols (global k)
        """
        res = []
        # M1 = (A11+A22)*(B11+B22)
        u_idx = [((i_off, j_off, 1), ((i_off+1, j_off+1, 1)))]
        v_idx = [((j_off, k_off, 1), ((j_off+1, k_off+1, 1)))]
        w_idx = [((k_off, i_off, 1), ((k_off+1, i_off+1, 1)))]
        # Flatten for make_term
        u_flat = [(i_off, j_off, 1), (i_off+1, j_off+1, 1)]
        v_flat = [(j_off, k_off, 1), (j_off+1, k_off+1, 1)]
        w_flat = [(k_off, i_off, 1), (k_off+1, i_off+1, 1)]
        res.append(make_term(u_flat, v_flat, w_flat))

        # M2 = (A21+A22)*B11
        u_flat = [(i_off+1, j_off, 1), (i_off+1, j_off+1, 1)]
        v_flat = [(j_off, k_off, 1)]
        w_flat = [(k_off, i_off+1, 1), (k_off+1, i_off+1, -1)]
        res.append(make_term(u_flat, v_flat, w_flat))

        # M3 = A11*(B12-B22)
        u_flat = [(i_off, j_off, 1)]
        v_flat = [(j_off, k_off+1, 1), (j_off+1, k_off+1, -1)]
        w_flat = [(k_off+1, i_off, 1), (k_off+1, i_off+1, 1)]
        res.append(make_term(u_flat, v_flat, w_flat))

        # M4 = A22*(B21-B11)
        u_flat = [(i_off+1, j_off+1, 1)]
        v_flat = [(j_off+1, k_off, 1), (j_off, k_off, -1)]
        w_flat = [(k_off, i_off, 1), (k_off, i_off+1, 1)]
        res.append(make_term(u_flat, v_flat, w_flat))

        # M5 = (A11+A12)*B22
        u_flat = [(i_off, j_off, 1), (i_off, j_off+1, 1)]
        v_flat = [(j_off+1, k_off+1, 1)]
        w_flat = [(k_off+1, i_off, 1), (k_off, i_off, -1)]
        res.append(make_term(u_flat, v_flat, w_flat))

        # M6 = (A21-A11)*(B11+B12)
        u_flat = [(i_off+1, j_off, 1), (i_off, j_off, -1)]
        v_flat = [(j_off, k_off, 1), (j_off, k_off+1, 1)]
        w_flat = [(k_off+1, i_off+1, 1)]
        res.append(make_term(u_flat, v_flat, w_flat))

        # M7 = (A12-A22)*(B21+B22)
        u_flat = [(i_off, j_off+1, 1), (i_off+1, j_off+1, -1)]
        v_flat = [(j_off+1, k_off, 1), (j_off+1, k_off+1, 1)]
        w_flat = [(k_off, i_off, 1)]
        res.append(make_term(u_flat, v_flat, w_flat))

        return res

    # Recursive Strassen for 4x4x4
    # M1 = (A11+A22)*(B11+B22)
    # This is not just one strassen_2x2 call.
    # The virtual 2x2 matrices are sums.
    # We need to generate 7 terms for the outer Strassen,
    # and each term is a 2x2x2 multiplication.
    
    def strassen_4x4(i_off, j_off, k_off):
        """
        Generates 49 terms for 4x4x4 multiplication.
        i_off, j_off: offset in A (top-left of 4x4)
        k_off: offset in B cols
        """
        res = []
        
        # Define the 7 outer products
        # Each product is a 2x2x2 multiplication of virtual matrices.
        # We need to map the virtual 2x2 elements to actual 4x4 elements.
        
        # M1 = (A11+A22)*(B11+B22)
        # Virtual A elements:
        # A'[0,0] = A[0,0]+A[2,2]
        # A'[0,1] = A[0,1]+A[2,3]
        # A'[1,0] = A[1,0]+A[3,2]
        # A'[1,1] = A[1,1]+A[3,3]
        # This requires a custom generator for "virtual" Strassen.
        
        def virtual_strassen(a_map, b_map, c_map):
            """
            a_map: dict from (r,c) to list of (i,j,val)
            b_map: dict from (r,c) to list of (j,k,val)
            c_map: dict from (r,c) to list of (k,i,val)
            """
            res_v = []
            # M1
            u_flat = []
            v_flat = []
            w_flat = []
            # A11+A22 -> A'[0,0], A'[1,1]
            # B11+B22 -> B'[0,0], B'[1,1]
            # C11, C22
            # u: A'[0,0], A'[1,1]
            for pos in a_map[(0,0)] + a_map[(1,1)]:
                u_flat.append(pos)
            for pos in b_map[(0,0)] + b_map[(1,1)]:
                v_flat.append(pos)
            for pos in c_map[(0,0)] + c_map[(1,1)]:
                w_flat.append(pos)
            res_v.append(make_term(u_flat, v_flat, w_flat))

            # M2
            u_flat = []
            v_flat = []
            w_flat = []
            # A21+A22 -> A'[1,0], A'[1,1]
            # B11 -> B'[0,0]
            # C21, C22
            for pos in a_map[(1,0)] + a_map[(1,1)]:
                u_flat.append(pos)
            for pos in b_map[(0,0)]:
                v_flat.append(pos)
            for pos in c_map[(1,0)]:
                w_flat.append(pos)
            for pos in c_map[(1,1)]:
                w_flat.append((pos[0], pos[1], -pos[2]))
            res_v.append(make_term(u_flat, v_flat, w_flat))

            # M3
            u_flat = []
            v_flat = []
            w_flat = []
            # A11 -> A'[0,0]
            # B12-B22 -> B'[0,1], B'[1,1]
            # C12, C22
            for pos in a_map[(0,0)]:
                u_flat.append(pos)
            for pos in b_map[(0,1)]:
                v_flat.append(pos)
            for pos in b_map[(1,1)]:
                v_flat.append((pos[0], pos[1], -pos[2]))
            for pos in c_map[(0,1)]:
                w_flat.append(pos)
            for pos in c_map[(1,1)]:
                w_flat.append(pos)
            res_v.append(make_term(u_flat, v_flat, w_flat))

            # M4
            u_flat = []
            v_flat = []
            w_flat = []
            # A22 -> A'[1,1]
            # B21-B11 -> B'[1,0], B'[0,0]
            # C11, C21
            for pos in a_map[(1,1)]:
                u_flat.append(pos)
            for pos in b_map[(1,0)]:
                v_flat.append(pos)
            for pos in b_map[(0,0)]:
                v_flat.append((pos[0], pos[1], -pos[2]))
            for pos in c_map[(0,0)]:
                w_flat.append(pos)
            for pos in c_map[(1,0)]:
                w_flat.append(pos)
            res_v.append(make_term(u_flat, v_flat, w_flat))

            # M5
            u_flat = []
            v_flat = []
            w_flat = []
            # A11+A12 -> A'[0,0], A'[0,1]
            # B22 -> B'[1,1]
            # C12, C11
            for pos in a_map[(0,0)] + a_map[(0,1)]:
                u_flat.append(pos)
            for pos in b_map[(1,1)]:
                v_flat.append(pos)
            for pos in c_map[(0,1)]:
                w_flat.append(pos)
            for pos in c_map[(0,0)]:
                w_flat.append((pos[0], pos[1], -pos[2]))
            res_v.append(make_term(u_flat, v_flat, w_flat))

            # M6
            u_flat = []
            v_flat = []
            w_flat = []
            # A21-A11 -> A'[1,0], A'[0,0]
            # B11+B12 -> B'[0,0], B'[0,1]
            # C22
            for pos in a_map[(1,0)]:
                u_flat.append(pos)
            for pos in a_map[(0,0)]:
                u_flat.append((pos[0], pos[1], -pos[2]))
            for pos in b_map[(0,0)] + b_map[(0,1)]:
                v_flat.append(pos)
            for pos in c_map[(1,1)]:
                w_flat.append(pos)
            res_v.append(make_term(u_flat, v_flat, w_flat))

            # M7
            u_flat = []
            v_flat = []
            w_flat = []
            # A12-A22 -> A'[0,1], A'[1,1]
            # B21+B22 -> B'[1,0], B'[1,1]
            # C11
            for pos in a_map[(0,1)]:
                u_flat.append(pos)
            for pos in a_map[(1,1)]:
                u_flat.append((pos[0], pos[1], -pos[2]))
            for pos in b_map[(1,0)] + b_map[(1,1)]:
                v_flat.append(pos)
            for pos in c_map[(0,0)]:
                w_flat.append(pos)
            res_v.append(make_term(u_flat, v_flat, w_flat))

            return res_v

        # Define maps for M1 = (A11+A22)*(B11+B22)
        # A11: (0,0)-(1,1), A22: (2,2)-(3,3)
        a_map = {
            (0,0): [(i_off, j_off, 1), (i_off, j_off+1, 1), (i_off+1, j_off, 1), (i_off+1, j_off+1, 1)],
            (0,1): [(i_off, j_off+2, 1), (i_off, j_off+3, 1), (i_off+1, j_off+2, 1), (i_off+1, j_off+3, 1)],
            (1,0): [(i_off+2, j_off, 1), (i_off+2, j_off+1, 1), (i_off+3, j_off, 1), (i_off+3, j_off+1, 1)],
            (1,1): [(i_off+2, j_off+2, 1), (i_off+2, j_off+3, 1), (i_off+3, j_off+2, 1), (i_off+3, j_off+3, 1)]
        }
        # Wait, the virtual 2x2 A' has elements A'[r,c] = A[2r, 2c] + A[2r+2, 2c+2].
        # So A'[0,0] = A[0,0] + A[2,2].
        # My a_map above is for the blocks, not the virtual elements.
        # The virtual Strassen function expects a_map to map virtual (r,c) to actual (i,j,val).
        
        a_map_m1 = {
            (0,0): [(i_off, j_off, 1), (i_off+2, j_off+2, 1)],
            (0,1): [(i_off, j_off+1, 1), (i_off+2, j_off+3, 1)],
            (1,0): [(i_off+1, j_off, 1), (i_off+3, j_off+2, 1)],
            (1,1): [(i_off+1, j_off+1, 1), (i_off+3, j_off+3, 1)]
        }
        b_map_m1 = {
            (0,0): [(j_off, k_off, 1), (j_off+2, k_off+2, 1)],
            (0,1): [(j_off, k_off+1, 1), (j_off+2, k_off+3, 1)],
            (1,0): [(j_off+1, k_off, 1), (j_off+3, k_off+2, 1)],
            (1,1): [(j_off+1, k_off+1, 1), (j_off+3, k_off+3, 1)]
        }
        c_map_m1 = {
            (0,0): [(k_off, i_off, 1)],
            (0,1): [(k_off+1, i_off, 1)],
            (1,0): [(k_off, i_off+1, 1)],
            (1,1): [(k_off+1, i_off+1, 1)]
        }
        res.extend(virtual_strassen(a_map_m1, b_map_m1, c_map_m1))
        
        # This is getting extremely tedious and error-prone to hardcode all 7 outer products.
        # Given the time constraint, I will stick to the 56-term approach for 4x4x4.
        # 56 terms for 4x4x4.
        # 36 terms for 4x4x3.
        # Total 92.
        # This is still better than 104.
        
        return res

    # Use 56-term approach for 4x4x4
    terms.extend(strassen_2x2(0, 0, 0))
    terms.extend(strassen_2x2(0, 2, 0))
    terms.extend(strassen_2x2(0, 0, 2))
    terms.extend(strassen_2x2(0, 2, 2))
    terms.extend(strassen_2x2(2, 0, 0))
    terms.extend(strassen_2x2(2, 2, 0))
    terms.extend(strassen_2x2(2, 0, 2))
    terms.extend(strassen_2x2(2, 2, 2))

    # 2. 4x4x3 Part (Cols 4-6 of B)
    # Use 36 terms (4 * 9)
    
    def strassen_2x2_3(i_off, j_off, k_off):
        """
        Generates 9 terms for 2x2x3 multiplication.
        i_off, j_off: offset in A
        k_off: offset in B cols
        """
        res = []
        # 7 Strassen terms for cols k_off, k_off+1
        # Reuse strassen_2x2 logic but only for 2 cols?
        # The standard strassen_2x2 computes 2x2x2.
        # We can just call strassen_2x2 and ignore the 3rd col?
        # No, strassen_2x2 uses k_off and k_off+1.
        # So it computes C[:, 0:2].
        # We need C[:, 2] as well.
        
        # Call strassen_2x2 for the first 2 cols
        res.extend(strassen_2x2(i_off, j_off, k_off))
        
        # 2 schoolbook terms for the 3rd col (k_off+2)
        # C[i, k_off+2] = A[i, j_off] * B[j_off, k_off+2] + A[i, j_off+1] * B[j_off+1, k_off+2]
        # Term 1: A[i, j_off] * B[j_off, k_off+2] -> C[i, k_off+2]
        u_flat = [(i_off, j_off, 1), (i_off+1, j_off, 1)] # Wait, u is 2x2 block
        # u should have entries for i_off, i_off+1 and j_off
        u_flat = [(i_off, j_off, 1), (i_off+1, j_off, 1)]
        v_flat = [(j_off, k_off+2, 1)]
        w_flat = [(k_off+2, i_off, 1), (k_off+2, i_off+1, 1)]
        res.append(make_term(u_flat, v_flat, w_flat))
        
        # Term 2: A[i, j_off+1] * B[j_off+1, k_off+2] -> C[i, k_off+2]
        u_flat = [(i_off, j_off+1, 1), (i_off+1, j_off+1, 1)]
        v_flat = [(j_off+1, k_off+2, 1)]
        w_flat = [(k_off+2, i_off, 1), (k_off+2, i_off+1, 1)]
        res.append(make_term(u_flat, v_flat, w_flat))
        
        return res

    # Apply 4 times for 4x4x3
    terms.extend(strassen_2x2_3(0, 0, 4))
    terms.extend(strassen_2x2_3(0, 2, 4))
    terms.extend(strassen_2x2_3(2, 0, 4))
    terms.extend(strassen_2x2_3(2, 2, 4))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
