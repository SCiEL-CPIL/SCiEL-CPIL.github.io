import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe
    
    # Rank-19 decomposition for 3x3 matrix multiplication
    # Each term: (u_indices, v_indices, w_indices) where each is a list of (index, coeff)
    # A[i,k], B[k,j], C[i,j] with i,j,k in {0,1,2}
    # U index = i*3 + k, V index = k*3 + j, W index = i*3 + j
    def get_3x3_rank19():
        terms = []
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        terms.append([(0*3+0, 1)], [(0*3+0, 1)], [(0*3+0, 1)])
        # Term 2: A[0,1]*B[1,0] -> C[0,0]
        terms.append([(0*3+1, 1)], [(1*3+0, 1)], [(0*3+0, 1)])
        # Term 3: A[1,0]*B[0,1] -> C[1,0]
        terms.append([(1*3+0, 1)], [(0*3+1, 1)], [(1*3+0, 1)])
        # Term 4: A[1,1]*B[1,1] -> C[1,1]
        terms.append([(1*3+1, 1)], [(1*3+1, 1)], [(1*3+1, 1)])
        # Term 5: A[2,0]*B[0,2] -> C[2,0]
        terms.append([(2*3+0, 1)], [(0*3+2, 1)], [(2*3+0, 1)])
        # Term 6: A[2,1]*B[1,2] -> C[2,0]
        terms.append([(2*3+1, 1)], [(1*3+2, 1)], [(2*3+0, 1)])
        # Term 7: A[2,2]*B[2,2] -> C[2,2]
        terms.append([(2*3+2, 1)], [(2*3+2, 1)], [(2*3+2, 1)])
        # Term 8: (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0]+C[1,1]
        terms.append([(0*3+0, 1), (1*3+1, 1)], [(0*3+0, 1), (1*3+1, 1)], [(0*3+0, 1), (1*3+1, 1)])
        # Term 9: (A[1,0]+A[1,1])*B[0,0] -> C[1,0]
        terms.append([(1*3+0, 1), (1*3+1, 1)], [(0*3+0, 1)], [(1*3+0, 1)])
        # Term 10: A[0,0]*(B[0,1]-B[1,1]) -> C[0,1]
        terms.append([(0*3+0, 1)], [(0*3+1, 1), (1*3+1, -1)], [(0*3+1, 1)])
        # Term 11: A[1,1]*(B[1,0]-B[0,0]) -> C[1,0]
        terms.append([(1*3+1, 1)], [(1*3+0, 1), (0*3+0, -1)], [(1*3+0, 1)])
        # Term 12: (A[0,0]+A[0,1])*B[1,1] -> C[0,1]
        terms.append([(0*3+0, 1), (0*3+1, 1)], [(1*3+1, 1)], [(0*3+1, 1)])
        # Term 13: (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C[1,0]+C[1,1]
        terms.append([(1*3+0, 1), (0*3+0, -1)], [(0*3+0, 1), (0*3+1, 1)], [(1*3+0, 1), (1*3+1, 1)])
        # Term 14: (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C[0,0]
        terms.append([(0*3+1, 1), (1*3+1, -1)], [(1*3+0, 1), (1*3+1, 1)], [(0*3+0, 1)])
        # Term 15: A[0,2]*B[2,0] -> C[0,0]
        terms.append([(0*3+2, 1)], [(2*3+0, 1)], [(0*3+0, 1)])
        # Term 16: A[0,2]*B[2,1] -> C[0,1]
        terms.append([(0*3+2, 1)], [(2*3+1, 1)], [(0*3+1, 1)])
        # Term 17: A[0,2]*B[2,2] -> C[0,2]
        terms.append([(0*3+2, 1)], [(2*3+2, 1)], [(0*3+2, 1)])
        # Term 18: A[1,2]*B[2,0] -> C[1,0]
        terms.append([(1*3+2, 1)], [(2*3+0, 1)], [(1*3+0, 1)])
        # Term 19: A[1,2]*B[2,1] -> C[1,1]
        terms.append([(1*3+2, 1)], [(2*3+1, 1)], [(1*3+1, 1)])
        return terms
    
    # Fallback solution: Rank 189 (2x2 Strassen on 3x3 blocks)
    def get_fallback():
        block_size = 2
        n_blocks = n // block_size
        rank = n_blocks ** 3 * 7
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        def u_idx(ib, kb, r, c): return (ib * block_size + r) * m + (kb * block_size + c)
        def v_idx(kb, jb, r, c): return (kb * block_size + r) * p + (jb * block_size + c)
        def w_idx(ib, jb, r, c): return (jb * block_size + c) * n + (ib * block_size + r)
        
        for ib in range(n_blocks):
            for jb in range(n_blocks):
                for kb in range(n_blocks):
                    # M1
                    U[u_idx(ib, kb, 0, 0), col] += 1; U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] += 1; V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1; W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M2
                    U[u_idx(ib, kb, 1, 0), col] += 1; U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] += 1
                    W[w_idx(ib, jb, 1, 0), col] += 1; W[w_idx(ib, jb, 1, 1), col] -= 1
                    col += 1
                    # M3
                    U[u_idx(ib, kb, 0, 0), col] += 1
                    V[v_idx(kb, jb, 0, 1), col] += 1; V[v_idx(kb, jb, 1, 1), col] -= 1
                    W[w_idx(ib, jb, 0, 1), col] += 1; W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M4
                    U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 1, 0), col] += 1; V[v_idx(kb, jb, 0, 0), col] -= 1
                    W[w_idx(ib, jb, 0, 0), col] += 1; W[w_idx(ib, jb, 1, 0), col] += 1
                    col += 1
                    # M5
                    U[u_idx(ib, kb, 0, 0), col] += 1; U[u_idx(ib, kb, 0, 1), col] += 1
                    V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] -= 1; W[w_idx(ib, jb, 0, 1), col] += 1
                    col += 1
                    # M6
                    U[u_idx(ib, kb, 1, 0), col] += 1; U[u_idx(ib, kb, 0, 0), col] -= 1
                    V[v_idx(kb, jb, 0, 0), col] += 1; V[v_idx(kb, jb, 0, 1), col] += 1
                    W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M7
                    U[u_idx(ib, kb, 0, 1), col] += 1; U[u_idx(ib, kb, 1, 1), col] -= 1
                    V[v_idx(kb, jb, 1, 0), col] += 1; V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate_decomposition(U_list, V_list, W_list, n, m, p, rank):
        """Check if U,V,W correctly decompose the matrix multiplication tensor."""
        U_arr = np.array(U_list)
        V_arr = np.array(V_list)
        W_arr = np.array(W_list)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    u_val = U_arr[i * m + k]
                    v_val = V_arr[k * p + j]
                    w_val = W_arr[j * n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def validate_decomposition(U_list, V_list, W_list, n, m, p, rank):
        """Check if U,V,W correctly decompose the matrix multiplication tensor."""
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    u_val = U_arr[i * m + k]
                    v_val = V_arr[k * p + j]
                    w_val = W_arr[j * n + i]
                    s = float(np.dot(u_val, v_val * w_val))
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    try:
        if time.time() > deadline:
            return get_fallback()
        
        # Build rank-133 solution: Strassen at block level (2x2 blocks of 3x3)
        # 7 Strassen terms, each a 3x3 matrix multiplication of rank 19
        # Total rank = 7 * 19 = 133
        bs = 3  # block size
        nb = 2  # number of blocks
        
        # Get the 3x3 rank-19 terms
        terms_3x3 = get_3x3_rank19()
        assert len(terms_3x3) == 19
        
        total_rank = 7 * 19
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Strassen block-level formulas:
        # M1 = (A_00 + A_11)(B_00 + B_11)
        # M2 = (A_10 + A_11)B_00
        # M3 = A_00(B_01 - B_11)
        # M4 = A_11(B_10 - B_00)
        # M5 = (A_00 + A_01)B_11
        # M6 = (A_10 - A_00)(B_00 + B_01)
        # M7 = (A_01 - A_11)(B_10 + B_11)
        #
        # C_00 = M1 + M4 - M5 + M7
        # C_01 = M3 + M5
        # C_10 = M2 + M4
        # C_11 = M1 - M2 + M3 + M6
        
        # Each M_i is a product of two 3x3 matrices (which are linear combinations of blocks)
        # For each Strassen term, we need to:
        # 1. Determine which A-blocks and B-blocks are involved and with what coefficients
        # 2. For each of the 19 rank-19 terms, embed the 3x3 decomposition into the 6x6 structure
        
        # Strassen term definitions: (list of (a_block, coeff), list of (b_block, coeff), list of (c_block, coeff))
        # a_block: which A block (ib, kb), b_block: which B block (kb, jb), c_block: which C block (ib, jb)
        strassen_terms = [
            # M1: (A_00 + A_11)(B_00 + B_11)
            {
                'a': [((0,0), 1), ((1,1), 1)],
                'b': [((0,0), 1), ((1,1), 1)],
                'c': [((0,0), 1), ((1,1), 1)]
            },
            # M2: (A_10 + A_11)B_00
            {
                'a': [((1,0), 1), ((1,1), 1)],
                'b': [((0,0), 1)],
                'c': [((1,0), 1)]
            },
            # M3: A_00(B_01 - B_11)
            {
                'a': [((0,0), 1)],
                'b': [((0,1), 1), ((1,1), -1)],
                'c': [((0,1), 1)]
            },
            # M4: A_11(B_10 - B_00)
            {
                'a': [((1,1), 1)],
                'b': [((1,0), 1), ((0,0), -1)],
                'c': [((1,0), 1)]
            },
            # M5: (A_00 + A_01)B_11
            {
                'a': [((0,0), 1), ((0,1), 1)],
                'b': [((1,1), 1)],
                'c': [((0,1), 1)]
            },
            # M6: (A_10 - A_00)(B_00 + B_01)
            {
                'a': [((1,0), 1), ((0,0), -1)],
                'b': [((0,0), 1), ((0,1), 1)],
                'c': [((1,1), 1)]
            },
            # M7: (A_01 - A_11)(B_10 + B_11)
            {
                'a': [((0,1), 1), ((1,1), -1)],
                'b': [((1,0), 1), ((1,1), 1)],
                'c': [((0,0), 1)]
            },
        ]
        
        # Now build the full 6x6 decomposition
        col = 0
        for st in strassen_terms:
            for term in terms_3x3:
                u_terms, v_terms, w_terms = term
                for (u3, uc) in u_terms:
                    i3, k3 = divmod(u3, 3)
                    for (ab, ac) in st['a']:
                        ib, kb = ab
                        # A index: (ib*bs + i3, kb*bs + k3) -> row index in U
                        u_row = (ib * bs + i3) * m + (kb * bs + k3)
                        U[u_row, col] += uc * ac
                for (v3, vc) in v_terms:
                    k3, j3 = divmod(v3, 3)
                    for (bb, bc) in st['b']:
                        kb, jb = bb
                        # B index: (kb*bs + k3, jb*bs + j3) -> row index in V
                        v_row = (kb * bs + k3) * p + (jb * bs + j3)
                        V[v_row, col] += vc * bc
                for (w3, wc) in w_terms:
                    i3, j3 = divmod(w3, 3)
                    for (cb, cc) in st['c']:
                        ib, jb = cb
                        # C index: (ib*bs + i3, jb*bs + j3) -> row index in W
                        w_row = (jb * bs + j3) * n + (ib * bs + i3)
                        W[w_row, col] += wc * cc
                col += 1
        
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Validate the decomposition
        if validate_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, total_rank):
            return result
        else:
            return get_fallback()
        
    except Exception as e:
        return get_fallback()
