import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Now apply Strassen to 6 disjoint 2x2x2 blocks
    # We need to remove the 8 schoolbook terms for each block and add 7 Strassen terms.
    # Since we are building from scratch, it's easier to just build the final matrices directly.
    
    # Let's rebuild.
    U = []
    V = []
    W = []
    
    # Define the blocks
    blocks = [
        (0, 0, 0), # i0, j0, k0
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    # We will add terms for each block.
    # For each block, add 7 Strassen terms.
    # For the remaining indices, add schoolbook terms.
    
    # To ensure disjointness, we track which (i,j,k) are covered by Strassen.
    covered = set()
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
                    
    # 1. Add Strassen terms for each block
    for i0, j0, k0 in blocks:
        # Strassen terms for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A22-A12)(B21+B22)
        
        # We need to express these in terms of U, V, W vectors.
        # The term is u \otimes v \otimes w.
        # U part: linear combo of A_{i,j}
        # V part: linear combo of B_{j,k}
        # W part: linear combo of C_{i,k} (actually just the coefficient for the output)
        
        # Let's define the 7 terms explicitly.
        # Indices: i in {0,1}, j in {0,1}, k in {0,1} relative to block.
        
        # Term 1: (A11+A22)(B11+B22)
        # U: [1,0,0,1] (for A11, A12, A21, A22) -> Global U indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        # V: [1,1,0,0] (for B11, B12, B21, B22) -> Global V indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
        # W: [1,0,0,0] (for C11, C12, C21, C22) -> Global W indices: (k0,i0), (k0+1,i0), (k0,i0+1), (k0+1,i0+1)
        # Wait, W index is k*n+i.
        # C11 -> (k0, i0)
        # C12 -> (k0+1, i0)
        # C21 -> (k0, i0+1)
        # C22 -> (k0+1, i0+1)
        
        # We need to add these to U, V, W.
        # Since we are building a list, we can just append.
        
        # Helper to add a term
        def add_term(u_vec, v_vec, w_vec):
            # u_vec: dict {global_u_idx: val}
            # v_vec: dict {global_v_idx: val}
            # w_vec: dict {global_w_idx: val}
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            
        # Define indices
        i1, i2 = i0, i0+1
        j1, j2 = j0, j0+1
        k1, k2 = k0, k0+1
        
        # U indices
        u11 = i1 * m + j1
        u12 = i1 * m + j2
        u21 = i2 * m + j1
        u22 = i2 * m + j2
        
        # V indices
        v11 = j1 * p + k1
        v12 = j1 * p + k2
        v21 = j2 * p + k1
        v22 = j2 * p + k2
        
        # W indices
        w11 = k1 * n + i1
        w12 = k2 * n + i1
        w21 = k1 * n + i2
        w22 = k2 * n + i2
        
        # Term 1: M1
        u = {u11: 1, u22: 1}
        v = {v11: 1, v22: 1}
        w = {w11: 1, w12: 1, w21: 1, w22: 1} # Wait, W is just the coefficient for C.
        # In Strassen, the result C is formed from M's.
        # C11 = M1 + M4 - M5 + M7
        # So for M1, the contribution to C11 is 1.
        # Does M1 contribute to other C's? No, M1 is a scalar.
        # The decomposition is T = sum u \otimes v \otimes w.
        # The term u \otimes v \otimes w contributes to T_{i,j,k} via u_i v_j w_k.
        # Here w is the vector for the output C.
        # So for M1, w should be the vector that distributes M1 to the C's.
        # M1 contributes to C11, C12, C21, C22?
        # No, M1 is a single multiplication.
        # The formula is C = sum_c M_c (linear combo of C basis).
        # So w_c is the vector of coefficients for C.
        # For M1: w = [1, 0, 0, 0]? No.
        # C11 = M1 + ...
        # So M1 contributes 1 to C11.
        # Does it contribute to C12? No.
        # So w1 = [1, 0, 0, 0] in local basis (C11, C12, C21, C22).
        
        # Let's re-verify Strassen.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M1: w = [1, 0, 0, 1] (C11, C12, C21, C22)
        # M2: w = [0, 0, 1, -1]
        # M3: w = [0, 1, 0, 1]
        # M4: w = [1, 0, 1, 0]
        # M5: w = [-1, 1, 0, 0]
        # M6: w = [0, 0, 0, 1]
        # M7: w = [1, 0, 0, 0]
        
        # Let's add these 7 terms.
        
        # M1
        u = {u11: 1, u22: 1}
        v = {v11: 1, v22: 1}
        w = {w11: 1, w22: 1}
        add_term(u, v, w)
        
        # M2
        u = {u21: 1, u22: 1}
        v = {v11: 1}
        w = {w21: 1, w22: -1}
        add_term(u, v, w)
        
        # M3
        u = {u11: 1}
        v = {v12: 1, v22: -1}
        w = {w12: 1, w22: 1}
        add_term(u, v, w)
        
        # M4
        u = {u22: 1}
        v = {v21: 1, v11: -1}
        w = {w11: 1, w21: 1}
        add_term(u, v, w)
        
        # M5
        u = {u11: 1, u12: 1}
        v = {v22: 1}
        w = {w11: -1, w12: 1}
        add_term(u, v, w)
        
        # M6
        u = {u21: 1, u11: -1}
        v = {v11: 1, v12: 1}
        w = {w22: 1}
        add_term(u, v, w)
        
        # M7
        u = {u22: 1, u12: -1}
        v = {v21: 1, v22: 1}
        w = {w11: 1}
        add_term(u, v, w)
        
    # 2. Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in covered:
                    u = {i * m + j: 1}
                    v = {j * p + k: 1}
                    w = {k * n + i: 1}
                    add_term(u, v, w)
                    
    # Convert lists of dicts to matrices
    final_rank = len(U)
    U_mat = np.zeros((n * m, final_rank))
    V_mat = np.zeros((m * p, final_rank))
    W_mat = np.zeros((p * n, final_rank))
    
    for c in range(final_rank):
        for idx, val in U[c].items():
            U_mat[idx, c] = val
        for idx, val in V[c].items():
            V_mat[idx, c] = val
        for idx, val in W[c].items():
            W_mat[idx, c] = val
            
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
