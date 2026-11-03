"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Block structure:
    # A (3x4): rows [2,1], cols [2,2]
    # B (4x6): rows [2,2], cols [2,2,2]
    # C (3x6): rows [2,1], cols [2,2,2]
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A01)*(B00+B10)
    # M7 = (A01-A11)*(B11+B01)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    terms = []  # Each term: (U_dict, V_dict, W_dict)
    
    def add_term(U_entries, V_entries, W_entries):
        U_d = {}
        for (i,j,c) in U_entries:
            if c != 0:
                U_d[(i,j)] = U_d.get((i,j), 0) + c
        V_d = {}
        for (j,k,c) in V_entries:
            if c != 0:
                V_d[(j,k)] = V_d.get((j,k), 0) + c
        W_d = {}
        for (k,i,c) in W_entries:
            if c != 0:
                W_d[(k,i)] = W_d.get((k,i), 0) + c
        terms.append((U_d, V_d, W_d))
    
    def strassen_terms(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        r0, r1 = A_rows
        c0, c1 = A_cols
        s0, s1 = B_rows
        t0, t1 = B_cols
        
        # M1 = (A[r0,c0]+A[r1,c1]) * (B[s0,t0]+B[s1,t1])
        add_term(
            [(r0,c0,1), (r1,c1,1)],
            [(s0,t0,1), (s1,t1,1)],
            [(t0,r0,1), (t1,r1,1)]
        )
        
        # M2 = (A[r1,c0]+A[r1,c1]) * B[s0,t0]
        add_term(
            [(r1,c0,1), (r1,c1,1)],
            [(s0,t0,1)],
            [(t0,r1,1), (t1,r1,-1)]
        )
        
        # M3 = A[r0,c0] * (B[s0,t1]-B[s1,t1])
        add_term(
            [(r0,c0,1)],
            [(s0,t1,1), (s1,t1,-1)],
            [(t1,r0,1), (t1,r1,1)]
        )
        
        # M4 = A[r1,c1] * (B[s1,t0]-B[s0,t0])
        add_term(
            [(r1,c1,1)],
            [(s1,t0,1), (s0,t0,-1)],
            [(t0,r1,1), (t0,r0,1)]
        )
        
        # M5 = (A[r0,c0]+A[r0,c1]) * B[s1,t1]
        add_term(
            [(r0,c0,1), (r0,c1,1)],
            [(s1,t1,1)],
            [(t1,r1,1), (t0,r0,-1)]
        )
        
        # M6 = (A[r1,c1]-A[r0,c1]) * (B[s0,t0]+B[s0,t1])
        add_term(
            [(r1,c1,1), (r0,c1,-1)],
            [(s0,t0,1), (s0,t1,1)],
            [(t1,r1,1)]
        )
        
        # M7 = (A[r0,c1]-A[r1,c1]) * (B[s1,t1]+B[s0,t1])
        add_term(
            [(r0,c1,1), (r1,c1,-1)],
            [(s1,t1,1), (s0,t1,1)],
            [(t0,r0,1)]
        )
    
    def direct_1x2_2x2_terms(A_row, A_cols, B_rows, B_cols, C_row, C_cols):
        r = A_row
        c0, c1 = A_cols
        s0, s1 = B_rows
        t0, t1 = B_cols
        
        add_term([(r,c0,1)], [(s0,t0,1)], [(t0,r,1)])
        add_term([(r,c1,1)], [(s1,t0,1)], [(t0,r,1)])
        add_term([(r,c0,1)], [(s0,t1,1)], [(t1,r,1)])
        add_term([(r,c1,1)], [(s1,t1,1)], [(t1,r,1)])
    
    col_groups = [(0,1), (2,3), (4,5)]
    
    for ci, (t0, t1) in enumerate(col_groups):
        strassen_terms((0,1), (0,1), (0,1), (t0,t1), (0,1), (t0,t1))
        strassen_terms((0,1), (2,3), (2,3), (t0,t1), (0,1), (t0,t1))
        direct_1x2_2x2_terms(2, (0,1), (0,1), (t0,t1), 2, (t0,t1))
        direct_1x2_2x2_terms(2, (2,3), (2,3), (t0,t1), 2, (t0,t1))
    
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (U_d, V_d, W_d) in enumerate(terms):
        for (i, j), coeff in U_d.items():
            U[i * m + j, c] = coeff
        for (j, k), coeff in V_d.items():
            V[j * p + k, c] = coeff
        for (k, i), coeff in W_d.items():
            W[k * n + i, c] = coeff
    
    def verify(U_mat, V_mat, W_mat, r):
        T_reconstructed = np.zeros((n, m, p))
        for c in range(r):
            u = U_mat[:, c].reshape(n, m)
            v = V_mat[:, c].reshape(m, p)
            w = W_mat[:, c].reshape(p, n)
            T_reconstructed += u[:, :, np.newaxis] * v[np.newaxis, :, :] * w[np.newaxis, np.newaxis, :]
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)
    
    if not verify(U, V, W, rank):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = rank_school
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
