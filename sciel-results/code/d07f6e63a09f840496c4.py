"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 145.0 # 5s margin
    
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to get index for U (A is n x m)
    def u_idx(i, j):
        return i * m + j
        
    # Helper to get index for V (B is m x p)
    def v_idx(j, k):
        return j * p + k
        
    # Helper to get index for W (C is n x p, trace convention W[k*n+i])
    def w_idx(i, k):
        return k * n + i

    # Represent a 2x2 matrix of linear forms as a 2x2 array of dicts {global_idx: coeff}
    # Or simpler: just lists of (row, col, coeff)
    
    # Define Outer Strassen Terms
    # Each term is (X, Y) where X, Y are 2x2 matrices of linear forms.
    # Linear form for A entry (r, c) is represented as (r, c, coeff)
    # Linear form for B entry (r, c) is represented as (r, c, coeff)
    
    def make_block_A(br, bc):
        """Returns 2x2 matrix of single A entries for block (br, bc)"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append([(br*2+i, bc*2+j, 1.0)])
            mat.append(row)
        return mat

    def make_block_B(br, bc):
        """Returns 2x2 matrix of single B entries for block (br, bc)"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append([(br*2+i, bc*2+j, 1.0)])
            mat.append(row)
            
    def add_mat(X, Y):
        """Add two 2x2 matrices of linear forms"""
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d = {}
                for (r, c, v) in X[i][j]:
                    d[(r,c)] = d.get((r,c), 0.0) + v
                for (r, c, v) in Y[i][j]:
                    d[(r,c)] = d.get((r,c), 0.0) + v
                res[i][j] = [(r, c, v) for (r,c), v in d.items() if v != 0.0]
        return res

    def sub_mat(X, Y):
        """Subtract Y from X"""
        neg_Y = [[[(r, c, -v) for (r,c,v) in Y[i][j]] for j in range(2)] for i in range(2)]
        return add_mat(X, neg_Y)

    # Outer Blocks
    A00 = make_block_A(0, 0)
    A01 = make_block_A(0, 1)
    A10 = make_block_A(1, 0)
    A11 = make_block_A(1, 1)
    
    B00 = make_block_B(0, 0)
    B01 = make_block_B(0, 1)
    B10 = make_block_B(1, 0)
    B11 = make_block_B(1, 1)

    # Outer Terms X_t, Y_t
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    outer_X = [
        add_mat(A00, A11),
        add_mat(A10, A11),
        A00,
        A11,
        add_mat(A00, A01),
        sub_mat(A10, A00),
        sub_mat(A01, A11)
    ]
    
    outer_Y = [
        add_mat(B00, B11),
        B00,
        sub_mat(B01, B11),
        sub_mat(B10, B00),
        B11,
        add_mat(B00, B01),
        add_mat(B10, B11)
    ]
    
    # Outer Contributions to C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # c_targets[t] = list of (block_row, block_col, coeff)
    c_targets = [
        [(0,0,1.0), (1,1,1.0)],       # M1
        [(1,0,1.0), (1,1,-1.0)],      # M2
        [(0,1,1.0), (1,1,1.0)],       # M3
        [(0,0,1.0), (1,0,1.0)],       # M4
        [(0,0,-1.0), (0,1,1.0)],      # M5
        [(1,1,1.0)],                  # M6
        [(0,0,1.0)]                   # M7
    ]
    
    # Inner Strassen Targets
    # For a 2x2 product Z = X Y
    # Z00 = N1 + N4 - N5 + N7
    # Z01 = N3 + N5
    # Z10 = N2 + N4
    # Z11 = N1 - N2 + N3 + N6
    # inner_targets[k] = list of (local_row, local_col, coeff)
    inner_targets = [
        [(0,0,1.0), (1,1,1.0)],       # N1
        [(1,0,1.0), (1,1,-1.0)],      # N2
        [(0,1,1.0), (1,1,1.0)],       # N3
        [(0,0,1.0), (1,0,1.0)],       # N4
        [(0,0,-1.0), (0,1,1.0)],      # N5
        [(1,1,1.0)],                  # N6
        [(0,0,1.0)]                   # N7
    ]
    
    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]
        
        # Define Inner Strassen factors for X, Y
        # N1 = (X00+X11)(Y00+Y11)
        # N2 = (X10+X11)Y00
        # N3 = X00(Y01-Y11)
        # N4 = X11(Y10-Y00)
        # N5 = (X00+X01)Y11
        # N6 = (X10-X00)(Y00+Y01)
        # N7 = (X01-X11)(Y10+Y11)
        
        # Helper to extract specific entries and combine
        def get_entry(mat, i, j):
            return mat[i][j]
            
        def combine_entries(mat, indices, signs):
            """
            indices: list of (i, j)
            signs: list of +1 or -1
            Returns list of (r, c, coeff)
            """
            d = {}
            for idx, sign in zip(indices, signs):
                i, j = idx
                for (r, c, v) in mat[i][j]:
                    key = (r, c)
                    d[key] = d.get(key, 0.0) + sign * v
            return [(r, c, v) for (r,c), v in d.items() if v != 0.0]
        
        # X factors for N1..N7
        X_N1 = combine_entries(X, [(0,0), (1,1)], [1.0, 1.0])
        X_N2 = combine_entries(X, [(1,0), (1,1)], [1.0, 1.0])
        X_N3 = combine_entries(X, [(0,0)], [1.0])
        X_N4 = combine_entries(X, [(1,1)], [1.0])
        X_N5 = combine_entries(X, [(0,0), (0,1)], [1.0, 1.0])
        X_N6 = combine_entries(X, [(1,0), (0,0)], [1.0, -1.0])
        X_N7 = combine_entries(X, [(0,1), (1,1)], [1.0, -1.0])
        
        # Y factors for N1..N7
        Y_N1 = combine_entries(Y, [(0,0), (1,1)], [1.0, 1.0])
        Y_N2 = combine_entries(Y, [(0,0)], [1.0])
        Y_N3 = combine_entries(Y, [(0,1), (1,1)], [1.0, -1.0])
        Y_N4 = combine_entries(Y, [(1,0), (0,0)], [1.0, -1.0])
        Y_N5 = combine_entries(Y, [(1,1)], [1.0])
        Y_N6 = combine_entries(Y, [(0,0), (0,1)], [1.0, 1.0])
        Y_N7 = combine_entries(Y, [(1,0), (1,1)], [1.0, 1.0])
        
        inner_X_factors = [X_N1, X_N2, X_N3, X_N4, X_N5, X_N6, X_N7]
        inner_Y_factors = [Y_N1, Y_N2, Y_N3, Y_N4, Y_N5, Y_N6, Y_N7]
        
        for k in range(7):
            # Fill U
            for (r, c, v) in inner_X_factors[k]:
                U[u_idx(r, c), col] += v
                
            # Fill V
            for (r, c, v) in inner_Y_factors[k]:
                V[v_idx(r, c), col] += v
                
            # Fill W
            # The product N_k contributes to Z (which is M_t)
            # Z[li, lj] gets contribution inner_targets[k]
            # M_t[li, lj] is at global block (br, bc) from c_targets[t]
            # Global C index: i = br*2 + li, k_col = bc*2 + lj
            
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k), col] += c_coeff * z_coeff
            
            col += 1
            
            if time.time() > _DEADLINE:
                # Should not happen for fixed size, but safety
                break
    
    assert col == rank, f"Rank mismatch: {col}"
    
    # Verification
    # Use integer matrices to ensure exactness
    # If verification fails, return naive (rank 64) as fallback to avoid 0 score
    # But rank 49 is the goal.
    
    if time.time() < _DEADLINE - 2.0:
        for _ in range(5):
            # Random integer matrices
            A_test = np.random.randint(-10, 10, (n, m)).astype(float)
            B_test = np.random.randint(-10, 10, (m, p)).astype(float)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_r = U[:, r]
                v_r = V[:, r]
                w_r = W[:, r]
                
                f_r = np.sum(u_r.reshape(n, m) * A_test)
                g_r = np.sum(v_r.reshape(m, p) * B_test)
                
                # w_r is length p*n, index k*n+i
                # C[i, k] += w_r[k*n+i] * f_r * g_r
                # Vectorized:
                # w_r reshaped to (p, n) is w_r[k, i]
                # We need C[i, k]. 
                # C += f_r * g_r * (w_r.reshape(p, n)).T
                C_computed += f_r * g_r * w_r.reshape(p, n).T
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                # Fallback to naive
                U_naive = np.zeros((n*m, n*m*p))
                V_naive = np.zeros((m*p, n*m*p))
                W_naive = np.zeros((p*n, n*m*p))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_naive[i*m+j, idx] = 1.0
                            V_naive[j*p+k, idx] = 1.0
                            W_naive[k*n+i, idx] = 1.0
                            idx += 1
                return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
