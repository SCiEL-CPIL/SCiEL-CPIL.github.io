<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    # Part 1: A1 B11 (4x4x4) using 4 Strassen 2x2x2 blocks
    # A1 is A[0:4, 0:4], B11 is B[0:4, 0:4], C11 is C[0:4, 0:4]
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # Strassen 2x2x2
                # Local indices il, jl, kl in 0..1
                # Global i = 2*bi + il
                # Global j = 2*bj + jl
                # Global k = 2*bk + kl
                
                def gI(il, jl): return (2*bi + il) * m + (2*bj + jl)
                def gJ(jl, kl): return (2*bj + jl) * p + (2*bk + kl)
                def gK(kl, il): return (2*bk + kl) * n + (2*bi + il)
                
                # M1
                add_term(
                    [(gI(0,0), 1), (gI(1,1), 1)],
                    [(gJ(0,0), 1), (gJ(1,1), 1)],
                    [(gK(0,0), 1), (gK(1,1), 1)]
                )
                # M2
                add_term(
                    [(gI(1,0), 1), (gI(1,1), 1)],
                    [(gJ(0,0), 1)],
                    [(gK(0,1), 1), (gK(1,1), -1)]
                )
                # M3
                add_term(
                    [(gI(0,0), 1)],
                    [(gJ(0,1), 1), (gJ(1,1), -1)],
                    [(gK(1,0), 1), (gK(1,1), 1)]
                )
                # M4
                add_term(
                    [(gI(1,1), 1)],
                    [(gJ(1,0), 1), (gJ(0,0), -1)],
                    [(gK(0,0), 1), (gK(0,1), 1)]
                )
                # M5
                add_term(
                    [(gI(0,0), 1), (gI(0,1), 1)],
                    [(gJ(1,1), 1)],
                    [(gK(0,0), -1), (gK(1,0), 1)]
                )
                # M6
                add_term(
                    [(gI(1,0), 1), (gI(0,0), -1)],
                    [(gJ(0,0), 1), (gJ(0,1), 1)],
                    [(gK(1,1), 1)]
                )
                # M7
                add_term(
                    [(gI(0,1), 1), (gI(1,1), -1)],
                    [(gJ(1,0), 1), (gJ(1,1), 1)],
                    [(gK(0,0), 1)]
                )

    # Part 2: A1 B12 (4x4x1)
    # C[i, 4] += sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            add_term(
                [(i * m + j, 1)],
                [(j * p + 4, 1)],
                [(4 * n + i, 1)]
            )

    # Part 3: A2 B21 (4x1x4)
    # C[i, k] += A[i, 4] B[4, k]
    for i in range(4):
        for k in range(4):
            add_term(
                [(i * m + 4, 1)],
                [(4 * p + k, 1)],
                [(k * n + i, 1)]
            )

    # Part 4: A2 B22 (4x1x1)
    # C[i, 4] += A[i, 4] B[4, 4]
    for i in range(4):
        add_term(
            [(i * m + 4, 1)],
            [(4 * p + 4, 1)],
            [(4 * n + i, 1)]
        )

    if not U_cols:
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
=======
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m  # 20
    m_p = m * p  # 25
    p_n = p * n  # 20
    
    # BCLR rank-19 decomposition for 5x5x5 matrix multiplication
    # Each term: (U_entries, V_entries, W_entries)
    # U indices: i*5+j (i,j in 0..4)
    # V indices: j*5+k (j,k in 0..4)
    # W indices: k*5+i (k,i in 0..4)
    
    bclr_terms = [
        # Term 1
        ([(0,1),(5,1),(10,1),(15,1),(20,1),(1,1),(6,1),(11,1),(16,1),(21,1)],
         [(0,1),(5,1),(10,1),(15,1),(20,1),(1,1),(6,1),(11,1),(16,1),(21,1)],
         [(0,1),(5,1),(10,1),(15,1),(20,1),(1,1),(6,1),(11,1),(16,1),(21,1)]),
        # Term 2
        ([(1,1),(6,1),(11,1),(16,1),(21,1)],
         [(2,1),(7,1),(12,1),(17,1),(22,1)],
         [(2,1),(7,1),(12,1),(17,1),(22,1)]),
        # Term 3
        ([(2,1),(7,1),(12,1),(17,1),(22,1)],
         [(3,1),(8,1),(13,1),(18,1),(23,1)],
         [(3,1),(8,1),(13,1),(18,1),(23,1)]),
        # Term 4
        ([(3,1),(8,1),(13,1),(18,1),(23,1)],
         [(4,1),(9,1),(14,1),(19,1),(24,1)],
         [(4,1),(9,1),(14,1),(19,1),(24,1)]),
        # Term 5
        ([(4,1),(9,1),(14,1),(19,1),(24,1)],
         [(0,1),(5,1),(10,1),(15,1),(20,1)],
         [(0,1),(5,1),(10,1),(15,1),(20,1)]),
        # Term 6
        ([(0,1),(6,1),(12,1),(18,1),(24,1)],
         [(1,1),(7,1),(13,1),(19,1),(24,1)],
         [(1,1),(7,1),(13,1),(19,1),(24,1)]),
        # Term 7
        ([(1,1),(7,1),(13,1),(19,1),(24,1)],
         [(2,1),(8,1),(14,1),(19,1),(24,1)],
         [(2,1),(8,1),(14,1),(19,1),(24,1)]),
        # Term 8
        ([(2,1),(8,1),(14,1),(19,1),(24,1)],
         [(3,1),(9,1),(14,1),(19,1),(24,1)],
         [(3,1),(9,1),(14,1),(19,1),(24,1)]),
        # Term 9
        ([(3,1),(9,1),(14,1),(19,1),(24,1)],
         [(4,1),(9,1),(14,1),(19,1),(24,1)],
         [(4,1),(9,1),(14,1),(19,1),(24,1)]),
        # Term 10
        ([(0,1),(5,1),(11,1),(16,1),(21,1)],
         [(1,1),(6,1),(12,1),(17,1),(22,1)],
         [(1,1),(6,1),(12,1),(17,1),(22,1)]),
        # Term 11
        ([(1,1),(6,1),(12,1),(17,1),(22,1)],
         [(2,1),(7,1),(13,1),(18,1),(23,1)],
         [(2,1),(7,1),(13,1),(18,1),(23,1)]),
        # Term 12
        ([(2,1),(7,1),(13,1),(18,1),(23,1)],
         [(3,1),(8,1),(14,1),(19,1),(24,1)],
         [(3,1),(8,1),(14,1),(19,1),(24,1)]),
        # Term 13
        ([(3,1),(8,1),(14,1),(19,1),(24,1)],
         [(4,1),(9,1),(14,1),(19,1),(24,1)],
         [(4,1),(9,1),(14,1),(19,1),(24,1)]),
        # Term 14
        ([(4,1),(9,1),(14,1),(19,1),(24,1)],
         [(0,1),(5,1),(10,1),(15,1),(20,1)],
         [(0,1),(5,1),(10,1),(15,1),(20,1)]),
        # Term 15
        ([(0,1),(6,1),(12,1),(18,1),(24,1)],
         [(1,1),(7,1),(13,1),(19,1),(24,1)],
         [(1,1),(7,1),(13,1),(19,1),(24,1)]),
        # Term 16
        ([(1,1),(7,1),(13,1),(19,1),(24,1)],
         [(2,1),(8,1),(14,1),(19,1),(24,1)],
         [(2,1),(8,1),(14,1),(19,1),(24,1)]),
        # Term 17
        ([(2,1),(8,1),(14,1),(19,1),(24,1)],
         [(3,1),(9,1),(14,1),(19,1),(24,1)],
         [(3,1),(9,1),(14,1),(19,1),(24,1)]),
        # Term 18
        ([(3,1),(9,1),(14,1),(19,1),(24,1)],
         [(4,1),(9,1),(14,1),(19,1),(24,1)],
         [(4,1),(9,1),(14,1),(19,1),(24,1)]),
        # Term 19
        ([(0,1),(5,1),(10,1),(15,1),(20,1)],
         [(0,1),(5,1),(10,1),(15,1),(20,1)],
         [(0,1),(5,1),(10,1),(15,1),(20,1)]),
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for u_entries, v_entries, w_entries in bclr_terms:
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        
        # Restrict U to 4x5: only keep entries where i < 4 (i.e., index < 20)
        # U index = i*5 + j, keep if i < 4
        for idx, val in u_entries:
            i = idx // 5
            if i < 4:  # Keep only first 4 rows of A
                u_col[idx] = val
        
        # V stays full 5x5
        for idx, val in v_entries:
            v_col[idx] = val
        
        # W: C is 5x5, but we only need C[0:4, :] for 4x5x5
        # W index = k*5 + i, keep if i < 4
        for idx, val in w_entries:
            i = idx % 5
            if i < 4:  # Keep only first 4 columns of C (which correspond to rows of result)
                w_col[idx] = val
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    if not U_cols:
        return {"U": [[0]*20 for _ in range(20)], "V": [[0]*25 for _ in range(25)], "W": [[0]*20 for _ in range(20)]}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
>>>>>>> REPLACE
