import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    def sanitize_matrix(M):
        """
        Ensures all entries are exactly integers or half-integers.
        Multiplies by 2, rounds to nearest integer, divides by 2.
        """
        M_2 = M * 2.0
        M_rounded = np.round(M_2)
        M_clean = M_rounded / 2.0
        return M_clean

    def verify(U, V, W, tol=1e-9):
        """
        Verifies that the decomposition exactly reproduces the matrix multiplication tensor.
        U: (n*m, rank) = (4, rank)
        V: (m*p, rank) = (18, rank)
        W: (p*n, rank) = (18, rank)
        
        The tensor T[i, j, k] = 1 if we are computing (A*B)[i,k] from A[i,j] and B[j,k].
        Indices:
        U index: i*m + j = i*2 + j
        V index: j*p + k = j*9 + k
        W index: k*n + i = k*2 + i
        
        We need sum_t U[u_idx, t] * V[v_idx, t] * W[w_idx, t] == 1 for all i,j,k.
        """
        rank = U.shape[1]
        # Vectorized check
        # Reshape U, V, W to (dim1, dim2, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # This is a tensor contraction.
        # Let's do it step by step to be safe and fast enough for rank ~30.
        # Intermediate: C[i, k, j, t] = U[i, j, t] * V[j, k, t] ? No.
        
        # Efficient check:
        # For each t, we have vectors u_t (4), v_t (18), w_t (18).
        # We want to check if sum_t (u_t[i,j] * v_t[j,k] * w_t[k,i]) == 1.
        
        # Let's construct the tensor T from the decomposition.
        # T[i, j, k] = sum_t U[i*2+j, t] * V[j*9+k, t] * W[k*2+i, t]
        
        # We can compute this by iterating over t and accumulating into a 2x2x9 array.
        T_calc = np.zeros((n, m, p))
        for t in range(rank):
            u_t = U[:, t]
            v_t = V[:, t]
            w_t = W[:, t]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_calc[i, j, k] += u_t[i*m + j] * v_t[j*p + k] * w_t[k*n + i]
                        
        # Target tensor T_target[i, j, k] = 1 for all i, j, k
        T_target = np.ones((n, m, p))
        
        if not np.allclose(T_calc, T_target, atol=tol):
            return False
        return True

    def build_rank32():
        """
        Build a rank-32 decomposition.
        Uses Strassen (rank 7) for 4 blocks of 2 columns each (columns 0-1, 2-3, 4-5, 6-7)
        and schoolbook (rank 4) for the last column (column 8).
        Total Rank = 7*4 + 4 = 32.
        """
        terms = []
        
        def add_strassen_block(k0, k1):
            # Standard Strassen decomposition for M_{2,2,2}
            # U indices: 0,1,2,3 (i*2+j)
            # V indices: j*2+k_local (0..3) -> mapped to global j*9 + k
            # W indices: k_local*2+i (0..3) -> mapped to global k*2 + i
            
            # The 7 Strassen terms for C = A*B where A, B, C are 2x2.
            # A = [[a00, a01], [a10, a11]]
            # B = [[b00, b01], [b10, b11]]
            # C = [[c00, c01], [c10, c11]]
            
            # k0, k1 are the global column indices corresponding to b00/b10 and b01/b11
            # So for the sub-matrix B_sub:
            # B_sub[0,0] = B[0, k0] -> V index 0*9 + k0 = k0
            # B_sub[0,1] = B[0, k1] -> V index 0*9 + k1 = k1
            # B_sub[1,0] = B[1, k0] -> V index 1*9 + k0 = 9 + k0
            # B_sub[1,1] = B[1, k1] -> V index 1*9 + k1 = 9 + k1
            
            # W indices for C_sub:
            # C_sub[0,0] = C[0, k0] -> W index k0*2 + 0 = 2*k0
            # C_sub[0,1] = C[0, k1] -> W index k1*2 + 0 = 2*k1
            # C_sub[1,0] = C[1, k0] -> W index k0*2 + 1 = 2*k0 + 1
            # C_sub[1,1] = C[1, k1] -> W index k1*2 + 1 = 2*k1 + 1
            
            # Strassen Formulas:
            # M1 = (a00 + a11)(b00 + b11)
            # M2 = (a10 + a11)b00
            # M3 = a00(b01 - b11)
            # M4 = a11(b10 - b00)
            # M5 = (a00 + a01)b11
            # M6 = (a10 - a00)(b00 + b01)
            # M7 = (a01 - a11)(b10 + b11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # We define terms as (U_dict, V_dict, W_dict)
            
            # Term 1: M1
            # U: a00, a11 -> indices 0, 3
            # V: b00, b11 -> indices k0, 9+k1
            # W: c00, c11 -> indices 2*k0, 2*k1+1
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            
            # Term 2: M2
            # U: a10, a11 -> indices 2, 3
            # V: b00 -> index k0
            # W: c10, c11 (with -1 for c11) -> indices 2*k0+1, 2*k1+1
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            
            # Term 3: M3
            # U: a00 -> index 0
            # V: b01, b11 (with -1 for b11) -> indices k1, 9+k1
            # W: c01, c11 -> indices 2*k1, 2*k1+1
            terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
            
            # Term 4: M4
            # U: a11 -> index 3
            # V: b10, b00 (with -1 for b00) -> indices 9+k0, k0
            # W: c10, c11 (Wait, M4 contributes to C00 and C10)
            # C00 = ... + M4 ...
            # C10 = ... + M4 ...
            # So W indices: 2*k0 (C00), 2*k0+1 (C10)
            terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
            
            # Term 5: M5
            # U: a00, a01 -> indices 0, 1
            # V: b11 -> index 9+k1
            # W: C00 (with -1), C01 -> indices 2*k0, 2*k1
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
            
            # Term 6: M6
            # U: a10, a00 (with -1) -> indices 2, 0
            # V: b00, b01 -> indices k0, k1
            # W: C11 -> index 2*k1+1
            terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
            
            # Term 7: M7
            # U: a01, a11 (with -1) -> indices 1, 3
            # V: b10, b11 -> indices 9+k0, 9+k1
            # W: C00 -> index 2*k0
            terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))

        # Apply Strassen to column pairs (0,1), (2,3), (4,5), (6,7)
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            add_strassen_block(k0, k1)
        
        # Handle last column (k=8) with schoolbook
        # C[i, 8] = sum_j A[i, j] B[j, 8]
        # Terms:
        # 1. A[0,0] * B[0,8] -> C[0,8]
        # 2. A[0,1] * B[1,8] -> C[0,8]
        # 3. A[1,0] * B[0,8] -> C[1,8]
        # 4. A[1,1] * B[1,8] -> C[1,8]
        
        k = 8
        # U indices: 0, 1, 2, 3
        # V indices: 0*9+8=8, 1*9+8=17
        # W indices: 8*2+0=16, 8*2+1=17
        
        terms.append(({0: 1}, {8: 1}, {16: 1}))
        terms.append(({1: 1}, {17: 1}, {16: 1}))
        terms.append(({2: 1}, {8: 1}, {17: 1}))
        terms.append(({3: 1}, {17: 1}, {17: 1}))

        rank = len(terms)
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    def build_rank63():
        """
        Safe fallback: Rank 63 decomposition.
        Tensor product of M_{2,2,2} (rank 7? No, schoolbook rank 4 is simpler for fallback? 
        Actually, the prompt suggests rank 63 from 2x1x9 tensor.
        Let's just use schoolbook for everything: Rank = 2*2*9 = 36.
        Wait, schoolbook is rank 36. 
        Why did the prompt say 63? 
        "tensor product of the standard 2x2x2 multiplication (rank 7) and the 2x1x9 'vector-matrix' multiplication tensor"
        Rank 7 * Rank 9 = 63.
        
        Let's implement Schoolbook Rank 36 as a safer, lower rank fallback than 63.
        """
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    u_idx = i*2 + j
                    v_idx = j*9 + k
                    w_idx = k*2 + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
        rank = len(terms)
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    # Try rank 32 first
    U, V, W = build_rank32()
    U = sanitize_matrix(U)
    V = sanitize_matrix(V)
    W = sanitize_matrix(W)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank 32 fails (shouldn't happen), try rank 36 (schoolbook)
    if time.time() > deadline:
        # If we're out of time, we have to return something. 
        # Returning an invalid result scores 0. 
        # Returning a valid result scores > 0.
        # We haven't verified rank 36 yet. 
        # But rank 32 construction is deterministic and fast. 
        # If verify failed, it's a bug in my code.
        pass

    U36, V36, W36 = build_rank63() # This is actually rank 36
    U36 = sanitize_matrix(U36)
    V36 = sanitize_matrix(V36)
    W36 = sanitize_matrix(W36)
    
    if verify(U36, V36, W36):
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    # If both fail, raise error (scores 0)
    raise ValueError("Failed to construct valid decomposition")

# EVOLVE-BLOCK-END
