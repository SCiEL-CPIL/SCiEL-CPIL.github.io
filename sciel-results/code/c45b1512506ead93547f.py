import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def construct():
    # Try to find a better decomposition for M_{3,2,2}
    # If found, use it to build M_{3,4,8}
    # Otherwise, use schoolbook
    
    deadline = time.time() + 110  # Leave 10s margin
    
    def find_decomp_322():
        # Target tensor for M_{3,2,2}
        # n=3, m=2, p=2
        # U shape (6, r), V shape (4, r), W shape (6, r)
        # Target: sum of u_c v_c w_c^T? No.
        # The tensor T is such that T[i*m+j, j*p+k, k*n+i] = 1?
        # No, the decomposition is T = sum u_c \otimes v_c \otimes w_c.
        # The entries of T are 1 at (i*m+j, j*p+k, k*n+i) for all i,j,k.
        # Here n=3, m=2, p=2.
        # Indices:
        # U: i*m+j -> 0..5
        # V: j*p+k -> 0..3
        # W: k*n+i -> 0..5
        
        # We want to find U, V, W of rank r such that 
        # sum_c U[:,c] \otimes V[:,c] \otimes W[:,c] == T_target
        
        # T_target is a 6x4x6 tensor.
        # T_target[a, b, c] = 1 if there exists i,j,k such that:
        # a = i*2+j, b = j*2+k, c = k*3+i
        # else 0.
        
        T_target = np.zeros((6, 4, 6), dtype=int)
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 3 + i
                    T_target[a, b, c] = 1
        
        # Try ranks 9, 10, 11
        for r in [9, 10, 11]:
            # Local search
            best_err = np.inf
            best_U = None
            best_V = None
            best_W = None
            
            # Initialize with random integers in [-2, 2]
            for _ in range(50):
                U = np.random.randint(-2, 3, (6, r))
                V = np.random.randint(-2, 3, (4, r))
                W = np.random.randint(-2, 3, (6, r))
                
                # Compute residual
                # T_sum = sum_c U[:,c] \otimes V[:,c] \otimes W[:,c]
                # This is equivalent to:
                # T_sum[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
                # We can use einsum
                T_sum = np.einsum('ak,bk,ck->abc', U, V, W)
                err = np.linalg.norm(T_sum - T_target)
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    if err == 0:
                        return best_U, best_V, best_W, r
                
                # Local search steps
                for step in range(100):
                    if time.time() > deadline:
                        return None
                    # Pick a random tensor and modify a random entry
                    c = np.random.randint(r)
                    # Modify U
                    a = np.random.randint(6)
                    best_U[a, c] += np.random.choice([-1, 1])
                    T_sum = np.einsum('ak,bk,ck->abc', best_U, best_V, best_W)
                    err_new = np.linalg.norm(T_sum - T_target)
                    if err_new < best_err:
                        best_err = err_new
                    else:
                        best_U[a, c] -= np.random.choice([-1, 1]) # Revert
                        # Try V
                        b = np.random.randint(4)
                        best_V[b, c] += np.random.choice([-1, 1])
                        T_sum = np.einsum('ak,bk,ck->abc', best_U, best_V, best_W)
                        err_new = np.linalg.norm(T_sum - T_target)
                        if err_new < best_err:
                            best_err = err_new
                        else:
                            best_V[b, c] -= np.random.choice([-1, 1])
                            # Try W
                            d = np.random.randint(6)
                            best_W[d, c] += np.random.choice([-1, 1])
                            T_sum = np.einsum('ak,bk,ck->abc', best_U, best_V, best_W)
                            err_new = np.linalg.norm(T_sum - T_target)
                            if err_new < best_err:
                                best_err = err_new
                            else:
                                best_W[d, c] -= np.random.choice([-1, 1])
                    
                    if best_err == 0:
                        return best_U, best_V, best_W, r
            
            if best_err == 0:
                return best_U, best_V, best_W, r
                
        return None

    # Run search
    decomp = find_decomp_322()
    
    if decomp:
        U_base, V_base, W_base, r_base = decomp
        # Build M_{3,4,8}
        # n=3, m=4, p=8
        # U shape (12, 8*r_base), V shape (32, 8*r_base), W shape (24, 8*r_base)
        
        rank = 8 * r_base
        U = np.zeros((12, rank), dtype=int)
        V = np.zeros((32, rank), dtype=int)
        W = np.zeros((24, rank), dtype=int)
        
        # 8 blocks of M_{3,2,2}
        # Block index b from 0 to 7
        # b determines which part of A and B
        # A (3x4) split into A1 (3x2), A2 (3x2)
        # B (4x8) split into B1 (2x8), B2 (2x8)
        # B1 split into B11 (2x4), B12 (2x4)
        # B11 split into B111 (2x2), B112 (2x2)
        # So 2 * 2 * 2 = 8 blocks.
        
        for b in range(8):
            # b = i_block * 4 + j_block * 2 + k_block?
            # Let's define the mapping
            # i_block: 0 for A1, 1 for A2
            # j_block: 0 for B1, 1 for B2
            # k_block: 0 for B11, 1 for B12? No.
            
            # Let's trace indices
            # A index: i (0..2), j (0..3)
            # j = j_block * 2 + j_local (0..1)
            # B index: j (0..3), k (0..7)
            # j = j_block * 2 + j_local
            # k = k_block * 2 + k_local (0..1)
            
            # j_block = b // 4
            # k_block = (b % 4) // 2
            # j_local = b % 2 ? No.
            
            # Let's just iterate
            for i_block in range(2): # A1, A2
                for j_block in range(2): # B1, B2
                    for k_block in range(2): # B11, B12 (within B1 or B2)
                        # Wait, B is 4x8.
                        # B1 is rows 0,1. B2 is rows 2,3.
                        # B1 is cols 0..7.
                        # B11 is cols 0..3. B12 is cols 4..7.
                        # B111 is cols 0,1. B112 is cols 2,3.
                        # So 2 (A blocks) * 2 (B row blocks) * 2 (B col blocks) * 2 (B sub-col blocks) = 16?
                        # No, M_{3,2,2} handles 2x2 B.
                        # B is 4x8.
                        # Split into 16 blocks of 2x2?
                        # 4 rows / 2 = 2. 8 cols / 2 = 4.
                        # 2 * 4 = 8 blocks.
                        # Yes.
                        
                        idx = i_block * 4 + j_block * 2 + k_block
                        # Wait, k_block needs to be 0..3 for 4 col blocks.
                        # My loop only goes to 2.
                        
        # Let's re-index properly
        # i_block: 0,1 (A rows 0-1, 2-3? No, A is 3x4. Split into 3x2, 3x2)
        # j_block: 0,1 (B rows 0-1, 2-3)
        # k_block: 0,1,2,3 (B cols 0-1, 2-3, 4-5, 6-7)
        
        for i_block in range(2):
            for j_block in range(2):
                for k_block in range(4):
                    c_offset = (i_block * 4 + j_block * 2 + k_block) * r_base
                    # U: A rows
                    # A row i_local (0..1) corresponds to A index i_block * 2 + i_local
                    # U_base row is i_local * 2 + j_local
                    # We need to map U_base to U
                    # U_base[i_local * 2 + j_local, c]
                    # U[ (i_block * 2 + i_local) * 4 + (j_block * 2 + j_local), c_offset + c ]
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            u_row_base = i_local * 2 + j_local
                            u_row_full = (i_block * 2 + i_local) * 4 + (j_block * 2 + j_local)
                            U[u_row_full, c_offset:c_offset+r_base] = U_base[u_row_base, :]
                            
                    # V: B rows
                    # V_base row is j_local * 2 + k_local
                    # V row is (j_block * 2 + j_local) * 8 + (k_block * 2 + k_local)
                    for j_local in range(2):
                        for k_local in range(2):
                            v_row_base = j_local * 2 + k_local
                            v_row_full = (j_block * 2 + j_local) * 8 + (k_block * 2 + k_local)
                            V[v_row_full, c_offset:c_offset+r_base] = V_base[v_row_base, :]
                            
                    # W: C cols
                    # W_base col is k_local * 3 + i_local
                    # W col is (k_block * 2 + k_local) * 3 + (i_block * 2 + i_local)
                    for k_local in range(2):
                        for i_local in range(2):
                            w_col_base = k_local * 3 + i_local
                            w_col_full = (k_block * 2 + k_local) * 3 + (i_block * 2 + i_local)
                            W[w_col_full, c_offset:c_offset+r_base] = W_base[w_col_base, :]
                            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
