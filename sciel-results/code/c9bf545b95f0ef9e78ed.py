import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1.0
    
    # Known optimal solution with rank 28 (Strassen-based)
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_strassen_block(col_offset, X_rows, Y_rows, Z_rows):
        """Add a Strassen 2x2x2 block at given column offset.
        X_rows: [x00, x01, x10, x11] indices into U (A elements)
        Y_rows: [y00, y01, y10, y11] indices into V (B elements)  
        Z_rows: [z00, z01, z10, z11] indices into W (C elements)
        """
        U = U_known
        V = V_known
        W = W_known
        
        # Strassen 7 products for 2x2*2x2 -> 2x2
        # M1 = (A00+A11)(B00+B11)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2 = A10*(B00+B01)
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1  # Wait, this should be just A10
        # Let me redo this properly
        
    # Let me define Strassen properly
    # For 2x2 * 2x2:
    # C00 = M1 + M4 - M2 + M6
    # C01 = M2 + M3
    # C10 = M5 + M4
    # C11 = M1 - M3 + M5 + M7
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = A10*(B00+B01)
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A11)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    def add_strassen_block_v2(col_offset, X_rows, Y_rows, Z_rows):
        """Add a Strassen 2x2x2 block.
        X_rows: [x00, x01, x10, x11] = A[0,0], A[0,1], A[1,0], A[1,1]
        Y_rows: [y00, y01, y10, y11] = B[0,0], B[0,1], B[1,0], B[1,1]
        Z_rows: [z00, z01, z10, z11] = C[0,0], C[0,1], C[1,0], C[1,1]
        """
        U = U_known
        V = V_known
        W = W_known
        
        # M1 = (A00+A11)(B00+B11)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2 = A10*(B00+B01)
        U[X_rows[2]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        V[Y_rows[1]][col_offset + 1] += 1
        W[Z_rows[0]][col_offset + 1] -= 1
        W[Z_rows[1]][col_offset + 1] += 1
        
        # M3 = A00*(B10-B11)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[2]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] -= 1
        
        # M4 = A11*(B01-B10)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[1]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5 = (A00+A01)*B11
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6 = (A10-A11)*(B00+B10)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[3]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[2]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[1]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1
    
    # Block 1: C[0,0:2] from A[0,0:2] * B[0:2,0:2] + A[0,2:4] * B[2:4,0:2]
    # Wait, I need to be more careful about the block structure.
    
    # A is 2x4: A[i,k] for i=0,1 and k=0,1,2,3
    # B is 4x4: B[k,j] for k=0,1,2,3 and j=0,1,2,3
    # C is 2x4: C[i,j] for i=0,1 and j=0,1,2,3
    # C[i,j] = sum_k A[i,k]*B[k,j]
    
    # Split k into {0,1} and {2,3}:
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j] + A[i,3]*B[3,j]
    
    # This is a sum of 4 terms, each involving a 2x1 (A column) times 1x4 (B row)
    # But we can group: for j in {0,1}:
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0] + A[i,3]*B[3,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1] + A[i,3]*B[3,1]
    
    # For the left half of C (j=0,1), we can view this as:
    # [C[i,0], C[i,1]] = A[i,0:2] * B[0:2,0:2] + A[i,2:4] * B[2:4,0:2]
    # where A[i,0:2] is 1x2, B[0:2,0:2] is 2x2, etc.
    
    # But Strassen works for 2x2 * 2x2. Let me think differently.
    
    # Actually, the standard approach for 2x4 * 4x4:
    # View A as [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # View B as [[B_tl, B_tr], [B_bl, B_br]] where each is 2x2
    # Then C = [A_left * B_tl + A_right * B_bl, A_left * B_tr + A_right * B_br]
    # C is 2x4 = [C_left | C_right] where C_left is 2x2, C_right is 2x2
    
    # So we need to compute 4 products of 2x2 matrices:
    # P1 = A_left * B_tl
    # P2 = A_right * B_bl
    # P3 = A_left * B_tr
    # P4 = A_right * B_br
    # C_left = P1 + P2
    # C_right = P3 + P4
    
    # Each 2x2*2x2 takes rank 7 with Strassen, so total rank = 4*7 = 28.
    
    # Can we do better? The question is whether we can share Strassen terms between
    # different products. For example, if some of the intermediate sums in Strassen
    # are the same across different products.
    
    # Let me set up the indices properly.
    # A[i,k]: i=0,1; k=0,1,2,3
    #   A_left: A[0,0], A[0,1], A[1,0], A[1,1]
    #   A_right: A[0,2], A[0,3], A[1,2], A[1,3]
    # B[k,j]: k=0,1,2,3; j=0,1,2,3
    #   B_tl: B[0,0], B[0,1], B[1,0], B[1,1]
    #   B_tr: B[0,2], B[0,3], B[1,2], B[1,3]
    #   B_bl: B[2,0], B[2,1], B[3,0], B[3,1]
    #   B_br: B[2,2], B[2,3], B[3,2], B[3,3]
    # C[i,j]: i=0,1; j=0,1,2,3
    #   C_left: C[0,0], C[0,1], C[1,0], C[1,1]
    #   C_right: C[0,2], C[0,3], C[1,2], C[1,3]
    
    # U indices (n*m = 8): x = i*4 + k
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    #   A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # V indices (m*p = 16): y = k*4 + j
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    #   B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    #   B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    #   B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    # W indices (p*n = 8): z = j*2 + i
    #   C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
    #   C[0,2]=4, C[1,2]=5, C[0,3]=6, C[1,3]=7
    
    # Block 1: A_left * B_tl -> contributes to C_left
    # A_left: [A[0,0], A[0,1], A[1,0], A[1,1]] = [0, 1, 4, 5]
    # B_tl: [B[0,0], B[0,1], B[1,0], B[1,1]] = [0, 1, 4, 5]
    # C_left: [C[0,0], C[0,1], C[1,0], C[1,1]] = [0, 2, 1, 3]
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_strassen_block_v2(0, X1, Y1, Z1)
    
    # Block 2: A_right * B_bl -> contributes to C_left
    # A_right: [A[0,2], A[0,3], A[1,2], A[1,3]] = [2, 3, 6, 7]
    # B_bl: [B[2,0], B[2,1], B[3,0], B[3,1]] = [8, 9, 12, 13]
    # C_left: [0, 2, 1, 3]
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_strassen_block_v2(7, X2, Y2, Z2)
    
    # Block 3: A_left * B_tr -> contributes to C_right
    # A_left: [0, 1, 4, 5]
    # B_tr: [B[0,2], B[0,3], B[1,2], B[1,3]] = [2, 3, 6, 7]
    # C_right: [C[0,2], C[0,3], C[1,2], C[1,3]] = [4, 6, 5, 7]
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_strassen_block_v2(14, X3, Y3, Z3)
    
    # Block 4: A_right * B_br -> contributes to C_right
    # A_right: [2, 3, 6, 7]
    # B_br: [B[2,2], B[2,3], B[3,2], B[3,3]] = [10, 11, 14, 15]
    # C_right: [4, 6, 5, 7]
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_strassen_block_v2(21, X4, Y4, Z4)
    
    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
            
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True
    
    # Verify the known solution
    if not check_solution(U_known, V_known, W_known, rank_known):
        raise ValueError("Known solution verification failed")
    
    best_rank = rank_known
    best_U, best_V, best_W = U_known, V_known, W_known
    
    # Try to find a lower rank solution using numerical optimization
    # We'll use a gradient-free approach: randomly initialize and refine
    
    def try_find_lower_rank(target_rank, max_iter=5000, seed=42):
        """Try to find a rank-target_rank decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize randomly
        U_rand = rng.randn(8, target_rank)
        V_rand = rng.randn(16, target_rank)
        W_rand = rng.randn(8, target_rank)
        
        # Scale to match the tensor norm
        T_norm = np.linalg.norm(T)
        for r in range(target_rank):
            scale = 1.0 / (np.linalg.norm(U_rand[:, r]) * np.linalg.norm(V_rand[:, r]) * np.linalg.norm(W_rand[:, r]) + 1e-10)
            U_rand[:, r] *= scale ** (1/3)
            V_rand[:, r] *= scale ** (1/3)
            W_rand[:, r] *= scale ** (1/3)
        
        best_err = np.inf
        best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Compute current error
            T_est = np.zeros((8, 16, 8), dtype=float)
            for r in range(target_rank):
                T_est += np.einsum('r,s,t->rst', U_rand[:, r], V_rand[:, r], W_rand[:, r])
            err = np.linalg.norm(T_est - T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
            
            if best_err < 1e-8:
                break
            
            # Random perturbation with decreasing step size
            step = 0.1 * (0.99 ** (it // 100))
            r_idx = rng.randint(target_rank)
            which = rng.randint(3)
            if which == 0:
                U_rand[:, r_idx] += step * rng.randn(8)
            elif which == 1:
                V_rand[:, r_idx] += step * rng.randn(16)
            else:
                W_rand[:, r_idx] += step * rng.randn(8)
        
        return best_err, best_U, best_V, best_W
    
    # Try rank 27
    if time.time() - start_time < DEADLINE - 30:
        err27, U27, V27, W27 = try_find_lower_rank(27, max_iter=3000, seed=1)
        if err27 < 1e-6:
            # Check if it's truly valid (within tolerance)
            U27_int = np.round(U27 * 2) / 2  # Try to snap to half-integers
            V27_int = np.round(V27 * 2) / 2
            W27_int = np.round(W27 * 2) / 2
            
            T_est = np.zeros((8, 16, 8), dtype=float)
            for r in range(27):
                T_est += np.einsum('r,s,t->rst', U27_int[:, r], V27_int[:, r], W27_int[:, r])
            
            if np.allclose(T_est, T, atol=1e-6):
                best_rank = 27
                best_U = U27_int.tolist()
                best_V = V27_int.tolist()
                best_W = W27_int.tolist()
    
    # Try rank 26
    if best_rank > 26 and time.time() - start_time < DEADLINE - 30:
        err26, U26, V26, W26 = try_find_lower_rank(26, max_iter=3000, seed=2)
        if err26 < 1e-6:
            U26_int = np.round(U26 * 2) / 2
            V26_int = np.round(V26 * 2) / 2
            W26_int = np.round(W26 * 2) / 2
            
            T_est = np.zeros((8, 16, 8), dtype=float)
            for r in range(26):
                T_est += np.einsum('r,s,t->rst', U26_int[:, r], V26_int[:, r], W26_int[:, r])
            
            if np.allclose(T_est, T, atol=1e-6):
                best_rank = 26
                best_U = U26_int.tolist()
                best_V = V26_int.tolist()
                best_W = W26_int.tolist()
    
    # Return the best solution found
    return {"U": best_U, "V": best_V, "W": best_W}
