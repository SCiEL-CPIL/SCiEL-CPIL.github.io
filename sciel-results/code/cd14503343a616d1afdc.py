import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 4
    
    # Target tensor T for 2x4x4
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1

    # Known rank 28 solution
    rank_known = 28
    U_known = np.zeros((8, rank_known))
    V_known = np.zeros((16, rank_known))
    W_known = np.zeros((8, rank_known))
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Strassen 2x2x2 block
        # This code is from the original program, kept for fallback
        U_known[X_rows[0]][col_offset + 0] += 1
        U_known[X_rows[3]][col_offset + 0] += 1
        V_known[Y_rows[0]][col_offset + 0] += 1
        V_known[Y_rows[3]][col_offset + 0] += 1
        W_known[Z_rows[0]][col_offset + 0] += 1
        W_known[Z_rows[3]][col_offset + 0] += 1
        
        U_known[X_rows[2]][col_offset + 1] += 1
        U_known[X_rows[3]][col_offset + 1] += 1
        V_known[Y_rows[0]][col_offset + 1] += 1
        W_known[Z_rows[2]][col_offset + 1] += 1
        W_known[Z_rows[3]][col_offset + 1] -= 1
        
        U_known[X_rows[0]][col_offset + 2] += 1
        V_known[Y_rows[1]][col_offset + 2] += 1
        V_known[Y_rows[3]][col_offset + 2] -= 1
        W_known[Z_rows[1]][col_offset + 2] += 1
        W_known[Z_rows[3]][col_offset + 2] += 1
        
        U_known[X_rows[3]][col_offset + 3] += 1
        V_known[Y_rows[2]][col_offset + 3] += 1
        V_known[Y_rows[0]][col_offset + 3] -= 1
        W_known[Z_rows[0]][col_offset + 3] += 1
        W_known[Z_rows[2]][col_offset + 3] += 1
        
        U_known[X_rows[0]][col_offset + 4] += 1
        U_known[X_rows[1]][col_offset + 4] += 1
        V_known[Y_rows[3]][col_offset + 4] += 1
        W_known[Z_rows[0]][col_offset + 4] -= 1
        W_known[Z_rows[1]][col_offset + 4] += 1
        
        U_known[X_rows[2]][col_offset + 5] += 1
        U_known[X_rows[0]][col_offset + 5] -= 1
        V_known[Y_rows[0]][col_offset + 5] += 1
        V_known[Y_rows[1]][col_offset + 5] += 1
        W_known[Z_rows[3]][col_offset + 5] += 1
        
        U_known[X_rows[1]][col_offset + 6] += 1
        U_known[X_rows[3]][col_offset + 6] -= 1
        V_known[Y_rows[2]][col_offset + 6] += 1
        V_known[Y_rows[3]][col_offset + 6] += 1
        W_known[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    X2 = [2, 3, 6, 7]
    Y11 = [0, 1, 4, 5]
    Y12 = [2, 3, 6, 7]
    Y21 = [8, 9, 12, 13]
    Y22 = [10, 11, 14, 15]
    Z11 = [0, 2, 1, 3]
    Z12 = [4, 6, 5, 7]
    
    add_block(0, X1, Y11, Z11)
    add_block(7, X2, Y21, Z11)
    add_block(14, X1, Y12, Z12)
    add_block(21, X2, Y22, Z12)

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
            
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    # Attempt to find rank 20 solution
    # Strategy: Find rank 10 for 2x4x2 subproblem
    start_time = time.time()
    deadline = start_time + 110

    def solve_242_rank10():
        # Local tensor for 2x4x2
        # A: 2x4 (8), B: 4x2 (8), C: 2x2 (4)
        # T_local[x, y, z] = 1 if C[z] depends on A[x] and B[y]
        # x = i*4 + k (i in 0,1; k in 0..3)
        # y = k*2 + j (k in 0..3; j in 0,1)
        # z = i*2 + j (i in 0,1; j in 0,1)
        T_local = np.zeros((8, 8, 4), dtype=int)
        for i in range(2):
            for k in range(4):
                for j in range(2):
                    x = i * 4 + k
                    y = k * 2 + j
                    z = i * 2 + j
                    T_local[x, y, z] = 1

        rank = 10
        best_res = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            U = np.zeros((8, rank))
            V = np.zeros((8, rank))
            W = np.zeros((4, rank))
            
            # Random init
            for r in range(rank):
                for _ in range(2):
                    U[random.randint(0,7), r] = random.choice([-1,1])
                for _ in range(2):
                    V[random.randint(0,7), r] = random.choice([-1,1])
                for _ in range(1):
                    W[random.randint(0,3), r] = random.choice([-1,1])
            
            # Local search
            for it in range(5000):
                if time.time() > deadline:
                    break
                
                # Compute residual
                T_est = np.zeros((8, 8, 4), dtype=float)
                for r in range(rank):
                    T_est += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                res = np.sum((T_est - T_local)**2)
                
                if res < best_res:
                    best_res = res
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    if res < 1e-9:
                        return best_U, best_V, best_W
                
                # Find best move
                best_move = None
                best_new_res = res
                
                # Try U moves
                for r in range(rank):
                    for row in range(8):
                        for delta in [-1, 1]:
                            # T_est += delta * outer(V[:,r], W[:,r])
                            diff = T_est - T_local
                            cross = np.sum(diff * np.outer(V[:, r], W[:, r]))
                            quad = np.sum(np.outer(V[:, r], W[:, r])**2)
                            new_res = res + 2 * delta * cross + quad
                            if new_res < best_new_res:
                                best_new_res = new_res
                                best_move = ('U', row, r, delta)
                
                # Try V moves
                for r in range(rank):
                    for row in range(8):
                        for delta in [-1, 1]:
                            diff = T_est - T_local
                            u_col = U[:, r]
                            w_col = W[:, r]
                            cross = np.sum(diff[:, row, :] * u_col[:, None] * w_col[None, :])
                            quad = np.sum((u_col[:, None] * w_col[None, :])**2)
                            new_res = res + 2 * delta * cross + quad
                            if new_res < best_new_res:
                                best_new_res = new_res
                                best_move = ('V', row, r, delta)
                
                # Try W moves
                for r in range(rank):
                    for row in range(4):
                        for delta in [-1, 1]:
                            diff = T_est - T_local
                            u_col = U[:, r]
                            v_col = V[:, r]
                            cross = np.sum(diff[:, :, row] * u_col[:, None] * v_col[None, :])
                            quad = np.sum((u_col[:, None] * v_col[None, :])**2)
                            new_res = res + 2 * delta * cross + quad
                            if new_res < best_new_res:
                                best_new_res = new_res
                                best_move = ('W', row, r, delta)
                
                if best_move is None:
                    break
                
                mat, row, r, delta = best_move
                if mat == 'U':
                    U[row, r] += delta
                elif mat == 'V':
                    V[row, r] += delta
                else:
                    W[row, r] += delta
            
        if best_res < 1e-9:
            return best_U, best_V, best_W
        return None

    U_242, V_242, W_242 = solve_242_rank10()
    
    if U_242 is not None:
        # Assemble rank 20 global solution
        U_global = np.zeros((8, 20))
        V_global = np.zeros((16, 20))
        W_global = np.zeros((8, 20))
        
        # First 10 cols: Left part (B cols 0,1)
        U_global[:, 0:10] = U_242
        # Map V_242 (8 rows) to V_global (16 rows)
        # Local y = k*2 + j. Global y = k*4 + j.
        for r in range(10):
            for local_row in range(8):
                k = local_row // 2
                j = local_row % 2
                global_row = k * 4 + j
                V_global[global_row, r] = V_242[local_row, r]
        
        # Map W_242 (4 rows) to W_global (8 rows)
        # Local z = i*2 + j. Global z = i + 2*j.
        # Note: In 2x4x2, z index is i*2+j?
        # Task contract: W[k*n+i, r]. Here n=2.
        # For 2x4x2, output is 2x2.
        # z = i + j*n = i + 2j.
        # My T_local definition: z = i*2 + j.
        # Let's check consistency.
        # T_local[x,y,z] where x=i*4+k, y=k*2+j, z=i*2+j.
        # If I use z=i*2+j, then for i=0,j=0 -> 0. i=0,j=1 -> 1. i=1,j=0 -> 2. i=1,j=1 -> 3.
        # Global W index for C_left (j=0,1):
        # j=0: z = i + 0 = i. (0,1)
        # j=1: z = i + 2. (2,3)
        # So global rows 0,1,2,3 are used.
        # My local z mapping:
        # local 0 (i=0,j=0) -> global 0.
        # local 1 (i=0,j=1) -> global 2.
        # local 2 (i=1,j=0) -> global 1.
        # local 3 (i=1,j=1) -> global 3.
        # So mapping: local 0->0, 1->2, 2->1, 3->3.
        for r in range(10):
            for local_row in range(4):
                i = local_row // 2
                j = local_row % 2
                global_row = i + 2 * j
                W_global[global_row, r] = W_242[local_row, r]
        
        # Next 10 cols: Right part (B cols 2,3)
        U_global[:, 10:20] = U_242
        # Map V_242 to V_global for right part
        # Global y = k*4 + (j+2)
        for r in range(10):
            for local_row in range(8):
                k = local_row // 2
                j = local_row % 2
                global_row = k * 4 + (j + 2)
                V_global[global_row, 10 + r] = V_242[local_row, r]
        
        # Map W_242 to W_global for right part
        # Global z = i + 2*(j+2) = i + 2j + 4
        for r in range(10):
            for local_row in range(4):
                i = local_row // 2
                j = local_row % 2
                global_row = i + 2 * j + 4
                W_global[global_row, 10 + r] = W_242[local_row, r]
        
        # Verify
        if check_solution(U_global.tolist(), V_global.tolist(), W_global.tolist(), 20):
            return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
    
    # Fallback to rank 28
    return {"U": U_known.tolist(), "V": V_known.tolist(), "W": W_known.tolist()}
