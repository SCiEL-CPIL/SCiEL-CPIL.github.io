import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1
    T_flat = T.flatten()

    rank_known = 28
    U_known = np.zeros((8, rank_known), dtype=int)
    V_known = np.zeros((16, rank_known), dtype=int)
    W_known = np.zeros((8, rank_known), dtype=int)
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
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
        T_est = np.einsum('ir,sr,tr->ist', U[:, :rank], V[:, :rank], W[:, :rank])
        return np.allclose(T_est, T, atol=1e-9)

    def residual_val(U, V, W, rank):
        T_est = np.einsum('ir,sr,tr->ist', U[:, :rank], V[:, :rank], W[:, :rank])
        return np.sum((T_est.flatten() - T_flat) ** 2)

    best_rank = 28
    best_U = U_known.tolist()
    best_V = V_known.tolist()
    best_W = W_known.tolist()
    
    start_time = time.time()
    deadline = start_time + 110

    # Structural change: Try to find rank 27 by allowing half-integer values
    # Scale by 2 to work with integers, then divide by 2 at the end
    # The key insight: if we allow half-integers, we might be able to absorb one term
    
    # First, try the original approach with random search
    for zero_idx in range(28):
        if time.time() > deadline:
            break
        
        U_np = U_known.copy()
        V_np = V_known.copy()
        W_np = W_known.copy()
        
        U_np[:, zero_idx] = 0
        V_np[:, zero_idx] = 0
        W_np[:, zero_idx] = 0
        
        rank = 27
        current_residual = residual_val(U_np, V_np, W_np, rank)
        
        # Use simulated annealing with half-integer support
        # Scale by 2: work with 2*U, 2*V, 2*W, target 4*T
        U_scaled = (2 * U_np).astype(int)
        V_scaled = (2 * V_np).astype(int)
        W_scaled = (2 * W_np).astype(int)
        T_scaled = 4 * T_flat
        
        def residual_scaled(U_s, V_s, W_s, r):
            T_est = np.einsum('ir,sr,tr->ist', U_s[:, :r], V_s[:, :r], W_s[:, :r])
            return np.sum((T_est.flatten() - T_scaled) ** 2)
        
        current_res_scaled = residual_scaled(U_scaled, V_scaled, W_scaled, rank)
        
        # Local search with half-integer steps (delta = 1 in scaled space = 0.5 in original)
        for it in range(2000):
            if time.time() > deadline:
                break
            
            r = random.randint(0, rank - 1)
            mat = random.choice(['U', 'V', 'W'])
            delta = random.choice([-2, -1, 1, 2])  # In scaled space
            
            if mat == 'U':
                row = random.randint(0, 7)
                old_val = U_scaled[row, r]
                U_scaled[row, r] += delta
            elif mat == 'V':
                row = random.randint(0, 15)
                old_val = V_scaled[row, r]
                V_scaled[row, r] += delta
            else:
                row = random.randint(0, 7)
                old_val = W_scaled[row, r]
                W_scaled[row, r] += delta
            
            new_res_scaled = residual_scaled(U_scaled, V_scaled, W_scaled, rank)
            
            if new_res_scaled < current_res_scaled:
                current_res_scaled = new_res_scaled
                if current_res_scaled < 1e-6:
                    # Convert back and check
                    U_check = (U_scaled / 2.0)
                    V_check = (V_scaled / 2.0)
                    W_check = (W_scaled / 2.0)
                    if check_solution(U_check, V_check, W_check, rank):
                        best_rank = 27
                        best_U = [[float(x) for x in row] for row in U_check]
                        best_V = [[float(x) for x in row] for row in V_check]
                        best_W = [[float(x) for x in row] for row in W_check]
                        break
            else:
                if mat == 'U':
                    U_scaled[row, r] = old_val
                elif mat == 'V':
                    V_scaled[row, r] = old_val
                else:
                    W_scaled[row, r] = old_val
        
        if best_rank == 27:
            break

    # If we found a better rank, return it; otherwise return the known rank-28
    return {"U": best_U, "V": best_V, "W": best_W}
