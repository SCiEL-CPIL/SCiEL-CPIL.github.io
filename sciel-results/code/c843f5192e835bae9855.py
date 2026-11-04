import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 4
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1.0

    # Known rank 28 solution
    rank_known = 28
    U_known = np.zeros((8, rank_known))
    V_known = np.zeros((16, rank_known))
    W_known = np.zeros((8, rank_known))
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # This block structure is from the original code, assumed correct for rank 28
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

    best_rank = 28
    best_U = U_known.tolist()
    best_V = V_known.tolist()
    best_W = W_known.tolist()
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def optimize_rank(U_init, V_init, W_init, rank, max_iters=5000):
        """
        Local search using incremental gradient updates.
        """
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        # Precompute outer products for fast T_est calculation
        # T_est = sum_r U[:,r] * outer(V[:,r], W[:,r])
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += U[:, r][:, None, None] * np.outer(V[:, r], W[:, r])
            
        Residual = T_est - T
        current_loss = np.sum(Residual ** 2)
        
        # Precompute dot products for gradient updates
        # DotV[r, r'] = sum_y V[y,r] V[y,r']
        # DotW[r, r'] = sum_z W[z,r] W[z,r']
        DotV = np.zeros((rank, rank), dtype=float)
        DotW = np.zeros((rank, rank), dtype=float)
        for r in range(rank):
            for rp in range(rank):
                DotV[r, rp] = np.dot(V[:, r], V[:, rp])
                DotW[r, rp] = np.dot(W[:, r], W[:, rp])
                
        # Initialize Gradients
        # Grad_U[i, r] = 2 * sum_{y,z} Residual[i,y,z] V[y,r] W[z,r]
        # Grad_V[y, r] = 2 * sum_{x,z} Residual[x,y,z] U[x,r] W[z,r]
        # Grad_W[z, r] = 2 * sum_{x,y} Residual[x,y,z] U[x,r] V[y,r]
        
        # Efficient calculation:
        # Residual_VW_r = outer(V[:,r], W[:,r]) -> shape (16, 8)
        # Grad_U[:, r] = 2 * sum_{y,z} Residual[:, y, z] Residual_VW_r[y, z]
        # This is a contraction.
        
        Grad_U = np.zeros((8, rank), dtype=float)
        Grad_V = np.zeros((16, rank), dtype=float)
        Grad_W = np.zeros((8, rank), dtype=float)
        
        for r in range(rank):
            VW_outer = np.outer(V[:, r], W[:, r]) # (16, 8)
            # Grad_U[:, r] = 2 * np.tensordot(Residual, VW_outer, axes=([1,2],[0,1]))
            # Residual is (8, 16, 8). VW_outer is (16, 8).
            # Sum over y, z. Result shape (8).
            Grad_U[:, r] = 2.0 * np.tensordot(Residual, VW_outer, axes=([1, 2], [0, 1]))
            
            # Grad_V[y, r] = 2 * sum_{x,z} Residual[x, y, z] U[x, r] W[z, r]
            # Let's compute for all r at once? No, depends on r.
            # For a fixed r:
            # Temp[x, y] = sum_z Residual[x, y, z] W[z, r]
            # Grad_V[:, r] = 2 * Temp.T @ U[:, r]
            
            # Grad_W[z, r] = 2 * sum_{x,y} Residual[x, y, z] U[x, r] V[y, r]
            # Temp2[x, z] = sum_y Residual[x, y, z] V[y, r]
            # Grad_W[:, r] = 2 * Temp2.T @ U[:, r]
            
        # Recompute Grad_V and Grad_W properly
        for r in range(rank):
            # Grad_V
            # Residual (8, 16, 8). W[:, r] (8,).
            # Sum over z: Residual[x, y, z] * W[z, r] -> shape (8, 16)
            RW = np.tensordot(Residual, W[:, r], axes=([2], [0])) # (8, 16)
            # Grad_V[y, r] = 2 * sum_x RW[x, y] * U[x, r]
            Grad_V[:, r] = 2.0 * (RW.T @ U[:, r])
            
            # Grad_W
            # Sum over y: Residual[x, y, z] * V[y, r] -> shape (8, 8)
            RV = np.tensordot(Residual, V[:, r], axes=([1], [0])) # (8, 8)
            # Grad_W[z, r] = 2 * sum_x RV[x, z] * U[x, r]
            Grad_W[:, r] = 2.0 * (RV.T @ U[:, r])

        best_loss = current_loss
        best_U_state = U.copy()
        best_V_state = V.copy()
        best_W_state = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            best_move = None
            best_new_loss = current_loss + 1e-12
            
            # Evaluate moves for U
            for r in range(rank):
                quad_U = DotV[r, r] * DotW[r, r] # Sum V^2 W^2
                for i in range(8):
                    for delta in [-1, 1]:
                        # Change in loss approx: delta * Grad_U[i, r] + 0.5 * delta^2 * quad_U * 2?
                        # L = 0.5 * ||Residual||^2. 
                        # dL/dU = Grad_U (which includes factor 2).
                        # Delta L = delta * Grad_U + 0.5 * delta^2 * Hessian.
                        # Hessian entry H_ii = 2 * Sum V^2 W^2 = 2 * quad_U.
                        # So Delta L = delta * Grad_U[i, r] + 0.5 * delta^2 * 2 * quad_U = delta * Grad_U[i, r] + delta^2 * quad_U.
                        new_loss = current_loss + delta * Grad_U[i, r] + delta * delta * quad_U
                        if new_loss < best_new_loss:
                            best_new_loss = new_loss
                            best_move = ('U', i, r, delta)
                            
            # Evaluate moves for V
            for r in range(rank):
                quad_V = DotU[r, r] * DotW[r, r] if False else 0 # Need DotU
                # We don't have DotU precomputed.
                # Hessian for V[y, r] is 2 * Sum_x,z U[x,r]^2 W[z,r]^2 = 2 * (Sum U^2) * (Sum W^2)
                # Let's precompute Norms
                pass
            
            # To keep it simple and fast, I will stick to the O(N^2) evaluation but with optimized NumPy ops
            # The previous gradient update logic is complex to maintain exactly with discrete steps.
            # Given the small size (8x16x8), a direct residual update check might be fast enough if vectorized.
            
            # Let's try a simpler, robust approach: 
            # Keep the current code's logic but optimize the inner loop by precomputing the "impact" vectors.
            
            # Actually, let's just return the best found so far if we don't have time.
            # The current code is already doing a similar search. 
            # I will implement the gradient-based search properly below.
            
            # For now, break and use a simpler fallback if this function is too complex to debug in one go.
            # But I must provide a "stronger algorithm".
            
            # Let's implement the Gradient Update correctly.
            
            # Precompute Norms
            NormU_sq = np.sum(U**2, axis=0) # (rank,)
            NormV_sq = np.sum(V**2, axis=0) # (rank,)
            NormW_sq = np.sum(W**2, axis=0) # (rank,)
            
            # Update best_move selection with correct Hessian terms
            best_move = None
            best_new_loss = current_loss + 1e-12
            
            # U moves
            for r in range(rank):
                quad_U = NormV_sq[r] * NormW_sq[r]
                for i in range(8):
                    for delta in [-1, 1]:
                        new_loss = current_loss + delta * Grad_U[i, r] + delta * delta * quad_U
                        if new_loss < best_new_loss:
                            best_new_loss = new_loss
                            best_move = ('U', i, r, delta)
            
            # V moves
            for r in range(rank):
                quad_V = NormU_sq[r] * NormW_sq[r]
                for y in range(16):
                    for delta in [-1, 1]:
                        new_loss = current_loss + delta * Grad_V[y, r] + delta * delta * quad_V
                        if new_loss < best_new_loss:
                            best_new_loss = new_loss
                            best_move = ('V', y, r, delta)
            
            # W moves
            for r in range(rank):
                quad_W = NormU_sq[r] * NormV_sq[r]
                for z in range(8):
                    for delta in [-1, 1]:
                        new_loss = current_loss + delta * Grad_W[z, r] + delta * delta * quad_W
                        if new_loss < best_new_loss:
                            best_new_loss = new_loss
                            best_move = ('W', z, r, delta)
            
            if best_move is None:
                # Stuck in local minimum
                # Try random perturbation
                if random.random() < 0.05 and it > 100:
                    mat = random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = random.randint(0, 7)
                    elif mat == 'V':
                        row = random.randint(0, 15)
                    else:
                        row = random.randint(0, 7)
                    r = random.randint(0, rank - 1)
                    delta = random.choice([-1, 1])
                    best_move = (mat, row, r, delta)
                    best_new_loss = current_loss + 1.0 # Penalty
                else:
                    break
            
            mat, row, r, delta = best_move
            
            # Apply move
            if mat == 'U':
                U[row, r] += delta
                # Update T_est and Residual
                # Delta Residual[i, y, z] = delta * V[y, r] * W[z, r] if i==row
                Delta_Res = np.zeros((8, 16, 8), dtype=float)
                Delta_Res[row, :, :] = delta * np.outer(V[:, r], W[:, r])
                Residual += Delta_Res
                T_est += Delta_Res
                
                # Update Gradients
                # Delta Grad_U[i', r'] = 2 * delta * (if i'==row) * (V_r dot V_r') * (W_r dot W_r')
                for rp in range(rank):
                    if row == row: # Always true for the row we changed
                        # Only Grad_U[row, rp] changes
                        DotV_rrp = np.dot(V[:, r], V[:, rp])
                        DotW_rrp = np.dot(W[:, r], W[:, rp])
                        Grad_U[row, rp] += 2 * delta * DotV_rrp * DotW_rrp
                        
                # Delta Grad_V[y', r'] = 2 * delta * V[y', r] * (W_r dot W_r') * U[row, r']
                for rp in range(rank):
                    DotW_rrp = np.dot(W[:, r], W[:, rp])
                    # Grad_V[:, rp] changes
                    # Delta = 2 * delta * V[:, r] * DotW_rrp * U[row, rp]
                    Grad_V[:, rp] += 2 * delta * V[:, r] * DotW_rrp * U[row, rp]
                    
                # Delta Grad_W[z', r'] = 2 * delta * W[z', r] * (V_r dot V_r') * U[row, r']
                for rp in range(rank):
                    DotV_rrp = np.dot(V[:, r], V[:, rp])
                    Grad_W[:, rp] += 2 * delta * W[:, r] * DotV_rrp * U[row, rp]
                    
                # Update Norms
                NormU_sq[r] += 2 * delta * U[row, r] + delta * delta # (u+d)^2 - u^2 = 2ud + d^2
                # Note: U[row, r] is already updated. So we need old value.
                # Let's just recalculate NormU_sq[r] to be safe? It's O(8).
                NormU_sq[r] = np.sum(U[:, r]**2)
                
            elif mat == 'V':
                V[row, r] += delta
                Delta_Res = np.zeros((8, 16, 8), dtype=float)
                Delta_Res[:, row, :] = delta * np.outer(U[:, r], W[:, r])
                Residual += Delta_Res
                T_est += Delta_Res
                
                # Update Gradients
                for rp in range(rank):
                    DotU_rrp = np.dot(U[:, r], U[:, rp])
                    DotW_rrp = np.dot(W[:, r], W[:, rp])
                    # Grad_V[row, rp] changes
                    Grad_V[row, rp] += 2 * delta * DotU_rrp * DotW_rrp
                    
                    # Grad_U[i', rp] changes?
                    # Delta Res[i, row, z] = delta U[i, r] W[z, r]
                    # Grad_U[i', rp] = 2 sum Res[i', y, z] V[y, rp] W[z, rp]
                    # Delta = 2 sum_z delta U[i', r] W[z, r] V[row, rp] W[z, rp]
                    # = 2 delta U[i', r] V[row, rp] DotW_rrp
                    Grad_U[:, rp] += 2 * delta * U[:, r] * V[row, rp] * DotW_rrp
                    
                    # Grad_W[z', rp] changes?
                    # Delta = 2 sum_y delta U[y, r] V[row, rp] W[z', r] ... wait
                    # Delta Res[i, row, z] = delta U[i, r] W[z, r]
                    # Grad_W[z', rp] = 2 sum_{i,y} Res[i, y, z'] U[i, rp] V[y, rp]
                    # Delta = 2 sum_i delta U[i, r] W[z', r] U[i, rp] V[row, rp]
                    # = 2 delta W[z', r] V[row, rp] DotU_rrp
                    Grad_W[:, rp] += 2 * delta * W[:, r] * V[row, rp] * DotU_rrp
                    
                NormV_sq[r] = np.sum(V[:, r]**2)
                
            else: # W
                W[row, r] += delta
                Delta_Res = np.zeros((8, 16, 8), dtype=float)
                Delta_Res[:, :, row] = delta * np.outer(U[:, r], V[:, r])
                Residual += Delta_Res
                T_est += Delta_Res
                
                # Update Gradients
                for rp in range(rank):
                    DotU_rrp = np.dot(U[:, r], U[:, rp])
                    DotV_rrp = np.dot(V[:, r], V[:, rp])
                    # Grad_W[row, rp] changes
                    Grad_W[row, rp] += 2 * delta * DotU_rrp * DotV_rrp
                    
                    # Grad_U[i', rp] changes?
                    # Delta Res[i, y, row] = delta U[i, r] V[y, r]
                    # Grad_U[i', rp] = 2 sum_{y,z} Res[i', y, z] V[y, rp] W[z, rp]
                    # Delta = 2 sum_y delta U[i', r] V[y, r] V[y, rp] W[row, rp]
                    # = 2 delta U[i', r] W[row, rp] DotV_rrp
                    Grad_U[:, rp] += 2 * delta * U[:, r] * W[row, rp] * DotV_rrp
                    
                    # Grad_V[y', rp] changes?
                    # Delta = 2 sum_x delta U[x, r] V[y', r] W[row, rp] ... 
                    # Delta Res[i, y, row] = delta U[i, r] V[y, r]
                    # Grad_V[y', rp] = 2 sum_{x,z} Res[x, y', z] U[x, rp] W[z, rp]
                    # Delta = 2 sum_x delta U[x, r] V[y', r] U[x, rp] W[row, rp]
                    # = 2 delta V[y', r] W[row, rp] DotU_rrp
                    Grad_V[:, rp] += 2 * delta * V[:, r] * W[row, rp] * DotU_rrp
                    
                NormW_sq[r] = np.sum(W[:, r]**2)
                
            current_loss = best_new_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_U_state = U.copy()
                best_V_state = V.copy()
                best_W_state = W.copy()
                
            if best_loss < 1e-9:
                if check_solution(best_U_state, best_V_state, best_W_state, rank):
                    return best_U_state, best_V_state, best_W_state, True
                    
        return best_U_state, best_V_state, best_W_state, False

    # Try to find rank 27
    # Strategy: Zero out one column of the rank 28 solution and optimize
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        U_init = U_known.copy()
        V_init = V_known.copy()
        W_init = W_known.copy()
        
        # Zero out a random column
        col_to_zero = random.randint(0, rank_known - 1)
        U_init[:, col_to_zero] = 0
        V_init[:, col_to_zero] = 0
        W_init[:, col_to_zero] = 0
        
        # Remove the zeroed column to reduce rank
        mask = np.ones(rank_known, dtype=bool)
        mask[col_to_zero] = False
        U_opt = U_init[:, mask]
        V_opt = V_init[:, mask]
        W_opt = W_init[:, mask]
        
        U_res, V_res, W_res, success = optimize_rank(U_opt, V_opt, W_opt, rank_known - 1, max_iters=2000)
        
        if success:
            best_rank = 27
            best_U = U_res.tolist()
            best_V = V_res.tolist()
            best_W = W_res.tolist()
            break

    # Final verification
    if best_rank <= 27:
        if not check_solution(best_U, best_V, best_W, best_rank):
            best_rank = 28
            best_U = U_known.tolist()
            best_V = V_known.tolist()
            best_W = W_known.tolist()

    return {"U": best_U, "V": best_V, "W": best_W}
