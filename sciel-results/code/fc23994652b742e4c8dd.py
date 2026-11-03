import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    # x in 0..7, y in 0..15, z in 0..7
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    # Current known solution
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    # Replicate the logic from the prompt to generate U_known, V_known, W_known
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    def check_solution(U, V, W, rank):
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Compute tensor T_est
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
            
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    # Start search for lower rank
    best_rank = rank_known
    best_U, best_V, best_W = U_known, V_known, W_known
    
    start_time = time.time()
    deadline = start_time + 110  # 40s margin before 150s limit
    
    # Try to find rank 27 by local search from the known rank-28 solution
    # Strategy: try to eliminate one term by adjusting others
    # Use a smooth approach: minimize the squared residual of the tensor constraint
    
    def tensor_residual(U, V, W, rank):
        """Compute the squared residual of the tensor decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
        
        return np.sum((T_est - T) ** 2)
    
    def local_search_rank27(U_init, V_init, W_init, max_iter=5000):
        """Local search to find a rank-27 solution."""
        rank = 27
        # Start from rank-28, zero out the last column
        U = [row[:] for row in U_init]
        V = [row[:] for row in V_init]
        W = [row[:] for row in W_init]
        
        # Zero out the last column (column 27)
        for i in range(8):
            U[i][27] = 0
        for i in range(16):
            V[i][27] = 0
        for i in range(8):
            W[i][27] = 0
        
        # Now we have a rank-28 solution with the last term zeroed out
        # This creates a residual. We need to adjust the other 27 terms to compensate.
        
        # Local search: randomly perturb entries and accept if residual decreases
        current_residual = tensor_residual(U, V, W, rank)
        best_residual = current_residual
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Choose a random term and a random matrix entry to perturb
            r = random.randint(0, rank - 1)
            mat = random.choice(['U', 'V', 'W'])
            
            if mat == 'U':
                row = random.randint(0, 7)
                old_val = U[row][r]
                delta = random.choice([-1, 1])
                U[row][r] += delta
            elif mat == 'V':
                row = random.randint(0, 15)
                old_val = V[row][r]
                delta = random.choice([-1, 1])
                V[row][r] += delta
            else:
                row = random.randint(0, 7)
                old_val = W[row][r]
                delta = random.choice([-1, 1])
                W[row][r] += delta
            
            new_residual = tensor_residual(U, V, W, rank)
            
            if new_residual < current_residual:
                current_residual = new_residual
                if new_residual < best_residual:
                    best_residual = new_residual
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
            else:
                # Revert
                if mat == 'U':
                    U[row][r] = old_val
                elif mat == 'V':
                    V[row][r] = old_val
                else:
                    W[row][r] = old_val
            
            # Check if we found an exact solution
            if best_residual < 1e-9:
                # Verify it's truly exact
                if check_solution(best_U, best_V, best_W, rank):
                    return best_U, best_V, best_W
        
        return None
    
    # Try multiple random restarts for rank 27
    for restart in range(50):
        if time.time() > deadline:
            break
        
        result = local_search_rank27(U_known, V_known, W_known, max_iter=2000)
        if result is not None:
            U27, V27, W27 = result
            if check_solution(U27, V27, W27, 27):
                best_rank = 27
                best_U, best_V, best_W = U27, V27, W27
                break
    
    # If rank 27 wasn't found, try a different approach: 
    # Use the structure of the solution to find a more compact representation
    # The 4 blocks have a similar structure. Maybe we can share some terms.
    
    # Alternative: try to find a rank-26 solution
    if best_rank == 28 and time.time() < deadline - 30:
        def local_search_rank26(U_init, V_init, W_init, max_iter=3000):
            rank = 26
            U = [row[:] for row in U_init]
            V = [row[:] for row in V_init]
            W = [row[:] for row in W_init]
            
            # Zero out the last two columns
            for i in range(8):
                U[i][26] = 0
                U[i][27] = 0
            for i in range(16):
                V[i][26] = 0
                V[i][27] = 0
            for i in range(8):
                W[i][26] = 0
                W[i][27] = 0
            
            current_residual = tensor_residual(U, V, W, rank)
            best_residual = current_residual
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                r = random.randint(0, rank - 1)
                mat = random.choice(['U', 'V', 'W'])
                
                if mat == 'U':
                    row = random.randint(0, 7)
                    old_val = U[row][r]
                    delta = random.choice([-1, 1])
                    U[row][r] += delta
                elif mat == 'V':
                    row = random.randint(0, 15)
                    old_val = V[row][r]
                    delta = random.choice([-1, 1])
                    V[row][r] += delta
                else:
                    row = random.randint(0, 7)
                    old_val = W[row][r]
                    delta = random.choice([-1, 1])
                    W[row][r] += delta
                
                new_residual = tensor_residual(U, V, W, rank)
                
                if new_residual < current_residual:
                    current_residual = new_residual
                    if new_residual < best_residual:
                        best_residual = new_residual
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                else:
                    if mat == 'U':
                        U[row][r] = old_val
                    elif mat == 'V':
                        V[row][r] = old_val
                    else:
                        W[row][r] = old_val
                
                if best_residual < 1e-9:
                    if check_solution(best_U, best_V, best_W, rank):
                        return best_U, best_V, best_W
            
            return None
        
        for restart in range(20):
            if time.time() > deadline:
                break
            result = local_search_rank26(U_known, V_known, W_known, max_iter=1500)
            if result is not None:
                U26, V26, W26 = result
                if check_solution(U26, V26, W26, 26):
                    best_rank = 26
                    best_U, best_V, best_W = U26, V26, W26
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
