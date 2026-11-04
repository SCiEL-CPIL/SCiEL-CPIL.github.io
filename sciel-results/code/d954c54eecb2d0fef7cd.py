import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    
    # Build the target tensor T
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1

    # Helper to verify a solution
    def verify(U, V, W):
        r = len(U[0]) if U else 0
        T_est = np.einsum('xr,yr,zr->xyz', U, V, W)
        return np.allclose(T_est, T, atol=1e-9)

    # Known rank 28 solution (Blockwise Strassen)
    def get_strassen_28():
        rank = 28
        U = np.zeros((8, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((8, rank), dtype=int)
        
        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1
            
            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1
            
            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1
            
            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1
            
            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1
            
            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1
            
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
        
        return U.tolist(), V.tolist(), W.tolist()

    best_U, best_V, best_W = get_strassen_28()
    best_rank = 28
    
    start_time = time.time()
    deadline = start_time + 140

    def search_rank_27_f2():
        r = 27
        # T_flat: T[x, y, z] flattened in C-order: index = (x*16 + y)*8 + z
        T_flat = T.reshape(-1)  # shape (1024,)
        
        # For given U, V, the linear system for W is:
        # For each (x, y, z): sum_r U[x,r] * V[y,r] * W[z,r] = T[x,y,z]
        # Unknowns: W[z, r] for z in 0..7, r in 0..26 -> 8*27 = 216 unknowns
        # Equations: 8*16*8 = 1024 equations
        # Coefficient matrix M[eq, var]: M[(x*16+y)*8+z, z*27+r] = U[x,r] * V[y,r]
        
        # Build M efficiently using broadcasting
        # M shape: (1024, 216)
        # M[(x*16+y)*8+z, z*27+r] = U[x,r] * V[y,r]
        
        # U[x,r] has shape (8, 27)
        # V[y,r] has shape (16, 27)
        # Product P[x,y,r] = U[x,r] * V[y,r] has shape (8, 16, 27)
        # We need to scatter P into M.
        
        # M[(x*16+y)*8+z, z*27+r] = P[x,y,r]
        # Let's build M by iterating over z and r.
        # For fixed z, r: M[?, z*27+r] = P[:, :, r] where ? = (x*16+y)*8+z
        # P[:,:,r] has shape (8,16). Reshape to (128,). 
        # The equation index for (x,y,z) is (x*16+y)*8+z.
        # For fixed z, the equations with that z are at positions where index % 8 == z.
        # These are: 0*8+z, 1*8+z, 2*8+z, ..., 127*8+z? No.
        # index = (x*16 + y)*8 + z.
        # As (x,y) ranges over 128 pairs, (x*16+y) ranges 0..127.
        # So for fixed z, the indices are z, 8+z, 16+z, ..., 127*8+z? No.
        # index = (x*16+y)*8 + z.
        # If x=0, y=0: 0*8+z = z.
        # If x=0, y=1: 1*8+z = 8+z.
        # ...
        # If x=7, y=15: (7*16+15)*8 + z = 127*8 + z = 1016 + z.
        # So for fixed z, the equation indices are z + 8*k for k=0..127.
        # The corresponding (x,y) for k is x=k//16, y=k%16.
        # So M[z + 8*k, z*27+r] = P[k//16, k%16, r] = P.reshape(128, 27)[k, r].
        
        # So we can build M column by column or use advanced indexing.
        # Let's build M as a numpy array of uint8 for speed.
        
        # Optimization: Precompute the equation indices for each (z, r) column?
        # Or just loop z, r and assign.
        
        # P = np.einsum('xr,yr->xyr', U, V) # (8, 16, 27)
        # P_flat = P.reshape(128, 27)
        
        # M = np.zeros((1024, 216), dtype=np.uint8)
        # for z in range(8):
        #     for r in range(27):
        #         col = z * 27 + r
        #         # indices: z + 8*k for k in 0..127
        #         M[z::8, col] = P_flat[:, r]
        
        # This is 8*27 = 216 slices of length 128. Fast.
        
        # Then solve M W_flat = T_flat over F2.
        # M is (1024, 216). T_flat is (1024,).
        
        # Gaussian elimination over F2 using bit operations or uint8 arrays.
        # Since 216 is small, we can represent each row as a Python int (bitmask).
        # 216 bits fits in a Python int easily.
        
        # Convert M to list of ints (rows as bitmasks)
        # M_row_bits[i] = sum(M[i, j] << j)
        
        # T_flat bits: T_bits[i] = sum(T_flat[i] << i) ? No, T_flat is the RHS.
        # We need to solve M W = T.
        
        # Let's implement GE on the augmented matrix [M | T].
        # Rows are ints. Col 0..215 are M, Col 216 is T.
        # Actually, we just need to check consistency and find a solution.
        
        # Fast GE:
        # pivot_row = 0
        # for col in range(216):
        #     find row >= pivot_row with bit col set
        #     if none, continue
        #     swap
        #     for row in all rows except pivot_row:
        #         if row has bit col set: row ^= pivot_row
        #     pivot_row += 1
        # Check consistency: for any row where M part is 0, T part must be 0.
        
        # This is O(1024 * 216) bit ops per restart. ~220k ops. Very fast.
        
        def solve_f2(M, T_flat):
            # M: (1024, 216) uint8
            # T_flat: (1024,) uint8
            # Convert to list of ints
            # Row i: bits 0..215 from M[i], bit 216 from T_flat[i]
            rows = []
            for i in range(1024):
                val = 0
                # Build bitmask for M[i]
                # M[i] is uint8 array of length 216
                # We can use numpy pack? Or just loop?
                # Looping 216 * 1024 = 221k iterations in Python is ~0.02s.
                # 5000 restarts * 0.02s = 100s. Too slow.
                
                # Need faster conversion.
                # M[i] is uint8.
                # We can view M as bytes and use int.from_bytes?
                # M[i] is 216 bytes (since dtype uint8).
                # int.from_bytes(M[i].tobytes(), 'little') gives a 216-bit int.
                # Then add T_flat[i] << 216.
                
                m_bits = int.from_bytes(M[i].tobytes(), 'little')
                t_bit = T_flat[i]
                rows.append(m_bits | (t_bit << 216))
            
            # GE
            pivot = 0
            for col in range(216):
                # Find pivot
                pivot_row = -1
                for i in range(pivot, 1024):
                    if (rows[i] >> col) & 1:
                        pivot_row = i
                        break
                if pivot_row == -1:
                    continue
                
                rows[pivot], rows[pivot_row] = rows[pivot_row], rows[pivot]
                
                # Eliminate
                for i in range(1024):
                    if i != pivot and ((rows[i] >> col) & 1):
                        rows[i] ^= rows[pivot]
                
                pivot += 1
                if pivot == 216:
                    break
            
            # Check consistency
            for i in range(1024):
                # Check if M part is 0
                m_part = rows[i] & ((1 << 216) - 1)
                if m_part == 0:
                    t_part = (rows[i] >> 216) & 1
                    if t_part != 0:
                        return None
            
            # Find a solution
            # W_flat has 216 entries.
            # Free variables are 0.
            # Pivot variables are determined.
            # We need to map columns to variables.
            # The GE process didn't track which column was the pivot for each row.
            # Let's redo GE tracking pivots.
            
            # Actually, for verification we just need *some* solution.
            # If consistent, we can find one by back substitution.
            # But we need to know the pivot columns.
            
            # Let's restart GE with tracking.
            rows = []
            for i in range(1024):
                m_bits = int.from_bytes(M[i].tobytes(), 'little')
                t_bit = T_flat[i]
                rows.append(m_bits | (t_bit << 216))
            
            pivot_cols = []
            pivot = 0
            for col in range(216):
                pivot_row = -1
                for i in range(pivot, 1024):
                    if (rows[i] >> col) & 1:
                        pivot_row = i
                        break
                if pivot_row == -1:
                    continue
                
                rows[pivot], rows[pivot_row] = rows[pivot_row], rows[pivot]
                pivot_cols.append(col)
                
                for i in range(1024):
                    if i != pivot and ((rows[i] >> col) & 1):
                        rows[i] ^= rows[pivot]
                
                pivot += 1
                if pivot == 216:
                    break
            
            # Back substitution
            W_flat = np.zeros(216, dtype=np.uint8)
            # pivot_cols has the column indices for rows 0..pivot-1
            # Row i corresponds to pivot_cols[i]
            # Equation: W[pivot_cols[i]] + sum_{j in free or higher pivot} coeff * W[j] = RHS
            # Since we did full elimination (Gauss-Jordan), the matrix is in reduced row echelon form.
            # Row i has a 1 in column pivot_cols[i] and 0 in all other pivot columns.
            # So W[pivot_cols[i]] = RHS - sum_{j not in pivot_cols, j < ...} ...
            # Actually, in RREF, Row i is: e_{pivot_cols[i]} + sum_{j in free_cols} c_{ij} e_j = b_i
            # So W[pivot_cols[i]] = b_i - sum_{j in free_cols} c_{ij} W[j]
            # Since free variables are 0, W[pivot_cols[i]] = b_i.
            
            for i in range(pivot):
                col = pivot_cols[i]
                rhs = (rows[i] >> 216) & 1
                W_flat[col] = rhs
            
            return W_flat

        # Main search loop
        for _ in range(20000):
            if time.time() > deadline:
                break
            
            U_rand = np.random.randint(0, 2, (8, 27), dtype=np.uint8)
            V_rand = np.random.randint(0, 2, (16, 27), dtype=np.uint8)
            
            # Build M
            # P = U[:, :, None] * V[None, :, :] -> (8, 16, 27)
            # This creates a large array. 8*16*27 = 3456 bytes. Fine.
            P = np.einsum('xr,yr->xyr', U_rand, V_rand, dtype=np.uint8)
            P_flat = P.reshape(128, 27)
            
            M = np.zeros((1024, 216), dtype=np.uint8)
            # M[z::8, z*27+r] = P_flat[:, r]
            # Vectorize this assignment
            # For each z, r:
            #   M[z::8, z*27+r] = P_flat[:, r]
            # This is 216 assignments.
            # Can we do it faster?
            # M shape (1024, 216).
            # M[i, j] where i = z + 8k, j = z*27 + r.
            # i = z + 8k => z = i % 8, k = i // 8.
            # j = z*27 + r => z = j // 27, r = j % 27.
            # So M[i, j] is non-zero only if i%8 == j//27.
            # Value is P_flat[k, r] where k=i//8, r=j%27.
            
            # Let's just loop z, r. It's fast enough.
            for z in range(8):
                for r in range(27):
                    M[z::8, z*27+r] = P_flat[:, r]
            
            W_flat = solve_f2(M, T_flat)
            if W_flat is not None:
                W = W_flat.reshape(8, 27)
                # Verify over Q (integers)
                U_int = U_rand.astype(int)
                V_int = V_rand.astype(int)
                W_int = W.astype(int)
                if verify(U_int, V_int, W_int):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None

    result = search_rank_27_f2()
    if result is not None:
        best_U, best_V, best_W = result
        best_rank = 27
    
    return {"U": best_U, "V": best_V, "W": best_W}
