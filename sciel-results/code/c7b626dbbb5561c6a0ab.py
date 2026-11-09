"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Schoolbook solution as fallback
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify_exact(U, V, W, rank):
        """Exact verification using Fraction arithmetic."""
        U_flat = U
        V_flat = V
        W_flat = W
        n_val = n
        m_val = m
        p_val = p
        rank_val = rank
        
        for i in range(n_val):
            for j in range(m_val):
                u_row = U_flat[i*m_val+j]
                for k in range(p_val):
                    v_row = V_flat[j*p_val+k]
                    w_row = W_flat[k*n_val+i]
                    s = Fraction(0)
                    for c in range(rank_val):
                        s += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if s != Fraction(1):
                        return False
        return True

    def als_optimize(U_init, V_init, W_init, rank, max_iter=50, mask_U=None, mask_V=None, mask_W=None):
        """Alternating least squares in float with optional masks (0/1) to constrain support."""
        U = [row[:] for row in U_init]
        V = [row[:] for row in V_init]
        W = [row[:] for row in W_init]
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p+k][c] * W[k*n+i][c]
                    b = np.ones(p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        if mask_U:
                            sol = np.where(mask_U[i*m+j], sol, 0)
                        U[i*m+j] = sol.tolist()
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i*m+j][c] * W[k*n+i][c]
                    b = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        if mask_V:
                            sol = np.where(mask_V[j*p+k], sol, 0)
                        V[j*p+k] = sol.tolist()
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                    b = np.ones(m)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        if mask_W:
                            sol = np.where(mask_W[k*n+i], sol, 0)
                        W[k*n+i] = sol.tolist()
                    except:
                        pass
        
        return U, V, W
    
    def snap_to_half_int(U, V, W, rank):
        """Snap values to nearest half-integer."""
        U_s = [[round(U[i][c] * 2) / 2 for c in range(rank)] for i in range(n*m)]
        V_s = [[round(V[i][c] * 2) / 2 for c in range(rank)] for i in range(m*p)]
        W_s = [[round(W[i][c] * 2) / 2 for c in range(rank)] for i in range(p*n)]
        return U_s, V_s, W_s

    # Strategy: Try rank 32 with constrained support (Strassen + Standard)
    target_rank = 32
    num_seeds = 5
    
    # Masks for rank 32:
    # Cols 0-22: V rows 9-11 (k=3) are 0, W rows 9-11 (k=3) are 0
    # Cols 23-31: V rows 0-8 (k=0,1,2) are 0, W rows 0-8 (k=0,1,2) are 0
    mask_V = [[1.0]*target_rank for _ in range(m*p)]
    mask_W = [[1.0]*target_rank for _ in range(p*n)]
    mask_U = [[1.0]*target_rank for _ in range(n*m)]
    
    for c in range(23):
        for r in range(9, 12): # V rows for k=3
            mask_V[r][c] = 0.0
        for r in range(9, 12): # W rows for k=3
            mask_W[r][c] = 0.0
            
    for c in range(23, 32):
        for r in range(0, 9): # V rows for k=0,1,2
            mask_V[r][c] = 0.0
        for r in range(0, 9): # W rows for k=0,1,2
            mask_W[r][c] = 0.0

    for seed in range(num_seeds):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed * 1000 + target_rank * 13)
        np.random.seed(seed * 1000 + target_rank * 13)
        
        U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
        
        # Apply masks to initial random values
        for i in range(n*m):
            for c in range(target_rank):
                if mask_U[i][c] == 0:
                    U[i][c] = 0.0
        for i in range(m*p):
            for c in range(target_rank):
                if mask_V[i][c] == 0:
                    V[i][c] = 0.0
        for i in range(p*n):
            for c in range(target_rank):
                if mask_W[i][c] == 0:
                    W[i][c] = 0.0

        U_opt, V_opt, W_opt = als_optimize(U, V, W, target_rank, max_iter=30, mask_U=mask_U, mask_V=mask_V, mask_W=mask_W)
        
        U_snap, V_snap, W_snap = snap_to_half_int(U_opt, V_opt, W_opt, target_rank)
        
        if verify_exact(U_snap, V_snap, W_snap, target_rank):
            best_rank = target_rank
            best_U, best_V, best_W = U_snap, V_snap, W_snap
            break

    # If rank 32 not found, try rank 33? Unlikely to be better than 32 if 32 is the true rank.
    # But if 32 fails, we return 36.

    def convert_output(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float):
                    if val == int(val):
                        new_row.append(int(val))
                    else:
                        new_row.append(val)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {
        "U": convert_output(best_U),
        "V": convert_output(best_V),
        "W": convert_output(best_W)
    }
# EVOLVE-BLOCK-END
