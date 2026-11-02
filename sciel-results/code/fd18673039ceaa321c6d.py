import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def schoolbook():
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

    def find_decomp(n, m, p, target_rank, max_restarts=1000):
        """
        Finds a decomposition of rank target_rank for n x m * m x p.
        Returns dict with U, V, W or None.
        """
        num_eqs = n * m * p
        num_vars_W = p * n * target_rank
        
        # Values for U and V
        vals = [-1, -0.5, 0, 0.5, 1]
        
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
                
            U_rand = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)])
            V_rand = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)])
            
            # Check if U and V are too sparse or zero
            if np.all(U_rand == 0) or np.all(V_rand == 0):
                continue
                
            A = np.zeros((num_eqs, num_vars_W))
            b = np.ones(num_eqs)
            
            # Build system
            # Equation index: (i, j, k) -> i*m*p + j*p + k
            # Variable index: (j, i, r) -> (j*n + i)*target_rank + r
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(target_rank):
                            u_val = U_rand[i * m + j, r]
                            v_val = V_rand[j * p + k, r]
                            if u_val != 0 and v_val != 0:
                                var_idx = (k * n + i) * target_rank + r
                                A[eq_idx, var_idx] = u_val * v_val
            
            try:
                w_sol, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if solution is close to half-integers
            w_rounded = np.round(w_sol * 2) / 2.0
            
            # Verify
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            u_val = U_rand[i * m + j, r]
                            v_val = V_rand[j * p + k, r]
                            w_val = w_rounded[(k * n + i) * target_rank + r]
                            val += u_val * v_val * w_val
                        T_recon[i, j, k] = val
                        
            if np.allclose(T_recon, 1.0, atol=1e-9):
                # Convert to lists
                U_list = [[U_rand[i * m + j, r] for r in range(target_rank)] for i in range(n) for j in range(m)]
                V_list = [[V_rand[j * p + k, r] for r in range(target_rank)] for j in range(m) for k in range(p)]
                W_list = [[w_rounded[(k * n + i) * target_rank + r] for r in range(target_rank)] for k in range(p) for i in range(n)]
                
                # Final strict verification
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0
                            for r in range(target_rank):
                                s_val += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                            if abs(s_val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return {"U": U_list, "V": V_list, "W": W_list}
            
        return None

    # Strategy 1: Find rank 14 for 3x3x2, then combine two to get rank 28 for 3x3x4
    # 3x3x2: n=3, m=3, p=2
    decomp_332 = find_decomp(3, 3, 2, 14, max_restarts=500)
    
    if decomp_332:
        # Construct 3x3x4 from two 3x3x2
        # U: 3x3 x 28. First 14 cols from decomp_332['U'], next 14 same.
        # V: 3x4 x 28. 
        #   V rows: k (0..2), j (0..3).
        #   For j=0,1 (first block), use decomp_332['V'][k, 0..1]
        #   For j=2,3 (second block), use decomp_332['V'][k, 0..1]
        # W: 4x3 x 28.
        #   W rows: j (0..3), i (0..2).
        #   For j=0,1, use decomp_332['W'][0..1, i]
        #   For j=2,3, use decomp_332['W'][0..1, i]
        
        U_332 = decomp_332['U']
        V_332 = decomp_332['V']
        W_332 = decomp_332['W']
        
        R = 14
        U_new = [[0]*28 for _ in range(9)]
        V_new = [[0]*28 for _ in range(12)]
        W_new = [[0]*28 for _ in range(12)]
        
        # Fill U
        for i in range(3):
            for k in range(3):
                for r in range(14):
                    U_new[i*3+k][r] = U_332[i*3+k][r]
                    U_new[i*3+k][r+14] = U_332[i*3+k][r]
                    
        # Fill V
        # V_332 is 3x2 (k, j). j in 0,1.
        for k in range(3):
            for j in range(2):
                for r in range(14):
                    # First block j=0,1
                    V_new[k*4+j][r] = V_332[k*2+j][r]
                    # Second block j=2,3
                    V_new[k*4+2+j][r+14] = V_332[k*2+j][r]
                    
        # Fill W
        # W_332 is 2x3 (j, i). j in 0,1.
        for j in range(2):
            for i in range(3):
                for r in range(14):
                    # First block j=0,1
                    W_new[j*3+i][r] = W_332[j*3+i][r]
                    # Second block j=2,3
                    W_new[(2+j)*3+i][r+14] = W_332[j*3+i][r]
                    
        return {"U": U_new, "V": V_new, "W": W_new}

    # Strategy 2: Try to find rank 24 for 3x3x4 directly
    decomp_334 = find_decomp(3, 3, 4, 24, max_restarts=200)
    if decomp_334:
        return decomp_334

    # Strategy 3: Try rank 28 directly for 3x3x4 (in case 332 failed but 334 rank 28 works)
    decomp_334_28 = find_decomp(3, 3, 4, 28, max_restarts=200)
    if decomp_334_28:
        return decomp_334_28

    return schoolbook()
# EVOLVE-BLOCK-END
