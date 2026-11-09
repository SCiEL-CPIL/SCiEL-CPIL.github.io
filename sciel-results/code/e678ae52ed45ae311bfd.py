import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Direct naive rank-nmp identity decomposition
    def get_naive_factors():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Each term corresponds to the (i,j,k) entry of the matmul tensor
                    # U[i*m + j, r] = 1
                    U[i * m + j][col_idx] = 1
                    # V[j*p + k, r] = 1
                    V[j * p + k][col_idx] = 1
                    # W[k*n + i, r] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen 2x2 fallback
    def get_strassen_factors():
        block_size = 2
        blocks_per_dim = 3 # 6 / 2
        rank = (blocks_per_dim ** 3) * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c][col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c][col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i][col_idx] += val
                        col_idx += 1
        return {"U": U, "V": V, "W": W}

    # Laderman 3x3 attempt
    def get_laderman_factors():
        block_size = 3
        blocks_per_dim = 2 # 6 / 3
        rank = (blocks_per_dim ** 3) * 23
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Correct Laderman's algorithm coefficients
        laderman_patterns = [
            # M1
            ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
            # M2
            ({(0,1): 1}, {(1,1): 1}, {(0,1): 1}),
            # M3
            ({(0,2): 1}, {(2,2): 1}, {(0,2): 1}),
            # M4
            ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
            # M5
            ({(1,1): 1}, {(1,1): 1}, {(1,1): 1}),
            # M6
            ({(1,2): 1}, {(2,2): 1}, {(1,2): 1}),
            # M7
            ({(2,0): 1}, {(0,0): 1}, {(2,0): 1}),
            # M8
            ({(2,1): 1}, {(1,1): 1}, {(2,1): 1}),
            # M9
            ({(2,2): 1}, {(2,2): 1}, {(2,2): 1}),
            # M10
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): -1, (1,1): 1}),
            # M11
            ({(1,1): 1, (2,2): 1}, {(1,1): 1, (2,2): 1}, {(1,1): -1, (2,2): 1}),
            # M12
            ({(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}, {(0,0): -1, (2,2): 1}),
            # M13
            ({(0,1): 1, (1,0): 1}, {(1,0): 1, (0,1): 1}, {(0,1): -1, (1,0): -1}),
            # M14
            ({(1,0): 1, (2,0): 1}, {(2,0): 1, (1,0): 1}, {(1,0): -1, (2,0): -1}),
            # M15
            ({(1,1): 1, (2,1): 1}, {(2,1): 1, (1,1): 1}, {(1,1): -1, (2,1): -1}),
            # M16
            ({(0,1): 1, (2,1): 1}, {(2,1): 1, (0,1): 1}, {(0,1): -1, (2,1): -1}),
            # M17
            ({(0,2): 1, (1,2): 1}, {(1,2): 1, (0,2): 1}, {(0,2): -1, (1,2): -1}),
            # M18
            ({(1,2): 1, (2,2): 1}, {(2,2): 1, (1,2): 1}, {(1,2): -1, (2,2): -1}),
            # M19
            ({(0,2): 1, (2,2): 1}, {(2,2): 1, (0,2): 1}, {(0,2): -1, (2,2): -1}),
            # M20
            ({(0,0): 1, (0,1): 1, (0,2): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,0): 1, (0,1): -1, (0,2): -1, (1,0): 1, (1,1): -1, (1,2): -1, (2,0): 1, (2,1): -1, (2,2): -1}),
            # M21
            ({(0,1): 1, (1,1): 1, (2,1): 1}, {(1,0): 1, (1,1): 1, (2,1): 1}, {(0,1): 1, (0,2): -1, (1,1): 1, (1,2): -1, (2,1): 1, (2,2): -1}),
            # M22
            ({(0,2): 1, (1,2): 1, (2,2): 1}, {(2,0): 1, (2,1): 1, (2,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}),
            # M23
            ({(0,0): 1, (1,0): 1, (2,0): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}, {(0,0): -1, (0,1): 1, (0,2): 1, (1,0): -1, (1,1): 1, (1,2): 1, (2,0): -1, (2,1): 1, (2,2): 1}),
        ]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in laderman_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c][col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c][col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i][col_idx] += val
                        col_idx += 1
        return {"U": U, "V": V, "W": W}

    def verify_factors(factors):
        U = np.array(factors["U"])
        V = np.array(factors["V"])
        W = np.array(factors["W"])
        
        # Multiple random test matrices for robustness
        np.random.seed(42)
        for trial in range(3):
            A = np.random.randint(-10, 10, (n, m))
            B = np.random.randint(-10, 10, (m, p))
            C = A @ B
            
            U_reshaped = U.reshape(n, m, -1)
            V_reshaped = V.reshape(m, p, -1)
            W_reshaped = W.reshape(p, n, -1)
            
            u_prod = np.einsum('ijr,ij->ir', U_reshaped, A)
            v_prod = np.einsum('jkr,jk->kr', V_reshaped, B)
            
            term = W_reshaped * u_prod[:, np.newaxis, :] * v_prod[np.newaxis, :, :]
            C_calc = np.sum(term, axis=2)
            
            if not np.allclose(C_calc, C, atol=1e-6):
                return False
        return True

    # Try Laderman first (lowest rank if valid)
    try:
        laderman_factors = get_laderman_factors()
        if verify_factors(laderman_factors):
            return laderman_factors
    except Exception:
        pass
    
    # Fallback to Strassen
    try:
        strassen_factors = get_strassen_factors()
        if verify_factors(strassen_factors):
            return strassen_factors
    except Exception:
        pass
    
    # Final fallback: naive rank-216 identity (always valid)
    return get_naive_factors()
# EVOLVE-BLOCK-END
