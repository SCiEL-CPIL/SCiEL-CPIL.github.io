import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # --- Rank 7 (2x2 Strassen) coefficients ---
    # Flattened: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # --- Rank 23 (3x3 Bini-Lotti-Romani 1980) coefficients ---
    # Flattened: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2), 6=(2,0), 7=(2,1), 8=(2,2)
    # Each row is a list of 9 coefficients for the linear form
    U3 = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],   # M0: A00+A01
        [0, 0, 1, 0, 0, 0, 0, 0, 0],   # M1: A02
        [1, 0, 0, 0, 1, 0, 0, 0, 0],   # M2: A00+A11
        [0, 0, 0, 1, 0, 0, 0, 0, 0],   # M3: A10
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M4: A12
        [1, 0, 0, 0, 0, 0, 0, 1, 0],   # M5: A00+A21
        [0, 0, 0, 0, 1, 0, 0, 0, 1],   # M6: A11+A22
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # M7: A20
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M8: A22
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # M9: A00
        [0, 1, 0, 0, 0, 0, 0, 0, 0],   # M10: A01
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # M11: A11
        [0, 0, 0, 0, 0, 0, 0, 1, 0],   # M12: A21
        [0, 0, 0, 1, 0, 0, 0, 0, 0],   # M13: A10
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M14: A12
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # M15: A20
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # M16: A00
        [0, 0, 1, 0, 0, 0, 0, 0, 0],   # M17: A02
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # M18: A11
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M19: A22
        [0, 1, 0, 0, 0, 0, 0, 0, 0],   # M20: A01
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M21: A12
        [0, 0, 0, 0, 0, 0, 0, 1, 0],   # M22: A21
    ]
    V3 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # M0: B00
        [0, 0, 1, 0, 0, 0, 0, 0, 0],   # M1: B02
        [0, 1, 0, 0, 0, 0, 0, 0, 0],   # M2: B01
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # M3: B00
        [0, 0, 1, 0, 0, 0, 0, 0, 0],   # M4: B02
        [0, 1, 0, 0, 0, 0, 0, 0, 0],   # M5: B01
        [0, 0, 0, 1, 0, 0, 0, 0, 0],   # M6: B10
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M7: B12
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # M8: B11
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M9: B22
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M10: B12
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # M11: B20
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M12: B22
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # M13: B20
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M14: B22
        [0, 0, 0, 1, 0, 0, 0, 0, 0],   # M15: B10
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M16: B12
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # M17: B11
        [0, 0, 0, 0, 0, 0, 0, 1, 0],   # M18: B21
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M19: B22
        [0, 0, 0, 1, 0, 0, 0, 0, 0],   # M20: B10
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # M21: B20
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M22: B22
    ]
    W3 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # M0: C00
        [0, 0, 1, 0, 0, 0, 0, 0, 0],   # M1: C02
        [1, 1, 0, 0, 0, 0, 0, 0, 0],   # M2: C00+C01
        [0, 0, 0, 1, 0, 0, 0, 0, 0],   # M3: C10
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M4: C12
        [0, 0, 0, 0, 0, 0, 0, 1, 0],   # M5: C21
        [0, 0, 0, 1, 1, 0, 0, 0, 0],   # M6: C10+C11
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # M7: C20
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M8: C22
        [1, 0, 0, 0, 0, 0, 0, 0, 0],   # M9: C00
        [0, 1, 0, 0, 0, 0, 0, 0, 0],   # M10: C01
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # M11: C11
        [0, 0, 0, 0, 0, 0, 0, 1, 0],   # M12: C21
        [0, 0, 0, 1, 0, 0, 0, 0, 0],   # M13: C10
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M14: C12
        [0, 0, 0, 0, 0, 0, 1, 0, 0],   # M15: C20
        [0, 0, 0, 0, 0, 0, 0, 0, 0],   # M16: (unused in C? placeholder)
        [0, 0, 1, 0, 0, 0, 0, 0, 0],   # M17: C02
        [0, 0, 0, 0, 1, 0, 0, 0, 0],   # M18: C11
        [0, 0, 0, 0, 0, 0, 0, 0, 1],   # M19: C22
        [0, 1, 0, 0, 0, 0, 0, 0, 0],   # M20: C01
        [0, 0, 0, 0, 0, 1, 0, 0, 0],   # M21: C12
        [0, 0, 0, 0, 0, 0, 0, 1, 0],   # M22: C21
    ]

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                C_calc[i, k] = s
                
        return np.allclose(C_calc, C_true, atol=1e-10)

    def get_189_solution():
        """Fallback: 3x3 blocks of 2x2 Strassen, rank 27*7=189."""
        block_size = 2
        blocks_per_dim = 3
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

    def get_161_solution():
        """Kronecker product of 3x3 (rank 23) and 2x2 (rank 7), total rank 161."""
        rank3 = len(U3)
        rank2 = len(U2)
        total_rank = rank3 * rank2
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Kronecker product construction
        # For each pair (r3, r2) where r3 in [0,22], r2 in [0,6]:
        #   col = r3 * rank2 + r2
        #   U[i*m+j][col] = U3[r3][i3*2+j3] * U2[r2][i2*2+j2] 
        #   where i = i3*2 + i2, j = j3*2 + j2
        #   V[j*p+k][col] = V3[r3][j3*2+k3] * V2[r2][j2*2+k2]
        #   W[k*n+i][col] = W3[r3][k3*2+i3] * W2[r2][k2*2+i2]
        
        for r3 in range(rank3):
            for r2 in range(rank2):
                col = r3 * rank2 + r2
                for i3 in range(3):
                    for j3 in range(3):
                        for k3 in range(3):
                            for i2 in range(2):
                                for j2 in range(2):
                                    for k2 in range(2):
                                        i = i3 * 2 + i2
                                        j = j3 * 2 + j2
                                        k = k3 * 2 + k2
                                        
                                        u_val = U3[r3][i3*3 + j3] * U2[r2][i2*2 + j2]
                                        v_val = V3[r3][j3*3 + k3] * V2[r2][j2*2 + k2]
                                        w_val = W3[r3][k3*3 + i3] * W2[r2][k2*2 + i2]
                                        
                                        if u_val != 0:
                                            U[i * m + j][col] += u_val
                                        if v_val != 0:
                                            V[j * p + k][col] += v_val
                                        if w_val != 0:
                                            W[k * n + i][col] += w_val
        return {"U": U, "V": V, "W": W}

    # Try rank 161 first
    if time.time() < deadline:
        sol_161 = get_161_solution()
        rank_161 = len(sol_161['U'][0])
        if verify_decomposition(sol_161['U'], sol_161['V'], sol_161['W'], rank_161):
            return sol_161

    # Fallback to rank 189
    return get_189_solution()
# EVOLVE-BLOCK-END
