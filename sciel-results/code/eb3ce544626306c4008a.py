"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for 6x6x6 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock limit

    def build_strassen_6x6():
        """Build the 6x6x6 decomposition using Strassen on 2x2 blocks (rank 189)."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_patterns = [
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
            ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
            ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
            ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
            ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
        ]

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val
                        col_idx += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_laderman_3x3_block():
        """Build 6x6x6 using Laderman's rank-23 algorithm on 3x3 blocks.
        6/3=2 blocks per dim, 8 block positions, 23 each = 184.
        """
        # Laderman's rank-23 decomposition for 3x3 matrix multiplication.
        # Each entry: (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict (i,j) -> coeff for A[i,j]
        # v_coeffs: dict (j,k) -> coeff for B[j,k]  
        # w_coeffs: dict (k,i) -> coeff for C[i,k] (i=row of C, k=col of C)
        
        laderman_terms = [
            # m1 = (a00+a10+a20)*(b00+b10+b20)
            ({(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(1,0):1,(2,0):1},
             {(0,0):1,(1,0):-1,(2,0):-1,(0,1):-1,(1,1):1,(2,1):1,(0,2):-1,(1,2):1,(2,2):1}),
            # m2 = (a00+a10)*(b10+b20)
            ({(0,0):1,(1,0):1}, {(1,0):1,(2,0):1},
             {(0,0):-1,(1,0):1,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):0,(1,2):0,(2,2):0}),
            # m3 = (a10+a20)*(b00+b10)
            ({(1,0):1,(2,0):1}, {(0,0):1,(1,0):1},
             {(0,0):-1,(1,0):0,(2,0):1,(0,1):0,(1,1):0,(2,1):0,(0,2):0,(1,2):0,(2,2):0}),
            # m4 = (a01+a11+a21)*(b01+b11+b21)
            ({(0,1):1,(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1},
             {(0,0):-1,(1,0):1,(2,0):1,(0,1):1,(1,1):-1,(2,1):-1,(0,2):-1,(1,2):1,(2,2):1}),
            # m5 = (a01+a11)*(b11+b21)
            ({(0,1):1,(1,1):1}, {(1,1):1,(2,1):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):-1,(1,1):1,(2,1):0,(0,2):0,(1,2):0,(2,2):0}),
            # m6 = (a11+a21)*(b01+b11)
            ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):-1,(1,1):0,(2,1):1,(0,2):0,(1,2):0,(2,2):0}),
            # m7 = (a02+a12+a22)*(b02+b12+b22)
            ({(0,2):1,(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1},
             {(0,0):-1,(1,0):1,(2,0):1,(0,1):-1,(1,1):1,(2,1):1,(0,2):1,(1,2):-1,(2,2):-1}),
            # m8 = (a02+a12)*(b12+b22)
            ({(0,2):1,(1,2):1}, {(1,2):1,(2,2):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):-1,(1,2):1,(2,2):0}),
            # m9 = (a12+a22)*(b02+b12)
            ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):-1,(1,2):0,(2,2):1}),
            # m10 = (a00+a01+a02)*(b20+b21+b22)
            ({(0,0):1,(0,1):1,(0,2):1}, {(2,0):1,(2,1):1,(2,2):1},
             {(0,0):-1,(1,0):-1,(2,0):-1,(0,1):1,(1,1):0,(2,1):0,(0,2):1,(1,2):0,(2,2):0}),
            # m11 = (a10+a11+a12)*(b20+b21+b22)
            ({(1,0):1,(1,1):1,(1,2):1}, {(2,0):1,(2,1):1,(2,2):1},
             {(0,0):1,(1,0):1,(2,0):1,(0,1):-1,(1,1):0,(2,1):0,(0,2):-1,(1,2):0,(2,2):0}),
            # m12 = (a20+a21+a22)*(b20+b21+b22)
            ({(2,0):1,(2,1):1,(2,2):1}, {(2,0):1,(2,1):1,(2,2):1},
             {(0,0):1,(1,0):-1,(2,0):-1,(0,1):1,(1,1):0,(2,1):0,(0,2):1,(1,2):0,(2,2):0}),
            # m13 = (a00+a02)*(b20+b21)
            ({(0,0):1,(0,2):1}, {(2,0):1,(2,1):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):1,(1,2):0,(2,2):0}),
            # m14 = (a01+a02)*(b20+b22)
            ({(0,1):1,(0,2):1}, {(2,0):1,(2,2):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):-1,(1,2):0,(2,2):0}),
            # m15 = (a10+a12)*(b21+b22)
            ({(1,0):1,(1,2):1}, {(2,1):1,(2,2):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):0,(1,2):1,(2,2):0}),
            # m16 = (a11+a12)*(b20+b21)
            ({(1,1):1,(1,2):1}, {(2,0):1,(2,1):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):0,(1,2):-1,(2,2):0}),
            # m17 = (a20+a21)*(b20+b22)
            ({(2,0):1,(2,1):1}, {(2,0):1,(2,2):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):0,(1,2):0,(2,2):1}),
            # m18 = (a21+a22)*(b20+b21)
            ({(2,1):1,(2,2):1}, {(2,0):1,(2,1):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):0,(1,2):0,(2,2):-1}),
            # m19 = (a00+a11)*(b00+b11)
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1},
             {(0,0):1,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):0,(1,2):0,(2,2):0}),
            # m20 = (a01+a12)*(b01+b12)
            ({(0,1):1,(1,2):1}, {(0,1):1,(1,2):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):1,(1,1):0,(2,1):0,(0,2):0,(1,2):0,(2,2):0}),
            # m21 = (a02+a10)*(b02+b10)
            ({(0,2):1,(1,0):1}, {(0,2):1,(1,0):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):1,(1,2):0,(2,2):0}),
            # m22 = (a10+a21)*(b10+b21)
            ({(1,0):1,(2,1):1}, {(1,0):1,(2,1):1},
             {(0,0):0,(1,0):1,(2,0):0,(0,1):0,(1,1):0,(2,1):0,(0,2):0,(1,2):0,(2,2):0}),
            # m23 = (a11+a22)*(b11+b22)
            ({(1,1):1,(2,2):1}, {(1,1):1,(2,2):1},
             {(0,0):0,(1,0):0,(2,0):0,(0,1):0,(1,1):1,(2,1):0,(0,2):0,(1,2):0,(2,2):0}),
        ]
        
        # Verify Laderman 3x3 correctness
        np.random.seed(999)
        A3 = np.random.randn(3, 3)
        B3 = np.random.randn(3, 3)
        C3_expected = A3 @ B3
        C3_computed = np.zeros((3, 3))
        for u_c, v_c, w_c in laderman_terms:
            M_u = sum(u_c.get((i, j), 0) * A3[i, j] for i in range(3) for j in range(3))
            M_v = sum(v_c.get((i, j), 0) * B3[i, j] for i in range(3) for j in range(3))
            M_r = M_u * M_v
            for (k, i), val in w_c.items():
                C3_computed[i, k] += M_r * val
        err3 = np.max(np.abs(C3_computed - C3_expected))
        
        if err3 > 1e-10:
            return None
        
        # Build 6x6x6 from 3x3 Laderman blocks
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in laderman_terms:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        
        # Verify full 6x6
        np.random.seed(777)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(rank):
            M_u = sum(U[i * m + j, r] * A_test[i, j] for i in range(n) for j in range(m))
            M_v = sum(V[j * p + k, r] * B_test[j, k] for j in range(m) for k in range(p))
            M_r = M_u * M_v
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += M_r * W[k * n + i, r]
        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-10:
            return None
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_winograd_strassen_kron():
        """
        Build 6x6x6 via Kronecker product of Winograd 3x3 (rank 5) and Strassen 2x2 (rank 7).
        Total rank = 5 * 7 = 35.
        """
        # Winograd 3x3 rank-5 decomposition
        # C = A*B for 3x3 matrices
        winograd_3x3 = [
            # Term 1: (a00+a10+a20) * (b00+b10+b20)
            ({(0, 0): 1, (1, 0): 1, (2, 0): 1}, {(0, 0): 1, (1, 0): 1, (2, 0): 1},
             {(0, 0): 1, (1, 0): 1, (2, 0): 1}),
            # Term 2: (a01+a11+a21) * (b01+b11+b21)
            ({(0, 1): 1, (1, 1): 1, (2, 1): 1}, {(0, 1): 1, (1, 1): 1, (2, 1): 1},
             {(0, 1): 1, (1, 1): 1, (2, 1): 1}),
            # Term 3: (a02+a12+a22) * (b02+b12+b22)
            ({(0, 2): 1, (1, 2): 1, (2, 2): 1}, {(0, 2): 1, (1, 2): 1, (2, 2): 1},
             {(0, 2): 1, (1, 2): 1, (2, 2): 1}),
            # Term 4: (a00+a10) * (b01+b11)
            ({(0, 0): 1, (1, 0): 1}, {(0, 1): 1, (1, 1): 1},
             {(0, 0): -1, (1, 1): -1, (0, 1): 1, (1, 0): 1}),
            # Term 5: (a01+a11) * (b00+b10)
            ({(0, 1): 1, (1, 1): 1}, {(0, 0): 1, (1, 0): 1},
             {(0, 0): -1, (1, 1): -1, (0, 1): 1, (1, 0): 1}),
        ]

        # Verify Winograd 3x3 correctness
        def verify_winograd():
            np.random.seed(123)
            A3 = np.random.randn(3, 3)
            B3 = np.random.randn(3, 3)
            C3_expected = A3 @ B3
            C3_computed = np.zeros((3, 3))
            for u_c, v_c, w_c in winograd_3x3:
                M_u = sum(u_c.get((i, j), 0) * A3[i, j] for i in range(3) for j in range(3))
                M_v = sum(v_c.get((i, j), 0) * B3[i, j] for i in range(3) for j in range(3))
                M_r = M_u * M_v
                for (k, i), val in w_c.items():
                    C3_computed[i, k] += M_r * val
            return np.max(np.abs(C3_computed - C3_expected))
        
        wg_err = verify_winograd()
        if wg_err > 1e-10:
            return None

        # Strassen 2x2 patterns
        strassen_2x2 = [
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
            ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
            ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
            ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
            ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
        ]

        rank = 5 * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for r3, (u3, v3, w3) in enumerate(winograd_3x3):
            for r2, (u2, v2, w2) in enumerate(strassen_2x2):
                r = r3 * 7 + r2
                
                # Fill U: index (i, j) = (a_i * 2 + a_ip, b_j * 2 + b_jp)
                for (a_i, b_j), val3 in u3.items():
                    for (a_ip, b_jp), val2 in u2.items():
                        i = a_i * 2 + a_ip
                        j = b_j * 2 + b_jp
                        U[i * m + j, r] += val3 * val2
                
                # Fill V: index (j, k) = (b_j * 2 + b_jp, c_k * 2 + c_kp)
                for (b_j, c_k), val3 in v3.items():
                    for (b_jp, c_kp), val2 in v2.items():
                        j = b_j * 2 + b_jp
                        k = c_k * 2 + c_kp
                        V[j * p + k, r] += val3 * val2
                
                # Fill W: index (k, i) = (c_k * 2 + c_kp, a_i * 2 + a_ip)
                for (c_k, a_i), val3 in w3.items():
                    for (c_kp, a_ip), val2 in w2.items():
                        k = c_k * 2 + c_kp
                        i = a_i * 2 + a_ip
                        W[k * n + i, r] += val3 * val2

        # Verify the full 6x6 decomposition
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            M_u = sum(U[i * m + j, r] * A_test[i, j] for i in range(n) for j in range(m))
            M_v = sum(V[j * p + k, r] * B_test[j, k] for j in range(m) for k in range(p))
            M_r = M_u * M_v
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += M_r * W[k * n + i, r]
        
        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-10:
            return None

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try Laderman 3x3 blocks first (rank 184)
    best_solution = None
    best_rank = float('inf')
    
    lad_result = build_laderman_3x3_block()
    if lad_result is not None:
        best_solution = lad_result
        best_rank = 184
    
    # Fallback to Strassen 6x6 (rank 189)
    if best_solution is None:
        best_solution = build_strassen_6x6()
        best_rank = 189

    return best_solution
# EVOLVE-BLOCK-END
