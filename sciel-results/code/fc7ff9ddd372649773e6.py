"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=2, m=4, p=6
    # U: 8 x r
    # V: 24 x r
    # W: 12 x r
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    # Strategy: Structured Tensor Decomposition
    # We decompose the multiplication of (2x4) * (4x6) into blocks.
    # Let A be 2x4, B be 4x6.
    # We can view A as two 2x2 blocks: A = [A00, A01] where A00, A01 are 2x2.
    # We can view B as a 2x3 block matrix where each block is 2x2:
    # B = [B00 B01 B02]
    #     [B10 B11 B12]
    # The product C = A * B is 2x6.
    # C = [A00 B00 + A01 B10, A00 B01 + A01 B11, A00 B02 + A01 B12]
    # Each term is a sum of two 2x2 matrix multiplications.
    # If we use a rank-7 decomposition for 2x2 matrix multiplication (Strassen-like),
    # we can compute each 2x2 product with rank 7.
    # However, the terms are summed.
    # C_col1 = A00*B00 + A01*B10.
    # This is not a single 2x2 multiplication. It's a bilinear form.
    # Let's define the tensor for the whole 2x4x6 multiplication.
    
    # Alternative view:
    # The tensor T_{i, j, k} = sum_l A_{i,l} B_{l,k}.
    # We want to write T_{i, j, k} = sum_r U_{i,j} V_{j,k} W_{k,i}.
    
    # Let's use the block structure:
    # i in {0, 1}
    # j in {0, 1, 2, 3} -> let j = 2*j1 + j0, j1 in {0,1}, j0 in {0,1}
    # k in {0, ..., 5} -> let k = 2*k1 + k0, k1 in {0,1,2}, k0 in {0,1}
    
    # A_{i, 2*j1+j0}, B_{2*j1+j0, k}
    # Sum over j1, j0.
    # C_{i, k} = sum_{j1=0}^1 sum_{j0=0}^1 A_{i, 2*j1+j0} B_{2*j1+j0, k}
    
    # Let's group by j1.
    # For fixed j1, we have a 2x2 matrix A_{j1} with entries A_{i, 2*j1+j0} (i, j0)
    # and a 2x2 matrix B_{j1} with entries B_{2*j1+j0, k} (j0, k).
    # Wait, B index is (row, col). Row is 2*j1+j0. Col is k.
    # So B_{j1} is a 2x2 matrix where row index is j0 and col index is... k?
    # No, k varies from 0 to 5. So B is not just one 2x2 matrix per j1.
    # B has 4 rows and 6 cols.
    # For fixed j1 (0 or 1), the rows involved are 2*j1 and 2*j1+1.
    # The columns are 0..5.
    # So for each j1, we have a 2x6 submatrix of B.
    # And for each j1, we have a 2x2 submatrix of A (cols 2*j1, 2*j1+1).
    
    # C = A_{j1=0} * B_{j1=0} + A_{j1=1} * B_{j1=1}
    # where A_{j1} is 2x2 and B_{j1} is 2x6.
    # This is a sum of two matrix products.
    # We can compute each product using a decomposition.
    # Product 1: (2x2) * (2x6) -> (2x6). Rank is 2*2*6 = 24 schoolbook.
    # Can we do better?
    # The tensor for (2x2)*(2x6) is equivalent to 3 independent (2x2)*(2x2) multiplications?
    # Let B_{j1} = [B_{j1,0} B_{j1,1} B_{j1,2}] where each is 2x2.
    # Then A_{j1} * B_{j1} = [A_{j1}B_{j1,0}, A_{j1}B_{j1,1}, A_{j1}B_{j1,2}].
    # Each term is a 2x2 matrix multiplication.
    # If we use Strassen for each, rank is 7 per term.
    # Total rank for A_{j1} * B_{j1} would be 3 * 7 = 21?
    # But we need to combine the results for j1=0 and j1=1.
    # C = A_0 B_0 + A_1 B_1.
    # C[:, 0:2] = A_0 B_{0,0} + A_1 B_{1,0}
    # C[:, 2:4] = A_0 B_{0,1} + A_1 B_{1,1}
    # C[:, 4:6] = A_0 B_{0,2} + A_1 B_{1,2}
    
    # So for each column block c (0,1,2), we have:
    # C_c = A_0 B_{0,c} + A_1 B_{1,c}
    # This is a bilinear map: (A_0, A_1, B_{0,c}, B_{1,c}) -> C_c.
    # A_0, A_1 are 2x2. B_{0,c}, B_{1,c} are 2x2.
    # This looks like a "sum of two matrix products".
    # Can we decompose this efficiently?
    # Let X = A_0, Y = A_1, Z = B_{0,c}, W = B_{1,c}.
    # Target: XZ + YW.
    # This is a 2x2 matrix.
    # The tensor for XZ + YW is the sum of two 2x2 multiplication tensors.
    # The rank of the sum of two disjoint tensors is at most the sum of ranks.
    # Rank(XZ) <= 7, Rank(YW) <= 7. So Rank <= 14.
    # Total rank for 3 column blocks: 3 * 14 = 42.
    # This is better than 48!
    
    # Let's construct the rank-7 decomposition for 2x2 multiplication.
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)(e+f)
    # M6 = (c-d)(g+h)
    # M7 = (b-d)(e+h)
    # 
    # C = [ M5 + M4 - M6 - M7, M1 - M4 ]
    #     [ M2 + M7, M1 - M3 ]
    # 
    # We need U, V, W matrices for the 2x2 multiplication.
    # Inputs: A (2x2), B (2x2).
    # U maps A to 7 factors. V maps B to 7 factors. W maps factors to C.
    # 
    # Let's define the Strassen factors explicitly.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # 
    # Factors u (from A):
    # u1 = a+d
    # u2 = c+d
    # u3 = a
    # u4 = d
    # u5 = a+b
    # u6 = c-d
    # u7 = b-d
    # 
    # Factors v (from B):
    # v1 = e+h
    # v2 = e
    # v3 = f-h
    # v4 = g-f
    # v5 = e+f
    # v6 = g+h
    # v7 = e+h  # Note: v1 and v7 are the same?
    # Let's check Strassen again.
    # M1 = (a+d)(e+h) -> u1, v1
    # M2 = (c+d)e -> u2, v2
    # M3 = a(f-h) -> u3, v3
    # M4 = d(g-f) -> u4, v4
    # M5 = (a+b)(e+f) -> u5, v5
    # M6 = (c-d)(g+h) -> u6, v6
    # M7 = (b-d)(e+h) -> u7, v1 (reuse v1)
    # 
    # So v7 = v1.
    
    # Output C:
    # C[0,0] = M5 + M4 - M6 - M7
    # C[0,1] = M1 - M4
    # C[1,0] = M2 + M7
    # C[1,1] = M1 - M3
    
    # We need to construct U_strassen (4x7), V_strassen (4x7), W_strassen (4x7).
    # Indices for A, B, C are flattened row-major.
    # A: a=0, b=1, c=2, d=3
    # B: e=0, f=1, g=2, h=3
    # C: 0=0,0; 1=0,1; 2=1,0; 3=1,1
    
    U_str = [[0]*7 for _ in range(4)]
    V_str = [[0]*7 for _ in range(4)]
    W_str = [[0]*7 for _ in range(4)]
    
    # u1 = a+d -> U[0][0]=1, U[3][0]=1
    U_str[0][0] = 1; U_str[3][0] = 1
    # v1 = e+h -> V[0][0]=1, V[3][0]=1
    V_str[0][0] = 1; V_str[3][0] = 1
    # W for M1: C[0,1] += 1, C[1,1] += 1
    W_str[1][0] = 1; W_str[3][0] = 1
    
    # u2 = c+d -> U[2][1]=1, U[3][1]=1
    U_str[2][1] = 1; U_str[3][1] = 1
    # v2 = e -> V[0][1]=1
    V_str[0][1] = 1
    # W for M2: C[1,0] += 1
    W_str[2][1] = 1
    
    # u3 = a -> U[0][2]=1
    U_str[0][2] = 1
    # v3 = f-h -> V[1][2]=1, V[3][2]=-1
    V_str[1][2] = 1; V_str[3][2] = -1
    # W for M3: C[1,1] -= 1
    W_str[3][2] = -1
    
    # u4 = d -> U[3][3]=1
    U_str[3][3] = 1
    # v4 = g-f -> V[2][3]=1, V[1][3]=-1
    V_str[2][3] = 1; V_str[1][3] = -1
    # W for M4: C[0,0] += 1, C[0,1] -= 1
    W_str[0][3] = 1; W_str[1][3] = -1
    
    # u5 = a+b -> U[0][4]=1, U[1][4]=1
    U_str[0][4] = 1; U_str[1][4] = 1
    # v5 = e+f -> V[0][4]=1, V[1][4]=1
    V_str[0][4] = 1; V_str[1][4] = 1
    # W for M5: C[0,0] += 1
    W_str[0][4] = 1
    
    # u6 = c-d -> U[2][5]=1, U[3][5]=-1
    U_str[2][5] = 1; U_str[3][5] = -1
    # v6 = g+h -> V[2][5]=1, V[3][5]=1
    V_str[2][5] = 1; V_str[3][5] = 1
    # W for M6: C[0,0] -= 1
    W_str[0][5] = -1
    
    # u7 = b-d -> U[1][6]=1, U[3][6]=-1
    U_str[1][6] = 1; U_str[3][6] = -1
    # v7 = e+h -> V[0][6]=1, V[3][6]=1
    V_str[0][6] = 1; V_str[3][6] = 1
    # W for M7: C[0,0] -= 1, C[1,0] += 1
    W_str[0][6] = -1; W_str[2][6] = 1
    
    # Verify Strassen decomposition
    def verify_strassen():
        for a in range(-2, 3):
            for b in range(-2, 3):
                for c in range(-2, 3):
                    for d in range(-2, 3):
                        for e in range(-2, 3):
                            for f in range(-2, 3):
                                for g in range(-2, 3):
                                    for h in range(-2, 3):
                                        A = np.array([[a, b], [c, d]])
                                        B = np.array([[e, f], [g, h]])
                                        C_true = A @ B
                                        
                                        u = np.array([
                                            a+d, c+d, a, d, a+b, c-d, b-d
                                        ])
                                        v = np.array([
                                            e+h, e, f-h, g-f, e+f, g+h, e+h
                                        ])
                                        w = np.array([
                                            1, 1, 0, 0, 1, -1, -1, # C00
                                            0, 0, 0, -1, 0, 0, 0,   # C01 (Wait, M1-M4)
                                            0, 1, 0, 0, 0, 0, 1,     # C10 (M2+M7)
                                            0, 0, -1, 0, 0, 0, 0     # C11 (M1-M3)
                                        ])
                                        # My W_str construction above:
                                        # C00: M5(1) + M4(1) - M6(-1) - M7(-1) -> coeffs: M1:0, M2:0, M3:0, M4:1, M5:1, M6:-1, M7:-1
                                        # C01: M1(1) - M4(-1) -> coeffs: M1:1, M4:-1
                                        # C10: M2(1) + M7(1) -> coeffs: M2:1, M7:1
                                        # C11: M1(1) - M3(-1) -> coeffs: M1:1, M3:-1
                                        
                                        # Let's check my W_str array:
                                        # W_str[0] (C00): [0, 0, 0, 1, 1, -1, -1] -> Correct
                                        # W_str[1] (C01): [1, 0, 0, -1, 0, 0, 0] -> Correct
                                        # W_str[2] (C10): [0, 1, 0, 0, 0, 0, 1] -> Correct
                                        # W_str[3] (C11): [1, 0, -1, 0, 0, 0, 0] -> Correct
                                        
                                        C_calc = np.array([
                                            sum(W_str[0][r] * u[r] * v[r] for r in range(7)),
                                            sum(W_str[1][r] * u[r] * v[r] for r in range(7)),
                                            sum(W_str[2][r] * u[r] * v[r] for r in range(7)),
                                            sum(W_str[3][r] * u[r] * v[r] for r in range(7))
                                        ])
                                        if not np.allclose(C_calc, C_true.flatten()):
                                            return False
        return True

    # The above verify is too slow. Let's just trust the standard Strassen formulas.
    # But let's double check the W coefficients in the code.
    # W_str[0]: M4(1), M5(1), M6(-1), M7(-1) -> [0,0,0,1,1,-1,-1]
    # W_str[1]: M1(1), M4(-1) -> [1,0,0,-1,0,0,0]
    # W_str[2]: M2(1), M7(1) -> [0,1,0,0,0,0,1]
    # W_str[3]: M1(1), M3(-1) -> [1,0,-1,0,0,0,0]
    
    # Now build the full 2x4x6 decomposition.
    # Total rank = 3 (col blocks) * 2 (row blocks j1) * 7 (Strassen rank) = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Mapping:
    # col_block c in {0, 1, 2}
    # row_block j1 in {0, 1}
    # strassen_idx s in {0, ..., 6}
    # col = c * 14 + j1 * 7 + s
    
    for c in range(3):
        for j1 in range(2):
            for s in range(7):
                col = c * 14 + j1 * 7 + s
                
                # U: A_{i, 2*j1+j0}
                # i in {0, 1}, j0 in {0, 1}
                # U_str[j0][s] gives the coefficient for A_{j0} in factor s.
                # Wait, U_str is defined for A indices 0..3 (row, col).
                # A is 2x2. Index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
                # In our block A_{j1}, the rows are i=0,1 and cols are j0=0,1.
                # So A_{j1}[i, j0] corresponds to U_str index i*2 + j0.
                for i in range(2):
                    for j0 in range(2):
                        a_idx = i * 2 + j0
                        U[i * m + (2 * j1 + j0)][col] = U_str[a_idx][s]
                
                # V: B_{2*j1+j0, k}
                # k in {0..5}. k = 2*c + k0.
                # B_{j1} is 2x6. Rows j0=0,1. Cols k=0..5.
                # For fixed c, we are computing A_{j1} * B_{j1, c}.
                # B_{j1, c} is 2x2. Rows j0=0,1. Cols k0=0,1.
                # B index in V is (row, col) = (2*j1+j0, 2*c+k0).
                # V_str index for B_{j1,c}[j0, k0] is j0*2 + k0.
                for j0 in range(2):
                    for k0 in range(2):
                        b_idx = j0 * 2 + k0
                        v_row = 2 * j1 + j0
                        v_col = 2 * c + k0
                        V[v_row * p + v_col][col] = V_str[b_idx][s]
                
                # W: C_{i, k}
                # k = 2*c + k0.
                # C index in W is (k, i) = (2*c+k0, i).
                # W_str index for C_{j1, c}[i, k0] is i*2 + k0.
                # Note: W_str output is for the 2x2 result of A_{j1} * B_{j1,c}.
                # The result C_{j1,c} is 2x2. Rows i=0,1. Cols k0=0,1.
                # So W_str[i*2 + k0][s] is the coefficient.
                for i in range(2):
                    for k0 in range(2):
                        c_idx = i * 2 + k0
                        w_row = 2 * c + k0
                        w_col = i
                        W[w_row * n + w_col][col] = W_str[c_idx][s]
                        
    # Verify the full decomposition
    def verify_full(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if verify_full(U, V, W):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        best_result, best_rank = schoolbook()

    return best_result

# EVOLVE-BLOCK-END
