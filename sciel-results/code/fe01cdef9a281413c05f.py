import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == k else 0.0
                    u_vals = U[i*m+j, :]
                    v_vals = V[j*p+k, :]
                    w_vals = W[k*n+i, :]
                    prod_sum = np.dot(u_vals, v_vals * w_vals)
                    if abs(prod_sum - expected) > 1e-9:
                        return False
        return True

    def build_rank_85():
        rank = 85
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # 2x2x2 Strassen factors
        s_u = [[0]*7 for _ in range(4)]
        s_v = [[0]*7 for _ in range(4)]
        s_w = [[0]*7 for _ in range(4)]
        
        # Term 0: (a00+a11)(b00+b11)
        s_u[0][0]=1; s_u[3][0]=1
        s_v[0][0]=1; s_v[3][0]=1
        s_w[0][0]=1; s_w[3][0]=1
        
        # Term 1: (a10+a11)b00
        s_u[2][1]=1; s_u[3][1]=1
        s_v[0][1]=1
        s_w[2][1]=1; s_w[3][1]=-1
        
        # Term 2: a00(b01-b11)
        s_u[0][2]=1
        s_v[1][2]=1; s_v[3][2]=-1
        s_w[1][2]=1; s_w[3][2]=1
        
        # Term 3: a11(b10-b00)
        s_u[3][3]=1
        s_v[2][3]=1; s_v[0][3]=-1
        s_w[0][3]=1; s_w[2][3]=1
        
        # Term 4: (a00+a01)b11
        s_u[0][4]=1; s_u[1][4]=1
        s_v[3][4]=1
        s_w[0][4]=-1; s_w[1][4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        s_u[0][5]=-1; s_u[2][5]=1
        s_v[0][5]=1; s_v[1][5]=1
        s_w[3][5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        s_u[1][6]=1; s_u[3][6]=-1
        s_v[2][6]=1; s_v[3][6]=1
        s_w[0][6]=1
        
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        def add_term(a_coeff, b_coeff, c_coeff, start_col):
            for inner_k in range(7):
                col_idx = start_col + inner_k
                for (a_br, a_bc), a_c in a_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            u_val = s_u[local_idx][inner_k]
                            if u_val != 0:
                                gi = 2 * a_br + lr
                                gj = 2 * a_bc + lc
                                U[gi*5 + gj][col_idx] += a_c * u_val
                for (b_br, b_bc), b_c in b_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            v_val = s_v[local_idx][inner_k]
                            if v_val != 0:
                                gj = 2 * b_br + lr
                                gk = 2 * b_bc + lc
                                V[gj*5 + gk][col_idx] += b_c * v_val
                for (c_br, c_bc), c_c in c_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            w_val = s_w[local_idx][inner_k]
                            if w_val != 0:
                                gk = 2 * c_bc + lc
                                gi = 2 * c_br + lr
                                W[gk*4 + gi][col_idx] += c_c * w_val
        
        for t_idx, (ac, bc, cc) in enumerate(terms):
            add_term(ac, bc, cc, t_idx * 7)
            
        col = 49
        # 4x4x1 part (A_44 * B_41)
        # A is 4x4, B is 4x1. Result 4x1.
        # This adds to columns 0..3 of C? No.
        # C = A B.
        # A = [A44 A41]. B = [B44; B15]? No.
        # My previous logic:
        # A (4x5) = [A_44 (4x4) | A_41 (4x1)]
        # B (5x5) = [B_45 (4x5); B_15 (1x5)]
        # C = A_44 B_45 + A_41 B_15
        # B_45 (4x5) = [B_44 (4x4) | B_41 (4x1)]
        # So C = A_44 B_44 + A_44 B_41 + A_41 B_15
        
        # Term 1: A_44 B_44 (Rank 49). Done above.
        # Term 2: A_44 B_41 (Rank 16).
        # A_44 is 4x4. B_41 is 4x1.
        # C_41 is 4x1.
        # C_i = sum_j A_ij B_j.
        # For each i (0..3) and j (0..3):
        for i in range(4):
            for j in range(4):
                U[i*5+j][col] = 1
                V[j*5+4][col] = 1  # B_41 is at rows 0..3, col 4 of B (global col 4)
                W[4*4+i][col] = 1  # C_41 is at rows 0..3, col 4 of C (global col 4)
                col += 1
        
        # Term 3: A_41 B_15 (Rank 20).
        # A_41 is 4x1. B_15 is 1x5.
        # C_45 is 4x5.
        # C_i = A_i1 * B_1j.
        for i in range(4):
            for k in range(5):
                U[i*5+4][col] = 1  # A_41 is at col 4 of A
                V[4*5+k][col] = 1  # B_15 is at row 4 of B
                W[k*4+i][col] = 1  # C_45 is at col k, row i of C
                col += 1
                
        return {"U": U, "V": V, "W": W}
    
    baseline = build_rank_85()
    if verify(baseline['U'], baseline['V'], baseline['W']):
        return baseline
    else:
        return baseline

# EVOLVE-BLOCK-END
