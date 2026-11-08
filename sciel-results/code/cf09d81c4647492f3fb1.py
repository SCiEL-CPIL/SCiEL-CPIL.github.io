import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 # 150s wall-clock limit
    
    n, m, p = DIMENSIONS
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for half in range(2):
        b_col_offset = half * 2
        
        for s_id in range(7):
            if time.time() > deadline - 5.0:
                return {"U": U, "V": V, "W": W}
                
            A_coeffs = {}
            B_coeffs = {}
            
            if s_id == 0:
                A_coeffs = {(0,0): 1, (1,1): 1}
                B_coeffs = {(0, b_col_offset): 1, (1, b_col_offset+1): 1}
            elif s_id == 1:
                A_coeffs = {(1,0): 1, (1,1): 1}
                B_coeffs = {(0, b_col_offset): 1}
            elif s_id == 2:
                A_coeffs = {(0,0): 1}
                B_coeffs = {(0, b_col_offset+1): 1, (1, b_col_offset+1): -1}
            elif s_id == 3:
                A_coeffs = {(1,1): 1}
                B_coeffs = {(1, b_col_offset): 1, (0, b_col_offset): -1}
            elif s_id == 4:
                A_coeffs = {(0,0): 1, (0,1): 1}
                B_coeffs = {(1, b_col_offset+1): 1}
            elif s_id == 5:
                A_coeffs = {(1,0): 1, (0,0): -1}
                B_coeffs = {(0, b_col_offset): 1, (0, b_col_offset+1): 1}
            elif s_id == 6:
                A_coeffs = {(0,1): 1, (1,1): -1}
                B_coeffs = {(1, b_col_offset): 1, (1, b_col_offset+1): 1}
            
            for (ai, aj), coeff in A_coeffs.items():
                for i in range(2):
                    for j in range(2):
                        U[ai*2+i][aj*2+j][col] = coeff
                        
            for (bj, bk), coeff in B_coeffs.items():
                for i in range(2):
                    for j in range(2):
                        gj = bj * 2 + i
                        gk = bk * 2 + j
                        V[gj * p + gk][col] = coeff
            
            W_coeffs = {}
            if half == 0:
                if s_id == 0: W_coeffs = {(0,0): 1, (1,1): 1}
                elif s_id == 1: W_coeffs = {(1,0): 1, (1,1): -1}
                elif s_id == 2: W_coeffs = {(0,1): 1, (1,1): 1}
                elif s_id == 3: W_coeffs = {(0,0): 1, (1,0): 1}
                elif s_id == 4: W_coeffs = {(0,0): -1, (0,1): 1}
                elif s_id == 5: W_coeffs = {(1,1): 1}
                elif s_id == 6: W_coeffs = {(0,0): 1}
            else:
                if s_id == 0: W_coeffs = {(0,2): 1, (1,3): 1}
                elif s_id == 1: W_coeffs = {(1,2): 1, (1,3): -1}
                elif s_id == 2: W_coeffs = {(0,3): 1, (1,3): 1}
                elif s_id == 3: W_coeffs = {(0,2): 1, (1,2): 1}
                elif s_id == 4: W_coeffs = {(0,2): -1, (0,3): 1}
                elif s_id == 5: W_coeffs = {(1,3): 1}
                elif s_id == 6: W_coeffs = {(0,2): 1}
            
            for (ci, cj), coeff in W_coeffs.items():
                for i in range(2):
                    for j in range(2):
                        gk = ci * 2 + i
                        gi = cj * 2 + j
                        W[gk * n + gi][col] = coeff
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
