"""Matrix multiplication 4x4x4 rank-minimal decomposition using Strassen's algorithm."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-49 tensor decomposition for 4x4x4 matrix multiplication
    using Strassen's algorithm applied to 2x2 block structure.
    
    The 4x4x4 multiplication tensor is decomposed by viewing the 4x4 matrices
    as 2x2 block matrices with 2x2 blocks. Strassen's algorithm gives rank 7
    for 2x2 multiplication, so the total rank is 7*7 = 49.
    
    U shape: (n*m, rank) = (16, 49)
    V shape: (m*p, rank) = (16, 49)  
    W shape: (p*n, rank) = (16, 49)
    
    Convention: W[k*n+i, r] where output C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen 2x2 decomposition coefficients
    # Standard Strassen rank-7 for 2x2:
    # The 2x2x2 tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    # 
    # Rank-7 decomposition:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    
    # 7 products:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    
    # C00 = p1 - p4 + p5 - p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # As rank-1 tensor decomposition, each product p_s contributes:
    # u_s (A-factor) * v_s (B-factor) * w_s (C-factor)
    
    strassen_terms = [
        # (u, v, w)
        ((1,0,0,1), (1,1,1,1), (1,0,0,1)),   # t1: p1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),   # t2: p2
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),   # t3: p3
        ((0,0,0,1), (0,-1,1,0), (-1,0,1,0)),  # t4: p4
        ((1,1,0,0), (0,0,0,1), (1,1,0,0)),    # t5: p5
        ((0,0,1,-1), (1,1,0,0), (0,0,0,1)),   # t6: p6
        ((0,1,0,-1), (0,0,1,1), (-1,0,0,0)),  # t7: p7
    ]
    
    # Block-level Strassen: for each s, define A_block_coeffs and B_block_coeffs
    # P0 = (A00+A11)(B00+B11), C: C00+=1, C11+=1
    # P1 = (A10+A11)B00, C: C10+=1, C11-=1
    # P2 = A00(B01-B11), C: C01+=1, C11+=1
    # P3 = A11(B10-B00), C: C00-=1, C10+=1
    # P4 = (A00+A01)B11, C: C00+=1, C01+=1
    # P5 = (A10-A00)(B00+B01), C: C11+=1
    # P6 = (A01-A11)(B10+B11), C: C00-=1
    
    A_block_coeffs = [
        [[1,0],[0,1]],   # P0
        [[0,0],[1,1]],   # P1
        [[1,0],[0,0]],   # P2
        [[0,0],[0,1]],   # P3
        [[1,1],[0,0]],   # P4
        [[-1,0],[1,0]],  # P5
        [[0,1],[0,-1]],  # P6
    ]
    
    B_block_coeffs = [
        [[1,0],[0,1]],   # P0
        [[1,0],[0,0]],   # P1
        [[0,1],[0,-1]],  # P2
        [[-1,0],[1,0]],  # P3
        [[0,0],[0,1]],   # P4
        [[1,1],[0,0]],   # P5
        [[0,0],[1,1]],   # P6
    ]
    
    C_block_coeffs = [
        [[1,0],[0,1]],   # P0: C00+=1, C11+=1
        [[0,0],[1,-1]],  # P1: C10+=1, C11-=1
        [[0,1],[0,1]],   # P2: C01+=1, C11+=1
        [[-1,0],[1,0]],  # P3: C00-=1, C10+=1
        [[1,1],[0,0]],   # P4: C00+=1, C01+=1
        [[0,0],[0,1]],   # P5: C11+=1
        [[-1,0],[0,0]],  # P6: C00-=1
    ]
    
    # Now construct the 49 terms.
    for s in range(7):
        for t in range(7):
            term_idx = s * 7 + t
            u_s, v_s, w_s = strassen_terms[t]
            
            # A_factor coefficients
            for i in range(2):
                for j in range(2):
                    u_val = u_s[i*2+j]
                    if u_val == 0:
                        continue
                    for a in range(2):
                        for b in range(2):
                            alpha = A_block_coeffs[s][a][b]
                            if alpha == 0:
                                continue
                            row_A = 2*a + i
                            col_A = 2*b + j
                            U[row_A * m + col_A][term_idx] += u_val * alpha
            
            # B_factor coefficients
            for i in range(2):
                for j in range(2):
                    v_val = v_s[i*2+j]
                    if v_val == 0:
                        continue
                    for b in range(2):
                        for c in range(2):
                            beta = B_block_coeffs[s][b][c]
                            if beta == 0:
                                continue
                            row_B = 2*b + i
                            col_B = 2*c + j
                            V[row_B * m + col_B][term_idx] += v_val * beta
            
            # C_factor coefficients
            for i in range(2):
                for j in range(2):
                    w_val = w_s[i*2+j]
                    if w_val == 0:
                        continue
                    for a in range(2):
                        for c in range(2):
                            gamma = C_block_coeffs[s][a][c]
                            if gamma == 0:
                                continue
                            row_C = 2*a + i
                            col_C = 2*c + j
                            W[col_C * n + row_C][term_idx] += gamma * w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
