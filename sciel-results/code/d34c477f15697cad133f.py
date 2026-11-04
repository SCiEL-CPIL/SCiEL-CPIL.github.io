import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    # Output C[i,k] = sum_t U[i*m+j... wait
    
    # Actually the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # C[i,k] = sum_{j,t} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Wait, let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For our problem: n=2, m=4, p=4
    # U: (8, rank), V: (16, rank), W: (8, rank)
    # C[i,k] = sum_{j=0}^{3} sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*2+i, c]
    
    # Rank = 28
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2 product X*Y:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)(y00)
    # M3 = (x00)(y10-y11)
    # M4 = (x11)(y01-y10)
    # M5 = (x00+x01)(y11)
    # M6 = (x10-x00)(y00+y10)
    # M7 = (x01-x11)(y01+y11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_terms(x_indices, y_indices, offset):
        """
        x_indices: list of 4 U-row indices for X = [[x00,x01],[x10,x11]]
        y_indices: list of 4 V-row indices for Y = [[y00,y01],[y10,y11]]
        offset: starting term index
        Returns list of (term_idx, U_row, U_val, V_row, V_val, [(W_row, W_val), ...])
        """
        x00, x01, x10, x11 = x_indices
        y00, y01, y10, y11 = y_indices
        terms = []
        
        # M1 = (x00+x11)(y00+y11)
        t = offset + 0
        terms.append((t, [(x00, 1), (x11, 1)], [(y00, 1), (y11, 1)], [(0, 1), (3, 1)]))  # C00+=1, C11+=1
        
        # M2 = (x10+x11)(y00)
        t = offset + 1
        terms.append((t, [(x10, 1), (x11, 1)], [(y00, 1)], [(1, 1), (3, -1)]))  # C10+=1, C11+=-1
        
        # M3 = (x00)(y10-y11)
        t = offset + 2
        terms.append((t, [(x00, 1)], [(y10, 1), (y11, -1)], [(2, 1), (3, 1)]))  # C01+=1, C11+=1
        
        # M4 = (x11)(y01-y10)
        t = offset + 3
        terms.append((t, [(x11, 1)], [(y01, 1), (y10, -1)], [(0, 1), (1, 1)]))  # C00+=1, C10+=1
        
        # M5 = (x00+x01)(y11)
        t = offset + 4
        terms.append((t, [(x00, 1), (x01, 1)], [(y11, 1)], [(0, -1), (2, 1)]))  # C00+=-1, C01+=1
        
        # M6 = (x10-x00)(y00+y10)
        t = offset + 5
        terms.append((t, [(x10, 1), (x00, -1)], [(y00, 1), (y10, 1)], [(3, 1)]))  # C11+=1
        
        # M7 = (x01-x11)(y01+y11)
        t = offset + 6
        terms.append((t, [(x01, 1), (x11, -1)], [(y01, 1), (y11, 1)], [(0, 1)]))  # C00+=1
        
        return terms
    
    # A1 = A[:,0:2], A2 = A[:,2:4]
    # A1 indices in U: row i*4+j for i in {0,1}, j in {0,1}
    # A1: x00=U[0*4+0]=U[0], x01=U[0*4+1]=U[1], x10=U[1*4+0]=U[4], x11=U[1*4+1]=U[5]
    # A2: x00=U[0*4+2]=U[2], x01=U[0*4+3]=U[3], x10=U[1*4+2]=U[6], x11=U[1*4+3]=U[7]
    
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # B[j,k] → V[j*4+k]
    # B11: y00=V[0*4+0]=V[0], y01=V[0*4+1]=V[1], y10=V[1*4+0]=V[4], y11=V[1*4+1]=V[5]
    # B12: y00=V[0*4+2]=V[2], y01=V[0*4+3]=V[3], y10=V[1*4+2]=V[6], y11=V[1*4+3]=V[7]
    # B21: y00=V[2*4+0]=V[8], y01=V[2*4+1]=V[9], y10=V[3*4+0]=V[12], y11=V[3*4+1]=V[13]
    # B22: y00=V[2*4+2]=V[10], y01=V[2*4+3]=V[11], y10=V[3*4+2]=V[14], y11=V[3*4+3]=V[15]
    
    # W: W[k*2+i] for C[i,k]
    # C[0,0]→W[0], C[0,1]→W[2], C[0,2]→W[4], C[0,3]→W[6]
    # C[1,0]→W[1], C[1,1]→W[3], C[1,2]→W[5], C[1,3]→W[7]
    
    # For A1*B11: outputs go to C[:,0:2], i.e., C[0,0],C[0,1],C[1,0],C[1,1] → W[0],W[2],W[1],W[3]
    # For A2*B21: outputs go to C[:,0:2] as well → W[0],W[2],W[1],W[3]
    # For A1*B12: outputs go to C[:,2:4], i.e., C[0,2],C[0,3],C[1,2],C[1,3] → W[4],W[6],W[5],W[7]
    # For A2*B22: outputs go to C[:,2:4] → W[4],W[6],W[5],W[7]
    
    # Let me define W row mappings for each block
    # For left block (cols 0,1): C00→W[0], C01→W[2], C10→W[1], C11→W[3]
    # For right block (cols 2,3): C00→W[4], C01→W[6], C10→W[5], C11→W[7]
    
    left_W = [0, 2, 1, 3]   # C00, C01, C10, C11
    right_W = [4, 6, 5, 7]  # C00, C01, C10, C11
    
    # A1*B11 (terms 0-6)
    x_A1 = [0, 1, 4, 5]
    y_B11 = [0, 1, 4, 5]
    for term in strassen_terms(x_A1, y_B11, 0):
        t, u_rows, v_rows, w_contribs = term
        for row, val in u_rows:
            U[row][t] += val
        for row, val in v_rows:
            V[row][t] += val
        for w_idx, val in w_contribs:
            W[left_W[w_idx]][t] += val
    
    # A2*B21 (terms 7-13)
    x_A2 = [2, 3, 6, 7]
    y_B21 = [8, 9, 12, 13]
    for term in strassen_terms(x_A2, y_B21, 7):
        t, u_rows, v_rows, w_contribs = term
        for row, val in u_rows:
            U[row][t] += val
        for row, val in v_rows:
            V[row][t] += val
        for w_idx, val in w_contribs:
            W[left_W[w_idx]][t] += val
    
    # A1*B12 (terms 14-20)
    x_A1 = [0, 1, 4, 5]
    y_B12 = [2, 3, 6, 7]
    for term in strassen_terms(x_A1, y_B12, 14):
        t, u_rows, v_rows, w_contribs = term
        for row, val in u_rows:
            U[row][t] += val
        for row, val in v_rows:
            V[row][t] += val
        for w_idx, val in w_contribs:
            W[right_W[w_idx]][t] += val
    
    # A2*B22 (terms 21-27)
    x_A2 = [2, 3, 6, 7]
    y_B22 = [10, 11, 14, 15]
    for term in strassen_terms(x_A2, y_B22, 21):
        t, u_rows, v_rows, w_contribs = term
        for row, val in u_rows:
            U[row][t] += val
        for row, val in v_rows:
            V[row][t] += val
        for w_idx, val in w_contribs:
            W[right_W[w_idx]][t] += val
    
    return {"U": U, "V": V, "W": W}
