import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    deadline = time.time() + 115  # safety margin before 120s
    
    # Rank 85 decomposition:
    # Block 1: rank 20  (A[:,4] * B[4,:])
    # Block 2: rank 16  (A[:,:4] * B[:4,4])
    # Block 3: rank 49  (A[:,:4] * B[:4,:4]) via Strassen 4x4x4 = 7 outer * 7 inner
    
    rank = 85
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    
    # Block 1: a4 b4^T (Rank 20)
    for i in range(4):
        for k in range(5):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            
    # Block 2: A' B''' (Rank 16)
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            
    # Block 3: A' B'' (Rank 49) via Strassen
    inner_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    def z4():
        return np.zeros((4, 4), dtype=np.float64)
    
    def add_block(mat, a, b, i_start, j_start, sign=1):
        for di in range(2):
            for dj in range(2):
                mat[a][b][i_start+di][j_start+dj] += sign
    
    alphas = [[ [z4() for _ in range(2)] for _ in range(2) ] for _ in range(7)]
    betas  = [[ [z4() for _ in range(2)] for _ in range(2) ] for _ in range(7)]
    gammas = [[ [z4() for _ in range(2)] for _ in range(2) ] for _ in range(7)]
    
    # Term 0: M1 = (A00+A11)(B00+B11) -> C00
    add_block(alphas[0], 0,0, 0,0, 1); add_block(alphas[0], 0,1, 0,2, 1)
    add_block(alphas[0], 1,0, 2,0, 1); add_block(alphas[0], 1,1, 2,2, 1)
    add_block(betas[0],  0,0, 0,0, 1); add_block(betas[0],  0,1, 0,2, 1)
    add_block(betas[0],  1,0, 2,0, 1); add_block(betas[0],  1,1, 2,2, 1)
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[0][a][b][a*2+di][b*2+dj] = 1
    
    # Term 1: M2 = (A10+A11)B00 -> C10
    add_block(alphas[1], 0,0, 2,0, 1); add_block(alphas[1], 0,1, 2,2, 1)
    add_block(alphas[1], 1,0, 3,0, 1); add_block(alphas[1], 1,1, 3,2, 1)
    add_block(betas[1],  0,0, 0,0, 1); add_block(betas[1],  0,1, 0,1, 1)
    add_block(betas[1],  1,0, 1,0, 1); add_block(betas[1],  1,1, 1,1, 1)
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[1][a][b][2+a*2+di][b*2+dj] = 1
    
    # Term 2: M3 = A00(B10-B11) -> C01
    add_block(alphas[2], 0,0, 0,0, 1); add_block(alphas[2], 0,1, 0,1, 1)
    add_block(alphas[2], 1,0, 1,0, 1); add_block(alphas[2], 1,1, 1,1, 1)
    add_block(betas[2],  0,0, 2,0, 1); add_block(betas[2],  0,1, 2,1, 1)
    add_block(betas[2],  1,0, 3,0, 1); add_block(betas[2],  1,1, 3,1, 1)
    add_block(betas[2],  0,0, 2,2,-1); add_block(betas[2],  0,1, 2,3,-1)
    add_block(betas[2],  1,0, 3,2,-1); add_block(betas[2],  1,1, 3,3,-1)
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[2][a][b][a*2+di][2+b*2+dj] = 1
    
    # Term 3: M4 = A11(B01-B00) -> C00 + C10
    add_block(alphas[3], 0,0, 2,2, 1); add_block(alphas[3], 0,1, 2,3, 1)
    add_block(alphas[3], 1,0, 3,2, 1); add_block(alphas[3], 1,1, 3,3, 1)
    add_block(betas[3],  0,0, 0,2, 1); add_block(betas[3],  0,1, 0,3, 1)
    add_block(betas[3],  1,0, 1,2, 1); add_block(betas[3],  1,1, 1,3, 1)
    add_block(betas[3],  0,0, 0,0,-1); add_block(betas[3],  0,1, 0,1,-1)
    add_block(betas[3],  1,0, 1,0,-1); add_block(betas[3],  1,1, 1,1,-1)
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[3][a][b][a*2+di][b*2+dj] = 1
                    gammas[3][a][b][2+a*2+di][b*2+dj] = 1
    
    # Term 4: M5 = (A00+A01)B11 -> -C00 + C01
    add_block(alphas[4], 0,0, 0,0, 1); add_block(alphas[4], 0,1, 0,1, 1)
    add_block(alphas[4], 1,0, 1,0, 1); add_block(alphas[4], 1,1, 1,1, 1)
    add_block(alphas[4], 0,0, 0,2, 1); add_block(alphas[4], 0,1, 0,3, 1)
    add_block(alphas[4], 1,0, 1,2, 1); add_block(alphas[4], 1,1, 1,3, 1)
    add_block(betas[4],  0,0, 2,2, 1); add_block(betas[4],  0,1, 2,3, 1)
    add_block(betas[4],  1,0, 3,2, 1); add_block(betas[4],  1,1, 3,3, 1)
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[4][a][b][a*2+di][b*2+dj] = -1
                    gammas[4][a][b][a*2+di][2+b*2+dj] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01) -> C11
    add_block(alphas[5], 0,0, 2,0, 1); add_block(alphas[5], 0,1, 2,1, 1)
    add_block(alphas[5], 1,0, 3,0, 1); add_block(alphas[5], 1,1, 3,1, 1)
    add_block(alphas[5], 0,0, 0,0,-1); add_block(alphas[5], 0,1, 0,1,-1)
    add_block(alphas[5], 1,0, 1,0,-1); add_block(alphas[5], 1,1, 1,1,-1)
    add_block(betas[5],  0,0, 0,0, 1); add_block(betas[5],  0,1, 0,1, 1)
    add_block(betas[5],  1,0, 1,0, 1); add_block(betas[5],  1,1, 1,1, 1)
    add_block(betas[5],  0,0, 0,2, 1); add_block(betas[5],  0,1, 0,3, 1)
    add_block(betas[5],  1,0, 1,2, 1); add_block(betas[5],  1,1, 1,3, 1)
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[5][a][b][2+a*2+di][2+b*2+dj] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11) -> C00
    add_block(alphas[6], 0,0, 0,2, 1); add_block(alphas[6], 0,1, 0,3, 1)
    add_block(alphas[6], 1,0, 1,2, 1); add_block(alphas[6], 1,1, 1,3, 1)
    add_block(alphas[6], 0,0, 2,2,-1); add_block(alphas[6], 0,1, 2,3,-1)
    add_block(alphas[6], 1,0, 3,2,-1); add_block(alphas[6], 1,1, 3,3,-1)
    add_block(betas[6],  0,0, 2,0, 1); add_block(betas[6],  0,1, 2,1, 1)
    add_block(betas[6],  1,0, 3,0, 1); add_block(betas[6],  1,1, 3,1, 1)
    add_block(betas[6],  0,0, 2,2, 1); add_block(betas[6],  0,1, 2,3, 1)
    add_block(betas[6],  1,0, 3,2, 1); add_block(betas[6],  1,1, 3,3, 1)
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[6][a][b][a*2+di][b*2+dj] = 1
    
    # Combine outer and inner Strassen terms
    for l in range(7):
        for mm in range(7):
            if time.time() > deadline:
                break
            s_c = np.array(inner_terms[mm][0], dtype=np.float64)
            t_c = np.array(inner_terms[mm][1], dtype=np.float64)
            u_c = np.array(inner_terms[mm][2], dtype=np.float64)
            
            U_term = np.zeros(nm, dtype=np.float64)
            V_term = np.zeros(mp, dtype=np.float64)
            W_term = np.zeros(pn, dtype=np.float64)
            
            for a in range(2):
                for b in range(2):
                    idx = a*2 + b
                    for i in range(4):
                        for j in range(4):
                            U_term[i*5+j] += s_c[idx] * alphas[l][a][b][i][j]
                    for j in range(4):
                        for k in range(4):
                            V_term[j*5+k] += t_c[idx] * betas[l][a][b][j][k]
                    for i in range(4):
                        for k in range(4):
                            W_term[k*4+i] += u_c[idx] * gammas[l][a][b][i][k]
                            
            U[:, col] = U_term
            V[:, col] = V_term
            W[:, col] = W_term
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
