import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Block definitions
    # i blocks: [0], [1, 2]
    i_blocks = [[0], [1, 2]]
    # j blocks: [0, 1], [2, 3], [4]
    j_blocks = [[0, 1], [2, 3], [4]]
    # k blocks: [0, 1], [2, 3], [4, 5]
    k_blocks = [[0, 1], [2, 3], [4, 5]]
    
    # Calculate total rank
    # Types:
    # (1, 2, 2) -> 4
    # (1, 1, 2) -> 2
    # (2, 2, 2) -> 7
    # (2, 1, 2) -> 4
    
    rank = 0
    for i_b in i_blocks:
        for k_b in k_blocks:
            for j_b in j_blocks:
                ni = len(i_b)
                nj = len(j_b)
                nk = len(k_b)
                if ni == 1 and nj == 2 and nk == 2:
                    rank += 4
                elif ni == 1 and nj == 1 and nk == 2:
                    rank += 2
                elif ni == 2 and nj == 2 and nk == 2:
                    rank += 7
                elif ni == 2 and nj == 1 and nk == 2:
                    rank += 4
                else:
                    raise ValueError(f"Unexpected block sizes: {ni}x{nj}x{nk}")
                    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for i_b in i_blocks:
        for k_b in k_blocks:
            for j_b in j_blocks:
                ni = len(i_b)
                nj = len(j_b)
                nk = len(k_b)
                
                if ni == 1 and nj == 2 and nk == 2:
                    # Schoolbook 1x2x2
                    for j_loc in range(nj):
                        for k_loc in range(nk):
                            i_loc = 0
                            i_glob = i_b[0]
                            j_glob = j_b[j_loc]
                            k_glob = k_b[k_loc]
                            
                            U[i_glob * m + j_glob][col] = 1
                            V[j_glob * p + k_glob][col] = 1
                            W[k_glob * n + i_glob][col] = 1
                            col += 1
                            
                elif ni == 1 and nj == 1 and nk == 2:
                    # Schoolbook 1x1x2
                    for k_loc in range(nk):
                        i_loc = 0
                        j_loc = 0
                        i_glob = i_b[0]
                        j_glob = j_b[0]
                        k_glob = k_b[k_loc]
                        
                        U[i_glob * m + j_glob][col] = 1
                        V[j_glob * p + k_glob][col] = 1
                        W[k_glob * n + i_glob][col] = 1
                        col += 1
                        
                elif ni == 2 and nj == 2 and nk == 2:
                    # Strassen 2x2x2
                    # p1 = A00(B01-B11)
                    # p2 = (A00+A10)B11
                    # p3 = (A10-A11)B00
                    # p4 = A11(B10-B00)
                    # p5 = (A00+A11)(B00+B11)
                    # p6 = (A01-A00)(B10+B11)
                    # p7 = (A10-A11)(B00+B01)
                    
                    # C00 = p5 - p3 - p7 + p4
                    # C01 = p1 + p7
                    # C10 = p2 + p3
                    # C11 = p5 - p1 - p2 + p6
                    
                    # We need to define U, V, W for each p_l
                    # U is 2x2, V is 2x2, W is 2x2 (local)
                    
                    # p1
                    U[i_b[0]*m + j_b[0]][col] = 1
                    V[j_b[0]*p + k_b[1]][col] = 1
                    V[j_b[1]*p + k_b[1]][col] = -1
                    W[k_b[1]*n + i_b[0]][col] = 1
                    W[k_b[0]*n + i_b[0]][col] = 0 # C01 gets p1, so W for k=1, i=0 is 1
                    # Wait, W index is k*n + i.
                    # C01 corresponds to i=0, k=1. W[1*n + 0] = 1.
                    # C00 corresponds to i=0, k=0. W[0*n + 0] = 0.
                    # So W[k_b[1]*n + i_b[0]] = 1.
                    col += 1
                    
                    # p2
                    U[i_b[0]*m + j_b[0]][col] = 1
                    U[i_b[1]*m + j_b[0]][col] = 1
                    V[j_b[1]*p + k_b[1]][col] = 1
                    W[k_b[0]*n + i_b[1]][col] = 1 # C10 gets p2
                    col += 1
                    
                    # p3
                    U[i_b[1]*m + j_b[0]][col] = 1
                    U[i_b[1]*m + j_b[1]][col] = -1
                    V[j_b[0]*p + k_b[0]][col] = 1
                    W[k_b[0]*n + i_b[1]][col] = 1 # C10 gets p3
                    W[k_b[0]*n + i_b[0]][col] = -1 # C00 gets -p3
                    col += 1
                    
                    # p4
                    U[i_b[1]*m + j_b[1]][col] = 1
                    V[j_b[1]*p + k_b[0]][col] = 1
                    V[j_b[0]*p + k_b[0]][col] = -1
                    W[k_b[0]*n + i_b[0]][col] = 1 # C00 gets p4
                    col += 1
                    
                    # p5
                    U[i_b[0]*m + j_b[0]][col] = 1
                    U[i_b[1]*m + j_b[1]][col] = 1
                    V[j_b[0]*p + k_b[0]][col] = 1
                    V[j_b[1]*p + k_b[1]][col] = 1
                    W[k_b[0]*n + i_b[0]][col] = 1 # C00 gets p5
                    W[k_b[1]*n + i_b[1]][col] = 1 # C11 gets p5
                    col += 1
                    
                    # p6
                    U[i_b[0]*m + j_b[1]][col] = 1
                    U[i_b[0]*m + j_b[0]][col] = -1
                    V[j_b[1]*p + k_b[0]][col] = 1
                    V[j_b[1]*p + k_b[1]][col] = 1
                    W[k_b[1]*n + i_b[1]][col] = 1 # C11 gets p6
                    col += 1
                    
                    # p7
                    U[i_b[1]*m + j_b[0]][col] = 1
                    U[i_b[1]*m + j_b[1]][col] = -1
                    V[j_b[0]*p + k_b[0]][col] = 1
                    V[j_b[0]*p + k_b[1]][col] = 1
                    W[k_b[1]*n + i_b[0]][col] = 1 # C01 gets p7
                    W[k_b[0]*n + i_b[0]][col] = -1 # C00 gets -p7
                    col += 1

                elif ni == 2 and nj == 1 and nk == 2:
                    # Schoolbook 2x1x2
                    for i_loc in range(ni):
                        for k_loc in range(nk):
                            j_loc = 0
                            i_glob = i_b[i_loc]
                            j_glob = j_b[0]
                            k_glob = k_b[k_loc]
                            
                            U[i_glob * m + j_glob][col] = 1
                            V[j_glob * p + k_glob][col] = 1
                            W[k_glob * n + i_glob][col] = 1
                            col += 1

    return {"U": U, "V": V, "W": W}
