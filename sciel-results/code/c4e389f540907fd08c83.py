import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen coefficients for 2x2x2
    # P[i, j, u], Q[j, k, u], G[i, k, u]
    P = np.zeros((2, 2, 7))
    Q = np.zeros((2, 2, 7))
    G = np.zeros((2, 2, 7))
    
    # u=0: (X00+X11)(Y00+Y11)
    P[0,0,0]=1; P[1,1,0]=1
    Q[0,0,0]=1; Q[1,1,0]=1
    G[0,0,0]=1; G[1,1,0]=1
    
    # u=1: (X10+X11)Y00
    P[1,0,1]=1; P[1,1,1]=1
    Q[0,0,1]=1
    G[1,0,1]=1; G[1,1,1]=-1
    
    # u=2: X00(Y01-Y11)
    P[0,0,2]=1
    Q[0,1,2]=1; Q[1,1,2]=-1
    G[0,1,2]=1; G[1,1,2]=1
    
    # u=3: X11(Y10-Y00)
    P[1,1,3]=1
    Q[1,0,3]=1; Q[0,0,3]=-1
    G[0,0,3]=1; G[1,0,3]=1
    
    # u=4: (X00+X01)Y11
    P[0,0,4]=1; P[0,1,4]=1
    Q[1,1,4]=1
    G[0,0,4]=-1; G[0,1,4]=1
    
    # u=5: (X10-X00)(Y00+Y01)
    P[1,0,5]=1; P[0,0,5]=-1
    Q[0,0,5]=1; Q[0,1,5]=1
    G[1,1,5]=1
    
    # u=6: (X01-X11)(Y10+Y11)
    P[0,1,6]=1; P[1,1,6]=-1
    Q[1,0,6]=1; Q[1,1,6]=1
    G[0,0,6]=1
    
    # Outer Strassen coefficients for 4x4x4
    # X[t, i, j], Y[t, j, k], D[t, i, k]
    X = np.zeros((7, 2, 2))
    Y = np.zeros((7, 2, 2))
    D = np.zeros((7, 2, 2))
    
    # t=0
    X[0,0,0]=1; X[0,1,1]=1
    Y[0,0,0]=1; Y[0,1,1]=1
    D[0,0,0]=1; D[0,1,1]=1
    
    # t=1
    X[1,1,0]=1; X[1,1,1]=1
    Y[1,0,0]=1
    D[1,1,0]=1; D[1,1,1]=-1
    
    # t=2
    X[2,0,0]=1
    Y[2,0,1]=1; Y[2,1,1]=-1
    D[2,0,1]=1; D[2,1,1]=1
    
    # t=3
    X[3,1,1]=1
    Y[3,1,0]=1; Y[3,0,0]=-1
    D[3,0,0]=1; D[3,1,0]=1
    
    # t=4
    X[4,0,0]=1; X[4,0,1]=1
    Y[4,1,1]=1
    D[4,0,0]=-1; D[4,0,1]=1
    
    # t=5
    X[5,1,0]=1; X[5,0,0]=-1
    Y[5,0,0]=1; Y[5,0,1]=1
    D[5,1,1]=1
    
    # t=6
    X[6,0,1]=1; X[6,1,1]=-1
    Y[6,1,0]=1; Y[6,1,1]=1
    D[6,0,0]=1

    col = 0
    for half in range(2):
        k_offset = half * 4
        for t in range(7):
            for u in range(7):
                for i in range(4):
                    for j in range(4):
                        i_block = i // 2
                        j_block = j // 2
                        coeff_u = P[i_block, j_block, u] * X[t, i_block, j_block]
                        if coeff_u != 0:
                            U[i * 4 + j][col] += coeff_u
                            
                        for k_local in range(4):
                            k_global = k_local + k_offset
                            k_block = k_local // 2
                            coeff_v = Q[j_block, k_block, u] * Y[t, j_block, k_block]
                            if coeff_v != 0:
                                V[j * 8 + k_global][col] += coeff_v
                            
                            i_block_w = i // 2
                            k_block_w = k_local // 2
                            coeff_w = G[i_block_w, k_block_w, u] * D[t, i_block_w, k_block_w]
                            if coeff_w != 0:
                                W[k_global * 4 + i][col] += coeff_w
                col += 1
                
    return {"U": U, "V": V, "W": W}
