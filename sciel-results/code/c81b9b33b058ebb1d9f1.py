"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A_left (2x2) * B_top[:,0:2] (2x2) via Strassen
    # a=U[0],b=U[1],c=U[2],d=U[3]; e=V[0],f=V[1],g=V[2],h=V[3]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[5], C[1,1]=W[6]
    U[0][col]+=1; U[3][col]+=1; V[0][col]+=1; V[3][col]+=1; W[0][col]+=1; W[6][col]+=1; col+=1
    U[2][col]+=1; U[3][col]+=1; V[0][col]+=1; W[5][col]+=1; W[6][col]+=-1; col+=1
    U[0][col]+=1; V[1][col]+=1; V[3][col]+=-1; W[1][col]+=1; W[6][col]+=1; col+=1
    U[3][col]+=1; V[2][col]+=1; V[0][col]+=-1; W[0][col]+=1; W[5][col]+=1; col+=1
    U[0][col]+=1; U[1][col]+=1; V[3][col]+=1; W[0][col]+=-1; W[1][col]+=1; col+=1
    U[2][col]+=1; U[0][col]+=-1; V[0][col]+=1; V[1][col]+=1; W[6][col]+=1; col+=1
    U[1][col]+=1; U[3][col]+=-1; V[2][col]+=1; V[3][col]+=1; W[0][col]+=1; col+=1
    
    # Part 2: A_left * B_top[:,2:4] via Strassen
    # e=V[4],f=V[5],g=V[6],h=V[7]; C[0,2]=W[2],C[0,3]=W[3],C[1,2]=W[7],C[1,3]=W[8]
    U[0][col]+=1; U[3][col]+=1; V[4][col]+=1; V[7][col]+=1; W[2][col]+=1; W[8][col]+=1; col+=1
    U[2][col]+=1; U[3][col]+=1; V[4][col]+=1; W[7][col]+=1; W[8][col]+=-1; col+=1
    U[0][col]+=1; V[5][col]+=1; V[7][col]+=-1; W[3][col]+=1; W[8][col]+=1; col+=1
    U[3][col]+=1; V[6][col]+=1; V[4][col]+=-1; W[2][col]+=1; W[7][col]+=1; col+=1
    U[0][col]+=1; U[1][col]+=1; V[7][col]+=1; W[2][col]+=-1; W[3][col]+=1; col+=1
    U[2][col]+=1; U[0][col]+=-1; V[4][col]+=1; V[5][col]+=1; W[8][col]+=1; col+=1
    U[1][col]+=1; U[3][col]+=-1; V[6][col]+=1; V[7][col]+=1; W[2][col]+=1; col+=1
    
    # Part 3: A_left * B_top[:,4:5] (2x2 x 2x1) schoolbook
    U[0][col]+=1; V[8][col]+=1; W[4][col]+=1; col+=1
    U[1][col]+=1; V[9][col]+=1; W[4][col]+=1; col+=1
    U[2][col]+=1; V[8][col]+=1; W[9][col]+=1; col+=1
    U[3][col]+=1; V[9][col]+=1; W[9][col]+=1; col+=1
    
    # Part 4: A_right * B_bottom (2x1 x 1x5) schoolbook
    for k in range(5):
        U[4][col]+=1; V[10+k][col]+=1; W[k][col]+=1; col+=1
        U[5][col]+=1; V[10+k][col]+=1; W[5+k][col]+=1; col+=1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
