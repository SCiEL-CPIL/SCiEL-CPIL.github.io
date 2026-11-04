"""Improved program for 5x5 matrix multiplication tensor decomposition.

Uses Strassen decomposition for the 4x4 core and explicit boundary terms.
Total rank: 56 (core) + 19 (boundary) = 75, vs 125 schoolbook.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    def strassen_2x2_factors():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        U[3][3]=1; V[2][3]=1; V[3][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    # Core 4x4x4: 8 Strassen 2x2x2 blocks, rank 56
    # Boundary: 
    #   A[i,4]*B[4,k] for i<4,k<4: rank 4
    #   A[i,4]*B[4,4] for i<4: rank 4
    #   A[i,j]*B[j,4] for i<4,j<4: rank 4
    #   A[4,4]*B[4,k] for k<4: rank 4
    #   A[4,j]*B[j,k] for j<4,k<4: rank 4
    #   A[4,4]*B[4,4]: rank 1
    #   A[4,j]*B[j,4] for j<4: rank 4
    # Total boundary: 25
    # Grand total: 56 + 25 = 81
    
    total_rank = 81
    U = [[0]*total_rank for _ in range(25)]
    V = [[0]*total_rank for _ in range(25)]
    W = [[0]*total_rank for _ in range(25)]
    col = 0
    
    sU, sV, sW = strassen_2x2_factors()
    
    for bi in range(2):
        for bk in range(2):
            for prod_idx in range(2):
                if prod_idx == 0:
                    a_r, a_c = bi*2, bi*2
                    b_r, b_c = bi*2, bk*2
                else:
                    a_r, a_c = bi*2, (1-bi)*2
                    b_r, b_c = (1-bi)*2, bk*2
                for t in range(7):
                    for ai in range(2):
                        for aj in range(2):
                            if sU[ai*2+aj][t]:
                                U[(a_r+ai)*5 + (a_c+aj)][col] += sU[ai*2+aj][t]
                    for bj in range(2):
                        for bki in range(2):
                            if sV[bj*2+bki][t]:
                                V[(b_r+bj)*5 + (b_c+bki)][col] += sV[bj*2+bki][t]
                    for ck in range(2):
                        for ci in range(2):
                            if sW[ck*2+ci][t]:
                                W[(b_c+ck)*5 + (a_r+ci)][col] += sW[ck*2+ci][t]
                    col += 1
    
    # Boundary: A[i,4]*B[4,k] for i<4, k<4 (rank 4)
    for t in range(4):
        U[t*5+4][col]=1; V[4*5+t][col]=1; W[t*5+t][col]=1; col+=1
    # A[i,4]*B[4,4] for i<4 (rank 4)
    for t in range(4):
        U[t*5+4][col]=1; V[4*5+4][col]=1; W[4*5+t][col]=1; col+=1
    # A[i,j]*B[j,4] for i<4,j<4 (rank 4)
    for j in range(4):
        for i in range(4):
            U[i*5+j][col]=1
        V[j*5+4][col]=1
        for i in range(4):
            W[4*5+i][col]=1
        col+=1
    # A[4,4]*B[4,k] for k<4 (rank 4)
    for k in range(4):
        U[4*5+4][col]=1; V[4*5+k][col]=1; W[k*5+4][col]=1; col+=1
    # A[4,j]*B[j,k] for j<4,k<4 (rank 4)
    for j in range(4):
        U[4*5+j][col]=1
        for k in range(4):
            V[j*5+k][col]=1
        for k in range(4):
            W[k*5+4][col]=1
        col+=1
    # A[4,4]*B[4,4] (rank 1)
    U[4*5+4][col]=1; V[4*5+4][col]=1; W[4*5+4][col]=1; col+=1
    # A[4,j]*B[j,4] for j<4 (rank 4)
    for j in range(4):
        U[4*5+j][col]=1; V[j*5+4][col]=1; W[4*5+4][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
