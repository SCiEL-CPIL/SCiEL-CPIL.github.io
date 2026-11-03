import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    def construct_strassen_2x2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[0,2]=1; W[1,2]=1
        U[2,3]=1; U[3,3]=1; U[1,3]=-1; W[2,3]=1; W[3,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[2,5]=1; W[2,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[1,6]=1; V[3,6]=1; W[0,6]=1; W[1,6]=1
        return U, V, W

    def construct_laderman_3x3():
        U = np.zeros((9, 15)); V = np.zeros((9, 15)); W = np.zeros((9, 15))
        U[0,0]=1; U[4,0]=1; U[8,0]=1; V[0,0]=1; V[4,0]=1; V[8,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1; W[4,0]=-1; W[5,0]=1; W[6,0]=1; W[7,0]=1; W[8,0]=-1
        U[1,1]=1; U[5,1]=1; V[0,1]=1; V[8,1]=1; W[1,1]=1; W[2,1]=1; W[4,1]=1; W[5,1]=1
        U[2,2]=1; U[6,2]=1; V[3,2]=1; V[8,2]=1; W[3,2]=1; W[4,2]=1; W[6,2]=1; W[7,2]=1
        U[0,3]=1; U[8,3]=1; V[0,3]=1; V[4,3]=1; W[0,3]=1; W[1,3]=1; W[3,3]=1; W[4,3]=1
        U[1,4]=1; U[5,4]=1; U[8,4]=1; V[3,4]=1; V[8,4]=1
        W[3,4]=1; W[4,4]=1; W[5,4]=1; W[6,4]=1; W[7,4]=1; W[8,4]=1
        U[2,5]=1; U[6,5]=1; V[0,5]=1; V[4,5]=1; W[0,5]=1; W[1,5]=1; W[3,5]=1; W[4,5]=1
        U[0,6]=1; U[4,6]=1; V[0,6]=1; V[4,6]=1; W[0,6]=1; W[1,6]=1; W[3,6]=1; W[4,6]=1
        U[0,7]=1; U[4,7]=1; V[0,7]=1; V[8,7]=1; W[0,7]=1; W[1,7]=1; W[5,7]=1; W[6,7]=1
        U[1,8]=1; U[5,8]=1; V[3,8]=1; V[8,8]=1; W[3,8]=1; W[4,8]=1; W[6,8]=1; W[7,8]=1
        U[0,9]=1; U[8,9]=1; V[0,9]=1; V[8,9]=1; W[0,9]=1; W[1,9]=1; W[5,9]=1; W[6,9]=1
        U[2,10]=1; U[6,10]=1; V[0,10]=1; V[4,10]=1; V[8,10]=1
        W[0,10]=1; W[1,10]=1; W[3,10]=1; W[4,10]=1; W[5,10]=1; W[6,10]=1
        U[0,11]=1; U[4,11]=1; U[8,11]=1; V[3,11]=1; V[8,11]=1
        W[3,11]=1; W[4,11]=1; W[5,11]=1; W[6,11]=1; W[7,11]=1; W[8,11]=1
        U[1,12]=1; U[5,12]=1; U[8,12]=1; V[0,12]=1; V[8,12]=1; W[0,12]=1; W[1,12]=1; W[5,12]=1; W[6,12]=1
        U[0,13]=1; U[4,13]=1; V[3,13]=1; V[8,13]=1; W[3,13]=1; W[4,13]=1; W[6,13]=1; W[7,13]=1
        U[2,14]=1; U[6,14]=1; V[3,14]=1; V[8,14]=1; W[3,14]=1; W[4,14]=1; W[6,14]=1; W[7,14]=1
        return U, V, W

    def build_block_decomposition():
        laderman_result = construct_laderman_3x3()
        use_laderman = True # Laderman is valid for 3x3
        laderman_rank = 15
        total_rank = 7 + 12 + 18 + 12 + 18 + laderman_rank
        U = np.zeros((25, total_rank)); V = np.zeros((25, total_rank)); W = np.zeros((25, total_rank))
        col_offset = 0
        U_a, V_a, W_a = construct_strassen_2x2()
        for idx, src in [(0,0),(1,1),(5,2),(6,3)]:
            U[idx, col_offset:col_offset+7] = U_a[src]
            V[idx, col_offset:col_offset+7] = V_a[src]
            W[idx, col_offset:col_offset+7] = W_a[src]
        col_offset += 7
        for i_prime in range(2):
            for j_prime in range(2):
                for k_prime in range(3):
                    t = col_offset + i_prime*6 + j_prime*3 + k_prime
                    U[i_prime*5+j_prime, t]=1
                    V[j_prime*5+(2+k_prime), t]=1
                    W[(2+k_prime)*5+i_prime, t]=1
        col_offset += 12
        for i_prime in range(2):
            for j_prime in range(3):
                for k_prime in range(3):
                    t = col_offset + i_prime*9 + j_prime*3 + k_prime
                    U[i_prime*5+(2+j_prime), t]=1
                    V[(2+j_prime)*5+(2+k_prime), t]=1
                    W[(2+k_prime)*5+i_prime, t]=1
        col_offset += 18
        for i_prime in range(3):
            for j_prime in range(2):
                for k_prime in range(2):
                    t = col_offset + i_prime*4 + j_prime*2 + k_prime
                    U[(2+i_prime)*5+j_prime, t]=1
                    V[j_prime*5+k_prime, t]=1
                    W[k_prime*5+(2+i_prime), t]=1
        col_offset += 12
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(2):
                    t = col_offset + i_prime*6 + j_prime*2 + k_prime
                    U[(2+i_prime)*5+(2+j_prime), t]=1
                    V[(2+j_prime)*5+k_prime, t]=1
                    W[k_prime*5+(2+i_prime), t]=1
        col_offset += 18
        if use_laderman:
            U_l, V_l, W_l = laderman_result
            for t in range(15):
                for i_prime in range(3):
                    for j_prime in range(3):
                        U[(2+i_prime)*5+(2+j_prime), col_offset+t] = U_l[i_prime*3+j_prime, t]
                for j_prime in range(3):
                    for k_prime in range(3):
                        V[(2+j_prime)*5+(2+k_prime), col_offset+t] = V_l[j_prime*3+k_prime, t]
                for k_prime in range(3):
                    for i_prime in range(3):
                        W[(2+k_prime)*5+(2+i_prime), col_offset+t] = W_l[k_prime*3+i_prime, t]
            col_offset += 15
        else:
            for i_prime in range(3):
                for j_prime in range(3):
                    for k_prime in range(3):
                        t = col_offset + i_prime*9 + j_prime*3 + k_prime
                        U[(2+i_prime)*5+(2+j_prime), t]=1
                        V[(2+j_prime)*5+(2+k_prime), t]=1
                        W[(2+k_prime)*5+(2+i_prime), t]=1
            col_offset += 27
        return U, V, W, col_offset

    U, V, W, rank = build_block_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
