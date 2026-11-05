import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# Hardcoded decompositions for small ranks

def get_strassen_2x2x2():
    # Returns U, V, W of shape (4,7), (4,7), (4,7)
    # C = A * B
    # A, B are 2x2. C is 2x2.
    # Indices for U, V: (i*m + j) -> i*2+j.
    # Indices for W: (k*n + i) -> k*3+i? No, W is (p*n, rank).
    # Task says W[k*n+i, r]. Here n=2, p=2. W is 4x7.
    # T = sum r U[i,j,r] V[j,k,r] W[k,i,r]
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Indices in A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices in B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices in C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[2][0] = -1; W[1][0] = -1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1
    
    return U, V, W

def get_laderman_3x3x3():
    # Rank 23
    # This is complex to hardcode. 
    # I will use a known 23-term decomposition.
    # U (9x23), V (9x23), W (9x23)
    # I will not hardcode the full 23 terms here due to length, but I will implement a placeholder or a known smaller decomposition if 23 is too long.
    # However, 23 is the best known.
    # For the purpose of this code, I will use a recursive construction that might be higher rank if I don't have the 23 terms, 
    # OR I can use the 17-term for 2x3x3 and combine.
    # Wait, if I don't have Laderman, I can split 3x3x3 into 2x3x3 + 1x3x3?
    # R(3,3,3) <= R(2,3,3) + R(1,3,3) + R(2,3,3) + R(1,3,3)? No.
    # Split 3=1+2.
    # R(3,3,3) <= R(1,3,3) + R(1,3,2) + R(2,3,3) + R(2,3,2)?
    # Actually, splitting 3x3x3 into 1x3x3, 1x3x2, 2x3x3, 2x3x2 is not direct.
    
    # I will implement a generic recursive decomposer that uses Strassen for 2x2x2 and diagonal for 1s.
    # It will naturally produce a valid decomposition, possibly rank > 23 for 3x3x3, but valid.
    pass

def construct():
    # Implement the 94-rank strategy
    pass
