import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Ranks for blocks
    # I: 0 (2x), 1 (1x)
    # J: 0 (2x), 1 (2x), 2 (1x)
    # K: 0 (2x), 1 (2x), 2 (2x), 3 (1x)
    
    # We will build U, V, W as lists of lists
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    rank = 86
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2 block
    def add_strassen_2x2(I_start, J_start, K_start, U, V, W, col):
        # I_start, J_start, K_start are starting indices for i, j, k
        # Sizes are 2, 2, 2
        # Strassen multipliers M1..M7
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B21-B22)
        # M4: A22(B12-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # M1
        U[I_start*m + J_start][col] = 1
        U[(I_start+1)*m + (J_start+1)][col] = 1
        V[J_start*p + K_start][col] = 1
        V[(J_start+1)*p + (K_start+1)][col] = 1
        W[K_start*n + I_start][col] = 1
        W[(K_start+1)*n + (I_start+1)][col] = 1
        col += 1
        
        # M2
        U[(I_start+1)*m + J_start][col] = 1
        U[(I_start+1)*m + (J_start+1)][col] = 1
        V[J_start*p + K_start][col] = 1
        W[K_start*n + (I_start+1)][col] = 1
        col += 1
        
        # M3
        U[I_start*m + J_start][col] = 1
        V[(J_start+1)*p + K_start][col] = 1
        V[(J_start+1)*p + (K_start+1)][col] = -1
        W[K_start*n + I_start][col] = 1
        W[(K_start+1)*n + I_start][col] = 1
        col += 1
        
        # M4
        U[(I_start+1)*m + (J_start+1)][col] = 1
        V[J_start*p + (K_start+1)][col] = 1
        V[J_start*p + K_start][col] = -1
        W[(K_start+1)*n + (I_start+1)][col] = 1
        W[K_start*n + (I_start+1)][col] = 1
        col += 1
        
        # M5
        U[I_start*m + J_start][col] = 1
        U[I_start*m + (J_start+1)][col] = 1
        V[(J_start+1)*p + (K_start+1)][col] = 1
        W[(K_start+1)*n + I_start][col] = 1
        col += 1
        
        # M6
        U[(I_start+1)*m + J_start][col] = 1
        U[I_start*m + J_start][col] = -1
        V[J_start*p + K_start][col] = 1
        V[J_start*p + (K_start+1)][col] = 1
        W[K_start*n + (I_start+1)][col] = 1
        W[(K_start+1)*n + (I_start+1)][col] = 1
        col += 1
        
        # M7
        U[I_start*m + (J_start+1)][col] = 1
        U[(I_start+1)*m + (J_start+1)][col] = -1
        V[(J_start+1)*p + K_start][col] = 1
        V[(J_start+1)*p + (K_start+1)][col] = 1
        W[K_start*n + I_start][col] = 1
        W[(K_start+1)*n + I_start][col] = 1
        col += 1
        
        return col

    # Helper to add a 2x1 block (A 2x2, B 2x1) -> Rank 4
    def add_2x2_2x1(I_start, J_start, K_start, U, V, W, col):
        # A is 2x2, B is 2x1. Output 2x1.
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        # 4 terms
        # Term 1: A11 B1 -> C1
        U[I_start*m + J_start][col] = 1
        V[J_start*p + K_start][col] = 1
        W[K_start*n + I_start][col] = 1
        col += 1
        # Term 2: A12 B2 -> C1
        U[I_start*m + (J_start+1)][col] = 1
        V[(J_start+1)*p + K_start][col] = 1
        W[K_start*n + I_start][col] = 1
        col += 1
        # Term 3: A21 B1 -> C2
        U[(I_start+1)*m + J_start][col] = 1
        V[J_start*p + K_start][col] = 1
        W[K_start*n + (I_start+1)][col] = 1
        col += 1
        # Term 4: A22 B2 -> C2
        U[(I_start+1)*m + (J_start+1)][col] = 1
        V[(J_start+1)*p + K_start][col] = 1
        W[K_start*n + (I_start+1)][col] = 1
        col += 1
        return col

    # Helper to add a 2x1 by 1x2 block (A 2x1, B 1x2) -> Rank 1
    def add_2x1_1x2(I_start, J_start, K_start, U, V, W, col):
        # A is 2x1, B is 1x2. Output 2x2.
        # C11 = A1 B1, C12 = A1 B2
        # C21 = A2 B1, C22 = A2 B2
        # Wait, Rank 1?
        # T_{i,j,k} = 1 for all i,k, j=0.
        # U[I, J]=1, V[J, K]=1, W[K, I]=1?
        # If we use 1 multiplier:
        # M = A1 * B1? No, we need to distribute to all 4 outputs.
        # The tensor is $u \otimes v \otimes w$?
        # $T_{i,0,k} = 1$.
        # This is NOT a rank 1 tensor in the sense of outer product of 3 vectors?
        # $T_{i,j,k} = U_i V_j W_k$.
        # If $j=0$, $T_{i,0,k} = U_i W_k$.
        # We want $T_{i,0,k} = 1$ for all $i,k$.
        # So $U_i = 1, W_k = 1$.
        # Yes, it is rank 1.
        U[I_start*m + J_start][col] = 1
        U[(I_start+1)*m + J_start][col] = 1
        V[J_start*p + K_start][col] = 1
        V[J_start*p + (K_start+1)][col] = 1
        W[K_start*n + I_start][col] = 1
        W[(K_start+1)*n + I_start][col] = 1
        W[K_start*n + (I_start+1)][col] = 1
        W[(K_start+1)*n + (I_start+1)][col] = 1
        col += 1
        return col

    # Helper to add a 1x2 by 2x2 block (A 1x2, B 2x2) -> Rank 4
    def add_1x2_2x2(I_start, J_start, K_start, U, V, W, col):
        # A is 1x2, B is 2x2. Output 1x2.
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        # 4 terms
        U[I_start*m + J_start][col] = 1
        V[J_start*p + K_start][col] = 1
        W[K_start*n + I_start][col] = 1
        col += 1
        U[I_start*m + (J_start+1)][col] = 1
        V[(J_start+1)*p + K_start][col] = 1
        W[K_start*n + I_start][col] = 1
        col += 1
        U[I_start*m + J_start][col] = 1
        V[J_start*p + (K_start+1)][col] = 1
        W[(K_start+1)*n + I_start][col] = 1
        col += 1
        U[I_start*m + (J_start+1)][col] = 1
        V[(J_start+1)*p + (K_start+1)][col] = 1
        W[(K_start+1)*n + I_start][col] = 1
        col += 1
        return col

    # Helper to add a 1x2 by 2x1 block (A 1x2, B 2x1) -> Rank 2
    def add_1x2_2x1(I_start, J_start, K_start, U, V, W, col):
        # A is 1x2, B is 2x1. Output 1x1.
        # C = A1 B1 + A2 B2
        # 2 terms
        U[I_start*m + J_start][col] = 1
        V[J_start*p + K_start][col] = 1
        W[K_start*n + I_start][col] = 1
        col += 1
        U[I_start*m + (J_start+1)][col] = 1
        V[(J_start+1)*p + K_start][col] = 1
        W[K_start*n + I_start][col] = 1
        col += 1
        return col

    # Helper to add a 1x1 by 1x2 block (A 1x1, B 1x2) -> Rank 1
    def add_1x1_1x2(I_start, J_start, K_start, U, V, W, col):
        # A is 1x1, B is 1x2. Output 1x2.
        # C1 = A1 B1, C2 = A1 B2
        # Rank 1?
        # T_{0,0,0}=1, T_{0,0,1}=1.
        # U_0 V_0 W_0 = 1, U_0 V_0 W_1 = 1.
        # Yes.
        U[I_start*m + J_start][col] = 1
        V[J_start*p + K_start][col] = 1
        V[J_start*p + (K_start+1)][col] = 1
        W[K_start*n + I_start][col] = 1
        W[(K_start+1)*n + I_start][col] = 1
        col += 1
        return col

    # Helper to add a 1x1 by 1x1 block -> Rank 1
    def add_1x1_1x1(I_start, J_start, K_start, U, V, W, col):
        U[I_start*m + J_start][col] = 1
        V[J_start*p + K_start][col] = 1
        W[K_start*n + I_start][col] = 1
        col += 1
        return col

    # Block Indices
    # I: 0 (rows 0,1), 1 (row 2)
    # J: 0 (cols 0,1), 1 (cols 2,3), 2 (col 4)
    # K: 0 (cols 0,1), 1 (cols 2,3), 2 (cols 4,5), 3 (col 6)

    # I=0, J=0
    col = add_strassen_2x2(0, 0, 0, U, V, W, col)
    col = add_strassen_2x2(0, 0, 1, U, V, W, col)
    col = add_strassen_2x2(0, 0, 2, U, V, W, col)
    col = add_2x2_2x1(0, 0, 3, U, V, W, col)

    # I=0, J=1
    col = add_strassen_2x2(0, 1, 0, U, V, W, col)
    col = add_strassen_2x2(0, 1, 1, U, V, W, col)
    col = add_strassen_2x2(0, 1, 2, U, V, W, col)
    col = add_2x2_2x1(0, 1, 3, U, V, W, col)

    # I=0, J=2
    col = add_2x1_1x2(0, 2, 0, U, V, W, col)
    col = add_2x1_1x2(0, 2, 1, U, V, W, col)
    col = add_2x1_1x2(0, 2, 2, U, V, W, col)
    col = add_1x1_1x1(0, 2, 3, U, V, W, col) # A 2x1, B 1x1 -> Output 2x1. Rank 1? 
    # Wait, A is 2x1, B is 1x1. C is 2x1.
    # C1 = A1 B1, C2 = A2 B1.
    # T_{i,0,0} = 1.
    # U_i V_0 W_0 = 1.
    # Yes, rank 1.
    # My helper add_1x1_1x1 assumes 1x1.
    # Let's make a helper for 2x1 by 1x1.
    # Actually, add_2x1_1x2 handles 2x1 by 1x2.
    # For 2x1 by 1x1, it's just scaling.
    # U[I,J]=1, V[J,K]=1, W[K,I]=1?
    # No, W has 2 rows.
    # W[K, I_start]=1, W[K, I_start+1]=1.
    # So:
    U[0*m + 2][col] = 1
    U[1*m + 2][col] = 1
    V[2*p + 3][col] = 1
    W[3*n + 0][col] = 1
    W[3*n + 1][col] = 1
    col += 1

    # I=1, J=0
    col = add_1x2_2x2(1, 0, 0, U, V, W, col)
    col = add_1x2_2x2(1, 0, 1, U, V, W, col)
    col = add_1x2_2x2(1, 0, 2, U, V, W, col)
    col = add_1x2_2x1(1, 0, 3, U, V, W, col)

    # I=1, J=1
    col = add_1x2_2x2(1, 1, 0, U, V, W, col)
    col = add_1x2_2x2(1, 1, 1, U, V, W, col)
    col = add_1x2_2x2(1, 1, 2, U, V, W, col)
    col = add_1x2_2x1(1, 1, 3, U, V, W, col)

    # I=1, J=2
    col = add_1x1_1x2(1, 2, 0, U, V, W, col)
    col = add_1x1_1x2(1, 2, 1, U, V, W, col)
    col = add_1x1_1x2(1, 2, 2, U, V, W, col)
    col = add_1x1_1x1(1, 2, 3, U, V, W, col)

    return {"U": U, "V": V, "W": W}
