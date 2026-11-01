"""Matrix multiplication 2x2 by 2x15: rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    N = n * m  # 4
    M = m * p  # 30
    P = p * n  # 30
    
    # We'll build the decomposition using Strassen for 2-column blocks
    # 15 = 7*2 + 1, so 7 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4)
    # Total rank = 7*7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    
    col = 0  # current column in U, V, W
    
    # Strassen decomposition for a 2x2 x 2x2 block
    # A is 2x2, B is 2x2, C is 2x2
    # C[0,0] = M0 + M3 + M5 - M4
    # C[0,1] = M4 + M6
    # C[1,0] = M1 + M3
    # C[1,1] = M0 - M1 + M2 + M5
    # where:
    # M0 = A[0,0] * B[0,0]
    # M1 = A[1,0] * B[0,0]
    # M2 = A[0,1] * B[1,1]
    # M3 = (A[1,0]-A[0,0]) * (B[0,0]+B[1,1])
    # M4 = (A[0,0]+A[1,1]) * (B[1,0]-B[1,1])
    # M5 = (A[1,1]-A[0,1]) * (B[0,1]+B[1,1])
    # M6 = (A[0,1]-A[1,1]) * (B[0,0]+B[1,0])
    
    def add_strassen_block(b_start):
        """Add Strassen decomposition for columns b_start, b_start+1"""
        nonlocal col
        # b_start and b_start+1 are the two column indices in B (0-indexed)
        # U indices: i*m + j where i=0,1 (rows of A), j=0,1 (cols of A)
        # V indices: j*p + k where j=0,1 (rows of B), k=b_start or b_start+1
        # W indices: k*n + i where k=b_start or b_start+1, i=0,1 (rows of C)
        
        # M0: A[0,0] * B[0, b_start] -> C[0, b_start]
        # u: A[0,0] -> U[0]
        # v: B[0, b_start] -> V[0*p + b_start] = V[b_start]
        # w: C[0, b_start] -> W[b_start*n + 0] = W[2*b_start]
        U[0][col] = 1
        V[b_start][col] = 1
        W[2*b_start][col] = 1
        col += 1
        
        # M1: A[1,0] * B[0, b_start] -> C[1, b_start]
        U[2][col] = 1
        V[b_start][col] = 1
        W[2*b_start + 1][col] = 1
        col += 1
        
        # M2: A[0,1] * B[1, b_start+1] -> C[1, b_start+1]
        U[1][col] = 1
        V[p + b_start + 1][col] = 1
        W[2*(b_start+1) + 1][col] = 1
        col += 1
        
        # M3: (A[1,0]-A[0,0]) * (B[0, b_start]+B[1, b_start+1]) -> C[0, b_start] and C[1, b_start]
        U[2][col] = 1
        U[0][col] = -1
        V[b_start][col] = 1
        V[p + b_start + 1][col] = 1
        W[2*b_start][col] = 1
        W[2*b_start + 1][col] = 1
        col += 1
        
        # M4: (A[0,0]+A[1,1]) * (B[1, b_start]-B[1, b_start+1]) -> C[0, b_start+1] and C[1, b_start+1]
        U[0][col] = 1
        U[3][col] = 1
        V[p + b_start][col] = 1
        V[p + b_start + 1][col] = -1
        W[2*(b_start+1)][col] = 1
        W[2*(b_start+1) + 1][col] = 1
        col += 1
        
        # M5: (A[1,1]-A[0,1]) * (B[0, b_start+1]+B[1, b_start+1]) -> C[0, b_start] and C[1, b_start+1]
        U[3][col] = 1
        U[1][col] = -1
        V[b_start + 1][col] = 1
        V[p + b_start + 1][col] = 1
        W[2*b_start][col] = 1
        W[2*(b_start+1) + 1][col] = 1
        col += 1
        
        # M6: (A[0,1]-A[1,1]) * (B[0, b_start]+B[1, b_start]) -> C[0, b_start+1] and C[1, b_start+1]
        U[1][col] = 1
        U[3][col] = -1
        V[b_start][col] = 1
        V[p + b_start][col] = 1
        W[2*(b_start+1)][col] = 1
        W[2*(b_start+1) + 1][col] = 1
        col += 1
    
    # Add 7 Strassen blocks for columns 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13
    for b in range(7):
        add_strassen_block(2*b)
    
    # Schoolbook for the last column (column 14)
    # C[0,14] = A[0,0]*B[0,14] + A[0,1]*B[1,14]
    # C[1,14] = A[1,0]*B[0,14] + A[1,1]*B[1,14]
    
    # A[0,0] * B[0,14] -> C[0,14]
    U[0][col] = 1
    V[14][col] = 1
    W[2*14][col] = 1
    col += 1
    
    # A[0,1] * B[1,14] -> C[0,14]
    U[1][col] = 1
    V[p + 14][col] = 1
    W[2*14][col] = 1
    col += 1
    
    # A[1,0] * B[0,14] -> C[1,14]
    U[2][col] = 1
    V[14][col] = 1
    W[2*14 + 1][col] = 1
    col += 1
    
    # A[1,1] * B[1,14] -> C[1,14]
    U[3][col] = 1
    V[p + 14][col] = 1
    W[2*14 + 1][col] = 1
    col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
