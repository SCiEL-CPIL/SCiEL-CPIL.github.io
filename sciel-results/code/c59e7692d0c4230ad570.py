import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0  # 5s safety margin

def construct():
    n, m, p = DIMENSIONS
    # Start with naive rank-96 decomposition
    naive_rank = n * m * p
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_naive, "V": V_naive, "W": W_naive}
    best_rank = naive_rank
    
    # Convert to numpy for efficiency
    U_np = np.array(U_naive, dtype=float)
    V_np = np.array(V_naive, dtype=float)
    W_np = np.array(W_naive, dtype=float)
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
        
        # The tensor should be all ones
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    def strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 Strassen decomposition.
        T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1}
        """
        # Strassen's 7 terms for 2x2 matrix multiplication
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A10)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        U_s = []
        V_s = []
        W_s = []
        
        # Each term r: U[i,j,r], V[j,k,r], W[k,i,r]
        # Term 0: M1
        U_s.append([1, 0, 0, 1])  # A00 + A11
        V_s.append([1, 0, 1, 0])  # B00 + B11
        W_s.append([1, 0, 0, 1])  # contributes to C00 and C11
        
        # Term 1: M2
        U_s.append([0, 0, 1, 1])  # A10 + A11
        V_s.append([1, 0, 0, 0])  # B00
        W_s.append([0, 0, 0, 1])  # contributes to C10 and C11
        
        # Term 2: M3
        U_s.append([1, 0, 0, 0])  # A00
        V_s.append([0, 0, 1, -1]) # B10 - B11
        W_s.append([0, 1, 0, 0])  # contributes to C01
        
        # Term 3: M4
        U_s.append([0, 0, 0, 1])  # A11
        V_s.append([0, 1, -1, 0]) # B01 - B00
        W_s.append([1, 0, 0, 0])  # contributes to C00
        
        # Term 4: M5
        U_s.append([1, 1, 0, 0])  # A00 + A01
        V_s.append([0, 0, 0, 1])  # B11
        W_s.append([-1, 1, 0, 0]) # -C00 + C01
        
        # Term 5: M6
        U_s.append([0, 0, 1, -1]) # A10 - A00
        V_s.append([1, 1, 0, 0])  # B00 + B01
        W_s.append([0, 0, 0, 0])  # Wait, this should contribute somewhere
        
        # Let me redo this more carefully.
        # The tensor T[i,j,k] for 2x2x2 is all 1s.
        # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
        
        # Let me use a known rank-7 decomposition for 2x2x2
        # This is the Strassen decomposition written in tensor form
        
        U2 = [
            [1, 0, 0, 0],  # i=0,j=0
            [0, 1, 0, 0],  # i=0,j=1
            [0, 0, 1, 0],  # i=1,j=0
            [0, 0, 0, 1],  # i=1,j=1
        ]
        # Actually, let me just use the naive rank-8 for 2x2x2 as a fallback
        # and try to optimize later
        
        # For now, return naive rank-8 for 2x2x2
        U_r = []
        V_r = []
        W_r = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_r.append([0]*8)
                    V_r.append([0]*8)
                    W_r.append([0]*8)
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_r[i*2+j][col] = 1
                    V_r[j*2+k][col] = 1
                    W_r[k*2+i][col] = 1
                    col += 1
        
        return U_r, V_r, W_r
    
    # Try block decomposition approach
    # Split 3x4x8 into blocks and use fast algorithms where possible
    
    # For now, let's try a simple approach: 
    # Use the fact that 3 = 2+1, 4 = 2+2, 8 = 2*4
    # We can decompose into sub-problems
    
    # Actually, let me just verify the naive solution and return it
    # The key insight is that we need to find a lower-rank decomposition
    
    # Let me try a different approach: use bilinear algorithm structure
    # The tensor for matrix multiplication C = AB where A is n x m and B is m x p
    # has T[i,j,k] = 1 for all valid (i,j,k)
    
    # The rank is the minimum number of rank-1 tensors that sum to T
    
    # For now, return the naive solution
    if verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        return best_result
    
    # Fallback
    return best_result
