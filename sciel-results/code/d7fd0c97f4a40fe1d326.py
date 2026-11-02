import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    S = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True

def _build_4x4x3_decomp():
    """
    Build a rank-12 exact decomposition for 4x4 * 4x3 multiplication.
    
    We use the Kronecker product structure:
    4x4x3 = (2x2x1) ⊗ (2x2x3)
    
    For 2x2x1: rank 4 (schoolbook)
    For 2x2x3: We use a rank-3 decomposition (optimal for 2x2x3).
    Total rank = 4 * 3 = 12.
    
    For 2x2x3 (C = A * B where A is 2x2, B is 2x3):
    We can decompose as:
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    A rank-3 decomposition for 2x2x3:
    Actually, let me think more carefully. The tensor for 2x2x3 has dimensions
    (2,2,3) with T[i,j,k] = sum_l A[i,l] B[l,j] where B is 2x3.
    
    Wait, the standard matrix multiplication tensor for M_{2,2} * M_{2,3} -> M_{2,3}:
    C[i,j] = sum_{k=0}^{1} A[i,k] * B[k,j]
    
    The tensor has rank... Let me just construct it directly.
    
    For 2x2x3: we need T such that T[i,j,k] = sum_l U[i,l] V[k,j,l] ... 
    
    Actually, let me use a different approach. The bilinear complexity of 2x2x3:
    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    
    This is just 4 multiplications (i in {0,1}, j in {0,1,2} gives 6 outputs,
    but each output needs 2 products, and products are shared).
    
    Products: A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,0]*B[0,2],
              A[0,1]*B[1,0], A[0,1]*B[1,1], A[0,1]*B[1,2],
              A[1,0]*B[0,0], A[1,0]*B[0,1], A[1,0]*B[0,2],
              A[1,1]*B[1,0], A[1,1]*B[1,1], A[1,1]*B[1,2]
    
    That's 12 products, but with sharing:
    P0 = A[0,0]*B[0,0], P1 = A[0,0]*B[0,1], P2 = A[0,0]*B[0,2],
    P3 = A[0,1]*B[1,0], P4 = A[0,1]*B[1,1], P5 = A[0,1]*B[1,2],
    P6 = A[1,0]*B[0,0], P7 = A[1,0]*B[0,1], P8 = A[1,0]*B[0,2],
    P9 = A[1,1]*B[1,0], P10 = A[1,1]*B[1,1], P11 = A[1,1]*B[1,2]
    
    C[0,0] = P0 + P3
    C[0,1] = P1 + P4
    C[0,2] = P2 + P5
    C[1,0] = P6 + P9
    C[1,1] = P7 + P10
    C[1,2] = P8 + P11
    
    So rank 12 for 2x2x3.
    
    For 4x4x3 = (2x2) ⊗ (2x2) * (2x2 ⊗ 2x3)... this is getting complex.
    
    Let me just use a direct approach: build the 4x4x3 decomposition via
    Kronecker product of 2x2x1 (rank 4) and 2x2x3 (rank 12).
    Total rank = 4 * 12 = 48. That's too much.
    
    Let me try a different strategy. For 4x4x9, I'll try to find a good
    decomposition by direct construction.
    
    Actually, let me just return the schoolbook for now and focus on the
    search strategy.
    """
    n, m, p = 4, 4, 3
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Schoolbook as default
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try ranks from a lower bound upward
    # Lower bound for 4x4x9: at least max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But realistically much higher. Known results suggest < 144.
    # Search from 90 to 143
    for r in range(90, best_rank):
        if time.time() > deadline:
            break
        
        # Try many random attempts with diverse entry sets
        for attempt in range(300):
            if time.time() > deadline:
                break
            
            # Generate random integer/half-integer matrices
            U = np.random.choice([-1, 0, 1, 2, -2], size=(n*m, r))
            V = np.random.choice([-1, 0, 1, 2, -2], size=(m*p, r))
            W = np.random.choice([-1, 0, 1, 2, -2], size=(p*n, r))
            
            if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    return best
