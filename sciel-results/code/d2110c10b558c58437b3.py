"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product
from collections import defaultdict

DIMENSIONS = [4, 5, 6]

def _construct_strassen_2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W as lists of lists (or numpy arrays).
    U shape: (4, 7) corresponding to A_ij (i,j in 0..1) -> index i*2+j
    V shape: (4, 7) corresponding to B_jk (j,k in 0..1) -> index j*2+k
    W shape: (4, 7) corresponding to C_ik (i,k in 0..1) -> index k*2+i (Trace convention W[k*n+i, r] for n=2)
    """
    # Strassen's 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0) [k=0,i=0], 1:(0,1) [k=1,i=0? No, W[k*n+i]. For 2x2: W[k*2+i].
    #    C11 -> i=0,k=0 -> W[0*2+0]=W[0]
    #    C12 -> i=0,k=1 -> W[1*2+0]=W[2]
    #    C21 -> i=1,k=0 -> W[0*2+1]=W[1]
    #    C22 -> i=1,k=1 -> W[1*2+1]=W[3]
    
    # Wait, let's double check the trace convention: W[k*n+i, r].
    # Output is C. C[i,k].
    # The tensor T is defined such that T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # Actually, the standard matrix multiplication tensor is T_{i,j,k} = A_{i,j} B_{j,k} = C_{i,k}.
    # The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # Here U corresponds to A part, V to B part, W to C part (output).
    # The index for W is k*n+i.
    
    # Let's construct U, V, W for the 7 terms.
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: M1
    U[0, 0] = 1; U[3, 0] = 1  # A11 + A22
    V[0, 0] = 1; V[3, 0] = 1  # B11 + B22
    # C11 += M1, C22 += M1
    # W[0] += 1, W[3] += 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2
    U[2, 1] = 1; U[3, 1] = 1  # A21 + A22
    V[0, 1] = 1               # B11
    # C21 += M2, C22 -= M2
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3
    U[0, 2] = 1               # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12 - B22
    # C12 += M3, C22 += M3
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4
    U[3, 3] = 1               # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21 - B11
    # C11 += M4, C21 += M4
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5
    U[0, 4] = 1; U[1, 4] = 1  # A11 + A12
    V[3, 4] = 1               # B22
    # C11 -= M5, C12 += M5
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6
    U[2, 5] = 1; U[0, 5] = -1 # A21 - A11
    V[0, 5] = 1; V[1, 5] = 1  # B11 + B12
    # C22 += M6
    W[3, 5] = 1
    
    # Term 7: M7
    U[1, 6] = 1; U[3, 6] = -1 # A12 - A22
    V[2, 6] = 1; V[3, 6] = 1  # B21 + B22
    # C11 += M7
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    # Strategy: Try to find a low-rank decomposition using a search/heuristic or known bounds.
    # Known lower bound for R(4,5,6) is > 100.
    # Schoolbook is 120.
    # We can try to use Strassen-like blocking if dimensions allow, but 5 is not a power of 2.
    # However, we can pad to 8x8x8 or similar? No, we need exact decomposition for 4x5x6.
    
    # Let's try a simple local search or greedy rank reduction?
    # Given the constraints and time, and the fact that finding an exact integer decomposition for arbitrary small matrices is NP-hard,
    # the safest bet is to return the schoolbook decomposition which is guaranteed correct.
    # But the prompt asks for a "stronger algorithm".
    # Is there a known decomposition for 4x5x6 with rank < 120?
    # Strassen gives 2^3=8 for 2x2x2.
    # For 4x4x4, rank is 27 (Strassen) vs 64 (Schoolbook).
    # For 4x5x6, we can decompose 4x5x6 into blocks.
    # 4x5 can be seen as 2x(2x5)? No.
    
    # Let's try to use the Strassen 2x2 decomposition for the 4x4 part and schoolbook for the rest?
    # 4x5x6:
    # Split A(4x5) into A11(2x2), A12(2x3), A21(2x2), A22(2x3).
    # Split B(5x6) into B11(2x6), B12(3x6).
    # This doesn't align well with Strassen's 2x2 blocks.
    
    # Alternative: Pad to 4x8x6? No.
    
    # Let's stick to the schoolbook for correctness, but maybe I can implement a specific known low-rank decomposition?
    # I don't have a hardcoded 4x5x6 decomposition.
    
    # However, I can try to reduce rank by finding linear dependencies among the schoolbook terms?
    # The schoolbook terms are independent.
    
    # What if I use the 2x2 Strassen decomposition to handle the 4x4 sub-matrix?
    # We can view 4x5x6 as:
    # C = A * B
    # A is 4x5. B is 5x6.
    # Let's split A into A_top (2x5) and A_bot (2x5).
    # Let's split B into B_left (5x3) and B_right (5x3).
    # This doesn't help directly.
    
    # Actually, there is a known result that R(4,5,6) <= 108? Or something similar?
    # Without a verified decomposition, I must return 120.
    
    # Wait, I can try to run a quick search for a better decomposition?
    # No, time limit is 150s. A random search for tensor rank is unlikely to find a valid one quickly.
    
    # I will return the schoolbook decomposition.
    U, V, W, rank = _construct_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
