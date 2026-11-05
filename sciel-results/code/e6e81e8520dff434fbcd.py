import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # We'll use a block decomposition approach.
    # Split the middle dimension m=4 into m1=3 and m2=1.
    # A (3x4) = [A1 (3x3) | A2 (3x1)]
    # B (4x5) = [B1 (3x5); B2 (1x5)]
    # C = A1*B1 + A2*B2
    
    # For A2*B2 (3x1x5): schoolbook rank 15
    # For A1*B1 (3x3x5): split B1 into B11(3x3) and B12(3x2)
    #   A1*B11 (3x3x3): use rank-26 decomposition
    #   A1*B12 (3x3x2): schoolbook rank 18
    # Total: 15 + 26 + 18 = 59
    
    rank = 59
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A2*B2 (3x1x5), schoolbook
    # A2[i,0] = A[i,3], B2[0,k] = B[3,k]
    # U index: i*4+3, V index: 3*5+k = 15+k, W index: k*3+i
    for i in range(n):
        for k in range(p):
            U[i * m + 3, col] = 1.0
            V[3 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    # col = 15
    
    # Part 2: A1*B12 (3x3x2), schoolbook
    # A1[i,j] for i=0..2, j=0..2; B12[j,k] for j=0..2, k=3..4
    # U index: i*4+j, V index: j*5+k, W index: k*3+i
    for i in range(n):
        for j in range(3):
            for k in range(3, p):  # k=3,4
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    # col = 15 + 18 = 33
    
    # Part 3: A1*B11 (3x3x3), rank-26 decomposition
    # X = A1 (3x3), Y = B11 (3x3)
    # X = [X11(2x2) X12(2x1); X21(1x2) X22(1x1)]
    # Y = [Y11(2x2) Y12(2x1); Y21(1x2) Y22(1x1)]
    # Z = X*Y
    # Z[0:2,0:2] = X11*Y11 + X12*Y21
    # Z[0:2,2]   = X11*Y12 + X12*Y22
    # Z[2,0:2]   = X21*Y11 + X22*Y21
    # Z[2,2]     = X21*Y12 + X22*Y22
    
    # Sub-part 3a: X11*Y11 (2x2x2), Laderman rank 7
    # X11 = [[a,b],[c,d]] = X[0,0],X[0,1],X[1,0],X[1,1]
    # Y11 = [[e,f],[g,h]] = Y[0,0],Y[0,1],Y[1,0],Y[1,1]
    # U indices: a=X[0,0]->U[0], b=X[0,1]->U[1], c=X[1,0]->U[4], d=X[1,1]->U[5]
    # V indices: e=Y[0,0]->V[0], f=Y[0,1]->V[1], g=Y[1,0]->V[5], h=Y[1,1]->V[6]
    # W indices for Z[0:2,0:2]:
    #   Z[0,0]->W[0], Z[0,1]->W[3], Z[1,0]->W[1], Z[1,1]->W[4]
    
    # Laderman 2x2x2 rank 7:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # The issue: m7 as written is a difference of two bilinear forms,
    # which would be rank 2, making total rank 8.
    # 
    # Actually, Laderman's algorithm for 2x2x2 IS rank 7.
    # The correct 7 bilinear forms are:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # Wait, I think the confusion is that in the tensor decomposition,
    # each "term" corresponds to one rank-1 tensor. The expression for m7
    # when expanded gives a specific combination of monomials, and it
    # corresponds to exactly ONE rank-1 tensor in the decomposition.
    #
    # Let me just implement it and verify numerically.
    # If it doesn't work, I'll fall back to schoolbook.
    
    # For safety and correctness, let me just use schoolbook for 3x3x3.
    # That gives rank 27 for 3x3x3, total = 15 + 18 + 27 = 60.
    
    # Actually, let me try a different approach. Let me implement the
    # rank-26 for 3x3x3 using the block decomposition, but use schoolbook
    # for the 2x2x2 part (rank 8 instead of 7).
    # Then 3x3x3 = 8+4+4+2+4+2+2+1 = 27. Same as schoolbook.
    
    # So the only way to beat 60 is to correctly implement Laderman's rank-7.
    # Let me do that carefully.
    
    # Laderman 2x2x2 (rank 7):
    # The 7 bilinear forms and their contributions:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # Reconstruction:
    # Z[0,0] = m1 + m2 + m5 + m6 - m7
    # Z[0,1] = m1 - m2 + m5 - m6 - m7
    # Z[1,0] = m3 + m4 - m5 + m6 + m7
    # Z[1,1] = m3 - m4 - m5 - m6 + m7
    #
    # Let me verify with a simple example:
    # a=1,b=0,c=0,d=0 (X=identity top-left)
    # e=1,f=0,g=0,h=0 (Y=identity top-left)
    # Z should be [[1,0],[0,0]]
    #
    # m1 = (1+0)(1+0) = 1
    # m2 = (1-0)(1-0) = 1
    # m3 = (0+0)(0+0) = 0
    # m4 = (0-0)(0-0) = 0
    # m5 = (1+0)(1+0) = 1
    # m6 = (0+0)(0+0) = 0
    # m7 = (1+0+1+0)*(0+0+0+0) - (1+0+1+0)*(0+0+0+0) = 0
    #
    # Z[0,0] = 1+1+1+0-0 = 3. Should be 1. WRONG.
    #
    # So the reconstruction formula I have is incorrect.
    # Let me look up the correct Laderman 2x2 formulas.
    #
    # Correct Laderman 2x2 rank-7:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # Z[0,0] = (m1 + m2)/2 + (m5 + m6 - m7)/2
    # Z[0,1] = (m1 - m2)/2 + (m5 - m6 - m7)/2
    # Z[1,0] = (m3 + m4)/2 + (-m5 + m6 + m7)/2
    # Z[1,1] = (m3 - m4)/2 + (-m5 - m6 + m7)/2
    #
    # Let me verify: a=1,b=0,c=0,d=0, e=1,f=0,g=0,h=0
    # m1=1, m2=1, m3=0, m4=0, m5=1, m6=0, m7=0
    # Z[0,0] = (1+1)/2 + (1+0-0)/2 = 1 + 0.5 = 1.5. Still wrong.
    #
    # I'm clearly getting the formulas wrong. Let me just use schoolbook
    # for everything to ensure correctness. Rank 60.
    
    # Reset and use schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
