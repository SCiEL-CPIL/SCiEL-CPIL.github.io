"""Rewritten program for matrix multiplication 2,2,8 tensor decomposition."""
import time
import random
import math

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank decomposition for multiplying a 2x2 matrix by a 2x8 matrix.
    We use Strassen-like algorithms adapted to this specific shape.
    
    The tensor T_{i,j,k} for (A_{i,j}, B_{j,k}, C_{i,k}) where:
    - A is 2x2 (i in 0..1, j in 0..1)
    - B is 2x8 (j in 0..1, k in 0..7)
    - C is 2x8 (i in 0..1, k in 0..7)
    
    W[k*n+i, r] is the output factor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # Try to find a low-rank decomposition
    # First, let's try the schoolbook as a baseline, then attempt optimization
    
    # Schoolbook: rank = n*m*p = 32
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Let's try a Strassen-inspired approach
    # For 2x2 * 2x8, we can split the 8 columns into 4 groups of 2
    # Then use Strassen on each 2x2 * 2x2 block
    
    # Strassen for 2x2 * 2x2 uses 7 multiplications
    # For 4 blocks: 4 * 7 = 28 multiplications
    
    # Let's construct this explicitly
    # Split B into 4 blocks of 2 columns each: B_0, B_1, B_2, B_3
    # C = A * B = [A*B_0 | A*B_1 | A*B_2 | A*B_3]
    
    # For each 2x2 * 2x2 block, use Strassen's algorithm:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We need to express this as a tensor decomposition
    # Each "multiplication" M_r corresponds to one rank-1 term
    
    # Let's build U, V, W for rank 28 (4 blocks * 7 Strassen terms)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # For block b (0..3), columns 2b and 2b+1 of B
    # Strassen indices: r = b*7 + s where s is 0..6
    
    for b in range(4):
        col1 = 2 * b      # First column index in B for this block
        col2 = 2 * b + 1  # Second column index in B for this block
        
        # Strassen terms for this block
        # M1 = (a+d)(e+h): U has a,d; V has e,h
        # M2 = (c+d)(e): U has c,d; V has e
        # M3 = (a)(f-h): U has a; V has f,-h
        # M4 = (d)(g-e): U has d; V has g,-e
        # M5 = (a+b)(h): U has a,b; V has h
        # M6 = (c-a)(e+g): U has c,-a; V has e,g
        # M7 = (b-d)(f+h): U has b,-d; V has f,h
        
        # A indices: i*m+j where i is row of A, j is col of A
        # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        # So: a is at U[0], b is at U[1], c is at U[2], d is at U[3]
        
        # B indices: j*p+k where j is row of B, k is col of B
        # For block b: col1 and col2
        # e = B[0, col1], f = B[0, col2], g = B[1, col1], h = B[1, col2]
        # So: e is at V[0*p+col1] = V[col1], f is at V[col2], g is at V[8+col1], h is at V[8+col2]
        
        # C indices: k*n+i where k is col of C, i is row of C
        # C[0, col1] is at W[col1*2+0] = W[2*col1]
        # C[0, col2] is at W[2*col2]
        # C[1, col1] is at W[2*col1+1]
        # C[1, col2] is at W[2*col2+1]
        
        # Strassen term 1: M1 = (a+d)(e+h)
        r = b * 7 + 0
        U[0][r] += 1  # a
        U[3][r] += 1  # d
        V[col1][r] += 1    # e
        V[8 + col2][r] += 1  # h
        # C[0,col1] += M1, C[0,col2] += M1, C[1,col1] -= M1, C[1,col2] += M1
        # Wait, let me recheck the Strassen formulas:
        # Result[0,0] = M1 + M4 - M5 + M7
        # Result[0,1] = M3 + M5
        # Result[1,0] = M2 + M4
        # Result[1,1] = M1 - M2 + M3 + M6
        
        # So for M1: contributes to C[0,col1] (+1), C[0,col2] (+1), C[1,col1] (-1), C[1,col2] (+1)
        W[2*col1][r] += 1    # C[0, col1]
        W[2*col2][r] += 1    # C[0, col2]
        W[2*col1 + 1][r] += -1  # C[1, col1]
        W[2*col2 + 1][r] += 1   # C[1, col2]
        
        # Strassen term 2: M2 = (c+d)(e)
        r = b * 7 + 1
        U[2][r] += 1  # c
        U[3][r] += 1  # d
        V[col1][r] += 1    # e
        # M2 contributes to: C[1,col1] (+1), C[1,col2] (-1)
        W[2*col1 + 1][r] += 1   # C[1, col1]
        W[2*col2 + 1][r] += -1  # C[1, col2]
        
        # Strassen term 3: M3 = (a)(f-h)
        r = b * 7 + 2
        U[0][r] += 1  # a
        V[col2][r] += 1      # f
        V[8 + col2][r] += -1  # -h
        # M3 contributes to: C[0,col2] (+1), C[1,col2] (+1)
        W[2*col2][r] += 1    # C[0, col2]
        W[2*col2 + 1][r] += 1  # C[1, col2]
        
        # Strassen term 4: M4 = (d)(g-e)
        r = b * 7 + 3
        U[3][r] += 1  # d
        V[8 + col1][r] += 1   # g
        V[col1][r] += -1      # -e
        # M4 contributes to: C[0,col1] (+1), C[1,col1] (+1)
        W[2*col1][r] += 1    # C[0, col1]
        W[2*col1 + 1][r] += 1  # C[1, col1]
        
        # Strassen term 5: M5 = (a+b)(h)
        r = b * 7 + 4
        U[0][r] += 1  # a
        U[1][r] += 1  # b
        V[8 + col2][r] += 1  # h
        # M5 contributes to: C[0,col1] (-1), C[0,col2] (+1)
        W[2*col1][r] += -1   # C[0, col1]
        W[2*col2][r] += 1    # C[0, col2]
        
        # Strassen term 6: M6 = (c-a)(e+g)
        r = b * 7 + 5
        U[2][r] += 1   # c
        U[0][r] += -1  # -a
        V[col1][r] += 1     # e
        V[8 + col1][r] += 1  # g
        # M6 contributes to: C[1,col2] (+1)
        W[2*col2 + 1][r] += 1  # C[1, col2]
        
        # Strassen term 7: M7 = (b-d)(f+h)
        r = b * 7 + 6
        U[1][r] += 1   # b
        U[3][r] += -1  # -d
        V[col2][r] += 1      # f
        V[8 + col2][r] += 1  # h
        # M7 contributes to: C[0,col1] (+1)
        W[2*col1][r] += 1  # C[0, col1]
    
    # Verify the decomposition
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's check a few random points
    
    # Actually, let's verify the full tensor
    # T[i,j,k] = (A*B)_{i,k} where A is n x m, B is m x p
    # T[i,j,k] = 1 if this is the standard matrix multiplication tensor
    
    # For standard matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for matmul)
    
    # The decomposition gives: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify by checking all (i,j,k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1  # Standard matmul tensor
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-END
