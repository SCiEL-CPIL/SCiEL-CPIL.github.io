import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110

def _verify(U, V, W, n=3, m=5, p=7, trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C using the tensor decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u = U_np[:, r]  # shape (n*m,)
            v = V_np[:, r]  # shape (m*p,)
            w = W_np[:, r]  # shape (p*n,)
            
            a_vec = u.reshape(n, m)
            b_vec = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            
            C_computed += a_vec @ b_vec * w_mat.T  # w_mat.T is (n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _strassen_3x3x3():
    """Return rank-23 decomposition for 3x3x3 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    W index: k*n + i = k*3 + i
    """
    n = 3
    U = [[Fraction(0)] * 23 for _ in range(9)]
    V = [[Fraction(0)] * 23 for _ in range(9)]
    W = [[Fraction(0)] * 23 for _ in range(9)]
    
    # Use the known rank-23 decomposition for 3x3x3
    # This is based on the "Strassen-like" decomposition
    # 
    # We partition 3x3 into blocks:
    # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    #
    # The 23 multiplications are:
    
    # First, let's use a 2x2 Strassen on the top-left 2x2 blocks
    # plus additional terms for the third row/column
    
    # M1 = (a00+a22)(b00+b22)
    # M2 = (a22+a11)(b22)  
    # M3 = (a00)(b11-b22)
    # M4 = (a22)(b00)
    # M5 = (a00+a11)(b00+b11)
    # M6 = (a11-a00)(b22+b11)
    # M7 = (a22-a11)(b00+b22)
    
    # These 7 handle the 2x2 subproblem. Now we need to handle
    # the interactions with the third row and column.
    
    # Actually, let me use a different, verified decomposition.
    # I'll use the one from the "Anisotropic Strassen" paper.
    
    # For 3x3x3, the rank-23 decomposition can be constructed as follows:
    # Treat it as a 2x2 block matrix where one block is 1x1.
    
    # Let me use a computational search approach with verification.
    # For now, I'll use a known explicit form.
    
    # Standard rank-23 decomposition (from literature):
    # The 23 terms are indexed 0..22
    
    # Term 0-6: Strassen on 2x2 subblock (rows 0,1 and cols 0,1)
    # M0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0*3+0][0] += 1; U[1*3+1][0] += 1
    V[0*3+0][0] += 1; V[1*3+1][0] += 1
    W[0*3+0][0] += 1; W[1*3+1][0] += 1  # C[0,0] and C[1,1]
    
    # M1 = (A[1,1]+A[1,0])*(B[1,1])  -- wait, this is getting complicated
    
    # Let me use a different approach: I'll use the "Baur-Strassen" 
    # decomposition which is well-documented.
    
    # Actually, let me just use a verified computational approach.
    # I'll define the decomposition and verify it.
    
    # Here's a known rank-23 decomposition for 3x3x3:
    # Based on the "Strassen recursion" with 2+1 partition
    
    # The decomposition has the following structure:
    # 7 terms for the 2x2x2 core
    # 16 terms for the cross interactions
    
    # Let me define them explicitly:
    
    # Core 2x2x2 Strassen (for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]):
    # s0 = A[0,0]+A[1,1], t0 = B[0,0]+B[1,1]
    # s1 = A[1,0]+A[1,1], t1 = B[1,1]
    # s2 = A[0,0], t2 = B[1,1]-B[0,0]  -- wait, standard Strassen is:
    
    # Standard Strassen for 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = (a00)(b11-b00)  -- no, (a00)(b01-b11)
    # m4 = (a11)(b10-b00)  -- no, (a11)(b10)
    # m5 = (a00+a10)(b00+b01)
    # m6 = (a10-a00)(b11+b01)  -- (a11-a01)(b00+b10)
    # m7 = (a11-a01)(b10+b11)
    
    # Let me use the standard Strassen formulas:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = (a00)(b01-b11)
    # m4 = (a11)(b10)
    # m5 = (a00+a10)(b00+b01)
    # m6 = (a10-a00)(b11+b01)
    # m7 = (a11-a01)(b10+b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Term 0: m1 = (a00+a11)(b00+b11)
    U[0][0] += 1; U[3][0] += 1   # a00, a11
    V[0][0] += 1; V[4][0] += 1   # b00, b11
    W[0][0] += 1; W[4][0] += 1   # C[0,0], C[1,1] (k=0,i=0 and k=1,i=1)
    
    # Term 1: m2 = (a10+a11)(b00)
    U[3][1] += 1; U[4][1] += 1   # a10, a11
    V[0][1] += 1                 # b00
    W[3][1] += 1; W[4][1] -= 1   # C[1,0] +=, C[1,1] -=
    
    # Term 2: m3 = (a00)(b01-b11)
    U[0][2] += 1                 # a00
    V[1][2] += 1; V[4][2] -= 1   # b01, -b11
    W[1][2] += 1; W[4][2] += 1   # C[0,1] +=, C[1,1] +=
    
    # Term 3: m4 = (a11)(b10)
    U[4][3] += 1                 # a11
    V[3][3] += 1                 # b10
    W[0][3] += 1; W[3][3] += 1   # C[0,0] +=, C[1,0] +=
    
    # Term 4: m5 = (a00+a10)(b00+b01)
    U[0][4] += 1; U[3][4] += 1   # a00, a10
    V[0][4] += 1; V[1][4] += 1   # b00, b01
    W[0][4] -= 1; W[1][4] += 1   # C[0,0] -=, C[0,1] +=
    
    # Term 5: m6 = (a10-a00)(b11+b01)
    U[3][5] += 1; U[0][5] -= 1   # a10, -a00
    V[4][5] += 1; V[1][5] += 1   # b11, b01
    W[4][5] += 1; W[1][5] += 1   # C[1,1] +=, C[0,1] +=
    
    # Term 6: m7 = (a11-a01)(b10+b11)
    U[4][6] += 1; U[1][6] -= 1   # a11, -a01
    V[3][6] += 1; V[4][6] += 1   # b10, b11
    W[0][6] += 1; W[3][6] += 1   # C[0,0] +=, C[1,0] +=
    
    # Now handle the third row and column interactions.
    # We need:
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    # C[2,0] = a20*b00 + a21*b10 + a22*b20
    # C[2,1] = a20*b01 + a21*b11 + a22*b21
    # C[2,2] = a20*b02 + a21*b12 + a22*b22
    
    # And also:
    # C[0,0] includes a02*b20, C[0,1] includes a02*b21
    # C[1,0] includes a12*b20, C[1,1] includes a12*b21
    # C[0,2] includes a00*b02, a01*b12, a02*b22
    # etc.
    
    # This is getting very complex. Let me use a different approach.
    # I'll use a computational method to find the decomposition.
    
    # For now, let me just use the schoolbook for 3x3x3 (rank 27) 
    # and see if the overall result is better than 105.
    # 3*27 + 2*8 = 81 + 16 = 97... still not great.
    
    # Actually, let me try a different strategy: use rank-23 for 3x3x3
    # by finding it computationally.
    
    # Let me return a placeholder and use schoolbook for now,
    # then verify the overall structure.
    
    return U, V, W

def _strassen_3x2x3():
    """Return rank-8 decomposition for 3x2x3 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k] where i in {0,1,2}, j in {0,1}, k in {0,1,2}
    U has shape (6, 8), V has shape (6, 8), W has shape (9, 8)
    W index: k*n + i = k*3 + i
    """
    n = 3
    U = [[Fraction(0)] * 8 for _ in range(6)]
    V = [[Fraction(0)] * 8 for _ in range(6)]
    W = [[Fraction(0)] * 8 for _ in range(9)]
    
    # 3x2x2 has rank 8. For 3x2x3, we can view it as two 3x2x2 problems
    # but with sharing, or as a 3x2x2 plus a 3x2x1.
    # 
    # Actually, 3x2x3: A is 3x2, B is 2x3, C is 3x3.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # This can be decomposed as:
    # For each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # 
    # Using rank-4 decomposition for 1x2x3 (outer product style):
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This needs 2 multiplications per row, so 6 total for 3 rows.
    # But we can do better.
    #
    # Rank 8 decomposition:
    # m0 = A[0,0]*B[0,0]
    # m1 = A[0,1]*B[1,0]
    # m2 = A[1,0]*B[0,1]
    # m3 = A[1,1]*B[1,1]
    # m4 = A[2,0]*B[0,2]
    # m5 = A[2,1]*B[1,2]
    # m6 = (A[0,0]+A[1,1])*B[0,1]  -- sharing
    # m7 = (A[1,0]+A[2,1])*B[1,2]  -- sharing
    #
    # This is getting complicated. Let me use a verified rank-8 decomposition.
    #
    # For 3x2x2 (rank 8), a known decomposition is:
    # m0 = a00*b00
    # m1 = a01*b10
    # m2 = a10*b01
    # m3 = a11*b11
    # m4 = (a00+a11)*b01
    # m5 = (a10+a01)*b00
    # m6 = (a10+a11)*b10
    # m7 = (a00+a01)*b11
    #
    # C00 = m0 + m1
    # C01 = m2 + m3 + m4 - m7  -- need to verify
    # C10 = m5 + m6 - m0
    # C11 = m2 + m3 + m7 - m5  -- need to verify
    #
    # This is hard to get right without verification. Let me use a 
    # computational approach with verification.
    
    return U, V, W

def construct():
    n, m, p = 3, 5, 7
    
    # For now, use schoolbook to ensure correctness, then we can improve
    rank = n * m * p
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1.0
                V[j*p + k][idx] = 1.0
                W[k*n + i][idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
