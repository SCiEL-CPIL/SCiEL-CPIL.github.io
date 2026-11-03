import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U, V, W):
        rank = len(U[0])
        # Check entries are half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False

        # Convert to integer representations (multiply by 2)
        U_int = [[int(round(2*val)) for val in row] for row in U]
        V_int = [[int(round(2*val)) for val in row] for row in V]
        W_int = [[int(round(2*val)) for val in row] for row in W]

        # Target tensor T[a,b,c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # Otherwise 0.

        # Check all (a,b,c) tuples
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                for c in range(p*n):
                    k1 = c // n
                    i2 = c % n
                    # Compute sum U[a,r] V[b,r] W[c,r]
                    s = 0
                    for r in range(rank):
                        s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                    # Target: 1 if (i1,j1,k2) matches and (j1==j2, i1==i2, k2==k1)
                    # i.e., a=i1*m+j1, b=j1*p+k2, c=k2*n+i1
                    target = 0
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        # Check if a, b, c correspond to the same (i,j,k)
                        # a = i*m + j => i=a//m, j=a%m
                        # b = j*p + k => j=b//p, k=b%p
                        # c = k*n + i => k=c//n, i=c%n
                        # So we need i1==i2, j1==j2, k1==k2
                        target = 1
                    # Sum should be target * 8 (since we multiplied by 2^3=8)
                    if s != target * 8:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    # U, V, W entries for a 2x2x2 tensor with indices (i,j,k) in {0,1}
    # Standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    def get_strassen_2x2x2():
        # Returns (U, V, W) for 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: a=i*2+j, b=j*2+k, c=k*2+i
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1: A00+A11, B00+B11, C00+=1, C11+=1
        # U: a=00 (i=0,j=0) -> 1, a=11 (i=1,j=1) -> 1
        U[0][0] = 1; U[3][0] = 1
        # V: b=00 (j=0,k=0) -> 1, b=11 (j=1,k=1) -> 1
        V[0][0] = 1; V[3][0] = 1
        # W: c=00 (k=0,i=0) -> 1, c=11 (k=1,i=1) -> 1
        W[0][0] = 1; W[3][0] = 1

        # M2: A10+A11, B00, C10+=1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # c=10 (k=1,i=0) -> Wait, c=k*2+i. C10 means i=1, k=0 -> c=0*2+1=1? No.
        # C[i,k]. C10 is i=1, k=0. c = k*2 + i = 0*2 + 1 = 1.
        # Let's re-verify index c = k*n + i. For n=2, c = k*2 + i.
        # C00: i=0, k=0 -> c=0
        # C01: i=0, k=1 -> c=2
        # C10: i=1, k=0 -> c=1
        # C11: i=1, k=1 -> c=3
        # So W[1][1] = 1 for M2.
        W[1][1] = 1

        # M3: A00, B10-B11, C01+=1
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1  # B10 (j=1,k=0) -> b=2, B11 (j=1,k=1) -> b=3
        W[2][2] = 1  # C01: i=0, k=1 -> c=2

        # M4: A11, B01-B00, C00+=1, C10+=1
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1  # B01 (j=0,k=1) -> b=1, B00 (j=0,k=0) -> b=0
        W[0][3] = 1; W[1][3] = 1   # C00 -> c=0, C10 -> c=1

        # M5: A00+A01, B11, C00-=1, C01+=1
        U[0][4] = 1; U[1][4] = 1   # A00 (i=0,j=0), A01 (i=0,j=1)
        V[3][4] = 1                # B11 (j=1,k=1) -> b=3
        W[0][4] = -1; W[2][4] = 1  # C00 -> c=0, C01 -> c=2

        # M6: A10-A00, B00+B01, C10+=1
        U[2][5] = 1; U[0][5] = -1  # A10 (i=1,j=0), A00 (i=0,j=0)
        V[0][5] = 1; V[1][5] = 1   # B00 (j=0,k=0), B01 (j=0,k=1)
        W[1][5] = 1                # C10 -> c=1

        # M7: A01-A11, B10+B11, C00-=1, C11+=1
        U[1][6] = 1; U[3][6] = -1  # A01 (i=0,j=1), A11 (i=1,j=1)
        V[2][6] = 1; V[3][6] = 1   # B10 (j=1,k=0), B11 (j=1,k=1)
        W[0][6] = -1; W[3][6] = 1  # C00 -> c=0, C11 -> c=3

        return U, V, W

    def construct_block_decomp():
        # Rank 28 construction
        # Blocks:
        # 1. Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1} -> Rank 7
        # 2. Rank 1 for i in {0,1}, j=2, k in {0,1} -> Rank 2
        # 3. Rank 1 for i=2, j in {0,1}, k in {0,1} -> Rank 2
        # 4. Rank 1 for i=2, j=2, k in {0,1} -> Rank 2
        # 5. Rank 9 for k=2 (all i,j) -> Rank 9
        # 6. Rank 9 for k=3 (all i,j) -> Rank 9
        # Total 7+2+2+2+9+9 = 31? Wait.
        # Let's re-evaluate.
        # For k in {0,1}:
        #   i in {0,1}, j in {0,1}: 2x2x2 tensor -> Rank 7
        #   i in {0,1}, j=2: 2x1x2 tensor -> Rank 2 (diagonal)
        #   i=2, j in {0,1}: 1x2x2 tensor -> Rank 2 (diagonal)
        #   i=2, j=2: 1x1x2 tensor -> Rank 2 (diagonal)
        #   Total for k in {0,1} is 7+2+2+2 = 13.
        #   Wait, is Rank(2x1x2) = 2?
        #   T[i,j,k] = A[i,j] B[j,k]. j=2 fixed.
        #   A[i,2] B[2,k]. This is an outer product of vector A[:,2] and vector B[2,:].
        #   Yes, rank 1? No, it's a sum of products.
        #   Wait, the tensor is T[i, j=2, k] = A[i,2] B[2,k].
        #   This is a matrix product of 2x1 and 1x2, resulting in 2x2.
        #   The tensor shape is (2, 1, 2).
        #   U shape: 2x1 (for i), V shape: 1x2 (for k), W shape: 2x1 (for i)?
        #   No, U is (n*m, R). Here n=2, m=1 (since j is fixed to 2 in the block? No, m is the inner dimension).
        #   In the full problem, m=3. We are partitioning the dimensions.
        #   Let's stick to the full dimensions.
        #   
        #   For i in {0,1}, j=2, k in {0,1}:
        #   We need terms for C[i,k] += A[i,2] B[2,k].
        #   This is 4 terms (i=0,1; k=0,1).
        #   Each term is rank 1: U[i*3+2, r]=1, V[2*4+k, r]=1, W[k*3+i, r]=1.
        #   So Rank 4.
        #   
        #   Similarly for i=2, j in {0,1}, k in {0,1}: Rank 4.
        #   i=2, j=2, k in {0,1}: Rank 2.
        #   
        #   So for k in {0,1}:
        #   Strassen part: 7.
        #   Cross terms: 4 (i<2, j=2) + 4 (i=2, j<2) + 2 (i=2, j=2) = 10.
        #   Total 17 for k in {0,1}.
        #   
        #   For k=2: Rank 9.
        #   For k=3: Rank 9.
        #   
        #   Total Rank = 17 + 9 + 9 = 35.
        #   
        #   Can we do better?
        #   Strassen for 2x2x2 is 7. Schoolbook is 8. Savings 1.
        #   Total schoolbook 36.
        #   Savings 1 -> Rank 35.
        #   
        #   Is there a better way?
        #   Maybe combine the cross terms?
        #   The cross terms for k in {0,1} are just the remaining part of the 3x3x2 multiplication.
        #   Rank(3x3x2) is at least ?
        #   If we use Strassen for the 2x2x2 core, the rest is 10 terms.
        #   Total 17.
        #   
        #   Let's try to see if we can reduce the cross terms.
        #   Actually, let's just implement Rank 35 first.
        #   
        #   Wait, is Rank(2x1x2) actually 2?
        #   The tensor is T[i, k] = A[i,2] B[2,k].
        #   This is a 2x2 matrix formed by outer product of v (2x1) and w (1x2).
        #   As a tensor T[i, 2, k] (in full dims), it has non-zero entries at (i, 2*4+k, k*3+i).
        #   This is a sum of 4 rank-1 tensors.
        #   So yes, Rank 4.
        
        U = [[0]*35 for _ in range(n*m)]
        V = [[0]*35 for _ in range(m*p)]
        W = [[0]*35 for _ in range(p*n)]
        
        # Block 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
        U_s, V_s, W_s = get_strassen_2x2x2()
        for r in range(7):
            for a_idx in range(4):
                if U_s[a_idx][r] != 0:
                    i = a_idx // 2
                    j = a_idx % 2
                    U[i*m+j][r] = U_s[a_idx][r]
            for b_idx in range(4):
                if V_s[b_idx][r] != 0:
                    j = b_idx // 2
                    k = b_idx % 2
                    V[j*p+k][r] = V_s[b_idx][r]
            for c_idx in range(4):
                if W_s[c_idx][r] != 0:
                    k = c_idx // 2
                    i = c_idx % 2
                    W[k*n+i][r] = W_s[c_idx][r]
        
        # Block 2: i in {0,1}, j=2, k in {0,1}
        # 4 terms
        col = 7
        for i in [0, 1]:
            for k in [0, 1]:
                U[i*m+2][col] = 1
                V[2*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        
        # Block 3: i=2, j in {0,1}, k in {0,1}
        # 4 terms
        for j in [0, 1]:
            for k in [0, 1]:
                U[2*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+2][col] = 1
                col += 1
        
        # Block 4: i=2, j=2, k in {0,1}
        # 2 terms
        for k in [0, 1]:
            U[2*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
            
        # Block 5: k=2 (all i,j)
        # 9 terms
        for i in range(3):
            for j in range(3):
                U[i*m+j][col] = 1
                V[j*p+2][col] = 1
                W[2*n+i][col] = 1
                col += 1
                
        # Block 6: k=3 (all i,j)
        # 9 terms
        for i in range(3):
            for j in range(3):
                U[i*m+j][col] = 1
                V[j*p+3][col] = 1
                W[3*n+i][col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}

    # First try the block construction
    res = construct_block_decomp()
    if verify_exact(res["U"], res["V"], res["W"]):
        return res

    # Fallback to schoolbook
    return schoolbook()
