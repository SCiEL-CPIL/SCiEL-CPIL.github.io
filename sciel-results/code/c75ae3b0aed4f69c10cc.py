"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    n, m, p = 3, 4, 7
    deadline = time.time() + 140  # 140s deadline (10s safety margin)
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W):
        """Check if decomposition is exact using integer/rational arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication (rank 7)."""
        # Strassen's 7 multiplications for 2x2 * 2x2
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        # M1 = (a11+a22)*(b11+b22)
        # M2 = (a21+a22)*b11
        # M3 = a11*(b12-b22)
        # M4 = a22*(b21-b11)
        # M5 = (a11+a12)*b22
        # M6 = (a21-a11)*(b11+b12)
        # M7 = (a12-a22)*(b21+b22)
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # For 2x2x2: n=2, m=2, p=2
        # U has 4 rows (i*m+j for i in 0..1, j in 0..1)
        # V has 4 rows (j*p+k for j in 0..1, k in 0..1)  
        # W has 4 rows (k*n+i for k in 0..1, i in 0..1)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a11+a22)*(b11+b22) -> contributes to c11, c22
        U[0][0] = 1  # a11
        U[3][0] = 1  # a22
        V[0][0] = 1  # b11
        V[3][0] = 1  # b22
        W[0][0] = 1  # c11 (k=0, i=0)
        W[3][0] = 1  # c22 (k=1, i=1)
        
        # M2 = (a21+a22)*b11 -> contributes to c21
        U[2][1] = 1  # a21
        U[3][1] = 1  # a22
        V[0][1] = 1  # b11
        W[2][1] = 1  # c21 (k=1, i=0)
        
        # M3 = a11*(b12-b22) -> contributes to c12, c22
        U[0][2] = 1  # a11
        V[1][2] = 1  # b12
        V[3][2] = -1 # b22
        W[1][2] = 1  # c12 (k=0, i=1)
        W[3][2] = 1  # c22 (k=1, i=1)
        
        # M4 = a22*(b21-b11) -> contributes to c11, c21
        U[3][3] = 1  # a22
        V[2][3] = 1  # b21
        V[0][3] = -1 # b11
        W[0][3] = 1  # c11 (k=0, i=0)
        W[2][3] = 1  # c21 (k=1, i=0)
        
        # M5 = (a11+a12)*b22 -> contributes to c11, c12
        U[0][4] = 1  # a11
        U[1][4] = 1  # a12
        V[3][4] = 1  # b22
        W[0][4] = -1 # c11 (k=0, i=0)
        W[1][4] = 1  # c12 (k=0, i=1)
        
        # M6 = (a21-a11)*(b11+b12) -> contributes to c22
        U[2][5] = 1  # a21
        U[0][5] = -1 # a11
        V[0][5] = 1  # b11
        V[1][5] = 1  # b12
        W[3][5] = 1  # c22 (k=1, i=1)
        
        # M7 = (a12-a22)*(b21+b22) -> contributes to c11, c22
        U[1][6] = 1  # a12
        U[3][6] = -1 # a22
        V[2][6] = 1  # b21
        V[3][6] = 1  # b22
        W[0][6] = 1  # c11 (k=0, i=0)
        W[3][6] = 1  # c22 (k=1, i=1)
        
        return U, V, W
    
    def validate_strassen():
        U, V, W = strassen_2x2x2()
        # Verify: for all i,j,k in {0,1}, sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] == 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j][c] * V[j*2+k][c] * W[k*2+i][c] for c in range(7))
                    if s != 1:
                        return False
        return True
    
    # Block decomposition approach:
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    # Split B (4x7) into B1 (2x7) and B2 (2x7)
    # C = A1*B1 + A2*B2
    # Each term is 3x2 * 2x7 = 3x7
    # Rank of 3x2x7: can we do better than schoolbook (42)?
    # Use Strassen-like: 3x2 * 2x7
    # Split 3 into 2+1, so (2x2 + 1x2) * (2x7)
    # 2x2 * 2x7: use Strassen on 2x2 part, rank 7 for 2x2x2, but we have 7 columns
    # Actually, 2x2 * 2x7: treat as 7 separate 2x2*2x1, each needs rank 2, total 14
    # But we can do better: 2x2 * 2x7 has tensor rank <= 2*7 = 14, but maybe less
    
    # Let's try a different approach: use known results
    # rank(3x4x7) <= rank(3x2x7) + rank(3x2x7) if we split the middle dimension
    # But we need to be careful about how we combine
    
    # Better: Use the fact that we can apply Strassen to overlapping blocks
    # Split B into three parts: B1 (4x2), B2 (4x2), B3 (4x3)
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bk is 3x4 * 4xk
    
    # For 3x4 * 4x2: split A into [A1(3x2) A2(3x2)], B into [B1(2x2); B2(2x2)]
    # C = A1*B1 + A2*B2, each 3x2*2x2
    # 3x2*2x2: split 3 into 2+1
    # (2x2+1x2)*2x2 = 2x2*2x2 + 1x2*2x2
    # 2x2*2x2: rank 7 (Strassen)
    # 1x2*2x2: rank 2 (schoolbook: 1*2*2=4, but actually rank 2 since it's outer product-like)
    # Wait, 1x2 * 2x2: C[0,k] = sum_j A[0,j]*B[j,k], this is rank 2 (one row of A times B)
    # So 3x2*2x2 has rank <= 7+2 = 9
    # Thus 3x4*4x2 has rank <= 9+9 = 18
    
    # Similarly, 3x4*4x3: split B into B1(4x2)+B2(4x1)
    # 3x4*4x2: rank <= 18 (as above)
    # 3x4*4x1: split A into A1(3x2)+A2(3x2), B into B1(2x1)+B2(2x1)
    # C = A1*B1 + A2*B2, each 3x2*2x1
    # 3x2*2x1: C[0,k] = sum_j A[0,j]*B[j,k], rank <= 2 (since it's a linear combination)
    # Actually 3x2*2x1 has tensor rank 2 (just two terms)
    # So 3x4*4x1 has rank <= 2+2 = 4
    # Thus 3x4*4x3 has rank <= 18+4 = 22
    
    # Total: 3x4*4x7 = 3x4*4x2 + 3x4*4x2 + 3x4*4x3 has rank <= 18+18+22 = 58
    
    # Let's try to construct this!
    
    def construct_3x2x2x2():
        """Construct decomposition for 3x2 * 2x2 with rank 9."""
        # C (3x2) = A (3x2) * B (2x2)
        # Split A into A1 (2x2) and A2 (1x2)
        # C = A1*B + A2*B
        # A1*B: 2x2*2x2, rank 7 (Strassen)
        # A2*B: 1x2*2x2, rank 2
        # Total rank: 9
        
        U = [[0]*9 for _ in range(6)]  # 3*2 = 6 rows
        V = [[0]*9 for _ in range(4)]  # 2*2 = 4 rows
        W = [[0]*9 for _ in range(6)]  # 2*3 = 6 rows
        
        # First 7 terms: Strassen for A1 (rows 0,1,2,3) * B
        U1, V1, W1 = strassen_2x2x2()
        for c in range(7):
            for r in range(4):
                U[r][c] = U1[r][c]
            for r in range(4):
                V[r][c] = V1[r][c]
            for r in range(4):
                W[r][c] = W1[r][c]
        
        # Next 2 terms: A2 (row 4,5) * B
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        # Term 7: A[2,0] * B[0,0] -> C[2,0]
        U[4][7] = 1
        V[0][7] = 1
        W[4][7] = 1  # k=0, i=2 -> index 0*3+2=2... wait, W index is k*n+i
        
        # W has p*n = 2*3 = 6 rows, indexed by k*n+i
        # k=0, i=2: index 0*3+2 = 2
        # k=1, i=2: index 1*3+2 = 5
        
        # Term 7: A[2,0]*B[0,0] -> C[2,0]
        U[4][7] = 1   # A row 2, col 0 -> index 2*2+0=4
        V[0][7] = 1   # B row 0, col 0 -> index 0*2+0=0
        W[2][7] = 1   # C row 2, col 0 -> k=0, i=2 -> index 0*3+2=2
        
        # Term 8: A[2,1]*B[1,0] -> C[2,0]
        U[5][8] = 1   # A row 2, col 1 -> index 2*2+1=5
        V[2][8] = 1   # B row 1, col 0 -> index 1*2+0=2
        W[2][8] = 1   # C row 2, col 0 -> k=0, i=2 -> index 2
        
        # Wait, I need to also cover C[2,1]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        # I only have 2 terms, but I need 4 contributions. Let me reconsider.
        
        # Actually, 1x2 * 2x2 has tensor rank 2? Let me check.
        # C[0,k] = sum_j A[0,j] * B[j,k]
        # This is: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        #           C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # Tensor: T[i,j,k] where i=0 always, j in {0,1}, k in {0,1}
        # T[0,j,k] = 1 for all j,k
        # This is a 1x2x2 tensor, which has rank 2 (it's the sum of two rank-1 tensors:
        # T = e_0 ⊗ e_0 ⊗ e_0 + e_0 ⊗ e_1 ⊗ e_0 + e_0 ⊗ e_0 ⊗ e_1 + e_0 ⊗ e_1 ⊗ e_1
        # Wait, that's 4 terms. Actually the tensor is T[i,j,k] = delta(i,0), so it's
        # e_0 ⊗ (e_0⊗e_0 + e_0⊗e_1 + e_1⊗e_0 + e_1⊗e_1) = e_0 ⊗ (e_0+e_1) ⊗ (e_0+e_1)
        # So it's rank 1! T = e_0 ⊗ f ⊗ g where f = e_0+e_1, g = e_0+e_1
        
        # So 1x2*2x2 has rank 1!
        # C[0,k] = (A[0,0]+A[0,1]) * (B[0,k]+B[1,k])... no that's not right.
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # Let M = A[0,0]+A[0,1], N_k = B[0,k]+B[1,k]
        # M*N_0 = A[0,0]*B[0,0] + A[0,0]*B[1,0] + A[0,1]*B[0,0] + A[0,1]*B[1,0]
        # That's not equal to C[0,0].
        
        # Let me re-examine. The tensor for 1x2 * 2x2 is:
        # T[i,j,k] = 1 if i=0 (always true for 1-row), so T = 1 ⊗ I_2 ⊗ I_2
        # This has rank 4 (it's the identity on the 2x2 part).
        # Actually no: T[i,j,k] = delta(i,0) * 1, so T = e_0 ⊗ (sum_{j,k} e_j ⊗ e_k)
        # = e_0 ⊗ (e_0+e_1) ⊗ (e_0+e_1)
        # This IS rank 1!
        
        # But wait, that means C[0,k] = sum_j A[0,j]*B[j,k] = (sum_j A[0,j]) * (sum_j B[j,k])?
        # No! sum_j A[0,j]*B[j,k] is NOT equal to (sum_j A[0,j])*(sum_j B[j,k])
        # The former is a dot product, the latter is a product of sums.
        
        # I'm confusing myself. Let me be precise.
        # The multiplication tensor M_{1,2,2} has M[i,j,k] = 1 for all valid i,j,k
        # i in {0}, j in {0,1}, k in {0,1}
        # So M has 4 non-zero entries, all equal to 1.
        # M = e_0 ⊗ e_0 ⊗ e_0 + e_0 ⊗ e_0 ⊗ e_1 + e_0 ⊗ e_1 ⊗ e_0 + e_0 ⊗ e_1 ⊗ e_1
        # This is e_0 ⊗ (e_0⊗e_0 + e_0⊗e_1 + e_1⊗e_0 + e_1⊗e_1)
        # = e_0 ⊗ (e_0+e_1) ⊗ (e_0+e_1)
        # So yes, it IS rank 1!
        
        # But then C[0,k] = sum_j A[0,j]*B[j,k] should equal (sum_j A[0,j])*(sum_j B[j,k])?
        # Let's check: (A[0,0]+A[0,1])*(B[0,0]+B[1,0]) = A[0,0]*B[0,0] + A[0,0]*B[1,0] + A[0,1]*B[0,0] + A[0,1]*B[1,0]
        # But C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # These are NOT equal in general!
        
        # I see my error. The tensor M[i,j,k] = 1 means that the (i,j) entry of A and (j,k) entry of B
        # contribute to the (i,k) entry of C. So for 1x2*2x2:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # The tensor T[i,j,k] has T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
        # This is NOT e_0 ⊗ (e_0+e_1) ⊗ (e_0+e_1)
        # (e_0+e_1) ⊗ (e_0+e_1) = e_0⊗e_0 + e_0⊗e_1 + e_1⊗e_0 + e_1⊗e_1
        # So e_0 ⊗ (e_0+e_1) ⊗ (e_0+e_1) = e_0⊗e_0⊗e_0 + e_0⊗e_0⊗e_1 + e_0⊗e_1⊗e_0 + e_0⊗e_1⊗e_1
        # Which IS exactly our tensor!
        
        # So the tensor IS rank 1, which means:
        # C[0,k] = (sum_j A[0,j]) * (sum_j B[j,k])
        # But that's not how matrix multiplication works!
        
        # I think I'm making a fundamental error in understanding the tensor.
        # Let me re-read the problem statement.
        
        # "The output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed by (k*n+i), meaning for C[i,k], we use W[k*n+i]
        
        # The decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
        # And the tensor form is: T[i,j,k] = 1 (for all valid i,j,k)
        # With the constraint: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # So for 1x2*2x2: n=1, m=2, p=2
        # U has 1*2=2 rows, V has 2*2=4 rows, W has 2*1=2 rows
        # T[0,j,k] = 1 for j in {0,1}, k in {0,1}
        # T[0,0,0] = sum_c U[0,c]*V[0,c]*W[0,c] = 1
        # T[0,0,1] = sum_c U[0,c]*V[1,c]*W[1,c] = 1
        # T[0,1,0] = sum_c U[1,c]*V[2,c]*W[0,c] = 1
        # T[0,1,1] = sum_c U[1,c]*V[3,c]*W[1,c] = 1
        
        # Can we do this with rank 1?
        # U[0,0]=a, U[1,0]=b, V[0,0]=c, V[1,0]=d, V[2,0]=e, V[3,0]=f, W[0,0]=g, W[1,0]=h
        # a*c*g = 1, a*d*h = 1, b*e*g = 1, b*f*h = 1
        # From first: a*c*g=1, second: a*d*h=1, so c*g = d*h
        # Third: b*e*g=1, fourth: b*f*h=1, so e*g = f*h
        # We have 8 variables, 4 equations, so solutions exist.
        # For example: a=1, b=1, c=1, d=1, e=1, f=1, g=1, h=1
        # Then: 1*1*1=1, 1*1*1=1, 1*1*1=1, 1*1*1=1. Works!
        
        # So 1x2*2x2 has rank 1!
        
        # Let me verify with actual matrix multiplication:
        # A = [[a0, a1]] (1x2), B = [[b00, b01], [b10, b11]] (2x2)
        # C = A*B = [[a0*b00 + a1*b10, a0*b01 + a1*b11]] (1x2)
        # With rank 1 decomposition:
        # C[0,0] = U[0]*V[0]*W[0] + U[1]*V[2]*W[0] = a0*1*1 + a1*1*1 = a0 + a1
        # But we need C[0,0] = a0*b00 + a1*b10
        # These are NOT equal unless b00=b10=1!
        
        # I see the problem. The tensor T[i,j,k]=1 means that the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
        # But in the decomposition, we're summing U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # This should equal 1 for each (i,j,k), but the actual computation is:
        # C[i,k] = sum_j (sum_c U[i*m+j,c]*W[k*n+i,c]) * (sum_c' ... )
        
        # No wait, I think I'm overcomplicating this. The tensor decomposition means:
        # The bilinear map (A,B) -> C has a tensor representation, and we're finding a rank decomposition of that tensor.
        
        # For now, let me just use schoolbook for the small cases and see if I can get a valid answer.
        
        # Given the complexity, let me just return schoolbook for now and try a simpler improvement.
        pass
    
    # Given the time constraints and complexity, let me try a simpler approach:
    # Use schoolbook but with a small optimization: merge terms where possible
    
    # Actually, let me try a direct search for rank 83, 82, ... using a smarter initialization
    
    def try_rank(target_rank):
        """Try to find a valid decomposition of given rank."""
        # Initialize with schoolbook, then try to remove/merge terms
        U = [[0]*target_rank for _ in range(n*m)]
        V = [[0]*target_rank for _ in range(m*p)]
        W = [[0]*target_rank for _ in range(p*n)]
        
        # Fill with schoolbook terms for the first target_rank terms
        for c in range(min(target_rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            U[i*m+j][c] = 1
            V[j*p+k][c] = 1
            W[k*n+i][c] = 1
        
        # For remaining terms, try to find useful combinations
        # This is a heuristic; in practice we'd need a proper search
        
        return U, V, W
    
    # Given the difficulty, let me just return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
