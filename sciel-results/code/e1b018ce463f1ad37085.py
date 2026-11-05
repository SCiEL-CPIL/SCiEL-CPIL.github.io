import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # Indices for A (2x3): i*3+j
    # Indices for B (3x10): j*10+k
    # Indices for C (2x10): k*2+i
    
    for b in range(5):
        base_col = b * 11
        # Outer product part: A[:, 0] * B[0, 2b:2b+2]
        # A[:, 0] is col 0. Indices: 0 (A[0,0]), 3 (A[1,0])
        # B[0, 2b:2b+2] is row 0, cols 2b, 2b+1. Indices: 10+2b, 10+2b+1
        # C[:, 2b:2b+2] is cols 2b, 2b+1. Indices: 2*2b+0, 2*2b+1 (for i=0,1)
        
        # Term 1: A[0,0] * B[0, 2b] * C[0, 2b]
        add_term(0, 10 + 2*b, 2*(2*b) + 0, 1, base_col)
        # Term 2: A[0,0] * B[0, 2b+1] * C[0, 2b+1]
        add_term(0, 10 + 2*b + 1, 2*(2*b + 1) + 0, 1, base_col + 1)
        # Term 3: A[1,0] * B[0, 2b] * C[1, 2b]
        add_term(3, 10 + 2*b, 2*(2*b) + 1, 1, base_col + 2)
        # Term 4: A[1,0] * B[0, 2b+1] * C[1, 2b+1]
        add_term(3, 10 + 2*b + 1, 2*(2*b + 1) + 1, 1, base_col + 3)
        
        # Strassen part: A[:, 1:3] * B[1:3, 2b:2b+2]
        # A submatrix:
        # A[0,1] (idx 1), A[0,2] (idx 2)
        # A[1,1] (idx 4), A[1,2] (idx 5)
        
        # B submatrix:
        # B[1, 2b] (idx 20+2b), B[1, 2b+1] (idx 20+2b+1)  <-- Row 1 of B is index 1? 
        # Wait. B is 3x10.
        # Row 0: 0..9
        # Row 1: 10..19
        # Row 2: 20..29
        # My previous mapping:
        # Outer product used B[0, k] -> 10+k? No.
        # B[j, k] index is j*10 + k.
        # Row 0 is j=0. Indices 0..9.
        # Row 1 is j=1. Indices 10..19.
        # Row 2 is j=2. Indices 20..29.
        
        # In the outer product part, I used B[0, 2b].
        # A[:, 0] is col 0.
        # The term is A[i,0] * B[0,k] * C[i,k].
        # This corresponds to the first row of B.
        # So B indices should be 0*10 + 2b = 2b.
        # Let's re-verify the outer product indices.
        # A[0,0] (idx 0). B[0, 2b] (idx 2b). C[0, 2b] (idx 2*2b+0).
        # I wrote 10+2b above. That is B[1, 2b].
        # The formula C = A B.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The outer product part used j=0.
        # So B index is 0*10 + k = k.
        # So B indices are 2b and 2b+1.
        
        # Let's correct the outer product terms.
        # I will rewrite the whole block generation to be safe.
        
        # Reset the block contributions? No, I'm building it up.
        # I'll just define the indices correctly now.
        
        # Outer product (j=0):
        # A[0,0] (0), B[0, 2b] (2b), C[0, 2b] (4b)
        # A[0,0] (0), B[0, 2b+1] (2b+1), C[0, 2b+1] (4b+2)
        # A[1,0] (3), B[0, 2b] (2b), C[1, 2b] (4b+1)
        # A[1,0] (3), B[0, 2b+1] (2b+1), C[1, 2b+1] (4b+3)
        
        # Strassen (j=1,2):
        # A[0,1] (1), A[0,2] (2)
        # A[1,1] (4), A[1,2] (5)
        # B[1, 2b] (10+2b), B[1, 2b+1] (10+2b+1)
        # B[2, 2b] (20+2b), B[2, 2b+1] (20+2b+1)
        # C[0, 2b] (4b), C[0, 2b+1] (4b+2)
        # C[1, 2b] (4b+1), C[1, 2b+1] (4b+3)
        
        # I will clear the previous incorrect additions by restarting the construction logic in the code.
        pass

    # Re-implementing correctly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(5):
        base_col = b * 11
        
        # Outer product (j=0)
        # A[0,0]=0, B[0,2b]=2b, C[0,2b]=4b
        U[0][base_col] += 1; V[2*b][base_col] += 1; W[4*b][base_col] += 1
        # A[0,0]=0, B[0,2b+1]=2b+1, C[0,2b+1]=4b+2
        U[0][base_col+1] += 1; V[2*b+1][base_col+1] += 1; W[4*b+2][base_col+1] += 1
        # A[1,0]=3, B[0,2b]=2b, C[1,2b]=4b+1
        U[3][base_col+2] += 1; V[2*b][base_col+2] += 1; W[4*b+1][base_col+2] += 1
        # A[1,0]=3, B[0,2b+1]=2b+1, C[1,2b+1]=4b+3
        U[3][base_col+3] += 1; V[2*b+1][base_col+3] += 1; W[4*b+3][base_col+3] += 1
        
        # Strassen (j=1,2)
        # A11=1, A12=2, A21=4, A22=5
        # B11=10+2b, B12=10+2b+1, B21=20+2b, B22=20+2b+1
        # C11=4b, C12=4b+2, C21=4b+1, C22=4b+3
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11(1), A22(5). V: B11, B22. W: C11, C22? No, W is for C.
        # The term is sum of products.
        # We add 4 rank-1 terms for each M? No, M is a scalar product of sums.
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # This contributes to C11, C12, C21, C22.
        # Actually, Strassen computes C from Ms.
        # It is easier to expand the Strassen formula into rank-1 terms directly?
        # No, Strassen reduces the number of multiplications.
        # The decomposition is:
        # Compute M1...M7.
        # Then C = linear combo of Ms.
        # This is a tensor decomposition where the "multiplication" step is the rank-1 term.
        # So for each M, we have one column in U, V, W.
        # U column contains coefficients for A.
        # V column contains coefficients for B.
        # W column contains coefficients for C.
        
        # M1 = (A11+A22)(B11+B22)
        # U[1] += 1, U[5] += 1
        # V[10+2b] += 1, V[20+2b] += 1
        # W[4b] += 1 (for C11), W[4b+2] += 1 (for C12)? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1 (col base_col+4):
        # U: A11(1), A22(5)
        # V: B11(10+2b), B22(20+2b+1)
        # W: C11(4b), C22(4b+3)
        U[1][base_col+4] += 1; U[5][base_col+4] += 1
        V[10+2*b][base_col+4] += 1; V[20+2*b+1][base_col+4] += 1
        W[4*b][base_col+4] += 1; W[4*b+3][base_col+4] += 1
        
        # M2 = (A21+A22)B11
        # U: A21(4), A22(5)
        # V: B11(10+2b)
        # W: C21(4b+1), C22(4b+3)
        U[4][base_col+5] += 1; U[5][base_col+5] += 1
        V[10+2*b][base_col+5] += 1
        W[4*b+1][base_col+5] += 1; W[4*b+3][base_col+5] += -1
        
        # M3 = A11(B12-B22)
        # U: A11(1)
        # V: B12(10+2b+1), B22(20+2b+1)
        # W: C12(4b+2), C22(4b+3)
        U[1][base_col+6] += 1
        V[10+2*b+1][base_col+6] += 1; V[20+2*b+1][base_col+6] += -1
        W[4*b+2][base_col+6] += 1; W[4*b+3][base_col+6] += 1
        
        # M4 = A22(B21-B11)
        # U: A22(5)
        # V: B21(20+2b), B11(10+2b)
        # W: C11(4b), C21(4b+1)
        U[5][base_col+7] += 1
        V[20+2*b][base_col+7] += 1; V[10+2*b][base_col+7] += -1
        W[4*b][base_col+7] += 1; W[4*b+1][base_col+7] += 1
        
        # M5 = (A11+A12)B22
        # U: A11(1), A12(2)
        # V: B22(20+2b+1)
        # W: C11(4b), C12(4b+2)
        U[1][base_col+8] += 1; U[2][base_col+8] += 1
        V[20+2*b+1][base_col+8] += 1
        W[4*b][base_col+8] += -1; W[4*b+2][base_col+8] += 1
        
        # M6 = (A21-A11)(B11+B12)
        # U: A21(4), A11(1)
        # V: B11(10+2b), B12(10+2b+1)
        # W: C22(4b+3)
        U[4][base_col+9] += 1; U[1][base_col+9] += -1
        V[10+2*b][base_col+9] += 1; V[10+2*b+1][base_col+9] += 1
        W[4*b+3][base_col+9] += 1
        
        # M7 = (A12-A22)(B21+B22)
        # U: A12(2), A22(5)
        # V: B21(20+2b), B22(20+2b+1)
        # W: C11(4b)
        U[2][base_col+10] += 1; U[5][base_col+10] += -1
        V[20+2*b][base_col+10] += 1; V[20+2*b+1][base_col+10] += 1
        W[4*b][base_col+10] += 1

    return {"U": U, "V": V, "W": W}
