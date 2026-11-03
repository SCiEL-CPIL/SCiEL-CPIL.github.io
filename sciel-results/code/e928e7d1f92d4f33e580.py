import numpy as np

def construct():
    n, m, p = 5, 5, 5
    # coeff[i, j, k] stores the coefficient for A[i,j] * B[j,k] contributing to C[i,k]
    # Note: The problem uses W[k*n+i], so C index is (k, i) in W?
    # Let's stick to C[i, k] for logic, and map to W index k*n+i at the end.
    coeff = np.zeros((n, m, p), dtype=int)
    
    # Helper to add a term A[i,j] * B[j,k] -> C[i,k] with coefficient c
    def add_term(i, j, k, c):
        coeff[i, j, k] += c
        
    # Strassen 2x2 block starting at (r, c) for A, (c, r2) for B?
    # A block: rows r..r+1, cols c..c+1
    # B block: rows c..c+1, cols r2..r2+1
    # C block: rows r..r+1, cols r2..r2+1
    def add_strassen_2x2(r, c, r2):
        # Local indices 0,1 map to global:
        # A: (r, c), (r, c+1), (r+1, c), (r+1, c+1)
        # B: (c, r2), (c, r2+1), (c+1, r2), (c+1, r2+1)
        # C: (r, r2), (r, r2+1), (r+1, r2), (r+1, r2+1)
        
        # Define linear forms for A and B blocks
        # A11, A12, A21, A22
        # B11, B12, B21, B22
        
        # M1 = (A11 + A22)(B11 + B22)
        # A_part: {(r,c):1, (r+1,c+1):1}
        # B_part: {(c,r2):1, (c+1,r2+1):1}
        # Expand product and add to C11, C22?
        # Wait, M1 is a 2x2 matrix.
        # M1[0,0] = A11*B11 + A11*B21 + A21*B11 + A21*B21? No.
        # M1 is the product of two 2x2 matrices X and Y.
        # X = A11+A22, Y = B11+B22.
        # X[0,0] = A[r,c] + A[r+1,c+1]
        # X[0,1] = A[r,c+1] + A[r+1,c] ? No, A12 is top-right.
        # A11 = A[r,c], A12 = A[r,c+1], A21 = A[r+1,c], A22 = A[r+1,c+1].
        # X = A11 + A22 => X[0,0]=A[r,c], X[0,1]=0, X[1,0]=0, X[1,1]=A[r+1,c+1].
        # Wait, A11+A22 is a matrix sum.
        # (A11 + A22) is a 2x2 matrix with diagonal elements A11, A22 and off-diagonals 0?
        # No, A11 is the top-left 1x1 block? No, in Strassen 2x2, A11 is 1x1.
        # So A11+A22 is a 1x1 matrix? No, Strassen works on 2x2 matrices.
        # A is 2x2. A11 is 1x1.
        # So X is 1x1? No.
        # In Strassen, A11, A12 etc are 1x1.
        # M1 is a scalar.
        # My previous "2x2 inner" assumption was for 4x4.
        # For 5x5, we can use Strassen on 2x2 blocks directly?
        # 5 is not divisible by 2.
        
        # Let's use the 4x4 core + remainder approach.
        # Core 4x4 uses Strassen (rank 49 if recursive, 56 if schoolbook inner).
        # Let's use Schoolbook inner for simplicity (rank 56).
        
        # We need to define the 7 Strassen products for the 4x4 block.
        # This is complex to code generically.
        
        # Alternative: Just use Schoolbook for the whole 5x5? Rank 125.
        # The seed is 125.
        # We need to beat 125.
        
        # Let's try a simpler improvement: 
        # Use the fact that 5x5 can be done with 2x2 Strassen blocks?
        # No.
        
        # Let's implement the Laderman 4x4 (rank 21) + remainder (rank 16) = 37.
        # Laderman coefficients are known.
        
        # Laderman 4x4:
        # A = [A11 A12; A21 A22] (2x2 blocks of 2x2)
        # 21 products.
        
        # This is too hard to code from scratch without errors.
        
        # Let's stick to the recursive Strassen for 4x4 (rank 49) + remainder (16) = 65.
        # We will implement a generic function to add a matmul decomposition.
        
        pass
