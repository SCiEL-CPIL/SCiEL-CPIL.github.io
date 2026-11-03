import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Rank 15 decomposition for 2x3x4
    # Part 1: j in {0, 1} (0-indexed), corresponds to A[:, 0:2] * B[0:2, :]
    # This is M_{2,2,4}, decomposed as two M_{2,2,2} blocks.
    # Part 2: j = 2, corresponds to A[:, 2] * B[2, :]
    # This is rank 1.
    
    R = 15
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    # Helper to add Strassen decomposition for 2x2x2
    # Indices for A (2x2): i in 0,1; j in 0,1
    # Indices for B (2x2): j in 0,1; k in 0,1
    # Indices for C (2x2): i in 0,1; k in 0,1
    
    def add_strassen_block(offset_U, offset_V, offset_W, j_base, k_base):
        # j_base is the index in m (3) for the first j of this block
        # k_base is the index in p (4) for the first k of this block
        # We are decomposing A[0:2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
        # Wait, the j index in A and B must match.
        # In M_{2,2,4}, A is 2x2 (j=0,1), B is 2x4 (j=0,1).
        # Block 1: k=0,1. B uses j=0,1. A uses j=0,1.
        # Block 2: k=2,3. B uses j=0,1. A uses j=0,1.
        
        # Strassen's 7 multiplications for 2x2x2
        # M1 = (A11+A21)(B11+B21)
        # M2 = (A21)(B11)
        # M3 = (A11)(B21-B11)
        # M4 = (A11+A12)(B11)
        # M5 = (A12-A11)(B21+B11)
        # M6 = (A11+A21)(B12-B11) -- Wait, standard Strassen:
        # M1 = (a11+a21)(b11+b21)
        # M2 = (a21)(b11)
        # M3 = (a11)(b21-b11)
        # M4 = (a11+a12)(b11)
        # M5 = (a12-a11)(b21+b11)
        # M6 = (a11+a21)(b12-b11)
        # M7 = (a12+a22)(b21) -- No, let's use a standard set.
        
        # Standard Strassen:
        # M1 = (a11+a21)(b11+b21)
        # M2 = (a21)(b11)
        # M3 = (a11)(b21-b11)
        # M4 = (a11+a12)(b11)
        # M5 = (a12-a11)(b21+b11)
        # M6 = (a11+a21)(b12-b11)
        # M7 = (a12+a22)(b21) -- This doesn't look right.
        
        # Let's use the explicit rank-7 factors for 2x2x2.
        # The factors are known.
        # U, V, W are 4x7 matrices.
        
        # i, j, k in 0,1
        # T[i,j,k] = 1 if i,j,k valid.
        
        # We can hardcode the 7 terms.
        # Term 1: (a11+a21)(b11+b21)
        # u = [1, 1]^T (for a11, a21) -> indices 0, 1 in A
        # v = [1, 1]^T (for b11, b21) -> indices 0, 1 in B
        # w = ? 
        # This contributes to c11, c12, c21, c22.
        
        # Actually, it is easier to use the known rank-7 decomposition matrices.
        # See e.g. "The Rank of 2x2x2 Matrix Multiplication"
        
        # Let's just use the schoolbook for 2x2x2 (rank 4) if I can't recall Strassen exactly?
        # No, I want rank 15. Schoolbook for 2x2x4 is 8. Total 8+1=9? No.
        # Schoolbook for 2x2 is 4. Two blocks = 8.
        # Total 8 + 1 = 9?
        # Wait, schoolbook for 2x2x2 is 4.
        # If I use schoolbook for the two 2x2x2 blocks, rank is 4+4=8.
        # Total rank 8+1=9.
        # Is rank 9 valid for 2x3x4?
        # Yes, if it's an exact decomposition.
        # Schoolbook is always exact.
        # So I can just use schoolbook for the sub-blocks!
        # Schoolbook for M_{2,2,4}:
        # A (2x2) * B (2x4).
        # C1 = A B[:, 0:2]. 4 mults.
        # C2 = A B[:, 2:4]. 4 mults.
        # Total 8 mults.
        # Plus 1 for j=2.
        # Total 9.
        
        # Why did I think Strassen was needed?
        # Because I wanted lower rank.
        # But 9 is much lower than 24.
        # Can I go lower?
        # Strassen for 2x2x2 is 7.
        # Two Strassen blocks: 7+7=14.
        # 14 > 8.
        # So Schoolbook is BETTER for small sizes?
        # No, Strassen is 7, Schoolbook is 4.
        # 7 > 4.
        # So Schoolbook is rank 4. Strassen is rank 7.
        # So Schoolbook is better!
        
        # So for 2x2x2, rank is 4 (schoolbook) or 7 (Strassen)?
        # Wait. Strassen reduces multiplications from 8 to 7.
        # Schoolbook for 2x2x2 is 4 multiplications?
        # No, schoolbook for 2x2 matrix multiplication is 8 multiplications.
        # $C_{11} = A_{11}B_{11} + A_{12}B_{21}$ (2 mults)
        # $C_{12} = A_{11}B_{12} + A_{12}B_{22}$ (2 mults)
        # $C_{21} = A_{21}B_{11} + A_{22}B_{21}$ (2 mults)
        # $C_{22} = A_{21}B_{12} + A_{22}B_{22}$ (2 mults)
        # Total 8.
        
        # So Schoolbook rank for 2x2x2 is 8.
        # Strassen rank is 7.
        # So Strassen is better.
        
        # Back to 2x2x4:
        # Block 1 (2x2x2): Rank 7.
        # Block 2 (2x2x2): Rank 7.
        # Total 14.
        
        # Can we do better for 2x2x4?
        # Maybe 13?
        # If I use Strassen for the whole 2x2x4?
        # It's not a standard tensor.
        
        # So Rank 15 (14+1) is a solid improvement over 24.
        # I will implement Rank 15 using Strassen for the two 2x2x2 blocks.
        
        pass

    # Implementation of Strassen Rank 7 for 2x2x2
    # We need to map the 7 terms to the global indices.
    
    # Strassen Terms for 2x2x2 (local indices 0,1)
    # u, v, w are vectors of length 2, 2, 2.
    
    # Term 1: (a0+a1)(b0+b1)
    # u = [1, 1], v = [1, 1], w = [1, 0, 1, 0]? No.
    # The output w must reconstruct the C matrix.
    # C = U V W^T?
    # C[i,k] = sum_r U[i,r] V[j,r] W[k,r]?
    # No, the tensor is T[i,j,k].
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    
    # For 2x2x2:
    # T[0,0,0] = A00 B00
    # T[0,0,1] = A00 B01
    # T[0,1,0] = A01 B10
    # T[0,1,1] = A01 B11
    # ...
    
    # Strassen decomposition:
    # M1 = (A00+A10)(B00+B10)
    # M2 = (A10)(B00)
    # M3 = (A00)(B10-B00)
    # M4 = (A00+A01)(B00)
    # M5 = (A01-A00)(B10+B00)
    # M6 = (A00+A10)(B01-B00)
    # M7 = (A01+A11)(B10) -- Wait, I need the correct 7 terms.
    
    # Correct Strassen:
    # M1 = (A00+A10)(B00+B10)
    # M2 = (A10)(B00)
    # M3 = (A00)(B10-B00)
    # M4 = (A00+A01)(B00)
    # M5 = (A01-A00)(B10+B00)
    # M6 = (A00+A10)(B01-B00)
    # M7 = (A01+A11)(B10) -- This is not symmetric.
    
    # Let's use the standard 7:
    # M1 = (a+b)(e+f)
    # M2 = (a+b)(g+h)
    # M3 = (c+d)(e)
    # M4 = (a-c)(f+h)
    # M5 = (b+d)(g)
    # M6 = (a-c)(e+f)
    # M7 = (b+d)(g+h)
    # Where A = [[a,b],[c,d]], B = [[e,f],[g,h]].
    
    # I will hardcode these 7 terms for the two blocks.
    
    # Block 1: j=0,1, k=0,1
    # Block 2: j=0,1, k=2,3
    
    # For Block 1:
    # j indices in global V: 0, 1
    # k indices in global W: 0, 1
    # i indices in global U: 0, 1
    
    # For Block 2:
    # j indices in global V: 0, 1
    # k indices in global W: 2, 3
    # i indices in global U: 0, 1
    
    # Term 1 (M1): (a+b)(e+f)
    # a=A00, b=A10, e=B00, f=B10
    # u = [1, 1] (A00, A10)
    # v = [1, 1] (B00, B10) -> v[0]=1, v[1]=1
    # w = ?
    # This term contributes to C00, C01, C10, C11.
    # C00 += M1
    # C01 += M1
    # C10 += M1
    # C11 += M1
    # So w = [1, 1] (for k=0,1)
    
    # Term 2 (M2): (a+b)(g+h)
    # g=B01, h=B11
    # u = [1, 1]
    # v = [1, 1] (B01, B11) -> v[0]=1, v[1]=1
    # w = ?
    # C01 += M2
    # C11 += M2
    # w = [0, 1] (for k=0,1)
    
    # Term 3 (M3): (c+d)(e)
    # c=A01, d=A11, e=B00
    # u = [0, 1] (A01, A11)
    # v = [1, 0] (B00, B10)
    # w = ?
    # C00 += M3
    # C10 += M3
    # w = [1, 0]
    
    # Term 4 (M4): (a-c)(f+h)
    # a=A00, c=A01, f=B01, h=B11
    # u = [1, -1] (A00, A01)
    # v = [1, 1] (B01, B11)
    # w = ?
    # C01 += M4
    # C11 += M4
    # w = [0, 1]
    
    # Term 5 (M5): (b+d)(g)
    # b=A10, d=A11, g=B01
    # u = [0, 1] (A10, A11)
    # v = [1, 0] (B01, B11)
    # w = ?
    # C01 += M5
    # C11 += M5
    # w = [0, 1]
    
    # Term 6 (M6): (a-c)(e+f)
    # a=A00, c=A01, e=B00, f=B01
    # u = [1, -1]
    # v = [1, 1] (B00, B01)
    # w = ?
    # C00 += M6
    # C01 += M6
    # w = [1, 1]
    
    # Term 7 (M7): (b+d)(g+h)
    # b=A10, d=A11, g=B01, h=B11
    # u = [0, 1]
    # v = [1, 1]
    # w = ?
    # C01 += M7
    # C11 += M7
    # w = [0, 1]
    
    # Wait, I need to check the C reconstruction.
    # C00 = M1 - M2 + M3 - M4 + M6 - M7 ?
    # This is getting complicated to verify mentally.
    
    # I will use a simpler, verified rank-7 decomposition if possible, or just use rank 8 (schoolbook) for the blocks if Strassen is too risky to code blindly.
    # Schoolbook for 2x2x2 is 4 terms? No, 8 terms.
    # 8+8+1 = 17.
    # 17 is still better than 24.
    
    # I will implement Rank 17 using Schoolbook for the two 2x2x2 blocks.
    # It is safer and easier to code correctly.
    
    R = 17
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    col = 0
    # Block 1: j=0,1; k=0,1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i, col] = 1
                V[j, col] = 1
                W[k, col] = 1
                col += 1
                
    # Block 2: j=0,1; k=2,3
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i, col] = 1
                V[j, col] = 1
                W[k+2, col] = 1
                col += 1
                
    # Block 3: j=2; k=0..3
    # A[:, 2] * B[2, :]
    # u = [1, 1] (A02, A12)
    # v = [0, 0, 1] (B02, B12, B22) -> v[2]=1
    # w = [1, 1, 1, 1] (B20, B21, B22, B23)
    U[0, col] = 1
    U[1, col] = 1
    V[2, col] = 1
    W[0, col] = 1
    W[1, col] = 1
    W[2, col] = 1
    W[3, col] = 1
    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
