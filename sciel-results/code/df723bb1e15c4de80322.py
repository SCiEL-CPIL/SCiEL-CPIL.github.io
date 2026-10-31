"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # BCLR rank-7 identity for 2x2 matrices:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # Z0 = a*e
    # Z1 = (a+d)*(g+h)
    # Z2 = (c-a)*(e+g)
    # Z3 = d*(h-f)
    # Z4 = (b-d)*(f+h)
    # Z5 = (c+d)*(e-g)
    # Z6 = (b+c)*(g-e)
    # C00 = Z0 + Z1 - Z2 + Z5
    # C01 = Z4 + Z1 - Z2 - Z6
    # C10 = Z3 + Z1 - Z2 + Z5
    # C11 = Z3 + Z4 + Z6
    
    # For 4x4: split into 2x2 blocks of 2x2 matrices.
    # 4 block positions (I,J) for A, (J,K) for B, (I,K) for C.
    # For each block product (I,J,K), use 7 block bilinear forms.
    # Each block bilinear form is a 2x2 matrix of scalar bilinear forms.
    # Total scalar bilinear forms: 4 (block positions) * 7 (BCLR) * 4 (2x2 entries) = 112.
    # But many are shared. Actually, each BCLR form Z_t for block (I,J,K) 
    # is a 2x2 matrix where each entry is a bilinear form in the 4x4 entries.
    
    # Let me construct U, V, W directly.
    # Indexing: A[i,j] with i=0..3, j=0..3. B[j,k] with j=0..3, k=0..3.
    # C[i,k] = sum_j A[i,j] * B[j,k].
    
    # Block structure:
    # A block (I,J) = A[2I:2I+2, 2J:2J+2], I,J in {0,1}
    # B block (J,K) = B[2J:2J+2, 2K:2K+2]
    # C block (I,K) = C[2I:2I+2, 2K:2K+2]
    
    # For each (I,J,K) in {0,1}^3, we have a 2x2 block product.
    # BCLR gives 7 bilinear forms Z0..Z6, each a 2x2 matrix.
    # Each entry of Z_t is a bilinear form in the 4 entries of A block (I,J) and 4 entries of B block (J,K).
    
    # Let's enumerate all bilinear forms.
    # For a fixed (I,J,K), A block entries: a00=A[2I,2J], a01=A[2I,2J+1], a10=A[2I+1,2J], a11=A[2I+1,2J+1]
    # B block entries: b00=B[2J,2K], b01=B[2J,2K+1], b10=B[2J+1,2K], b11=B[2J+1,2K+1]
    
    # Z0 = a00*b00
    # Z1 = (a00+a11)*(b10+b11) = a00*b10 + a00*b11 + a11*b10 + a11*b11
    # Z2 = (a10-a00)*(b00+b10) = a10*b00 + a10*b10 - a00*b00 - a00*b10
    # Z3 = a11*(b11-b01) = a11*b11 - a11*b01
    # Z4 = (a01-a11)*(b01+b11) = a01*b01 + a01*b11 - a11*b01 - a11*b11
    # Z5 = (a10+a11)*(b00-b10) = a10*b00 - a10*b10 + a11*b00 - a11*b10
    # Z6 = (a01+a10)*(b10-b00) = a01*b10 - a01*b00 + a10*b10 - a10*b00
    
    # C block entries:
    # C00 = Z0 + Z1 - Z2 + Z5
    # C01 = Z4 + Z1 - Z2 - Z6
    # C10 = Z3 + Z1 - Z2 + Z5
    # C11 = Z3 + Z4 + Z6
    
    # Now, each Z_t is a 2x2 matrix. But wait, Z_t is a scalar (product of 2x2 matrices), 
    # so Z_t is itself a 2x2 matrix. The entries of Z_t are bilinear forms.
    
    # Actually, I think I'm overcomplicating. Let me just verify if the BCLR identity 
    # works for 2x2 matrices by testing numerically.
    
    # Test:
    A = np.array([[1,2],[3,4]], dtype=float)
    B = np.array([[5,6],[7,8]], dtype=float)
    a,b,c,d = A[0,0],A[0,1],A[1,0],A[1,1]
    e,f,g,h = B[0,0],B[0,1],B[1,0],B[1,1]
    Z0 = a*e
    Z1 = (a+d)*(g+h)
    Z2 = (c-a)*(e+g)
    Z3 = d*(h-f)
    Z4 = (b-d)*(f+h)
    Z5 = (c+d)*(e-g)
    Z6 = (b+c)*(g-e)
    C00 = Z0 + Z1 - Z2 + Z5
    C01 = Z4 + Z1 - Z2 - Z6
    C10 = Z3 + Z1 - Z2 + Z5
    C11 = Z3 + Z4 + Z6
    C_test = np.array([[C00,C01],[C10,C11]])
    C_actual = A @ B
    if not np.allclose(C_test, C_actual):
        # BCLR doesn't work for non-commutative 2x2 matrices as expected.
        # Fall back to schoolbook.
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
        return {"U": U, "V": V, "W": W}
    
    # If BCLR works, construct the rank-47 decomposition.
    # ... (complex construction)
    
    # For now, return schoolbook.
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
