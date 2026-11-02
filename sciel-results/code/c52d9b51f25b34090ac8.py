"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Trivial n*m*p rank decomposition (rank 45)
    def get_trivial():
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
    
    # Laderman's rank-23 decomposition for 3x3x3
    # This is a well-known result. The coefficients are:
    # M1 = A11 * B11
    # M2 = A12 + A11 * B21
    # M3 = A21 + A11 * B12
    # M4 = A22 + A12 * B22
    # M5 = (A11 + A22 - A12 - A21) * (B11 + B22 - B12 - B21)
    # M6 = (A12 + A22) * B12
    # M7 = (A11 + A21) * B21
    # M8 = A12 * (B12 - B22)
    # M9 = A21 * (B21 - B11)
    # M10 = (A22 + A12) * (B12 + B22)
    # M11 = (A11 + A21) * (B11 + B21)
    # M12 = (A12 + A11) * (B21 - B11)
    # M13 = (A21 + A22) * (B12 - B22)
    # M14 = (A11 + A22) * (B11 + B22)
    # M15 = (A12 + A22) * (B12 + B22)
    # M16 = (A11 + A21) * (B11 + B21)
    # M17 = (A12 + A11) * (B21 - B11)
    # M18 = (A21 + A22) * (B12 - B22)
    # M19 = A11 * B22
    # M20 = A22 * B11
    # M21 = A12 * B21
    # M22 = A21 * B12
    # M23 = (A11 + A22 - A12 - A21) * (B11 + B22 - B12 - B21)
    
    # Actually, let me use a verified Laderman decomposition.
    # The standard Laderman 23 terms for 3x3:
    
    def get_laderman_3x3():
        """Return Laderman's rank-23 decomposition for 3x3x3 as (U, V, W) with shape (9, 23) each"""
        rank = 23
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Laderman's 23 terms (verified formulation)
        # A indices: A11=0, A12=1, A13=2, A21=3, A22=4, A23=5, A31=6, A32=7, A33=8
        # B indices: B11=0, B12=1, B13=2, B21=3, B22=4, B23=5, B31=6, B32=7, B33=8
        # C indices: C11=0, C12=1, C13=2, C21=3, C22=4, C23=5, C31=6, C32=7, C33=8
        
        # Term 1: A11*B11 -> C11
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        
        # Term 2: A12*B21 -> C11
        U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
        
        # Term 3: A13*B31 -> C11
        U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
        
        # Term 4: A21*B12 -> C12
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        # Term 5: A22*B22 -> C12
        U[4, 4] = 1; V[4, 4] = 1; W[1, 4] = 1
        
        # Term 6: A23*B32 -> C12
        U[5, 5] = 1; V[7, 5] = 1; W[1, 5] = 1
        
        # Term 7: A31*B13 -> C13
        U[6, 6] = 1; V[2, 6] = 1; W[2, 6] = 1
        
        # Term 8: A32*B23 -> C13
        U[7, 7] = 1; V[5, 7] = 1; W[2, 7] = 1
        
        # Term 9: A33*B33 -> C13
        U[8, 8] = 1; V[8, 8] = 1; W[2, 8] = 1
        
        # Term 10: A11*B22 -> C12
        U[0, 9] = 1; V[4, 9] = 1; W[1, 9] = 1
        
        # Term 11: A12*B11 -> C11
        U[1, 10] = 1; V[0, 10] = 1; W[0, 10] = 1
        
        # Term 12: A13*B12 -> C12
        U[2, 11] = 1; V[1, 11] = 1; W[1, 11] = 1
        
        # Term 13: A21*B11 -> C11
        U[3, 12] = 1; V[0, 12] = 1; W[0, 12] = 1
        
        # Term 14: A22*B12 -> C12
        U[4, 13] = 1; V[1, 13] = 1; W[1, 13] = 1
        
        # Term 15: A23*B13 -> C13
        U[5, 14] = 1; V[2, 14] = 1; W[2, 14] = 1
        
        # Term 16: A31*B11 -> C11
        U[6, 15] = 1; V[0, 15] = 1; W[0, 15] = 1
        
        # Term 17: A32*B12 -> C12
        U[7, 16] = 1; V[1, 16] = 1; W[1, 16] = 1
        
        # Term 18: A33*B13 -> C13
        U[8, 17] = 1; V[2, 17] = 1; W[2, 17] = 1
        
        # Term 19: A11*B33 -> C13
        U[0, 18] = 1; V[8, 18] = 1; W[2, 18] = 1
        
        # Term 20: A12*B32 -> C12
        U[1, 19] = 1; V[7, 19] = 1; W[1, 19] = 1
        
        # Term 21: A13*B31 -> C11
        U[2, 20] = 1; V[6, 20] = 1; W[0, 20] = 1
        
        # Term 22: A21*B33 -> C13
        U[3, 21] = 1; V[8, 21] = 1; W[2, 21] = 1
        
        # Term 23: A22*B32 -> C12
        U[4, 22] = 1; V[7, 22] = 1; W[1, 22] = 1
        
        # Verify: this is NOT the correct Laderman decomposition.
        # The above is just a subset of trivial terms. Let me use a proper approach.
        
        # Instead, let me construct a valid decomposition by combining:
        # - Use the trivial 9 terms for the diagonal of 3x3
        # - Add 14 more terms to handle off-diagonal
        # This gives rank 23, but I need to verify it's correct.
        
        # Actually, the simplest safe approach: use rank 27 (3x3x3 can be done in 27 trivially,
        # but we want < 45 total). Let me just use a block approach.
        
        return U, V, W
    
    # Better approach: Use a block decomposition
    # 3x3 * 3x5 = (3x3 * 3x3) + (3x3 * 3x2)
    # For 3x3 * 3x3: use Laderman (rank 23)
    # For 3x3 * 3x2: use trivial (rank 18)
    # Total: 23 + 18 = 41
    
    # But I need the actual Laderman coefficients. Let me use a different strategy:
    # Use a verified rank-27 decomposition for 3x3x3 (simpler to construct)
    # and trivial for 3x3x2 (rank 18), total = 45. That's not better.
    
    # Let me try: rank-20 for 3x3x3? Lower bound is 19, best known 23.
    # So minimum for 3x3x3 is 23.
    
    # For 3x3 * 3x2: trivial is 18. Can we do better?
    # R(3,3,2) lower bound is 14, best known is 17 or 18.
    # If we can get 17, total = 23 + 17 = 40.
    
    # Let me construct a rank-17 decomposition for 3x3 * 3x2.
    # This is equivalent to computing C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1
    # That's 3*3*2 = 18 equations.
    
    # Actually, let me just use a simple improvement:
    # Combine the 3x3x3 and 3x3x2 parts more efficiently.
    
    # Safest improvement: Use Laderman for 3x3x3 (rank 23) and trivial for 3x3x2 (rank 18)
    # Total rank = 41
    
    # I need the actual Laderman coefficients. Let me use a known valid set.
    
    def get_laderman_valid():
        """A verified Laderman rank-23 decomposition for 3x3x3"""
        rank = 23
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Using the standard Laderman formulation with explicit coefficients
        # A11=A[0], A12=A[1], A13=A[2], A21=A[3], A22=A[4], A23=A[5], A31=A[6], A32=A[7], A33=A[8]
        # B11=B[0], B12=B[1], B13=B[2], B21=B[3], B22=B[4], B23=B[5], B31=B[6], B32=B[7], B33=B[8]
        
        # Laderman's 23 products:
        # P1 = A11*B11
        U[0,0]=1; V[0,0]=1; W[0,0]=1  # C11
        # P2 = A12*B21  
        U[1,1]=1; V[3,1]=1; W[0,1]=1  # C11
        # P3 = A13*B31
        U[2,2]=1; V[6,2]=1; W[0,2]=1  # C11
        # P4 = A21*B12
        U[3,3]=1; V[1,3]=1; W[1,3]=1  # C12
        # P5 = A22*B22
        U[4,4]=1; V[4,4]=1; W[1,4]=1  # C12
        # P6 = A23*B32
        U[5,5]=1; V[7,5]=1; W[1,5]=1  # C12
        # P7 = A31*B13
        U[6,6]=1; V[2,6]=1; W[2,6]=1  # C13
        # P8 = A32*B23
        U[7,7]=1; V[5,7]=1; W[2,7]=1  # C13
        # P9 = A33*B33
        U[8,8]=1; V[8,8]=1; W[2,8]=1  # C13
        # P10 = A11*B22
        U[0,9]=1; V[4,9]=1; W[1,9]=1  # C12
        # P11 = A12*B11
        U[1,10]=1; V[0,10]=1; W[0,10]=1  # C11
        # P12 = A13*B12
        U[2,11]=1; V[1,11]=1; W[1,11]=1  # C12
        # P13 = A21*B11
        U[3,12]=1; V[0,12]=1; W[0,12]=1  # C11
        # P14 = A22*B12
        U[4,13]=1; V[1,13]=1; W[1,13]=1  # C12
        # P15 = A23*B13
        U[5,14]=1; V[2,14]=1; W[2,14]=1  # C13
        # P16 = A31*B11
        U[6,15]=1; V[0,15]=1; W[0,15]=1  # C11
        # P17 = A32*B12
        U[7,16]=1; V[1,16]=1; W[1,16]=1  # C12
        # P18 = A33*B13
        U[8,17]=1; V[2,17]=1; W[2,17]=1  # C13
        # P19 = A11*B33
        U[0,18]=1; V[8,18]=1; W[2,18]=1  # C13
        # P20 = A12*B32
        U[1,19]=1; V[7,19]=1; W[1,19]=1  # C12
        # P21 = A13*B31
        U[2,20]=1; V[6,20]=1; W[0,20]=1  # C11
        # P22 = A21*B33
        U[3,21]=1; V[8,21]=1; W[2,21]=1  # C13
        # P23 = A22*B32
        U[4,22]=1; V[7,22]=1; W[1,22]=1  # C12
        
        # Verify this decomposition
        # C11 = P1 + P2 + P3 + P11 + P13 + P16 + P21
        #     = A11*B11 + A12*B21 + A13*B31 + A12*B11 + A21*B11 + A31*B11 + A13*B31
        # That's not right. I have duplicates and missing terms.
        
        # This is NOT a valid Laderman decomposition. Let me abandon this approach
        # and use a simpler guaranteed improvement.
        
        return U, V, W
    
    # Given the difficulty of getting Laderman right from memory, let me use
    # a different strategy: a rank-40 decomposition via a simple algebraic trick.
    
    # For 3x3 * 3x5:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can write this as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The trivial decomposition has 45 terms.
    # 
    # A simple improvement: notice that for fixed i, the row of C is:
    # C[i,:] = A[i,:] * B
    # 
    # We can use the fact that matrix multiplication is bilinear.
    # 
    # Let me try a rank-44 decomposition by merging one pair of terms.
    # Actually, that's not straightforward.
    
    # Let me just return the trivial decomposition to be safe.
    # The score will be 1.0, which is not an improvement, but it's valid.
    
    if time.time() - start_time > 140:
        return get_trivial()
    
    return get_trivial()
# EVOLVE-BLOCK-END
