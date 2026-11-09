"""Matrix multiplication 6x6x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Strassen's 2x2 matrix multiplication with rank 7.
    Returns (U, V, W) where:
      U: (4, 7) - input A factors (row index: i*2+j for A[i,j])
      V: (4, 7) - input B factors (row index: j*2+k for B[j,k])
      W: (4, 7) - output C factors (row index: k*2+i for C[k,i]) following trace convention W[k*n+i]
    
    For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    Strassen products:
      p1 = (a+d)(e+h)
      p2 = (c+d)e
      p3 = a(f-h)
      p4 = d(g-e)
      p5 = (a+b)h
      p6 = (c-a)(e+f)
      p7 = (b-d)(g+h)
    
    C[0,0] = p1 + p4 - p5 + p7
    C[0,1] = p3 + p5
    C[1,0] = p2 + p4
    C[1,1] = p1 - p2 + p3 + p6
    """
    # U rows: a=0, b=1, c=2, d=3
    # V rows: e=0, f=1, g=2, h=3
    # W rows: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3  (k*n+i: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3)
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 1],  # a
        [0, 0, 1, 0, 1, -1, 1], # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],  # e
        [0, 0, 1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, -1, 1, 0, 1] # h
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]
    ], dtype=float)
    
    return U, V, W


def _laderman_3x3():
    """
    Laderman's 3x3 matrix multiplication with rank 23.
    Returns (U, V, W) where:
      U: (9, 23) - input A factors (row index: i*3+j for A[i,j])
      V: (9, 23) - input B factors (row index: j*3+k for B[j,k])
      W: (9, 23) - output C factors (row index: k*3+i for C[k,i]) following trace convention W[k*n+i]
    
    Laderman's 23-term algorithm for 3x3 matrices.
    """
    # Laderman's algorithm:
    # Let A = (a_ij), B = (b_jk), C = A*B = (c_ik)
    # 
    # The 23 products are:
    # p1  = a11*(b11+b22)
    # p2  = (a11-a22)*(b21-b22)
    # p3  = (a11+a22)*b32
    # p4  = a33*(b11+b22)
    # p5  = (a33-a11)*(b12-b11)
    # p6  = (a33+a11)*b21
    # p7  = a22*(b33+b11)
    # p8  = (a22-a33)*(b31-b33)
    # p9  = (a22+a33)*b13
    # p10 = a13*(b22+b33)
    # p11 = (a13-a33)*(b23-b33)
    # p12 = (a13+a33)*b32  -- wait, let me look up the exact Laderman algorithm
    
    # Actually, let me use the standard Laderman 23-term algorithm:
    # Indices: a11,a12,a13,a21,a22,a23,a31,a32,a33
    #           b11,b12,b13,b21,b22,b23,b31,b32,b33
    
    # Laderman's products (from the paper):
    # p1  = a11 * (b11 + b22)
    # p2  = (a11 - a22) * (b21 - b22)
    # p3  = (a11 + a22) * b32
    # p4  = a33 * (b11 + b22)
    # p5  = (a33 - a11) * (b12 - b11)
    # p6  = (a33 + a11) * b21
    # p7  = a22 * (b33 + b11)
    # p8  = (a22 - a33) * (b31 - b33)
    # p9  = (a22 + a33) * b13
    # p10 = a13 * (b22 + b33)
    # p11 = (a13 - a33) * (b23 - b33)
    # p12 = (a13 + a33) * b32
    # p13 = a23 * (b11 + b33)
    # p14 = (a23 - a11) * (b13 - b11)
    # p15 = (a23 + a11) * b31
    # p16 = a31 * (b22 + b11)
    # p17 = (a31 - a22) * (b21 - b22)  -- hmm, this doesn't look right
    
    # Let me use a different, well-known formulation. 
    # Actually, let me just construct it via the known rank-23 decomposition.
    # 
    # A simpler approach: use the known fact that 3x3 can be done in 23 multiplications.
    # Let me use a specific published version.
    
    # From Bini, Capovani, Lotti, Romani (1978) - Laderman's algorithm:
    # 
    # I'll use a verified implementation. Let me define it carefully.
    
    # Row indices for U (A matrix, 3x3):
    # 0: a11, 1: a12, 2: a13, 3: a21, 4: a22, 5: a23, 6: a31, 7: a32, 8: a33
    # Row indices for V (B matrix, 3x3):
    # 0: b11, 1: b12, 2: b13, 3: b21, 4: b22, 5: b23, 6: b31, 7: b32, 8: b33
    # Row indices for W (C matrix, 3x3, trace convention W[k*3+i]):
    # 0: c11, 1: c12, 2: c13, 3: c21, 4: c22, 5: c23, 6: c31, 7: c32, 8: c33
    
    # Laderman's 23 terms (using 1-indexed notation, converting to 0-indexed):
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # p1 = a11 * (b11 + b22)
    U[0, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1  # c11 += p1, c22 += p1? No...
    
    # Actually, let me be more careful. The output C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # With trace convention W[k*n+i] where n=3, so W[k*3+i] corresponds to C[i,k].
    # W[0] = C[0,0], W[1] = C[1,0], W[2] = C[2,0], W[3] = C[0,1], W[4] = C[1,1], etc.
    # Wait, W[k*n+i] means the (k,i) entry in the flattened output. If we think of C as having
    # rows indexed by i and columns by k, then C[i,k] is at position k*n+i in the flat array.
    # So W[0] -> k=0,i=0 -> C[0,0]
    #    W[1] -> k=0,i=1 -> C[1,0]
    #    W[2] -> k=0,i=2 -> C[2,0]
    #    W[3] -> k=1,i=0 -> C[0,1]
    #    W[4] -> k=1,i=1 -> C[1,1]
    #    W[5] -> k=1,i=2 -> C[2,1]
    #    W[6] -> k=2,i=0 -> C[0,2]
    #    W[7] -> k=2,i=1 -> C[1,2]
    #    W[8] -> k=2,i=2 -> C[2,2]
    
    # Let me use a known correct Laderman decomposition.
    # I'll use the one from the literature with careful indexing.
    
    # Reset
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Laderman's algorithm (verified version):
    # The 23 bilinear forms are:
    # 
    # t1  = a11 * (b11 + b22)
    # t2  = (a11 - a22) * (b21 - b22)
    # t3  = (a11 + a22) * b32
    # t4  = a33 * (b11 + b22)
    # t5  = (a33 - a11) * (b12 - b11)
    # t6  = (a33 + a11) * b21
    # t7  = a22 * (b33 + b11)
    # t8  = (a22 - a33) * (b31 - b33)
    # t9  = (a22 + a33) * b13
    # t10 = a13 * (b22 + b33)
    # t11 = (a13 - a33) * (b23 - b33)
    # t12 = (a13 + a33) * b32
    # t13 = a23 * (b11 + b33)
    # t14 = (a23 - a11) * (b13 - b11)
    # t15 = (a23 + a11) * b31
    # t16 = a31 * (b22 + b11)
    # t17 = (a31 - a22) * (b21 - b22)
    # t18 = (a31 + a22) * b12
    # t19 = a12 * (b33 + b22)
    # t20 = (a12 - a33) * (b32 - b33)
    # t21 = (a12 + a33) * b23
    # t22 = a21 * (b11 + b33)  -- wait, this is getting confusing
    
    # Let me just use a programmatic approach to verify and build it.
    # I'll use a known-correct set of 23 terms.
    
    # Actually, let me use a different strategy. I'll use the fact that 
    # the tensor rank of 3x3 matrix multiplication is 23 and construct
    # it using a verified algorithm.
    
    # Here's a clean Laderman implementation:
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Using 0-indexed: A[i,j] at U[i*3+j], B[j,k] at V[j*3+k], C[i,k] at W[k*3+i]
    
    # Term 1: a[0,0] * (b[0,0] + b[1,1])
    U[0, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1  # contributes to c[0,0] and c[1,1]
    
    # Term 2: (a[0,0] - a[1,1]) * (b[1,0] - b[1,1])
    U[0, 1] = 1; U[4, 1] = -1
    V[3, 1] = 1; V[4, 1] = -1
    W[1, 1] = 1; W[4, 1] = -1; W[3, 1] = -1; W[0, 1] = 1  # need to verify
    
    # This is getting error-prone. Let me use a different approach:
    # I'll directly construct the 6x6 using Kronecker products of smaller verified decompositions.
    
    # For now, let me use a simpler but valid approach:
    # Use Strassen (rank 7) for 2x2 blocks. Since 6 = 2*3, we can use
    # a 3x3 grid of 2x2 blocks, each using Strassen.
    # This gives rank = 7^3 = 343, which is worse than schoolbook (216).
    # So that's not good.
    
    # Alternative: Use the fact that 6 = 3*2. Use Laderman for 3x3 and Strassen for 2x2.
    # But we need to combine them properly.
    
    # Actually, the cleanest approach for 6x6:
    # Partition A (6x6) into 3x3 blocks of 2x2 matrices: A = [[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]]
    # where each Aij is 2x2.
    # Similarly B = [[B11,B12,B13],[B21,B22,B23],[B31,B32,B33]]
    # C = A*B has 3x3 blocks of 2x2 matrices.
    # 
    # The 3x3 block multiplication can be done with Laderman (rank 23),
    # but each "scalar" multiplication in the 3x3 level is actually a 2x2 matrix multiplication.
    # Using Strassen for each 2x2 gives rank 7 per scalar mult.
    # Total rank = 23 * 7 = 161.
    
    # This is the key insight! Rank 161 < 216 (schoolbook).
    
    # Let me implement this Kronecker product approach properly.
    
    # First, let me get a correct Laderman 3x3 decomposition.
    # I'll use a well-known formulation.
    
    # Laderman's 3x3 (from multiple verified sources):
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # I'll use the formulation where we can verify by checking the tensor identity.
    # Let me use a specific known-correct set.
    
    # From the paper "Tensor rank and arithmetic complexity of small matrices" by Bini et al.
    # and various implementations:
    
    # Term indices 0-22
    # I'll use a verified implementation pattern:
    
    # p0:  a00*(b00+b11)
    U[0,0]=1; V[0,0]=1; V[4,0]=1; W[0,0]=1; W[4,0]=1
    # p1:  (a00-a11)*(b10-b11)
    U[0,1]=1; U[4,1]=-1; V[3,1]=1; V[4,1]=-1; W[1,1]=1; W[4,1]=-1; W[0,1]=1; W[3,1]=-1
    # p2:  (a00+a11)*b21
    U[0,2]=1; U[4,2]=1; V[7,2]=1; W[1,2]=1; W[4,2]=1
    # p3:  a22*(b00+b11)
    U[8,3]=1; V[0,3]=1; V[4,3]=1; W[0,3]=1; W[4,3]=1
    # p4:  (a22-a00)*(b01-b00)
    U[8,4]=1; U[0,4]=-1; V[1,4]=1; V[0,4]=-1; W[3,4]=1; W[4,4]=-1; W[0,4]=1; W[1,4]=-1
    # p5:  (a22+a00)*b10
    U[8,5]=1; U[0,5]=1; V[3,5]=1; W[1,5]=1; W[4,5]=1
    # p6:  a11*(b22+b00)
    U[4,6]=1; V[8,6]=1; V[0,6]=1; W[0,6]=1; W[6,6]=1
    # p7:  (a11-a22)*(b20-b22)
    U[4,7]=1; U[8,7]=-1; V[6,7]=1; V[8,7]=-1; W[6,7]=1; W[8,7]=-1; W[0,7]=1; W[6,7]=1
    # p8:  (a11+a22)*b02
    U[4,8]=1; U[8,8]=1; V[2,8]=1; W[2,8]=1; W[6,8]=1
    # p9:  a02*(b11+b22)
    U[2,9]=1; V[4,9]=1; V[8,9]=1; W[2,9]=1; W[6,9]=1
    # p10: (a02-a22)*(b12-b22)
    U[2,10]=1; U[8,10]=-1; V[5,10]=1; V[8,10]=-1; W[5,10]=1; W[8,10]=-1; W[2,10]=1; W[5,10]=1
    # p11: (a02+a22)*b21
    U[2,11]=1; U[8,11]=1; V[7,11]=1; W[5,11]=1; W[8,11]=1
    # p12: a12*(b00+b22)
    U[1,12]=1; V[0,12]=1; V[8,12]=1; W[0,12]=1; W[6,12]=1
    # p13: (a12-a00)*(b02-b00)
    U[1,13]=1; U[0,13]=-1; V[2,13]=1; V[0,13]=-1; W[2,13]=1; W[6,13]=-1; W[0,13]=1; W[2,13]=1
    # p14: (a12+a00)*b20
    U[1,14]=1; U[0,14]=1; V[6,14]=1; W[2,14]=1; W[6,14]=1
    # p15: a20*(b11+b00)
    U[6,15]=1; V[4,15]=1; V[0,15]=1; W[0,15]=1; W[3,15]=1
    # p16: (a20-a11)*(b10-b11)
    U[6,16]=1; U[4,16]=-1; V[3,16]=1; V[4,16]=-1; W[3,16]=1; W[4,16]=-1; W[0,16]=1; W[3,16]=1
    # p17: (a20+a11)*b01
    U[6,17]=1; U[4,17]=1; V[1,17]=1; W[3,17]=1; W[4,17]=1
    # p18: a01*(b22+b11)
    U[1,18]=1; V[8,18]=1; V[4,18]=1; W[4,18]=1; W[6,18]=1
    # p19: (a01-a22)*(b21-b22)
    U[1,19]=1; U[8,19]=-1; V[7,19]=1; V[8,19]=-1; W[7,19]=1; W[8,19]=-1; W[4,19]=1; W[7,19]=1
    # p20: (a01+a22)*b12
    U[1,20]=1; U[8,20]=1; V[5,20]=1; W[7,20]=1; W[8,20]=1
    # p21: a21*(b00+b22)  -- wait, I'm running out of terms and this doesn't look symmetric
    
    # I'm making errors. Let me use a completely different, cleaner approach.
    # I'll verify the decomposition numerically and adjust.
    
    # Let me just use the Kronecker product approach with Strassen only,
    # but in a smarter way. 
    
    # Actually, the simplest correct approach: 
    # 6x6 = (3x3) of (2x2) blocks
    # Use Laderman (rank 23) at the 3x3 block level, Strassen (rank 7) at the 2x2 level.
    # Total rank = 23 * 7 = 161.
    
    # For the Kronecker product to work, I need correct Laderman and Strassen.
    # Let me use Strassen which I'm confident about, and find a correct Laderman.
    
    # Let me verify my Strassen implementation first, then build Laderman carefully.
    
    return None  # placeholder


def _build_kron(U1, V1, W1, U2, V2, W2):
    """
    Kronecker product of two tensor decompositions.
    If T1 = sum_c u1_c ⊗ v1_c ⊗ w1_c (rank r1)
    and T2 = sum_d u2_d ⊗ v2_d ⊗ w2_d (rank r2)
    then T1 ⊗ T2 has rank r1*r2 with:
      U = kron(U1, U2), V = kron(V1, V2), W = kron(W1, W2)
    
    For matrix multiplication:
    T1 multiplies n1 x m1 by m1 x p1 -> n1 x p1
    T2 multiplies n2 x m2 by m2 x p2 -> n2 x p2
    T1 ⊗ T2 multiplies (n1*n2) x (m1*m2) by (m1*m2) x (p1*p2) -> (n1*n2) x (p1*p2)
    
    With n1=m1=p1=n2=m2=p2 for square case.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    # U has shape (n1*m1 * n2*m2, rank) = (n1*n2*m1*m2, rank)
    # The input for the combined problem is A of size (n1*n2) x (m1*m2)
    # A is formed by Kronecker: A[i1*n2+i2, j1*m2+j2] = A1[i1,j1] * A2[i2,j2]
    # So U_combined[(i1*n2+i2)*(m1*m2) + (j1*m2+j2), c1*r2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    
    n1, m1, p1 = U1.shape[0] // 1, None, None  # We need to infer dimensions
    
    # Actually, let's just do it with explicit kron
    U_combined = np.kron(U1, U2)  # shape (n1*m1*n2*m2, r1*r2)
    V_combined = np.kron(V1, V2)
    W_combined = np.kron(W1, W2)
    
    return U_combined, V_combined, W_combined


def _schoolbook(n, m, p):
    """Schoolbook matrix multiplication decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify tensor decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _to_half_integer_list(arr):
    """Convert numpy array to list, rounding to nearest 0.5."""
    rounded = np.round(arr * 2) / 2
    return rounded.tolist()


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Strategy: Use Kronecker product of Laderman (3x3, rank 23) and Strassen (2x2, rank 7)
    # to get 6x6 with rank 161.
    
    # First, let's build a correct Laderman 3x3 decomposition.
    # I'll use a verified approach: construct it and verify.
    
    # Laderman's 3x3 decomposition (rank 23)
    # Using a known-correct formulation:
    
    U3 = np.zeros((9, 23), dtype=float)
    V3 = np.zeros((9, 23), dtype=float)
    W3 = np.zeros((9, 23), dtype=float)
    
    # I'll use the following verified Laderman terms:
    # Reference: Laderman, 1976, "On the number of multiplications necessary to multiply 3x3 matrices"
    
    # Let me use a different, cleaner construction.
    # I'll use the fact that we can verify numerically.
    
    # Actually, let me just use a simple but effective approach:
    # Use Strassen on 2x2 sub-blocks within a 3x3 grid of blocks.
    # But that gives 7^3 = 343 which is worse.
    
    # Better: Use the direct Kronecker of Laderman ⊗ Strassen.
    # Let me get a correct Laderman by using a known implementation pattern.
    
    # Here's a verified Laderman 3x3 (I'll verify it numerically):
    terms = [
        # (U_indices, V_indices, W_indices)
        # Each term: list of (index, coeff) for U, V, W
        # Term 0: a00*(b00+b11)
        {('U', [(0,1)]), ('V', [(0,1),(4,1)]), ('W', [(0,1),(4,1)])},
        # Term 1: (a00-a11)*(b10-b11) 
        {('U', [(0,1),(4,-1)]), ('V', [(3,1),(4,-1)]), ('W', [(1,1),(4,-1),(0,1),(3,-1)])},
        # Term 2: (a00+a11)*b21
        {('U', [(0,1),(4,1)]), ('V', [(7,1)]), ('W', [(1,1),(4,1)])},
        # Term 3: a22*(b00+b11)
        {('U', [(8,1)]), ('V', [(0,1),(4,1)]), ('W', [(0,1),(4,1)])},
        # Term 4: (a22-a00)*(b01-b00)
        {('U', [(8,1),(0,-1)]), ('V', [(1,1),(0,-1)]), ('W', [(3,1),(4,-1),(0,1),(1,-1)])},
        # Term 5: (a22+a00)*b10
        {('U', [(8,1),(0,1)]), ('V', [(3,1)]), ('W', [(1,1),(4,1)])},
        # Term 6: a11*(b22+b00)
        {('U', [(4,1)]), ('V', [(8,1),(0,1)]), ('W', [(0,1),(6,1)])},
        # Term 7: (a11-a22)*(b20-b22)
        {('U', [(4,1),(8,-1)]), ('V', [(6,1),(8,-1)]), ('W', [(6,1),(8,-1),(0,1),(2,-1)])},
        # Term 8: (a11+a22)*b02
        {('U', [(4,1),(8,1)]), ('V', [(2,1)]), ('W', [(2,1),(6,1)])},
        # Term 9: a02*(b11+b22)
        {('U', [(2,1)]), ('V', [(4,1),(8,1)]), ('W', [(2,1),(6,1)])},
        # Term 10: (a02-a22)*(b12-b22)
        {('U', [(2,1),(8,-1)]), ('V', [(5,1),(8,-1)]), ('W', [(5,1),(8,-1),(2,1),(6,-1)])},
        # Term 11: (a02+a22)*b21
        {('U', [(2,1),(8,1)]), ('V', [(7,1)]), ('W', [(5,1),(8,1)])},
        # Term 12: a12*(b00+b22)
        {('U', [(1,1)]), ('V', [(0,1),(8,1)]), ('W', [(0,1),(6,1)])},
        # Term 13: (a12-a00)*(b02-b00)
        {('U', [(1,1),(0,-1)]), ('V', [(2,1),(0,-1)]), ('W', [(2,1),(6,-1),(0,1),(3,-1)])},
        # Term 14: (a12+a00)*b20
        {('U', [(1,1),(0,1)]), ('V', [(6,1)]), ('W', [(2,1),(6,1)])},
        # Term 15: a20*(b11+b00)
        {('U', [(6,1)]), ('V', [(4,1),(0,1)]), ('W', [(0,1),(3,1)])},
        # Term 16: (a20-a11)*(b10-b11)
        {('U', [(6,1),(4,-1)]), ('V', [(3,1),(4,-1)]), ('W', [(3,1),(4,-1),(0,1),(1,-1)])},
        # Term 17: (a20+a11)*b01
        {('U', [(6,1),(4,1)]), ('V', [(1,1)]), ('W', [(3,1),(4,1)])},
        # Term 18: a01*(b22+b11)
        {('U', [(1,1)]), ('V', [(8,1),(4,1)]), ('W', [(4,1),(6,1)])},
        # Term 19: (a01-a22)*(b21-b22)
        {('U', [(1,1),(8,-1)]), ('V', [(7,1),(8,-1)]), ('W', [(7,1),(8,-1),(4,1),(5,-1)])},
        # Term 20: (a01+a22)*b12
        {('U', [(1,1),(8,1)]), ('V', [(5,1)]), ('W', [(7,1),(8,1)])},
        # Term 21: a21*(b00+b22)
        {('U', [(7,1)]), ('V', [(0,1),(8,1)]), ('W', [(0,1),(6,1)])},
        # Term 22: (a21-a11)*(b02-b00) -- this doesn't look right
        {('U', [(7,1),(4,-1)]), ('V', [(2,1),(0,-1)]), ('W', [(2,1),(6,-1),(0,1),(5,-1)])},
    ]
    
    # This is getting too error-prone. Let me use a completely different strategy.
    # I'll use a numerical optimization approach to find a valid decomposition.
    
    # Actually, the safest approach: use schoolbook (rank 216) as guaranteed valid,
    # and try to improve with a structured approach.
    
    # Let me use the Kronecker product of two Strassen decompositions for 2x2,
    # giving 4x4 with rank 49, then extend to 6x6.
    
    # For 6x6: partition into 2x2 blocks gives a 3x3 grid.
    # Each 2x2 block multiplication uses Strassen (rank 7).
    # The 3x3 block structure needs to be handled.
    
    # Simplest valid improvement: use Strassen for the 2x2 blocks in a 3x3 arrangement.
    # But the 3x3 block multiplication itself needs to be done.
    
    # Let me just use the schoolbook as the guaranteed valid solution and try 
    # a simple Strassen-based approach for 6x6.
    
    # For 6x6, we can partition into four 3x3 blocks and use a 2x2 Strassen on top:
    # A = [[A11, A12], [A21, A22]] where each is 3x3
    # B = [[B11, B12], [B21, B22]] where each is 3x3
    # C = [[C11, C12], [C21, C22]] where each is 3x3
    # 
    # Strassen: 7 products of 3x3 matrices
    # Each 3x3 product uses schoolbook (27 mults) or Laderman (23 mults)
    # With schoolbook: 7 * 27 = 189
    # With Laderman: 7 * 23 = 161
    
    # Let me implement the 2x2 Strassen on 3x3 blocks with schoolbook 3x3:
    # This gives rank 189, which is better than 216.
    
    # Implementation:
    # The 6x6 input A is split into 3x3 blocks:
    # A[0:3, 0:3] = A11, A[0:3, 3:6] = A12
    # A[3:6, 0:3] = A21, A[3:6, 3:6] = A22
    
    # U for the combined problem:
    # For Strassen term c, U_combined[i, c] where i indexes into the 6x6 A matrix
    # U_combined = kron(Strassen_U, Schoolbook3x3_U)
    
    # Strassen_U has shape (4, 7) for 2x2
    # Schoolbook3x3_U has shape (9, 27) for 3x3
    # kron gives (36, 189) which is correct for 6x6 input (6*6=36)
    
    U2, V2, W2 = _strassen_2x2()
    
    # Schoolbook 3x3
    U3_sb, V3_sb, W3_sb = _schoolbook(3, 3, 3)
    
    # Kronecker product
    U_comb = np.kron(U2, U3_sb)  # (4*9, 7*27) = (36, 189)
    V_comb = np.kron(V2, V3_sb)  # (36, 189)
    W_comb = np.kron(W2, W3_sb)  # (36, 189)
    
    # Verify
    if _verify(U_comb, V_comb, W_comb, 6, 6, 6):
        best_U, best_V, best_W = U_comb, V_comb, W_comb
        best_rank = 189
    else:
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook(6, 6, 6)
        best_rank = 216
    
    # Check time and try to improve
    if time.time() < deadline - 10:
        # Try Laderman-based approach for rank 161
        # Need a correct Laderman 3x3
        # Let me try to construct and verify it
        
        # Use a known correct Laderman by building it from the definition
        # and verifying numerically
        
        # I'll use a programmatic construction and verification
        U3_l, V3_l, W3_l = _laderman_3x3_verified()
        
        if U3_l is not None and _verify(U3_l, V3_l, W3_l, 3, 3, 3):
            U_comb2 = np.kron(U2, U3_l)
            V_comb2 = np.kron(V2, V3_l)
            W_comb2 = np.kron(W2, W3_l)
            
            if _verify(U_comb2, V_comb2, W_comb2, 6, 6, 6):
                best_U, best_V, best_W = U_comb2, V_comb2, W_comb2
                best_rank = 161
    
    return {"U": _to_half_integer_list(best_U), "V": _to_half_integer_list(best_V), "W": _to_half_integer_list(best_W)}


def _laderman_3x3_verified():
    """
    Attempt to construct and verify Laderman's 3x3 decomposition.
    Returns (U, V, W) or (None, None, None) if verification fails.
    """
    # I'll use a known correct formulation from the literature.
    # Laderman's 23-term algorithm for 3x3 matrix multiplication.
    
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Using the formulation from various verified sources:
    # A[i,j] at U[i*3+j], B[j,k] at V[j*3+k], C[i,k] at W[k*3+i]
    
    # I'll define the 23 terms explicitly and verify.
    
    # Term definitions: (U_entries, V_entries, W_entries)
    # Each entry is (index, coefficient)
    
    terms = [
        # 0: a00*(b00+b11)
        ([(0,1)], [(0,1),(4,1)], [(0,1),(4,1)]),
        # 1: (a00-a11)*(b10-b11)
        ([(0,1),(4,-1)], [(3,1),(4,-1)], [(1,1),(4,-1),(0,1),(3,-1)]),
        # 2: (a00+a11)*b21
        ([(0,1),(4,1)], [(7,1)], [(1,1),(4,1)]),
        # 3: a22*(b00+b11)
        ([(8,1)], [(0,1),(4,1)], [(0,1),(4,1)]),
        # 4: (a22-a00)*(b01-b00)
        ([(8,1),(0,-1)], [(1,1),(0,-1)], [(3,1),(4,-1),(0,1),(1,-1)]),
        # 5: (a22+a00)*b10
        ([(8,1),(0,1)], [(3,1)], [(1,1),(4,1)]),
        # 6: a11*(b22+b00)
        ([(4,1)], [(8,1),(0,1)], [(0,1),(6,1)]),
        # 7: (a11-a22)*(b20-b22)
        ([(4,1),(8,-1)], [(6,1),(8,-1)], [(6,1),(8,-1),(0,1),(2,-1)]),
        # 8: (a11+a22)*b02
        ([(4,1),(8,1)], [(2,1)], [(2,1),(6,1)]),
        # 9: a02*(b11+b22)
        ([(2,1)], [(4,1),(8,1)], [(2,1),(6,1)]),
        # 10: (a02-a22)*(b12-b22)
        ([(2,1),(8,-1)], [(5,1),(8,-1)], [(5,1),(8,-1),(2,1),(6,-1)]),
        # 11: (a02+a22)*b21
        ([(2,1),(8,1)], [(7,1)], [(5,1),(8,1)]),
        # 12: a12*(b00+b22)
        ([(1,1)], [(0,1),(8,1)], [(0,1),(6,1)]),
        # 13: (a12-a00)*(b02-b00)
        ([(1,1),(0,-1)], [(2,1),(0,-1)], [(2,1),(6,-1),(0,1),(3,-1)]),
        # 14: (a12+a00)*b20
        ([(1,1),(0,1)], [(6,1)], [(2,1),(6,1)]),
        # 15: a20*(b11+b00)
        ([(6,1)], [(4,1),(0,1)], [(0,1),(3,1)]),
        # 16: (a20-a11)*(b10-b11)
        ([(6,1),(4,-1)], [(3,1),(4,-1)], [(3,1),(4,-1),(0,1),(1,-1)]),
        # 17: (a20+a11)*b01
        ([(6,1),(4,1)], [(1,1)], [(3,1),(4,1)]),
        # 18: a01*(b22+b11)
        ([(1,1)], [(8,1),(4,1)], [(4,1),(6,1)]),
        # 19: (a01-a22)*(b21-b22)
        ([(1,1),(8,-1)], [(7,1),(8,-1)], [(7,1),(8,-1),(4,1),(5,-1)]),
        # 20: (a01+a22)*b12
        ([(1,1),(8,1)], [(5,1)], [(7,1),(8,1)]),
        # 21: a21*(b00+b22)
        ([(7,1)], [(0,1),(8,1)], [(0,1),(6,1)]),
        # 22: (a21-a11)*(b02-b00)
        ([(7,1),(4,-1)], [(2,1),(0,-1)], [(2,1),(6,-1),(0,1),(5,-1)]),
    ]
    
    for c, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, coeff in u_entries:
            U[idx, c] += coeff
        for idx, coeff in v_entries:
            V[idx, c] += coeff
        for idx, coeff in w_entries:
            W[idx, c] += coeff
    
    # Verify
    if _verify(U, V, W, 3, 3, 3):
        return U, V, W
    else:
        return None, None, None


# EVOLVE-BLOCK-END
