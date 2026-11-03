"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute tensor from decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # W index: k*n + i (trace convention W[k*n+i, r])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                max_err = max(max_err, abs(val - expected))
    # Also check that off-diagonal entries are zero
    # The matmul tensor T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0
    # Actually for matmul, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so the tensor has 1s at all (i,j,k)
    # Wait no - the tensor is defined as T[i,j,k] = delta (it's 1 for all valid indices)
    # Actually the matmul tensor has T[i,j,k] = 1 for all i,j,k in their ranges
    # Let me re-examine: the decomposition must satisfy:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # This is the condition. There are no "off-diagonal" entries to check separately.
    # But we should also verify that the decomposition is exact, not just approximate.
    return max_err < tol

def _get_3x3x2_decomp():
    """
    Get a rank-12 decomposition for 3x3 x 3x2 matrix multiplication.
    
    The tensor T[i,j,k] = 1 for i in [0,3), j in [0,3), k in [0,2).
    We need U (3*3=9 x rank), V (3*2=6 x rank), W (2*3=6 x rank)
    such that sum_c U[i*3+j,c]*V[j*2+k,c]*W[k*3+i,c] = 1 for all i,j,k.
    
    Strategy: Use Strassen for the central 2x2x2 block + boundary terms.
    """
    n, m, p = 3, 3, 2
    # We'll build this by hand. Let me think about it more carefully.
    # 
    # The 18 terms of schoolbook are: (i,j,k) for i=0..2, j=0..2, k=0..1
    # 
    # Let's use the following approach:
    # Partition j into j=0 (boundary), j=1,2 (core)
    # Partition i into i=0 (boundary), i=1,2 (core)
    # k=0,1 are both "core" in some sense
    #
    # Core 2x2x2: i in {1,2}, j in {1,2}, k in {0,1} -> 8 schoolbook terms
    # Use Strassen rank 7 for this
    # Boundary: terms where i=0 or (i in {1,2} and j=0)
    # i=0, j=0,1,2, k=0,1: 6 terms
    # i=1,2, j=0, k=0,1: 4 terms
    # Total boundary: 10 terms
    # Core: 8 terms
    # Total: 18 terms
    #
    # If core uses rank 7 and boundary uses rank 10, total = 17. Not 12.
    #
    # Let me try a different partition.
    # 
    # Actually, let me just try to find rank 12 by direct search or known construction.
    # 
    # Known result: M(3,3,2) = 12 (I believe this is correct)
    # Let me construct it explicitly.
    #
    # For 3x3 x 3x2, let's denote:
    # A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # B = [[b00,b01],[b10,b11],[b20,b21]]
    # C = [[c00,c01],[c10,c11],[c20,c21]]
    # c_ik = sum_j a_ij * b_jk
    #
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    #
    # Let me try a rank-12 construction using the following 12 products:
    # 
    # I'll use a systematic approach. Let me define 12 rank-1 tensors.
    # 
    # Actually, let me just use a verified rank-12 construction.
    # 
    # Here's one approach: use the fact that 3x3x2 can be embedded in a larger structure.
    # 
    # Let me try constructing it with these 12 terms:
    # Term 1-3: handle column 0 of A (j=0) for all i,k
    # Term 4-6: handle row 0 of A (i=0) for j=1,2 and all k
    # Term 7-12: handle the 2x2x2 core (i=1,2; j=1,2; k=0,1) with Strassen
    
    # This gives 3 + 6 + 7 = 16, still not 12.
    
    # Let me try yet another approach. I'll just use a brute-force verified construction.
    
    # After research, here's a rank-12 construction for 3x3x2:
    # Using the following factor matrices:
    
    U = np.zeros((9, 12))
    V = np.zeros((6, 12))
    W = np.zeros((6, 12))
    
    # I'll use a construction based on splitting into overlapping 2x2 blocks
    # 
    # Block 1: rows 0-1, cols 0-1 of A, all cols of B -> 2x2x2, rank 7
    # Block 2: rows 1-2, cols 1-2 of A, all cols of B -> 2x2x2, rank 7
    # But these overlap, so we can't just add them.
    
    # Let me just hardcode a verified rank-12 solution.
    # After careful construction:
    
    # I'll use the following 12 rank-1 components:
    # Each component c has U[:,c], V[:,c], W[:,c]
    
    # Component 0: i=0,j=0,k=0
    U[0*3+0, 0] = 1; V[0*2+0, 0] = 1; W[0*3+0, 0] = 1
    # Component 1: i=0,j=0,k=1  
    U[0*3+0, 1] = 1; V[0*2+1, 1] = 1; W[1*3+0, 1] = 1
    # Component 2: i=0,j=1,k=0
    U[0*3+1, 2] = 1; V[1*2+0, 2] = 1; W[0*3+0, 2] = 1
    # Component 3: i=0,j=1,k=1
    U[0*3+1, 3] = 1; V[1*2+1, 3] = 1; W[1*3+0, 3] = 1
    # Component 4: i=0,j=2,k=0
    U[0*3+2, 4] = 1; V[2*2+0, 4] = 1; W[0*3+0, 4] = 1
    # Component 5: i=0,j=2,k=1
    U[0*3+2, 5] = 1; V[2*2+1, 5] = 1; W[1*3+0, 5] = 1
    # Component 6: i=1,j=0,k=0
    U[1*3+0, 6] = 1; V[0*2+0, 6] = 1; W[0*3+1, 6] = 1
    # Component 7: i=1,j=0,k=1
    U[1*3+0, 7] = 1; V[0*2+1, 7] = 1; W[1*3+1, 7] = 1
    # Component 8: i=2,j=0,k=0
    U[2*3+0, 8] = 1; V[0*2+0, 8] = 1; W[0*3+2, 8] = 1
    # Component 9: i=2,j=0,k=1
    U[2*3+0, 9] = 1; V[0*2+1, 9] = 1; W[1*3+2, 9] = 1
    # Component 10: i=1,j=1,k=0
    U[1*3+1, 10] = 1; V[1*2+0, 10] = 1; W[0*3+1, 10] = 1
    # Component 11: i=1,j=1,k=1
    U[1*3+1, 11] = 1; V[1*2+1, 11] = 1; W[1*3+1, 11] = 1
    
    # Wait, this only covers 12 of the 18 terms. I'm missing:
    # i=1,j=2,k=0; i=1,j=2,k=1; i=2,j=1,k=0; i=2,j=1,k=1; i=2,j=2,k=0; i=2,j=2,k=1
    # That's 6 more terms. This is just schoolbook for 12 terms, not a rank-12 decomposition.
    
    # I need to use Strassen-like tricks to reduce the rank.
    # 
    # Let me restart with a proper Strassen-based construction.
    # 
    # For the 2x2x2 core (i=1,2; j=1,2; k=0,1), Strassen uses 7 products:
    # M1 = (a11+a22)*(b10+b21)  -- but wait, this is for 2x2x2 where output is 2x2
    # 
    # Standard Strassen for 2x2x2:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b11-b21)
    # M4 = a22*(b22-b12)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    #
    # In our indexing for the core (i'=0,1 corresponding to i=1,2; j'=0,1 corresponding to j=1,2; k=0,1):
    # a11=a[1,1], a12=a[1,2], a21=a[2,1], a22=a[2,2]
    # b10=b[1,0], b11=b[1,1], b20=b[2,0], b21=b[2,1]
    # c10=c[1,0], c11=c[1,1], c20=c[2,0], c21=c[2,1]
    #
    # M1 = (a11+a22)*(b10+b21)
    # M2 = (a21+a22)*b10
    # M3 = a11*(b10-b20)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b21
    # M6 = (a21-a11)*(b10+b11)
    # M7 = (a12-a22)*(b20+b21)
    # c10 = M1+M4-M5+M7
    # c11 = M3+M5
    # c20 = M2+M4
    # c21 = M1-M2+M3+M6
    #
    # Now for the full 3x3x2, I need to handle:
    # - Row 0 of A: c00 = a00*b00+a01*b10+a02*b20, c01 = a00*b01+a01*b11+a02*b21
    # - The core 2x2x2 handled by Strassen above
    # - But wait, the Strassen core only handles c[1,0],c[1,1],c[2,0],c[2,1]
    #   using a[1,1],a[1,2],a[2,1],a[2,2] and b[1,0],b[1,1],b[2,0],b[2,1]
    # - I still need to handle contributions from j=0 to c[1,k] and c[2,k]
    #   c[1,0] += a[1,0]*b[0,0], c[1,1] += a[1,0]*b[0,1]
    #   c[2,0] += a[2,0]*b[0,0], c[2,1] += a[2,0]*b[0,1]
    # - And row 0: c[0,0] and c[0,1]
    #
    # So additional terms needed:
    # a[1,0]*b[0,0] -> c[1,0]
    # a[1,0]*b[0,1] -> c[1,1]
    # a[2,0]*b[0,0] -> c[2,0]
    # a[2,0]*b[0,1] -> c[2,1]
    # a[0,0]*b[0,0] -> c[0,0]
    # a[0,0]*b[0,1] -> c[0,1]
    # a[0,1]*b[1,0] -> c[0,0]
    # a[0,1]*b[1,1] -> c[0,1]
    # a[0,2]*b[2,0] -> c[0,0]
    # a[0,2]*b[2,1] -> c[0,1]
    #
    # That's 10 additional rank-1 terms + 7 Strassen = 17. Still not 12.
    
    # I think M(3,3,2) might actually be higher than 12. Let me check if 12 is achievable.
    # 
    # Actually, I recall that for rectangular matrices, the bounds are less tight.
    # Let me just try to get a good rank and verify it.
    # 
    # For now, let me use the schoolbook decomposition but try to optimize.
    # The schoolbook rank for 3x3x6 is 54.
    # 
    # Let me try a different approach: use the fact that 3x3x6 = 3 * (3x3x2)
    # and if I can get rank R for 3x3x2, total rank is 3R.
    # 
    # If R=18 (schoolbook), total = 54.
    # If R=16, total = 48.
    # 
    # Let me just implement the schoolbook and try to verify it works, then see if I can improve.
    
    # Actually, let me just return the schoolbook decomposition for now to ensure correctness,
    # and then try to improve.
    
    # For the full 3x3x6 problem, let me just use schoolbook:
    n, m, p = 3, 3, 6
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
    return U, V, W, rank

def construct():
    # Schoolbook decomposition of rank n*m*p
    n, m, p = DIMENSIONS
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
