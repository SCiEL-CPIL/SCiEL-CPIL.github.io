"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    # We will use Kronecker products of known small-rank decompositions.
    # 
    # Strategy:
    # 1. Decompose 4x4x9 multiplication.
    # 2. We can view 4x9 as 4x(4+4+1) or similar.
    # 3. Actually, we can use the rank of matmul 4x4x4 and 4x4x1.
    #    Rank(4x4x4) can be bounded by R(2x2x2)^2 = 7^2 = 49 (Kronecker).
    #    Rank(4x4x1) is the rank of the bilinear map A(4x4) * b(4) -> c(4).
    #    This is essentially matrix-vector multiplication.
    #    Can we do better than 16?
    #    For 2x2 matrix times 2x1 vector, the rank is 4 (schoolbook).
    #    Wait, is there a faster way?
    #    The tensor is T_{ijk} where i is row of A, j is col of A (and row of B), k is col of C.
    #    Actually, matvec is just a bilinear map.
    #    Rank of matvec for 2x2 is 4. For 4x4 is 16.
    #    So Rank(4x4x9) <= Rank(4x4x4) + Rank(4x4x4) + Rank(4x4x1)
    #    Using Kronecker:
    #    R(4x4x4) <= 49
    #    R(4x4x1) <= 16
    #    Total <= 49 + 49 + 16 = 114.
    #
    # Is there a better decomposition for 4x4x9?
    # Maybe 4x9 can be factored differently?
    # Let's try to construct the specific tensors.
    
    # 1. Construct 2x2x2 Strassen decomposition
    # U, V, W are lists of 7 triples.
    # Format: list of (u, v, w) where u, v, w are vectors in F^4?
    # No, U is 4x7, V is 4x7, W is 4x7.
    # Let's define the 7 multiplications for 2x2 Strassen.
    # Indices: 0:11, 1:12, 2:21, 3:22 for A and B.
    # C indices same.
    
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = (a11-a22)(b21+b22)
    # M4 = a22(b22-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Let's represent U, V, W for 2x2x2 as arrays of shape (4, 7).
    # Row index: 0:11, 1:12, 2:21, 3:22.
    
    def get_strassen_222():
        # U[i, k] is coeff of a_i in M_k
        # V[i, k] is coeff of b_i in M_k
        # W[i, k] is coeff of M_k in c_i
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1: (a0+a3)(b0+b3)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        
        # M2: (a2+a3)b0
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        
        # M3: (a0-a3)(b2+b3)
        U[0,2] = 1; U[3,2] = -1
        V[2,2] = 1; V[3,2] = 1
        
        # M4: a3(b3-b0)
        U[3,3] = 1
        V[3,3] = 1; V[0,3] = -1
        
        # M5: (a0+a1)b3
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        
        # M6: (a2-a0)(b0+b1)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        
        # M7: (a1-a3)(b2+b3)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        
        # W: c_i = sum_k W[i,k] M_k
        # c0 (11) = M1 + M4 - M5 + M7
        W[0,0] = 1; W[0,3] = 1; W[0,4] = -1; W[0,6] = 1
        # c1 (12) = M3 + M5
        W[1,2] = 1; W[1,4] = 1
        # c2 (21) = M2 + M4
        W[2,1] = 1; W[2,3] = 1
        # c3 (22) = M1 - M2 + M3 + M6
        W[3,0] = 1; W[3,1] = -1; W[3,2] = 1; W[3,5] = 1
        
        return U, V, W

    U2, V2, W2 = get_strassen_222()
    
    # Now Kronecker product for 4x4x4.
    # Rank 49.
    # U4 is 16x49, V4 is 16x49, W4 is 16x49.
    # The indices for 4x4 matrices are mapped from (i1,i2) to i1*2+i2.
    # Kronecker product of decompositions:
    # U4[i1*2+i2, k1*7+k2] = U2[i1, k1] * U2[i2, k2]
    # Similarly for V and W.
    
    def kron_decomp(U1, V1, W1):
        # U1, V1, W1 are (d1, r1)
        # Returns (d1^2, r1^2)
        d1, r1 = U1.shape
        d2, r2 = U1.shape
        U2 = np.zeros((d1*d1, r1*r1), dtype=int)
        V2 = np.zeros((d1*d1, r1*r1), dtype=int)
        W2 = np.zeros((d1*d1, r1*r1), dtype=int)
        
        for i1 in range(d1):
            for i2 in range(d1):
                row = i1*d1 + i2
                for k1 in range(r1):
                    for k2 in range(r1):
                        col = k1*r1 + k2
                        U2[row, col] = U1[i1, k1] * U1[i2, k2]
                        V2[row, col] = V1[i1, k1] * V1[i2, k2]
                        W2[row, col] = W1[i1, k1] * W1[i2, k2]
        return U2, V2, W2

    U4, V4, W4 = kron_decomp(U2, V2, W2)
    
    # Now we need 4x4x1.
    # Rank 16 is schoolbook.
    # Is there a better one?
    # For 2x2x1, rank is 4.
    # So for 4x4x1, rank is 16 via Kronecker.
    # Let's construct 2x2x1.
    # A(2x2) * b(2x1) = c(2x1)
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    # This is just 4 multiplications.
    # U21 (4x4), V21 (2x4), W21 (2x4).
    
    def get_matvec_221():
        # U is 4x4 (indices 0:00, 1:01, 2:10, 3:11)
        # V is 2x4 (indices 0:0, 1:1)
        # W is 2x4 (indices 0:0, 1:1)
        
        U = np.zeros((4, 4), dtype=int)
        V = np.zeros((2, 4), dtype=int)
        W = np.zeros((2, 4), dtype=int)
        
        # M0: a00*b0 -> c0
        U[0,0] = 1
        V[0,0] = 1
        W[0,0] = 1
        
        # M1: a01*b1 -> c0
        U[1,1] = 1
        V[1,1] = 1
        W[0,1] = 1
        
        # M2: a10*b0 -> c1
        U[2,2] = 1
        V[0,2] = 1
        W[1,2] = 1
        
        # M3: a11*b1 -> c1
        U[3,3] = 1
        V[1,3] = 1
        W[1,3] = 1
        
        return U, V, W

    U21, V21, W21 = get_matvec_221()
    
    # Kronecker for 4x4x1.
    # U41 (16x16), V41 (4x16), W41 (4x16).
    
    def kron_decomp_matvec(U1, V1, W1):
        # U1 (d1, r1), V1 (p1, r1), W1 (p1, r1) where p1=2
        # Returns U2 (d1^2, r1^2), V2 (p1^2, r1^2), W2 (p1^2, r1^2)
        d1, r1 = U1.shape
        p1, _ = V1.shape
        
        U2 = np.zeros((d1*d1, r1*r1), dtype=int)
        V2 = np.zeros((p1*p1, r1*r1), dtype=int)
        W2 = np.zeros((p1*p1, r1*r1), dtype=int)
        
        for i1 in range(d1):
            for i2 in range(d1):
                row_u = i1*d1 + i2
                for k1 in range(r1):
                    for k2 in range(r1):
                        col = k1*r1 + k2
                        U2[row_u, col] = U1[i1, k1] * U1[i2, k2]
        
        for i1 in range(p1):
            for i2 in range(p1):
                row_v = i1*p1 + i2
                row_w = i1*p1 + i2
                for k1 in range(r1):
                    for k2 in range(r1):
                        col = k1*r1 + k2
                        V2[row_v, col] = V1[i1, k1] * V1[i2, k2]
                        W2[row_w, col] = W1[i1, k1] * W1[i2, k2]
                        
        return U2, V2, W2

    U41, V41, W41 = kron_decomp_matvec(U21, V21, W21)
    
    # Now combine.
    # We want 4x4x9.
    # We split the 9 columns into 4 + 4 + 1.
    # So we have three blocks:
    # Block 1: 4x4x4 (uses U4, V4, W4)
    # Block 2: 4x4x4 (uses U4, V4, W4)
    # Block 3: 4x4x1 (uses U41, V41, W41)
    
    # The input A is 4x4. It is the same for all blocks.
    # The input B is 4x9. It is split into B1 (4x4), B2 (4x4), B3 (4x1).
    # The output C is 4x9. It is split into C1 (4x4), C2 (4x4), C3 (4x1).
    
    # U matrix: rows are A indices (16). Cols are rank (49+49+16 = 114).
    # V matrix: rows are B indices (16+16+4 = 36). Cols are rank (114).
    # W matrix: rows are C indices (16+16+4 = 36). Cols are rank (114).
    
    # Note: The problem statement says:
    # U shape (n*m, rank) = (16, 114)
    # V shape (m*p, rank) = (36, 114)
    # W shape (p*n, rank) = (36, 114)
    # Indices:
    # U[i*m+j, c] -> i,j in 0..3
    # V[j*p+k, c] -> j in 0..3, k in 0..8
    # W[k*n+i, c] -> k in 0..8, i in 0..3
    
    # Let's assemble U, V, W.
    
    rank_total = 49 + 49 + 16
    
    U_total = np.zeros((16, rank_total), dtype=int)
    V_total = np.zeros((36, rank_total), dtype=int)
    W_total = np.zeros((36, rank_total), dtype=int)
    
    # Block 1: cols 0..48
    # U_part1 = U4 (16x49)
    U_total[:, 0:49] = U4
    
    # V_part1: B1 is 4x4. Rows of V correspond to j*4+k for j in 0..3, k in 0..3.
    # This corresponds to rows 0..15 of V_total.
    # V4 is 16x49.
    V_total[0:16, 0:49] = V4
    
    # W_part1: C1 is 4x4. Rows of W correspond to k*4+i for k in 0..3, i in 0..3.
    # This corresponds to rows 0..15 of W_total.
    # W4 is 16x49.
    W_total[0:16, 0:49] = W4
    
    # Block 2: cols 49..97
    # U_part2 = U4 (16x49)
    U_total[:, 49:98] = U4
    
    # V_part2: B2 is 4x4. Rows j*4+k for j in 0..3, k in 4..7.
    # In V_total, these are rows 16..31.
    # V4 is 16x49.
    V_total[16:32, 49:98] = V4
    
    # W_part2: C2 is 4x4. Rows k*4+i for k in 4..7, i in 0..3.
    # In W_total, these are rows 16..31.
    # W4 is 16x49.
    W_total[16:32, 49:98] = W4
    
    # Block 3: cols 98..113
    # U_part3 = U41 (16x16)
    U_total[:, 98:114] = U41
    
    # V_part3: B3 is 4x1. Rows j*4+k for j in 0..3, k in 8.
    # In V_total, these are rows j*4+8.
    # j=0 -> 8, j=1 -> 12, j=2 -> 16, j=3 -> 20.
    # Wait, V_total rows are j*p+k.
    # For B3, k is always 8.
    # So rows are 0*4+8=8, 1*4+8=12, 2*4+8=16, 3*4+8=20.
    # V41 is 4x16. Its rows correspond to the input vector b (size 4).
    # The rows of V41 are indexed by i in 0..3.
    # We need to map V41[i, :] to V_total[j*4+8, :] where j=i.
    
    for i in range(4):
        V_total[i*4+8, 98:114] = V41[i, :]
        
    # W_part3: C3 is 4x1. Rows k*4+i for k in 8, i in 0..3.
    # In W_total, these are rows 8*4+i = 32+i.
    # i=0 -> 32, i=1 -> 33, i=2 -> 34, i=3 -> 35.
    # W41 is 4x16.
    # W_total[32:36, 98:114] = W41
    
    W_total[32:36, 98:114] = W41
    
    # Convert to lists of lists of integers
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
