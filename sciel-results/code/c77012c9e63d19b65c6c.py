"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-28 decomposition for (2x4)*(4x4) matrix multiplication.
    
    Strategy: Partition into 2x2 blocks and use Strassen for each of the
    4 block products (2x2 * 2x2). Each uses 7 rank-1 terms, giving rank 28.
    
    A (2x4) = [A0 | A1] where A0, A1 are 2x2
    B (4x4) = [[B00, B01], [B10, B11]] where each is 2x2
    
    C = A * B:
    C[:, 0:2] = A0 * B00 + A1 * B10
    C[:, 2:4] = A0 * B01 + A1 * B11
    
    Each 2x2 * 2x2 product uses Strassen (7 multiplications).
    Total: 4 * 7 = 28 multiplications.
    
    Strassen for X*Y (both 2x2):
    M1 = (x00+x11)*(y00+y11)
    M2 = (x10+x11)*y00
    M3 = x00*(y01-y11)
    M4 = x11*(y10-y00)
    M5 = (x00+x01)*y11
    M6 = (x10-x00)*(y00+y01)
    M7 = (x01-x11)*(y10+y11)
    
    Result:
    Z[0,0] = M1+M4-M5+M7
    Z[0,1] = M3+M5
    Z[1,0] = M2+M4
    Z[1,1] = M1-M2+M3+M6
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index conventions:
    # U index for A[i,j]: i*m + j = i*4 + j
    # V index for B[j,k]: j*p + k = j*4 + k
    # W index for C[i,k]: k*n + i = k*2 + i (official trace convention W[k*n+i,r])
    
    # Block index arrays:
    # A0 = A[:, 0:2]:
    #   A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # A1 = A[:, 2:4]:
    #   A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B00 = B[0:2, 0:2]:
    #   B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # B01 = B[0:2, 2:4]:
    #   B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # B10 = B[2:4, 0:2]:
    #   B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # B11 = B[2:4, 2:4]:
    #   B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    
    # W indices for C:
    # C[0,0] -> 0*2+0 = 0
    # C[0,1] -> 1*2+0 = 2
    # C[1,0] -> 0*2+1 = 1
    # C[1,1] -> 1*2+1 = 3
    # C[0,2] -> 2*2+0 = 4
    # C[0,3] -> 3*2+0 = 6
    # C[1,2] -> 2*2+1 = 5
    # C[1,3] -> 3*2+1 = 7
    
    def add_strassen(prod_id, x_idx, y_idx, z_idx):
        """
        Add 7 Strassen terms for a 2x2 * 2x2 product.
        
        x_idx: [[x00, x01], [x10, x11]] - U indices
        y_idx: [[y00, y01], [y10, y11]] - V indices
        z_idx: [[z00, z01], [z10, z11]] - W indices
        """
        col = prod_id * 7
        x00, x01 = x_idx[0][0], x_idx[0][1]
        x10, x11 = x_idx[1][0], x_idx[1][1]
        y00, y01 = y_idx[0][0], y_idx[0][1]
        y10, y11 = y_idx[1][0], y_idx[1][1]
        z00, z01 = z_idx[0][0], z_idx[0][1]
        z10, z11 = z_idx[1][0], z_idx[1][1]
        
        # M1 = (x00+x11)*(y00+y11)
        # Contributes to: Z[0,0] += M1, Z[1,1] += M1
        U[x00, col] += 1
        U[x11, col] += 1
        V[y00, col] += 1
        V[y11, col] += 1
        W[z00, col] += 1
        W[z11, col] += 1
        
        # M2 = (x10+x11)*y00
        # Contributes to: Z[1,0] += M2
        U[x10, col+1] += 1
        U[x11, col+1] += 1
        V[y00, col+1] += 1
        W[z10, col+1] += 1
        
        # M3 = x00*(y01-y11)
        # Contributes to: Z[0,1] += M3
        U[x00, col+2] += 1
        V[y01, col+2] += 1
        V[y11, col+2] -= 1
        W[z01, col+2] += 1
        
        # M4 = x11*(y10-y00)
        # Contributes to: Z[0,0] += M4, Z[1,0] += M4
        U[x11, col+3] += 1
        V[y10, col+3] += 1
        V[y00, col+3] -= 1
        W[z00, col+3] += 1
        W[z10, col+3] += 1
        
        # M5 = (x00+x01)*y11
        # Contributes to: Z[0,0] -= M5, Z[0,1] += M5
        U[x00, col+4] += 1
        U[x01, col+4] += 1
        V[y11, col+4] += 1
        W[z00, col+4] -= 1
        W[z01, col+4] += 1
        
        # M6 = (x10-x00)*(y00+y01)
        # Contributes to: Z[1,1] += M6
        U[x10, col+5] += 1
        U[x00, col+5] -= 1
        V[y00, col+5] += 1
        V[y01, col+5] += 1
        W[z11, col+5] += 1
        
        # M7 = (x01-x11)*(y10+y11)
        # Contributes to: Z[0,0] += M7
        U[x01, col+6] += 1
        U[x11, col+6] -= 1
        V[y10, col+6] += 1
        V[y11, col+6] += 1
        W[z00, col+6] += 1
    
    # Product 1: A0 * B00 -> C[:, 0:2]
    # A0: [[0,1],[4,5]], B00: [[0,1],[4,5]]
    # C[:,0:2]: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    add_strassen(0,
                 [[0,1],[4,5]],    # A0
                 [[0,1],[4,5]],    # B00
                 [[0,2],[1,3]])    # C[:,0:2]
    
    # Product 2: A1 * B10 -> C[:, 0:2]
    # A1: [[2,3],[6,7]], B10: [[8,9],[12,13]]
    add_strassen(1,
                 [[2,3],[6,7]],    # A1
                 [[8,9],[12,13]],  # B10
                 [[0,2],[1,3]])    # C[:,0:2]
    
    # Product 3: A0 * B01 -> C[:, 2:4]
    # A0: [[0,1],[4,5]], B01: [[2,3],[6,7]]
    # C[:,2:4]: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    add_strassen(2,
                 [[0,1],[4,5]],    # A0
                 [[2,3],[6,7]],    # B01
                 [[4,6],[5,7]])    # C[:,2:4]
    
    # Product 4: A1 * B11 -> C[:, 2:4]
    # A1: [[2,3],[6,7]], B11: [[10,11],[14,15]]
    add_strassen(3,
                 [[2,3],[6,7]],    # A1
                 [[10,11],[14,15]], # B11
                 [[4,6],[5,7]])    # C[:,2:4]
    
    # Verify the decomposition
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0 if j == k else 0.0
                error += abs(val - expected)
    
    if error > 1e-10:
        # Fallback to schoolbook if Strassen construction is wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
