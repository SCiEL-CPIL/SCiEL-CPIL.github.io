"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen's algorithm applied to each pair of columns.
    For 2x2 x 2xp with p even, group columns into p/2 pairs.
    Each pair uses Strassen's rank-7 decomposition for 2x2x2.
    Total rank = (p/2) * 7 = 6 * 7 = 42 for p=12.
    
    The decomposition satisfies:
    For all i in [0,2), j in [0,2), k in [0,12):
      sum_c U[i*2+j, c] * V[j*12+k, c] * W[k*2+i, c] = 1
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # For each pair t (columns 2t and 2t+1), apply Strassen's 7-term decomposition.
    # A is 2x2: A[i,j] for i,j in {0,1}
    # B_pair is 2x2: B[j, k_local] for j in {0,1}, k_local in {0,1}
    #   where global column k = 2*t + k_local
    # C_pair is 2x2: C[i, k_local] = sum_j A[i,j] * B[j, k_local]
    
    # Strassen for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: U[i*m+j, c] = U[i*2+j, c]
    # V indices: V[j*p+k, c] = V[j*12+k, c]
    # W indices: W[k*n+i, c] = W[k*2+i, c]
    
    for t in range(num_pairs):
        base = t * 7
        k0 = 2 * t      # first column in pair
        k1 = 2 * t + 1  # second column in pair
        
        for s in range(7):
            c = base + s
            
            if s == 0:
                # M1 = (A00+A11)*(B00+B11)
                U[0, c] += 1   # A[0,0]
                U[3, c] += 1   # A[1,1]
                V[0*p + k0, c] += 1   # B[0, k0] = B00
                V[1*p + k1, c] += 1   # B[1, k1] = B11
                # C00 += M1, C11 += M1
                W[k0*n + 0, c] += 1   # C[0, k0]
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 1:
                # M2 = (A10+A11)*B00
                U[2, c] += 1   # A[1,0]
                U[3, c] += 1   # A[1,1]
                V[0*p + k0, c] += 1   # B[0, k0]
                # C10 += M2, C11 -= M2
                W[k0*n + 1, c] += 1   # C[1, k0]
                W[k1*n + 1, c] += -1  # C[1, k1]
                
            elif s == 2:
                # M3 = A00*(B01-B11)
                U[0, c] += 1   # A[0,0]
                V[0*p + k1, c] += 1    # B[0, k1] = B01
                V[1*p + k1, c] += -1   # B[1, k1] = B11
                # C01 += M3, C11 += M3
                W[k1*n + 0, c] += 1   # C[0, k1]
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 3:
                # M4 = A11*(B10-B00)
                U[3, c] += 1   # A[1,1]
                V[1*p + k0, c] += 1    # B[1, k0] = B10
                V[0*p + k0, c] += -1   # B[0, k0] = B00
                # C00 += M4, C10 += M4
                W[k0*n + 0, c] += 1   # C[0, k0]
                W[k0*n + 1, c] += 1   # C[1, k0]
                
            elif s == 4:
                # M5 = (A00+A01)*B11
                U[0, c] += 1   # A[0,0]
                U[1, c] += 1   # A[0,1]
                V[1*p + k1, c] += 1   # B[1, k1] = B11
                # C00 -= M5, C01 += M5
                W[k0*n + 0, c] += -1  # C[0, k0]
                W[k1*n + 0, c] += 1   # C[0, k1]
                
            elif s == 5:
                # M6 = (A10-A00)*(B00+B01)
                U[2, c] += 1   # A[1,0]
                U[0, c] += -1  # A[0,0]
                V[0*p + k0, c] += 1    # B[0, k0] = B00
                V[0*p + k1, c] += 1    # B[0, k1] = B01
                # C11 += M6
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 6:
                # M7 = (A01-A11)*(B10+B11)
                U[1, c] += 1   # A[0,1]
                U[3, c] += -1  # A[1,1]
                V[1*p + k0, c] += 1    # B[1, k0] = B10
                V[1*p + k1, c] += 1    # B[1, k1] = B11
                # C00 += M7
                W[k0*n + 0, c] += 1   # C[0, k0]
    
    # Convert to lists as expected by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
