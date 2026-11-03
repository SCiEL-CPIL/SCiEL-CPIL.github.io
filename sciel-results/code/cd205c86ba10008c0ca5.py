import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    time_limit = 140.0

    def elapsed():
        return time.time() - t_start

    def time_exceeded():
        return elapsed() > time_limit

    # Strassen 2x2 decomposition
    # Rank 7
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def make_strassen_2x2():
        # U indices: (i, j) for 2x2 A. i,j in 0..1
        # V indices: (j, k) for 2x2 B. j,k in 0..1
        # W indices: (k, i) for 2x2 C. k,i in 0..1
        # Flattened: U[i*2+j], V[j*2+k], W[k*2+i]
        
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # M1: (A00+A11)(B00+B11)
        # A00: U[0], A11: U[3]
        # B00: V[0], B11: V[3]
        # C00: W[0], C11: W[3]
        # Term 1: A00 B00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A00 B11
        U[0, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
        # Term 3: A11 B00
        U[3, 2] = 1; V[0, 2] = 1; W[3, 2] = 1
        # Term 4: A11 B11
        U[3, 3] = 1; V[3, 3] = 1; W[3, 3] = 1
        
        # Wait, Strassen uses intermediate sums.
        # The tensor decomposition must be linear in A and B.
        # So we cannot just put sums in U and V.
        # We must expand the products.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This requires 4 terms for M1 alone?
        # No, Strassen's algorithm computes 7 products.
        # Each product is a bilinear form.
        # But the products are of SUMS of entries.
        # To express this as a tensor decomposition $\sum U V W$,
        # we need to expand the sums.
        # This would increase the rank.
        # Strassen's rank 7 is for the TENSOR rank.
        # The tensor rank 7 means we can write the tensor as sum of 7 rank-1 tensors.
        # The rank-1 tensors correspond to the 7 products.
        # But the inputs to the products are linear combinations of A and B.
        # So $U$ and $V$ can have non-zero entries in multiple positions.
        
        # Let's define the 7 products:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # U for P1: A00=1, A11=1. Indices 0, 3.
        U[0, 0] = 1; U[3, 0] = 1
        # V for P1: B00=1, B11=1. Indices 0, 3.
        V[0, 0] = 1; V[3, 0] = 1
        # W for P1: C00 += P1, C11 += P1.
        # C00 index 0, C11 index 3.
        W[0, 0] = 1; W[3, 0] = 1
        
        # P2
        U[2, 1] = 1; U[3, 1] = 1 # A10, A11
        V[0, 1] = 1 # B00
        W[2, 1] = 1 # C10 (index 2 in W[k*2+i] -> k=1, i=0 -> 2)
        
        # P3
        U[0, 2] = 1 # A00
        V[1, 2] = 1; V[3, 2] = -1 # B01, -B11
        W[1, 2] = 1 # C01 (index 1 in W -> k=0, i=1? No. W[k*2+i]. C01 is k=0, i=1? No. C is 2x2. C[i][k].
        # W index is k*2+i.
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 3
        # So C01 is W[2]. C10 is W[1].
        W[2, 2] = 1
        
        # P4
        U[3, 3] = 1 # A11
        V[2, 3] = 1; V[0, 3] = -1 # B10, -B00
        W[1, 3] = 1 # C10 is W[1]
        
        # P5
        U[0, 4] = 1; U[1, 4] = 1 # A00, A01
        V[3, 4] = 1 # B11
        W[3, 4] = 1 # C11 is W[3]
        
        # P6
        U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
        V[0, 5] = 1; V[1, 5] = 1 # B00, B01
        W[2, 5] = 1 # C01 is W[2]
        
        # P7
        U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
        V[2, 6] = 1; V[3, 6] = 1 # B10, B11
        W[3, 6] = 1 # C11 is W[3]
        
        return U, V, W

    U2, V2, W2 = make_strassen_2x2()
    
    # Kronecker product to get 4x4
    # U4 shape (16, 49)
    # V4 shape (16, 49)
    # W4 shape (16, 49)
    
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)
    
    idx = 0
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    # This loop is wrong. We need to iterate over the 7 terms.
                    pass
    
    # Correct Kronecker construction
    # Term r in 4x4 is combination of terms r1, r2 in 2x2.
    # U4[i,j, r] = U2[i1,j1, r1] * U2[i2,j2, r2]
    # where i = 2*i1 + i2, j = 2*j1 + j2, r = 7*r1 + r2
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            U4[i * 4 + j, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                            
                            V4[j * 4 + r, r] = V2[j1 * 2 + r1, r1] * V2[j2 * 2 + r2, r2] # Wait, V index is j,k.
                            # V2 index is j*2+k.
                            # V4 index is j*4+k.
                            # j = j1*2+j2, k = k1*2+k2?
                            # V2[j1, k1] * V2[j2, k2]
                            # V4[j, k]
                            
                            # Let's redo V and W mapping carefully.
                            pass

    # Simpler: Just build the 4x4 factors by iterating all indices.
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            U4[i * 4 + j, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V4[j * 4 + k, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                                    W4[k * 4 + i, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]

    # Now map to 4x5x5
    U = np.zeros((20, 85), dtype=np.int64)
    V = np.zeros((25, 85), dtype=np.int64)
    W = np.zeros((20, 85), dtype=np.int64)
    
    # Copy 4x4 part (indices 0..3 for i,j,k)
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, 0:49] = U4[i * 4 + j, 0:49]
            
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, 0:49] = V4[j * 4 + k, 0:49]
            
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
            
    # Add schoolbook terms for j=4 (A col 5, B row 5)
    # A_i4 * B_4k
    idx = 49
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4, idx] = 1
            V[4 * 5 + k, idx] = 1
            W[k * 4 + i, idx] = 1
            idx += 1
            
    # Add schoolbook terms for k=4 (B col 5) with j<4
    # A_ij * B_j4
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, idx] = 1
            V[j * 5 + 4, idx] = 1
            W[4 * 4 + i, idx] = 1
            idx += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    construct()
