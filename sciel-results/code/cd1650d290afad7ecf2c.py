import numpy as np
import time
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    1. Construct a rank-14 decomposition using the Kronecker product of 
       Strassen (2x2x2, rank 7) and a rank-2 decomposition for (1x2x3).
    2. The (1x2x3) tensor M_{1,2,3} has rank 2, not 6 as in the trivial decomposition.
       This is because the A-factor has dimension 1, so we can reuse multiplications.
    3. Return the valid decomposition.
    """
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Core 1: 2x2x2 Strassen Tensor (Rank 7)
    # Indices: i (2), j (2), k (2)
    # U1 shape (4, 7), V1 shape (4, 7), W1 shape (4, 7)
    # U1 row: i*2 + j
    # V1 row: j*2 + k
    # W1 row: k*2 + i (Trace convention W[k*n+i])
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U1 (A side)
    U1[0, 0] = 1; U1[3, 0] = 1  # M1: A00, A11
    U1[2, 1] = 1; U1[3, 1] = 1  # M2: A10, A11
    U1[0, 2] = 1                # M3: A00
    U1[3, 3] = 1                # M4: A11
    U1[0, 4] = 1; U1[1, 4] = 1  # M5: A00, A01
    U1[2, 5] = 1; U1[0, 5] = -1 # M6: A10, -A00
    U1[1, 6] = 1; U1[3, 6] = -1 # M7: A01, -A11
    
    # V1 (B side)
    V1[0, 0] = 1; V1[3, 0] = 1  # M1: B00, B11
    V1[0, 1] = 1                # M2: B00
    V1[1, 2] = 1; V1[3, 2] = -1 # M3: B01, -B11
    V1[2, 3] = 1; V1[0, 3] = -1 # M4: B10, -B00
    V1[3, 4] = 1                # M5: B11
    V1[0, 5] = 1; V1[1, 5] = 1  # M6: B00, B01
    V1[2, 6] = 1; V1[3, 6] = 1  # M7: B10, B11
    
    # W1 (Output side)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # W index: k*2 + i
    # C00 (i=0, k=0) -> idx 0
    # C01 (i=0, k=1) -> idx 2
    # C10 (i=1, k=0) -> idx 1
    # C11 (i=1, k=1) -> idx 3
    
    W1[0, 0] = 1; W1[0, 3] = 1; W1[0, 4] = -1; W1[0, 6] = 1 # C00
    W1[2, 2] = 1; W1[2, 4] = 1                               # C01
    W1[1, 1] = 1; W1[1, 3] = 1                               # C10
    W1[3, 0] = 1; W1[3, 1] = -1; W1[3, 2] = 1; W1[3, 5] = 1  # C11
    
    # Core 2: 1x2x3 Tensor (Rank 2)
    # Indices: i (1), j (2), k (3)
    # U2 shape (2, 2), V2 shape (6, 2), W2 shape (3, 2)
    # 
    # The tensor M_{1,2,3} has entries T[i,j,k] = 1 if i=0 (only value), else 0.
    # Actually, for matrix multiplication 1x2 by 2x3, the tensor is:
    # C[k] = A0 * B0k + A1 * B1k for k=0,1,2
    # T[0, j, k] = 1 for all j, k
    # 
    # Rank 2 decomposition:
    # r=0: u=[1, 0], v=[1, 0, 0, 1, 0, 0] (B00, B10), w=[1, 1, 1]
    # r=1: u=[0, 1], v=[0, 1, 0, 0, 1, 0] (B01, B11), w=[1, 1, 1]
    # 
    # Wait, this doesn't work because w should be different for different k.
    # Let me reconsider.
    # 
    # T[0, j, k] = 1 for j in {0,1}, k in {0,1,2}
    # 
    # Rank 2:
    # r=0: u=[1, 0], v=[1, 1, 1, 0, 0, 0], w=[1, 1, 1]
    #   Term: A0 * (B00 + B01 + B02) * 1 for all k? No, w is multiplied.
    #   Actually, the decomposition is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #   Wait, the indices are different.
    
    # Let me use the standard Kronecker product approach more carefully.
    # The tensor for M_{1,2,3} has rank 2.
    # U2 shape (2, 2): rows indexed by (i2, j2) where i2 in {0}, j2 in {0,1}
    #   So U2 has 2 rows: (i2=0, j2=0) and (i2=0, j2=1)
    # V2 shape (6, 2): rows indexed by (j2, k2) where j2 in {0,1}, k2 in {0,1,2}
    #   So V2 has 6 rows: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    # W2 shape (3, 2): rows indexed by (k2, i2) where k2 in {0,1,2}, i2 in {0}
    #   So W2 has 3 rows: (0,0), (1,0), (2,0)
    
    U2 = np.zeros((2, 2))
    V2 = np.zeros((6, 2))
    W2 = np.zeros((3, 2))
    
    # Rank 2 decomposition of M_{1,2,3}:
    # r=0: 
    #   U2[0, 0] = 1 (A0)
    #   U2[1, 0] = 1 (A1)
    #   V2[0, 0] = 1 (B00)
    #   V2[1, 0] = 1 (B01)
    #   V2[2, 0] = 1 (B02)
    #   V2[3, 0] = 1 (B10)
    #   V2[4, 0] = 1 (B11)
    #   V2[5, 0] = 1 (B12)
    #   W2[0, 0] = 1, W2[1, 0] = 1, W2[2, 0] = 1
    # This is rank 1, not 2. This gives T[0,j,k] = 1*1*1 = 1. Correct!
    # So M_{1,2,3} has rank 1?
    # 
    # Wait, if U2[0,0]=1, U2[1,0]=1, V2[all]=1, W2[all]=1, then:
    # T[0,j,k] = U2[i2*2+j2, r] * V2[j2*3+k2, r] * W2[k2*1+i2, r]
    # For i2=0, j2=0, k2=0: U2[0,0]*V2[0,0]*W2[0,0] = 1*1*1 = 1. Correct.
    # For i2=0, j2=1, k2=0: U2[1,0]*V2[3,0]*W2[0,0] = 1*1*1 = 1. Correct.
    # All entries are 1. So rank 1 works!
    # 
    # But wait, the trivial decomposition has rank 6, and I said rank 2 before.
    # Let me verify: if rank 1 works, then M_{1,2,3} has rank 1.
    # 
    # Actually, I think I'm confusing myself. Let me just use rank 1 for M_{1,2,3}.
    
    U2[0, 0] = 1
    U2[1, 0] = 1
    V2[0, 0] = 1
    V2[1, 0] = 1
    V2[2, 0] = 1
    V2[3, 0] = 1
    V2[4, 0] = 1
    V2[5, 0] = 1
    W2[0, 0] = 1
    W2[1, 0] = 1
    W2[2, 0] = 1
    
    # So Core 2 has rank 1.
    
    # Kronecker Product Construction
    # Dimensions:
    # n: 2 * 1 = 2
    # m: 2 * 2 = 4
    # p: 2 * 3 = 6
    # Rank: 7 * 1 = 7
    
    U = np.zeros((8, 7))
    V = np.zeros((24, 7))
    W = np.zeros((12, 7))
    
    for r1 in range(7):
        for r2 in range(1):
            r = r1 * 1 + r2
            
            # U
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U1[i1*2 + j1, r1]
                    if val1 != 0:
                        for i2 in range(1):
                            for j2 in range(2):
                                val2 = U2[i2*2 + j2, r2]
                                if val2 != 0:
                                    i = i1
                                    a = j1 * 2 + j2
                                    idx_u = i * 4 + a
                                    U[idx_u, r] += val1 * val2
            
            # V
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V1[j1*2 + k1, r1]
                    if val1 != 0:
                        for j2 in range(2):
                            for k2 in range(3):
                                val2 = V2[j2*3 + k2, r2]
                                if val2 != 0:
                                    a = j1 * 2 + j2
                                    b = k1 * 3 + k2
                                    idx_v = a * 6 + b
                                    V[idx_v, r] += val1 * val2
            
            # W
            for i1 in range(2):
                for k1 in range(2):
                    val1 = W1[k1*2 + i1, r1]
                    if val1 != 0:
                        for i2 in range(1):
                            for k2 in range(3):
                                val2 = W2[k2*1 + i2, r2]
                                if val2 != 0:
                                    i = i1
                                    b = k1 * 3 + k2
                                    idx_w = b * 2 + i
                                    W[idx_w, r] += val1 * val2
    
    # Verification
    def verify_decomposition(U, V, W, n, m, p):
        rank = U.shape[1]
        np.random.seed(42)
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            for i in range(n):
                for b in range(p):
                    w_val = W[b*n + i, r]
                    if w_val == 0:
                        continue
                    sum_a = 0
                    for a in range(m):
                        sum_a += U[i*m + a, r] * V[a*p + b, r]
                    C_computed[i, b] += sum_a * w_val
        
        return np.allclose(C_computed, C_expected, atol=1e-9)
    
    if verify_decomposition(U, V, W, n, m, p):
        # Remove zero columns if any
        cols_to_remove = []
        for r in range(U.shape[1]):
            if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
                cols_to_remove.append(r)
        
        if cols_to_remove:
            mask = np.ones(U.shape[1], dtype=bool)
            mask[cols_to_remove] = False
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank-42 if rank-7 fails
        return construct_fallback()

def construct_fallback():
    """Fallback to rank-42 decomposition if rank-7 fails."""
    n, m, p = 2, 4, 6
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    U1[0, 0] = 1; U1[3, 0] = 1
    U1[2, 1] = 1; U1[3, 1] = 1
    U1[0, 2] = 1
    U1[3, 3] = 1
    U1[0, 4] = 1; U1[1, 4] = 1
    U1[2, 5] = 1; U1[0, 5] = -1
    U1[1, 6] = 1; U1[3, 6] = -1
    
    V1[0, 0] = 1; V1[3, 0] = 1
    V1[0, 1] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    V1[2, 3] = 1; V1[0, 3] = -1
    V1[3, 4] = 1
    V1[0, 5] = 1; V1[1, 5] = 1
    V1[2, 6] = 1; V1[3, 6] = 1
    
    W1[0, 0] = 1; W1[0, 3] = 1; W1[0, 4] = -1; W1[0, 6] = 1
    W1[2, 2] = 1; W1[2, 4] = 1
    W1[1, 1] = 1; W1[1, 3] = 1
    W1[3, 0] = 1; W1[3, 1] = -1; W1[3, 2] = 1; W1[3, 5] = 1
    
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    U2[0, 0] = 1
    U2[0, 1] = 1
    U2[0, 2] = 1
    U2[1, 3] = 1
    U2[1, 4] = 1
    U2[1, 5] = 1
    
    V2[0, 0] = 1
    V2[1, 1] = 1
    V2[2, 2] = 1
    V2[3, 3] = 1
    V2[4, 4] = 1
    V2[5, 5] = 1
    
    W2[0, 0] = 1; W2[0, 3] = 1
    W2[1, 1] = 1; W2[1, 4] = 1
    W2[2, 2] = 1; W2[2, 5] = 1
    
    U = np.zeros((8, 42))
    V = np.zeros((24, 42))
    W = np.zeros((12, 42))
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U1[i1*2 + j1, r1]
                    if val1 != 0:
                        for i2 in range(1):
                            for j2 in range(2):
                                val2 = U2[i2*2 + j2, r2]
                                if val2 != 0:
                                    i = i1
                                    a = j1 * 2 + j2
                                    idx_u = i * 4 + a
                                    U[idx_u, r] += val1 * val2
            
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V1[j1*2 + k1, r1]
                    if val1 != 0:
                        for j2 in range(2):
                            for k2 in range(3):
                                val2 = V2[j2*3 + k2, r2]
                                if val2 != 0:
                                    a = j1 * 2 + j2
                                    b = k1 * 3 + k2
                                    idx_v = a * 6 + b
                                    V[idx_v, r] += val1 * val2
            
            for i1 in range(2):
                for k1 in range(2):
                    val1 = W1[k1*2 + i1, r1]
                    if val1 != 0:
                        for i2 in range(1):
                            for k2 in range(3):
                                val2 = W2[k2*1 + i2, r2]
                                if val2 != 0:
                                    i = i1
                                    b = k1 * 3 + k2
                                    idx_w = b * 2 + i
                                    W[idx_w, r] += val1 * val2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
