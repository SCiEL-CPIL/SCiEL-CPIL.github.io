import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def construct():
    # Strassen 2x2x2 decomposition
    # U, V are 4x7, W is 4x7
    # Indices 0,1,2,3 for 2x2 matrix
    # U[0,0]=1, U[0,1]=1, U[0,2]=0, U[0,3]=0, U[0,4]=0, U[0,5]=0, U[0,6]=0
    # ...
    
    U2 = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0]
    ], dtype=int)
    
    V2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0]
    ], dtype=int)
    
    # W is for output C.
    # C00 = M1 + M2 - M4 + M7
    # C01 = M3 + M4
    # C10 = M2 - M4 - M5 + M6
    # C11 = M1 + M3 + M5 - M7
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, -1]
    ], dtype=int)
    
    # Verify 2x2
    # T_{i,j,k} = sum_r U2[i,r] V2[j,r] W2[k,r]
    
    # Kronecker to 4x4x4
    # Rank 49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # U4 is 16x49
    # V4 is 16x49
    # W4 is 16x49 (but we only need first 16 rows for 4x4x4, wait W is p x rank)
    # W4 indices 0..15 correspond to k in 0..3.
    
    # Now extend to 4x4x5
    # We need W to be 20x50 (or 20x49)
    # Let's try rank 50.
    rank = 50
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((20, rank), dtype=int)
    
    # Copy first 49
    U[:, :49] = U4
    V[:, :49] = V4
    W[:16, :49] = W4
    # W[16:20, :49] = 0
    
    # 50th term for the 5th column (index 4 in k, which is 19 in flattened W? No)
    # W indices: k*n + i.
    # For 4x4x5, n=4, p=5.
    # W shape (p*n, rank) = (20, rank).
    # Index r in W corresponds to k*n + i.
    # The 5th column of B is k=4.
    # The output indices for k=4 are 4*4+0=16, 17, 18, 19.
    # So W[16:20, 49] should be set.
    
    # We need T_{i,j,4} = 1 for all i,j.
    # Term 49: U[:, 49] * V[:, 49] * W[:, 49]
    # Set U[:, 49] = 1 (all 16 rows? No, i is 0..3. U is 16x... wait)
    # U index is i*m + j. m=4.
    # U has 16 rows. i=0..3, j=0..3.
    # For the 5th column, we need the sum over j to work.
    # The term is U_{i,j,49} V_{i,j,49} W_{k,i,49}.
    # We want this to contribute 1 to C_{i,4} for all i.
    # C_{i,4} = sum_j A_{i,j} B_{j,4}.
    # The tensor term is 1 for all i,j.
    # So U_{i,j,49} V_{i,j,49} W_{4,i,49} = 1.
    # Let U_{i,j,49} = 1, V_{i,j,49} = 1, W_{4,i,49} = 1.
    # W index is 4*4 + i = 16 + i.
    
    U[:, 49] = 1
    V[:, 49] = 1
    W[16:20, 49] = 1
    # W[0:16, 49] = 0 (already)
    
    return {"U": U, "V": V, "W": W}
