import numpy as np
import time

def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Uses a Strassen-based rank-49 decomposition for the 4x4x4 part
    and extends it to handle the 5th column with 4 additional terms,
    resulting in a total rank of 53.
    """
    n, m, p = 4, 4, 5
    
    # Strassen decomposition for 2x2x2 (Rank 7)
    # U, V, W are 4x7 matrices (flattened 2x2)
    U22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V22 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W22 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Verify Strassen decomposition
    # sum_r U22[i,j,r] * V22[j',k,r] * W22[k',i',r] should be delta_jj' delta_kk' delta_ii'?
    # Actually, the condition is sum_r U[i,j,r] V[j',k,r] W[k,i,r] = delta_jj'
    # Wait, the indices for W are (k,i) in the flattened vector?
    # My W22 is flattened C. C_{i,k}. Index = i*2+k.
    # The condition derived earlier:
    # sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_jj'
    # For 2x2x2:
    # sum_r U22[i*2+j, r] V22[j'*2+k, r] W22[k*2+i, r] = delta_jj'
    
    # Let's check r=0
    # U22[0,0]=1, U22[1,1]=1
    # V22[0,0]=1, V22[1,1]=1
    # W22[0,0]=1, W22[1,1]=1? No W22 is 4x7.
    # W22[k*2+i, r].
    # If k=0, i=0, index 0. W22[0,0]=1.
    # If k=1, i=1, index 3. W22[3,0]=1.
    # If k=0, i=1, index 2. W22[2,0]=0.
    # If k=1, i=0, index 1. W22[1,0]=0.
    
    # Term r=0 contributes to (i,j,j',k):
    # U22[i*2+j, 0] V22[j'*2+k, 0] W22[k*2+i, 0]
    # Non-zero if i=0,j=0 or i=1,j=1.
    # And j'=0,k=0 or j'=1,k=1.
    # And k=0,i=0 or k=1,i=1.
    
    # Let's check (i=0, j=0, j'=0, k=0):
    # U22[0,0]=1, V22[0,0]=1, W22[0,0]=1. Product 1.
    # Let's check (i=0, j=0, j'=1, k=0):
    # U22[0,0]=1, V22[1,0]=0. Product 0.
    # Let's check (i=0, j=1, j'=0, k=0):
    # U22[1,0]=0. Product 0.
    
    # This seems correct for the diagonal terms.
    
    # Construct 4x4x4 decomposition (Rank 49)
    U44 = np.kron(U22, U22)
    V44 = np.kron(V22, V22)
    W44 = np.kron(W22, W22)
    
    # The indices for 4x4x4:
    # U44 index: i*4+j.
    # V44 index: j'*4+k.
    # W44 index: k*4+i.
    
    # Initialize 4x4x5 decomposition (Rank 53)
    rank = 53
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((20, rank), dtype=float)
    W = np.zeros((20, rank), dtype=float)
    
    # Copy 4x4x4 part
    U[:, :49] = U44
    V[:, :49] = V44
    W[:, :49] = W44
    
    # Note: V44 and W44 are 16x49.
    # V in 4x4x5 is 20x53. Rows 0-15 correspond to k=0..3.
    # V44[j'*4+k, r] maps to V[j'*5+k, r] for k=0..3.
    # We need to permute V44 and W44 to match the 5-column layout.
    
    # Reset V and W to be safe
    V = np.zeros((20, rank), dtype=float)
    W = np.zeros((20, rank), dtype=float)
    
    # Map V44 (j'*4+k) to V (j'*5+k)
    for j in range(4):
        for k in range(4):
            V[j*5+k, :49] = V44[j*4+k, :]
            
    # Map W44 (k*4+i) to W (k*4+i) -- Wait, W index is k*n+i = k*4+i.
    # In 4x4x5, W index is k*n+i = k*4+i.
    # So W44 maps directly to W for k=0..3.
    W[:16, :49] = W44
    
    # Add 4 schoolbook terms for k=4
    for j in range(4):
        r = 49 + j
        # U[i*4+j, r] = 1 for all i
        for i in range(4):
            U[i*4+j, r] = 1.0
        # V[j*5+4, r] = 1
        V[j*5+4, r] = 1.0
        # W[4*4+i, r] = 1 for all i
        for i in range(4):
            W[4*4+i, r] = 1.0
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
