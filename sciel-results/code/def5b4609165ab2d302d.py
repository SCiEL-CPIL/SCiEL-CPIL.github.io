import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    # U2[i, j, r], V2[j, k, r], W2[k, i, r]
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Build 4x4x4 factors using tensor product (Kronecker)
    # U4[i, j, R] where R = 7*s + r
    # U4[i, j, 7*s+r] = U2[i//2, j//2, s] * U2[i%2, j%2, r]
    
    # Use broadcasting: U2 reshaped to (2,2,1, 1,1,7) * (1,1,7, 2,2,1)
    # Actually: U4[i1*2+i0, j1*2+j0, 7*s+r] = U2[i1,j1,s] * U2[i0,j0,r]
    
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            # U4[i, j, :] = U2[i1, j1, :] * U2[i0, j0, :] element-wise in the 7-dim sense
            # But we need the Kronecker: for each (s,r), U4[i,j,7*s+r] = U2[i1,j1,s]*U2[i0,j0,r]
            # This is the outer product of U2[i1,j1,:] and U2[i0,j0,:] reshaped
            U4[i, j, :] = (U2[i1, j1, :][:, None] * U2[i0, j0, :][None, :]).flatten()
    
    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            V4[j, k, :] = (V2[j1, k1, :][:, None] * V2[j0, k0, :][None, :]).flatten()
    
    for k in range(4):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            W4[k, i, :] = (W2[k1, i1, :][:, None] * W2[k0, i0, :][None, :]).flatten()

    # Build 4x4x8 factors
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U4_flat = U4.reshape(16, 49)
    V4_flat = V4.reshape(16, 49)
    W4_flat = W4.reshape(16, 49)
    
    # U is duplicated
    U[:, :49] = U4_flat
    U[:, 49:] = U4_flat
    
    # V: V[j*8+k, r]
    # For k in 0..3: V[j*8+k, :49] = V4[j, k, :]
    # For k in 4..7: V[j*8+k, 49:] = V4[j, k-4, :]
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :49] = V4[j, k, :]
            V[j * 8 + (4 + k), 49:] = V4[j, k, :]
    
    # W: W[k*4+i, r]
    # For k in 0..3: W[k*4+i, :49] = W4[k, i, :]
    # For k in 4..7: W[k*4+i, 49:] = W4[k-4, i, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W4[k, i, :]
            W[(4 + k) * 4 + i, 49:] = W4[k, i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
