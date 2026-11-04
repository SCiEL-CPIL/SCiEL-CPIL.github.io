import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def construct():
    """
    Rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Built from Kronecker product of Strassen 2x2x2 (rank 7) -> 4x4x4 (rank 49),
    then replicated for the 8-wide output (two independent 4x4x4 blocks).
    """
    # Strassen 2x2x2: U,V,W each (4,7)
    U2 = np.zeros((4, 7), dtype=np.int16)
    V2 = np.zeros((4, 7), dtype=np.int16)
    W2 = np.zeros((4, 7), dtype=np.int16)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # M2 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1; W2[3,1]=-1
    # M3 = A00(B01-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1; W2[3,2]=1
    # M4 = A11(B10-B00)
    U2[3,3]=1; V2[0,3]=-1; V2[2,3]=1; W2[0,3]=1; W2[1,3]=1
    # M5 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U2[0,5]=-1; U2[2,5]=1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1

    # Build 4x4x4 via loop method (rank 49)
    R4 = 49
    U4 = np.zeros((16, R4), dtype=np.int16)
    V4 = np.zeros((16, R4), dtype=np.int16)
    W4 = np.zeros((16, R4), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    
    # Assemble 4x4x8: rank 98
    R8 = 98
    U = np.zeros((16, R8), dtype=np.int16)
    V = np.zeros((32, R8), dtype=np.int16)
    W = np.zeros((32, R8), dtype=np.int16)
    
    # Block 0: output cols 0-3
    U[:, 0:R4] = U4
    for j in range(4):
        for kl in range(4):
            V[j*8+kl, 0:R4] = V4[j*4+kl, :]
    for kl in range(4):
        for i in range(4):
            W[kl*4+i, 0:R4] = W4[kl*4+i, :]
    
    # Block 1: output cols 4-7
    U[:, R4:R8] = U4
    for j in range(4):
        for kl in range(4):
            V[j*8+(4+kl), R4:R8] = V4[j*4+kl, :]
    for kl in range(4):
        for i in range(4):
            W[(4+kl)*4+i, R4:R8] = W4[kl*4+i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
