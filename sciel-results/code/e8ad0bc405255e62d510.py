import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 140.0

def _build_strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _build_rank98():
    U2, V2, W2 = _build_strassen_2x2x2()
    
    U4 = np.zeros((16, 49), dtype=np.int16)
    V4 = np.zeros((16, 49), dtype=np.int16)
    W4 = np.zeros((16, 49), dtype=np.int16)
    
    # Precompute index mappings
    idx = np.arange(4)
    i0 = idx // 2
    i1 = idx % 2
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            u0 = U2[:, c0]
            u1 = U2[:, c1]
            v0 = V2[:, c0]
            v1 = V2[:, c1]
            w0 = W2[:, c0]
            w1 = W2[:, c1]
            
            for i in range(4):
                for j in range(4):
                    val = u0[i0[i]*2 + i1[j]] * u1[i0[j]*2 + i1[i]]
                    if val != 0:
                        U4[i*4+j, c] = val
            
            for j in range(4):
                for k in range(4):
                    val = v0[i0[j]*2 + i1[k]] * v1[i0[k]*2 + i1[j]]
                    if val != 0:
                        V4[j*4+k, c] = val
            
            for k in range(4):
                for i in range(4):
                    val = w0[i0[k]*2 + i1[i]] * w1[i0[i]*2 + i1[k]]
                    if val != 0:
                        W4[k*4+i, c] = val
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
            V[j*8+4+k, 49:98] = V4[j*4+k, :]
            W[k*4+i, 0:49] = W4[k*4+i, :] if False else None
            for i in range(4):
                W[k*4+i, 0:49] = W4[k*4+i, :]
                W[(4+k)*4+i, 49:98] = W4[k*4+i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    rank = 128
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U[i*4+j][col] = 1
                V[j*8+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    try:
        if time.time() < DEADLINE:
            return _build_rank98()
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
