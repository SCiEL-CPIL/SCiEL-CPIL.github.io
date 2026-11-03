import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[idx_u(i,j), idx_v(j,k), idx_w(k,i)] = 1.0

    def make_rank26():
        R = 26
        U = np.zeros((n*m, R), dtype=np.float64)
        V = np.zeros((m*p, R), dtype=np.float64)
        W = np.zeros((p*n, R), dtype=np.float64)
        
        U[idx_u(0,0),0]=1; U[idx_u(1,1),0]=1
        V[idx_v(0,0),0]=1; V[idx_v(1,1),0]=1
        W[idx_w(0,0),0]=1; W[idx_w(1,1),0]=1
        
        U[idx_u(1,0),1]=1; U[idx_u(1,1),1]=1
        V[idx_v(0,0),1]=1
        W[idx_w(0,1),1]=1; W[idx_w(1,1),1]=-1
        
        U[idx_u(0,0),2]=1
        V[idx_v(0,1),2]=1; V[idx_v(1,1),2]=-1
        W[idx_w(1,0),2]=1; W[idx_w(1,1),2]=1
        
        U[idx_u(1,1),3]=1
        V[idx_v(1,0),3]=1; V[idx_v(0,0),3]=-1
        W[idx_w(0,0),3]=1; W[idx_w(0,1),3]=1
        
        U[idx_u(0,0),4]=1; U[idx_u(0,1),4]=1
        V[idx_v(1,1),4]=1
        W[idx_w(0,0),4]=-1; W[idx_w(1,0),4]=1
        
        U[idx_u(1,0),5]=1; U[idx_u(0,0),5]=-1
        V[idx_v(0,0),5]=1; V[idx_v(0,1),5]=1
        W[idx_w(1,1),5]=1
        
        U[idx_u(0,1),6]=1; U[idx_u(1,1),6]=-1
        V[idx_v(1,0),6]=1; V[idx_v(1,1),6]=1
        W[idx_w(0,0),6]=1
        
        border_elements = [
            (0,0,2),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
            (1,0,2),(1,1,2),(1,2,0),(1,2,1),(1,2,2),
            (2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),
            (2,2,0),(2,2,1),(2,2,2)
        ]
        col = 7
        for (i,j,k) in border_elements:
            U[idx_u(i,j),col]=1
            V[idx_v(j,k),col]=1
            W[idx_w(k,i),col]=1
            col += 1
        return U, V, W

    U26, V26, W26 = make_rank26()
    T26 = np.einsum('ar,br,cr->abc', U26, V26, W26)
    
    if not np.allclose(T26, target, atol=1e-10):
        R27 = 27
        U27 = np.zeros((n*m, R27), dtype=np.float64)
        V27 = np.zeros((m*p, R27), dtype=np.float64)
        W27 = np.zeros((p*n, R27), dtype=np.float64)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j),r]=1
                    V27[idx_v(j,k),r]=1
                    W27[idx_w(k,i),r]=1
                    r += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
