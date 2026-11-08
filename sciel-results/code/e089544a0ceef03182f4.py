import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    m1, m2 = 2, 1
    p1 = 2
    num_blocks = p // p1  # 5
    
    # Strassen decomposition for 2x2x2
    # u_c: 4 entries (i*2+j), v_c: 4 entries (j*2+k), w_c: 4 entries (k*2+i)
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # Term 1: u=E00+E11, v=E00+E11, w=E00+E11
    S_U[0,0]=1; S_U[3,0]=1
    S_V[0,0]=1; S_V[3,0]=1
    S_W[0,0]=1; S_W[3,0]=1
    
    # Term 2: u=E10+E11, v=E00, w=E10
    S_U[2,1]=1; S_U[3,1]=1
    S_V[0,1]=1
    S_W[1,1]=1  # w[0*2+1]=w[1] for C10, k=0,i=1
    
    # Term 3: u=E00, v=E01-E11, w=E01
    S_U[0,2]=1
    S_V[1,2]=1; S_V[3,2]=-1
    S_W[2,2]=1  # w[1*2+0]=w[2] for C01, k=1,i=0
    
    # Term 4: u=E11, v=E10-E00, w=E00+E10
    S_U[3,3]=1
    S_V[2,3]=1; S_V[0,3]=-1
    S_W[0,3]=1; S_W[1,3]=1  # C00 and C10
    
    # Term 5: u=E00+E10, v=E11, w=-E00+E01
    S_U[0,4]=1; S_U[2,4]=1
    S_V[3,4]=1
    S_W[0,4]=-1; S_W[2,4]=1  # -C00 + C01
    
    # Term 6: u=E10-E00, v=E00+E10, w=E11
    S_U[2,5]=1; S_U[0,5]=-1
    S_V[0,5]=1; S_V[2,5]=1
    S_W[3,5]=1  # C11, k=1,i=1
    
    # Term 7: u=E11-E01, v=E01+E11, w=E00+E11
    S_U[3,6]=1; S_U[1,6]=-1
    S_V[1,6]=1; S_V[3,6]=1
    S_W[0,6]=1; S_W[3,6]=1  # C00 and C11
    
    # Verify
    def verify():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for c in range(7):
                        total += S_U[i*2+j,c] * S_V[j*2+k,c] * S_W[k*2+i,c]
                    if abs(total - 1) > 1e-10:
                        return False, (i,j,k,total)
        return True, None
    
    ok, info = verify()
    if not ok:
        # Fall back to schoolbook for 2x2x2
        rank = num_blocks * 8 + 1
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for block in range(num_blocks):
            k_start = block * p1
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[i*m+j][col] = 1
                        V[j*p+(k_start+k)][col] = 1
                        W[(k_start+k)*n+i][col] = 1
                        col += 1
        # Part 2
        for i in range(2):
            U[i*m+2][col] = 1
        for k in range(10):
            V[2*10+k][col] = 1
        for k in range(10):
            for i in range(2):
                W[k*2+i][col] = 1
        return {"U": U, "V": V, "W": W}
    
    # Use Strassen: rank = 5*7 + 1 = 36
    rank = num_blocks * 7 + 1
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for block in range(num_blocks):
        k_start = block * p1
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+j][col] = S_U[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V[j*p+(k_start+k)][col] = S_V[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[(k_start+k)*n+i][col] = S_W[k*2+i][c]
            col += 1
    
    # Part 2: 2x1x10, rank 1
    for i in range(2):
        U[i*m+2][col] = 1
    for k in range(10):
        V[2*10+k][col] = 1
    for k in range(10):
        for i in range(2):
            W[k*2+i][col] = 1
    
    return {"U": U, "V": V, "W": W}
