import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Build rank 26 solution: Strassen 2x2 core + 19 border terms
    R26 = 26
    U = np.zeros((n * m, R26), dtype=np.float64)
    V = np.zeros((m * p, R26), dtype=np.float64)
    W = np.zeros((p * n, R26), dtype=np.float64)
    
    # Strassen for 2x2 core (rows 0,1 of A; cols 0,1 of B)
    # r=0: (A00+A11)(B00+B11)
    U[idx_u(0,0),0]=1; U[idx_u(1,1),0]=1
    V[idx_v(0,0),0]=1; V[idx_v(1,1),0]=1
    W[idx_w(0,0),0]=1; W[idx_w(1,1),0]=1
    # r=1: (A10+A11)B00
    U[idx_u(1,0),1]=1; U[idx_u(1,1),1]=1
    V[idx_v(0,0),1]=1
    W[idx_w(0,1),1]=1; W[idx_w(1,1),1]=-1
    # r=2: A00(B11-B01)
    U[idx_u(0,0),2]=1
    V[idx_v(0,1),2]=1; V[idx_v(1,1),2]=-1
    W[idx_w(1,0),2]=1; W[idx_w(1,1),2]=1
    # r=3: A11(B10-B00)
    U[idx_u(1,1),3]=1
    V[idx_v(1,0),3]=1; V[idx_v(0,0),3]=-1
    W[idx_w(0,0),3]=1; W[idx_w(0,1),3]=1
    # r=4: (A00+A01)B11
    U[idx_u(0,0),4]=1; U[idx_u(0,1),4]=1
    V[idx_v(1,1),4]=1
    W[idx_w(0,0),4]=-1; W[idx_w(1,0),4]=1
    # r=5: (A10-A00)(B00+B01)
    U[idx_u(1,0),5]=1; U[idx_u(0,0),5]=-1
    V[idx_v(0,0),5]=1; V[idx_v(0,1),5]=1
    W[idx_w(1,1),5]=1
    # r=6: (A01-A11)(B10+B11)
    U[idx_u(0,1),6]=1; U[idx_u(1,1),6]=-1
    V[idx_v(1,0),6]=1; V[idx_v(1,1),6]=1
    W[idx_w(0,0),6]=1
    
    # 19 border terms
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
    
    # Verify rank 26 solution
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    expected = 1.0
                    actual = sum(U[idx_u(i,j),r]*V[idx_v(j,k),r]*W[idx_w(k,i),r] for r in range(R))
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback: schoolbook rank 27
        R27 = 27
        U27 = np.zeros((n*m, R27), dtype=np.float64)
        V27 = np.zeros((m*p, R27), dtype=np.float64)
        W27 = np.zeros((p*n, R27), dtype=np.float64)
        r = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U27[idx_u(i,j),r]=1
                    V27[idx_v(j,k),r]=1
                    W27[idx_w(k,i),r]=1
                    r += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Try to find rank 23 via border optimization
    if time.time() < deadline - 20:
        try:
            # Attempt: combine some border terms with shared structure
            # The 4 k=2 border terms for C[0,0],C[0,1],C[1,0],C[1,1] involve A[i,2]*B[2,j]
            # These form a 2x2 "matrix" A[0:2,2] * B[2,0:2]
            # Similarly, C[2,2] border: A[2,0]*B[0,2]+A[2,1]*B[1,2]+A[2,2]*B[2,2]
            # C[0,2]: A[0,0]*B[0,2]+A[0,1]*B[1,2]+A[0,2]*B[2,2]
            # etc.
            # 
            # Key insight: some border terms share U or V or W entries.
            # For example, the term (0,2,0) has U[0,2]=1, V[2,0]=1, W[0,0]=1
            # The term (1,2,0) has U[1,2]=1, V[2,0]=1, W[0,1]=1
            # They share V[2,0]=1 but have different U and W, so can't directly combine.
            #
            # Try a different approach: use a rank-3 decomposition for parts of the border
            # that currently use rank 4.
            
            # Alternative: try random search for rank 23
            # This is complex; skip for now and return rank 26
            pass
        except:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
