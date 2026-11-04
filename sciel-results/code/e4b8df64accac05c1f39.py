"""Matrix multiplication tensor decomposition for (2,3,10) using Strassen on (2,2,10) sub-block."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Try to find a rank-50 decomposition by solving for (2,3,2) rank-10 blocks
    # If successful, total rank = 5 * 10 = 50
    # Otherwise fall back to rank-55 Strassen+schoolbook
    
    def find_rank10_232(max_iter=5000, tol=1e-10):
        """Find a rank-10 decomposition of the (2,3,2) multiplication tensor."""
        # Tensor: T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1}
        # Equation: sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        # U: (6, 10), V: (6, 10), W: (4, 10)
        
        best_err = float('inf')
        best_U = best_V = best_W = None
        
        for trial in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Initialize with Strassen-like structure + schoolbook, then perturb
            U = np.zeros((6, 10))
            V = np.zeros((6, 10))
            W = np.zeros((4, 10))
            
            # 7 Strassen terms for (2,2,2) part
            # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
            U[0,0]=1; U[4,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
            # M2 = (A[1,0]+A[1,1])*B[0,0]
            U[3,1]=1; U[4,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
            # M3 = A[0,0]*(B[0,1]-B[1,1])
            U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
            # M4 = A[1,1]*(B[1,0]-B[0,0])
            U[4,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
            # M5 = (A[0,0]+A[0,1])*B[1,1]
            U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
            # M6 = (A[1,1]-A[0,0])*(B[0,0]+B[0,1])
            U[4,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
            # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
            U[1,6]=1; U[4,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[3,6]=-1
            # 3 schoolbook terms for j=2 (we need 4 but will try to absorb one)
            # M8 = A[0,2]*B[2,0]
            U[2,7]=1; V[4,7]=1; W[0,7]=1
            # M9 = A[0,2]*B[2,1]
            U[2,8]=1; V[5,8]=1; W[2,8]=1
            # M10 = A[1,2]*B[2,0]
            U[5,9]=1; V[4,9]=1; W[1,9]=1
            # Missing: A[1,2]*B[2,1] -> C[1,1] (W[3,*])
            # We'll try to absorb this into existing terms
            
            # Add small random perturbation to help escape local minima
            U += np.random.randn(6, 10) * 0.1
            V += np.random.randn(6, 10) * 0.1
            W += np.random.randn(4, 10) * 0.1
            
            # ALS optimization
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Update U: for each (i,j), U[i*3+j,:] = argmin ||T_slice - W_col * V_row||
                # T[i,j,k] = sum_r U[i*3+j,r] * V[j*2+k,r] * W[k*2+i,r]
                # For fixed i,j: T[i,j,k] = sum_r U[i*3+j,r] * (V[j*2+k,r] * W[k*2+i,r])
                # Let B_rk = V[j*2+k,r] * W[k*2+i,r], then T[i,j,:] = U[i*3+j,:] @ B^T
                # So U[i*3+j,:] = T[i,j,:] @ B @ (B^T B)^{-1}
                
                for i in range(2):
                    for j in range(3):
                        # B[r,k] = V[j*2+k, r] * W[k*2+i, r]
                        B = np.zeros((10, 2))
                        for k in range(2):
                            B[:, k] = V[j*2+k, :] * W[k*2+i, :]
                        # T[i,j,k] = 1 for all k
                        t = np.ones(2)
                        # Solve: t = B @ U[i*3+j, :].T  =>  U[i*3+j, :].T = B^+ @ t
                        try:
                            U[i*3+j, :] = np.linalg.lstsq(B, t, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), V[j*2+k,:] = argmin
                for j in range(3):
                    for k in range(2):
                        # T[i,j,k] = sum_r U[i*3+j,r] * V[j*2+k,r] * W[k*2+i,r]
                        # For fixed j,k: T[:,j,k] = V[j*2+k,:] @ C^T where C[r,i] = U[i*3+j,r]*W[k*2+i,r]
                        C = np.zeros((10, 2))
                        for i in range(2):
                            C[:, i] = U[i*3+j, :] * W[k*2+i, :]
                        t = np.ones(2)
                        try:
                            V[j*2+k, :] = np.linalg.lstsq(C, t, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), W[k*2+i,:] = argmin
                for k in range(2):
                    for i in range(2):
                        # T[i,j,k] = sum_r U[i*3+j,r] * V[j*2+k,r] * W[k*2+i,r]
                        # For fixed k,i: T[i,:,k] = W[k*2+i,:] @ D^T where D[r,j] = U[i*3+j,r]*V[j*2+k,r]
                        D = np.zeros((10, 3))
                        for j in range(3):
                            D[:, j] = U[i*3+j, :] * V[j*2+k, :]
                        t = np.ones(3)
                        try:
                            W[k*2+i, :] = np.linalg.lstsq(D, t, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                err = 0.0
                for i in range(2):
                    for j in range(3):
                        for k in range(2):
                            val = sum(U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] for r in range(10))
                            err += (val - 1.0) ** 2
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < tol:
                    break
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    U10, V10, W10, err10 = find_rank10_232()
    
    if U10 is not None and err10 < 1e-6:
        # Found rank-10 (2,3,2) decomposition! Build rank-50 (2,3,10)
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for pair in range(5):
            k0 = pair * 2
            k1 = pair * 2 + 1
            for r in range(10):
                col = pair * 10 + r
                # U: (6, 10) -> U[i*3+j, r]
                for i in range(2):
                    for j in range(3):
                        U[i*3+j][col] = U10[i*3+j][r]
                # V: (6, 10) -> V[j*2+k, r], but our V is (30, 50) with V[j*10+k, col]
                for j in range(3):
                    for k in range(2):
                        V[j*10+k0+k][col] = V10[j*2+k][r]
                # W: (4, 10) -> W[k*2+i, r], our W is (20, 50) with W[k*2+i, col]
                for k in range(2):
                    for i in range(2):
                        W[(k0+k)*2+i][col] = W10[k*2+i][r]
        
        # Verify
        random.seed(42)
        def verify(U, V, W, n, m, p, max_checks=10):
            rank = len(U[0])
            for _ in range(max_checks):
                A = [[random.uniform(-2, 2) for _ in range(m)] for _ in range(n)]
                B = [[random.uniform(-2, 2) for _ in range(p)] for _ in range(m)]
                C = [[0.0]*p for _ in range(n)]
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            C[i][k] += A[i][j] * B[j][k]
                Cc = [[0.0]*p for _ in range(n)]
                for i in range(n):
                    for k in range(p):
                        acc = 0.0
                        for r in range(rank):
                            ua = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                            vb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                            acc += W[k*n+i][r] * ua * vb
                        Cc[i][k] = acc
                for i in range(n):
                    for k in range(p):
                        if abs(C[i][k] - Cc[i][k]) > 1e-6:
                            return False
            return True
        
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-55 Strassen + schoolbook
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        U[0][col]=1; U[4][col]=1; V[k0][col]=1; V[10+k1][col]=1; W[2*k0][col]=1; W[2*k1+1][col]=1; col+=1
        U[3][col]=1; U[4][col]=1; V[k0][col]=1; W[2*k0+1][col]=1; W[2*k1+1][col]=-1; col+=1
        U[0][col]=1; V[k1][col]=1; V[10+k1][col]=-1; W[2*k1][col]=1; W[2*k1+1][col]=1; col+=1
        U[4][col]=1; V[10+k0][col]=1; V[k0][col]=-1; W[2*k0][col]=1; W[2*k0+1][col]=1; col+=1
        U[0][col]=1; U[1][col]=1; V[10+k1][col]=1; W[2*k0][col]=-1; W[2*k1][col]=1; col+=1
        U[4][col]=1; U[0][col]=-1; V[k0][col]=1; V[k1][col]=1; W[2*k1+1][col]=1; col+=1
        U[1][col]=1; U[4][col]=-1; V[10+k0][col]=1; V[10+k1][col]=1; W[2*k0][col]=1; W[2*k1+1][col]=-1; col+=1
    
    for i in range(2):
        for k in range(10):
            U[i*3+2][col]=1; V[20+k][col]=1; W[k*2+i][col]=1; col+=1
    
    assert col == rank
    random.seed(42)
    def verify(U, V, W, n, m, p, max_checks=20):
        rank = len(U[0])
        for _ in range(max_checks):
            A = [[random.uniform(-2, 2) for _ in range(m)] for _ in range(n)]
            B = [[random.uniform(-2, 2) for _ in range(p)] for _ in range(m)]
            C = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C[i][k] += A[i][j] * B[j][k]
            Cc = [[0.0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0.0
                    for r in range(rank):
                        ua = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                        vb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                        acc += W[k*n+i][r] * ua * vb
                    Cc[i][k] = acc
            for i in range(n):
                for k in range(p):
                    if abs(C[i][k] - Cc[i][k]) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
