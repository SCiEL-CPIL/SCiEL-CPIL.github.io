import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i

    start_time = time.time()
    deadline = start_time + 110  # 110s budget, leave 10s margin
    
    # First, build the rank-36 decomposition as a starting point
    rank36 = 36
    U36 = np.zeros((n * m, rank36), dtype=np.float64)
    V36 = np.zeros((m * p, rank36), dtype=np.float64)
    W36 = np.zeros((p * n, rank36), dtype=np.float64)
    
    col = 0
    def strassen_block(y, z, w):
        nonlocal col
        y00,y01,y10,y11 = y
        z00,z01,z10,z11 = z
        w00,w01,w10,w11 = w
        U36[u_idx(*y00)][col]+=1; U36[u_idx(*y11)][col]+=1; V36[v_idx(*z00)][col]+=1; V36[v_idx(*z11)][col]+=1; W36[w_idx(*w00)][col]+=1; W36[w_idx(*w11)][col]+=1; col+=1
        U36[u_idx(*y10)][col]+=1; U36[u_idx(*y11)][col]+=1; V36[v_idx(*z00)][col]+=1; W36[w_idx(*w10)][col]+=1; W36[w_idx(*w11)][col]-=1; col+=1
        U36[u_idx(*y00)][col]+=1; V36[v_idx(*z01)][col]+=1; V36[v_idx(*z11)][col]-=1; W36[w_idx(*w01)][col]+=1; W36[w_idx(*w11)][col]+=1; col+=1
        U36[u_idx(*y11)][col]+=1; V36[v_idx(*z10)][col]+=1; V36[v_idx(*z00)][col]-=1; W36[w_idx(*w00)][col]+=1; W36[w_idx(*w10)][col]+=1; col+=1
        U36[u_idx(*y00)][col]+=1; U36[u_idx(*y01)][col]+=1; V36[v_idx(*z11)][col]+=1; W36[w_idx(*w00)][col]-=1; W36[w_idx(*w01)][col]+=1; col+=1
        U36[u_idx(*y10)][col]+=1; U36[u_idx(*y00)][col]-=1; V36[v_idx(*z00)][col]+=1; V36[v_idx(*z01)][col]+=1; W36[w_idx(*w11)][col]+=1; col+=1
        U36[u_idx(*y01)][col]+=1; U36[u_idx(*y11)][col]-=1; V36[v_idx(*z10)][col]+=1; V36[v_idx(*z11)][col]+=1; W36[w_idx(*w00)][col]+=1; col+=1

    def sb_2x2x1(y, z, w):
        nonlocal col
        y00,y01,y10,y11 = y
        z0,z1 = z
        w0,w1 = w
        U36[u_idx(*y00)][col]+=1; V36[v_idx(*z0)][col]+=1; W36[w_idx(*w0)][col]+=1; col+=1
        U36[u_idx(*y01)][col]+=1; V36[v_idx(*z1)][col]+=1; W36[w_idx(*w0)][col]+=1; col+=1
        U36[u_idx(*y10)][col]+=1; V36[v_idx(*z0)][col]+=1; W36[w_idx(*w1)][col]+=1; col+=1
        U36[u_idx(*y11)][col]+=1; V36[v_idx(*z1)][col]+=1; W36[w_idx(*w1)][col]+=1; col+=1

    y1=[(0,0),(0,1),(1,0),(1,1)]; z1=[(0,0),(0,1),(1,0),(1,1)]; w1=[(0,0),(1,0),(0,1),(1,1)]
    y2=[(0,2),(0,3),(1,2),(1,3)]; z2=[(2,0),(2,1),(3,0),(3,1)]; w2=[(0,0),(1,0),(0,1),(1,1)]
    y3=[(0,0),(0,1),(1,0),(1,1)]; z3=[(0,2),(0,3),(1,2),(1,3)]; w3=[(2,0),(3,0),(2,1),(3,1)]
    y4=[(0,2),(0,3),(1,2),(1,3)]; z4=[(2,2),(2,3),(3,2),(3,3)]; w4=[(2,0),(3,0),(2,1),(3,1)]
    y5=[(0,0),(0,1),(1,0),(1,1)]; z5=[(0,4),(1,4)]; w5=[(4,0),(4,1)]
    y6=[(0,2),(0,3),(1,2),(1,3)]; z6=[(2,4),(3,4)]; w6=[(4,0),(4,1)]
    
    strassen_block(y1,z1,w1)
    strassen_block(y2,z2,w2)
    strassen_block(y3,z3,w3)
    strassen_block(y4,z4,w4)
    sb_2x2x1(y5,z5,w5)
    sb_2x2x1(y6,z6,w6)
    
    # Now try to reduce to rank 35
    # Idea: try to express one column as a linear combination of others
    # For each column r, try to find alpha such that:
    #   sum_r' != r  alpha_{r'} * (U[:,r'] ⊗ V[:,r'] ⊗ W[:,r']) = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # This is a system of 40 equations in 34 unknowns (alpha_{r'} for r' != r)
    
    # Build the tensor from the decomposition
    def tensor_from_decomp(U, V, W):
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        return T
    
    T = tensor_from_decomp(U36, V36, W36)
    
    # Check that T is all ones
    # For each candidate column to remove, set up the linear system
    target_rank = 35
    found = False
    best_result = None
    
    for remove_r in range(rank36):
        if time.time() > deadline:
            break
        
        # Try to find alpha such that column r is a linear combo of others
        # T[i,j,k] = sum_{r' != r} alpha_{r'} * U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r']
        # + U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We want: sum_{r' != r} alpha_{r'} * U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] 
        #          = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # This is a linear system in alpha
        # A[i,j,k, r'] = U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] for r' != r
        # b[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        keep_cols = [c for c in range(rank36) if c != remove_r]
        A = np.zeros((n*m*p, len(keep_cols)))
        b = np.zeros(n*m*p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ci, cr in enumerate(keep_cols):
                        A[idx, ci] = U36[i*m+j, cr] * V36[j*p+k, cr] * W36[k*n+i, cr]
                    b[idx] = U36[i*m+j, remove_r] * V36[j*p+k, remove_r] * W36[k*n+i, remove_r]
                    idx += 1
        
        # Solve A @ alpha = b
        try:
            alpha, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if the solution is good
            residual = np.linalg.norm(A @ alpha - b)
            if residual < 1e-8:
                # Great! We can remove column remove_r
                # New decomposition has rank 35
                U_new = U36[:, keep_cols].copy()
                V_new = V36[:, keep_cols].copy()
                W_new = W36[:, keep_cols].copy()
                
                # Verify
                T_new = tensor_from_decomp(U_new, V_new, W_new)
                if np.allclose(T_new, 1.0, atol=1e-8):
                    found = True
                    best_result = (U_new, V_new, W_new)
                    break
        except:
            pass
    
    if found:
        U, V, W = best_result
    else:
        U, V, W = U36, V36, W36
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return result
