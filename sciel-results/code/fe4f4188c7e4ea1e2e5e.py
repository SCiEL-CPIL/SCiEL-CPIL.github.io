import time
import numpy as np

n, m, p = 2, 4, 6
_START = time.time()
_DEADLINE = _START + 110.0  # Leave 10s safety margin for 120s limit

def _elapsed():
    return time.time() - _START

def _build_tensor():
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _res(U, V, W, T):
    # Compute sum_r U[:,r] V[:,r] W[:,r] and compare to T
    # T is (n*m, m*p, n*p)
    # U is (n*m, R), V is (m*p, R), W is (n*p, R)
    # Result R_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r]
    # Using einsum is efficient
    return np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)

def _check(U, V, W, T, tol=1e-7):
    if _elapsed() > _DEADLINE:
        return False
    return _res(U, V, W, T) < tol

def _strassen_2x2x2_coefficients():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U: (4, 7) for A (2x2)
    V: (4, 7) for B (2x2)
    W: (4, 7) for C (2x2)
    Indices: A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1 # C00 gets +M1, C11 gets +M1
    
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1 # C10 gets +M2
    W[3,1]=1 # C11 gets +M2? No, C11 = M1 - M3 + M6 - M7. C10 = M2 + M7.
    # Let's use standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M2 - M4 + M5
    # C12 = M3 + M4
    # C21 = M2 + M7
    # C22 = M1 - M3 + M6 - M7
    
    # Reset and fill correctly
    U.fill(0); V.fill(0); W.fill(0)
    
    # M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1 # Wait, C21=M2+M7, C22=M1-M3+M6-M7. C11=M1-M2-M4+M5.
    # W row 0 (C11): M1(1), M2(-1), M4(-1), M5(1)
    # W row 1 (C12): M3(1), M4(1)
    # W row 2 (C21): M2(1), M7(1)
    # W row 3 (C22): M1(1), M3(-1), M6(1), M7(-1)
    
    W[0,0]=1; W[0,1]=-1; W[0,3]=-1; W[0,4]=1
    W[1,2]=1; W[1,3]=1
    W[2,1]=1; W[2,6]=1
    W[3,0]=1; W[3,2]=-1; W[3,5]=1; W[3,6]=-1
    
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    
    # M3
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    
    # M4
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    
    # M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    
    # M6
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    
    # M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    
    return U, V, W

def _construct_rank42():
    """
    Construct the rank-42 decomposition.
    Split j into {0,1} and {2,3}.
    Split k into {0,1}, {2,3}, {4,5}.
    For each block, apply Strassen 2x2x2.
    """
    R = 42
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((n*p, R))
    
    U_str, V_str, W_str = _strassen_2x2x2_coefficients()
    
    col_offset = 0
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(3):
            k_start = k_block * 2
            
            # Map Strassen indices to global indices
            # A indices: i * 4 + j. j in {j_start, j_start+1}
            # Strassen A idx: 0->j_start, 1->j_start+1, 2->j_start, 3->j_start+1 (row 1)
            # Actually Strassen U is (4,7) where rows are (0,0),(0,1),(1,0),(1,1).
            # Global A index for (i, j) is i*4 + j.
            # For fixed j_block, we iterate i in {0,1}.
            # Strassen row 0 (i=0, j_local=0) -> Global i=0, j=j_start -> idx 0*4+j_start
            # Strassen row 1 (i=0, j_local=1) -> Global i=0, j=j_start+1 -> idx 0*4+j_start+1
            # Strassen row 2 (i=1, j_local=0) -> Global i=1, j=j_start -> idx 1*4+j_start
            # Strassen row 3 (i=1, j_local=1) -> Global i=1, j=j_start+1 -> idx 1*4+j_start+1
            
            a_indices = [j_start, j_start+1, 4+j_start, 4+j_start+1]
            
            # B indices: j * 6 + k. j in {j_start, j_start+1}, k in {k_start, k_start+1}
            # Strassen V idx: 0->(j0,k0), 1->(j0,k1), 2->(j1,k0), 3->(j1,k1)
            # Global B index for (j, k) is j*6 + k.
            # Strassen row 0 (j_local=0, k_local=0) -> Global j=j_start, k=k_start -> idx j_start*6+k_start
            # Strassen row 1 (j_local=0, k_local=1) -> Global j=j_start, k=k_start+1 -> idx j_start*6+k_start+1
            # Strassen row 2 (j_local=1, k_local=0) -> Global j=j_start+1, k=k_start -> idx (j_start+1)*6+k_start
            # Strassen row 3 (j_local=1, k_local=1) -> Global j=j_start+1, k=k_start+1 -> idx (j_start+1)*6+k_start+1
            
            b_indices = [j_start*6+k_start, j_start*6+k_start+1, 
                         (j_start+1)*6+k_start, (j_start+1)*6+k_start+1]
            
            # C indices: k * 2 + i. k in {k_start, k_start+1}, i in {0,1}
            # Strassen W idx: 0->(i0,k0), 1->(i0,k1), 2->(i1,k0), 3->(i1,k1)
            # Wait, Strassen W is defined for C (2x2).
            # C indices in Strassen: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
            # Global C index for (k, i) is k*2 + i.
            # Strassen row 0 (i=0, k_local=0) -> Global i=0, k=k_start -> idx k_start*2+0
            # Strassen row 1 (i=0, k_local=1) -> Global i=0, k=k_start+1 -> idx (k_start+1)*2+0
            # Strassen row 2 (i=1, k_local=0) -> Global i=1, k=k_start -> idx k_start*2+1
            # Strassen row 3 (i=1, k_local=1) -> Global i=1, k=k_start+1 -> idx (k_start+1)*2+1
            
            c_indices = [k_start*2+0, (k_start+1)*2+0, 
                         k_start*2+1, (k_start+1)*2+1]
            
            # Fill U, V, W
            for r in range(7):
                for s in range(4):
                    U[a_indices[s], col_offset+r] = U_str[s, r]
                    V[b_indices[s], col_offset+r] = V_str[s, r]
                    W[c_indices[s], col_offset+r] = W_str[s, r]
                    
            col_offset += 7
            
    return U, V, W

def _als_optimize(U, V, W, T, target_rank, max_iters=500):
    """
    Try to optimize U, V, W to reduce rank or improve fit.
    """
    if U.shape[1] != target_rank:
        # Truncate or pad
        if U.shape[1] > target_rank:
            U = U[:, :target_rank]
            V = V[:, :target_rank]
            W = W[:, :target_rank]
        else:
            # This shouldn't happen if we start from higher rank
            pass
            
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Update U
        U_new = np.zeros_like(U)
        for a in range(n*m):
            # U[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2
            num = 0
            den = 0
            for b in range(m*p):
                for c in range(n*p):
                    if T[a,b,c] != 0:
                        num += T[a,b,c] * V[b,r] * W[c,r] # Wait, this is not right.
                        # The equation is sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                        # For fixed a, we want to find U[a,:] such that for all b,c:
                        # sum_r U[a,r] (V[b,r] W[c,r]) = T[a,b,c]
                        # This is a linear system for U[a,:].
                        # Let M[a,b,c,r] = V[b,r] W[c,r].
                        # We want sum_r U[a,r] M[a,b,c,r] = T[a,b,c].
                        # This is underdetermined if rank < number of (b,c) pairs.
                        # We can solve least squares for each a.
                        pass
        # This naive loop is too slow. Use vectorization.
        # For each a, we have a system:
        # sum_r U[a,r] * (V[:,r] * W[:,r] flattened?) No.
        # T[a, :, :] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # Let X_a be the matrix where X_a[:, r] = V[:,r] * W[:,r] ? No.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Reshape T[a] to (m*p * n*p).
        # Reshape V and W?
        # This is a tensor train / ALS step.
        
        # Efficient ALS for this specific structure:
        # For fixed a, we want to minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
        # This is equivalent to solving for U[a,:] in:
        # sum_r U[a,r] (V[:,r] * W[:,r] elementwise product flattened?)
        # No, V[b,r] W[c,r] is not a simple outer product of vectors of length R.
        # It is a 3rd order tensor.
        
        # Let's use the property:
        # For fixed a, let M_a be a matrix of size (m*p * n*p) x R.
        # M_a[(b,c), r] = V[b,r] * W[c,r].
        # Then T[a, :, :] flattened is M_a @ U[a, :].
        # We can solve U[a, :] = pinv(M_a) @ T[a, :, :].flat
        
        for a in range(n*m):
            M_a = np.zeros((m*p * n*p, target_rank))
            for b in range(m*p):
                for c in range(n*p):
                    M_a[b * n*p + c, :] = V[b, :] * W[c, :]
            T_a = T[a, :, :].flatten()
            # Solve least squares
            try:
                U[a, :] = np.linalg.lstsq(M_a, T_a, rcond=None)[0]
            except:
                pass

        # Update V
        V_new = np.zeros_like(V)
        for b in range(m*p):
            M_b = np.zeros((n*m * n*p, target_rank))
            for a in range(n*m):
                for c in range(n*p):
                    M_b[a * n*p + c, :] = U[a, :] * W[c, :]
            T_b = T[:, b, :].flatten()
            try:
                V[b, :] = np.linalg.lstsq(M_b, T_b, rcond=None)[0]
            except:
                pass
        V = V_new

        # Update W
        W_new = np.zeros_like(W)
        for c in range(n*p):
            M_c = np.zeros((n*m * m*p, target_rank))
            for a in range(n*m):
                for b in range(m*p):
                    M_c[a * m*p + b, :] = U[a, :] * V[b, :]
            T_c = T[:, :, c].flatten()
            try:
                W[c, :] = np.linalg.lstsq(M_c, T_c, rcond=None)[0]
            except:
                pass
        W = W_new

    return U, V, W

def construct():
    T = _build_tensor()
    
    # 1. Get the known rank-42 solution
    U42, V42, W42 = _construct_rank42()
    best_rank = 42
    best_U, best_V, best_W = U42, V42, W42
    
    # Verify rank 42
    if not _check(U42, V42, W42, T):
        # If construction failed, fallback to schoolbook
        U_s = np.zeros((n*m, n*m*p))
        V_s = np.zeros((m*p, n*m*p))
        W_s = np.zeros((n*p, n*m*p))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, c] = 1
                    V_s[j*p+k, c] = 1
                    W_s[k*n+i, c] = 1
                    c += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # 2. Try to reduce rank using ALS
    # Start from 41 down to 36
    for rank in range(41, 35, -1):
        if _elapsed() > _DEADLINE - 20:
            break
            
        # Initialize from truncated rank-42 solution
        U_init = U42[:, :rank]
        V_init = V42[:, :rank]
        W_init = W42[:, :rank]
        
        # Run ALS
        U_opt, V_opt, W_opt = _als_optimize(U_init, V_init, W_init, T, rank, max_iters=200)
        
        # Check
        if _check(U_opt, V_opt, W_opt, T, tol=1e-5):
            best_rank = rank
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            break # Found lower rank, stop
            
    # 3. Try random initializations for even lower ranks if time permits
    if _elapsed() < _DEADLINE - 30:
        for rank in range(35, 29, -1):
            if _elapsed() > _DEADLINE - 30:
                break
            for trial in range(5):
                if _elapsed() > _DEADLINE - 30:
                    break
                np.random.seed(trial * 100 + rank)
                U_r = np.random.randn(n*m, rank)
                V_r = np.random.randn(m*p, rank)
                W_r = np.random.randn(n*p, rank)
                
                U_opt, V_opt, W_opt = _als_optimize(U_r, V_r, W_r, T, rank, max_iters=100)
                
                if _check(U_opt, V_opt, W_opt, T, tol=1e-5):
                    best_rank = rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    break
            if best_rank == rank:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
