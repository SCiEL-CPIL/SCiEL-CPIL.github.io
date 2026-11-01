"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # Strategy: Use Strassen-like rank-7 decomposition for 2x2x2 sub-blocks
    # 6 blocks of 2 columns (12 cols) + 1 block of 1 column
    # Total rank: 6*7 + 4 = 46
    
    # First, find the rank-7 decomposition for a 2x2x2 sub-tensor
    # Sub-tensor S of shape (4,4,4) with local indices:
    # A: 0=(i=0,k=0), 1=(i=0,k=1), 2=(i=1,k=0), 3=(i=1,k=1)
    # B: 0=(k=0,j0), 1=(k=0,j1), 2=(k=1,j0), 3=(k=1,j1)
    # C: 0=(j0,i=0), 1=(j0,i=1), 2=(j1,i=0), 3=(j1,i=1)
    
    S = np.zeros((4, 4, 4), dtype=float)
    S[0, 0, 0] = 1  # (i=0,k=0), (k=0,j0), (j0,i=0)
    S[0, 1, 2] = 1  # (i=0,k=0), (k=0,j1), (j1,i=0)
    S[1, 2, 0] = 1  # (i=0,k=1), (k=1,j0), (j0,i=0)
    S[1, 3, 2] = 1  # (i=0,k=1), (k=1,j1), (j1,i=0)
    S[2, 0, 1] = 1  # (i=1,k=0), (k=0,j0), (j0,i=1)
    S[2, 1, 3] = 1  # (i=1,k=0), (k=0,j1), (j1,i=1)
    S[3, 2, 1] = 1  # (i=1,k=1), (k=1,j0), (j0,i=1)
    S[3, 3, 3] = 1  # (i=1,k=1), (k=1,j1), (j1,i=1)
    
    # Solve for rank-7 decomposition: find U, V, W of shape (4,7)
    # such that sum_t U[:,t] outer V[:,t] outer W[:,t] = S
    # This is a nonlinear system. Use numerical optimization.
    
    # To break scaling symmetry, fix V[0,0]=1, V[1,1]=1, V[2,2]=1, V[3,3]=1
    # and set other V entries to 0 initially, then optimize.
    
    # Actually, let's use a simpler approach: parameterize and use least squares
    # iteratively.
    
    # Initialize with random values
    np.random.seed(42)
    U_sub = np.random.randn(4, 7) * 0.1
    V_sub = np.random.randn(4, 7) * 0.1
    W_sub = np.random.randn(4, 7) * 0.1
    
    # Fix scaling: set first row of V to have norm 1 for each column
    for t in range(7):
        if np.abs(V_sub[0, t]) > 1e-10:
            V_sub[:, t] /= V_sub[0, t]
            V_sub[0, t] = 1.0
    
    # Iterative optimization
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        # Update U: for each t, U[:,t] = argmin ||S - sum_t' U[:,t']*V[:,t']'W[:,t']'||
        # This is linear in U[:,t] for fixed V,W
        # S = sum_t U[:,t] outer (V[:,t] outer W[:,t])
        # Vectorize: S_vec = sum_t U[:,t] ⊗ (V[:,t] ⊙ W[:,t])
        # where ⊙ is element-wise and ⊗ is outer product
        
        # For each t, the contribution is U[:,t] outer (V[:,t] * W[:,t])
        # Let M_t = V[:,t] * W[:,t] (element-wise product, shape (4,4) after outer)
        # Actually, V[:,t] outer W[:,t] is a 4x4 matrix
        # S[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # For fixed V,W, this is linear in U:
        # S[a,b,c] = sum_t U[a,t] * (V[b,t] * W[c,t])
        # So S_vec = U @ G where G[a,t,b,c] = V[b,t]*W[c,t]
        # Reshape: S_flat (64,) = U (4,7) @ G_flat (7, 64/4=16)? No.
        
        # Let me think differently. Fix V and W, solve for U.
        # S[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # For fixed b,c, this is linear in U[:,t]:
        # S[a,b,c] = sum_t U[a,t] * (V[b,t] * W[c,t])
        # So for each (b,c), we have a linear system in U[:,t]:
        # S[:,b,c] = U @ (V[b,:]*W[c,:])
        # where (V[b,:]*W[c,:]) is a 7-vector
        
        # Stack all (b,c): S_reshaped (4, 4, 4) -> for each (b,c), S[:,b,c] (4,)
        # A_reshaped (4, 4*4) where A[:, b*4+c] = V[b,:]*W[c,:]
        # Then S_reshaped (4, 16) = U (4,7) @ A_reshaped.T (7, 16)
        # So U = S_reshaped @ pinv(A_reshaped.T)
        
        A_mat = np.zeros((7, 16))
        for b in range(4):
            for c in range(4):
                A_mat[:, b*4+c] = V_sub[b, :] * W_sub[c, :]
        
        S_reshaped = S.reshape(4, 16)
        U_new = S_reshaped @ np.linalg.pinv(A_mat.T)
        U_sub = U_new
        
        # Similarly update V
        B_mat = np.zeros((7, 16))
        for a in range(4):
            for c in range(4):
                B_mat[:, a*4+c] = U_sub[a, :] * W_sub[c, :]
        S_reshaped2 = S.reshape(4, 4, 4).transpose(1, 0, 2).reshape(4, 16)
        V_new = S_reshaped2 @ np.linalg.pinv(B_mat.T)
        V_sub = V_new
        
        # Update W
        C_mat = np.zeros((7, 16))
        for a in range(4):
            for b in range(4):
                C_mat[:, a*4+b] = U_sub[a, :] * V_sub[b, :]
        S_reshaped3 = S.reshape(4, 4, 4).transpose(2, 0, 1).reshape(4, 16)
        W_new = S_reshaped3 @ np.linalg.pinv(C_mat.T)
        W_sub = W_new
    
    # Verify the decomposition
    reconstructed = np.zeros((4, 4, 4))
    for t in range(7):
        reconstructed += np.einsum('a,b,c->abc', U_sub[:, t], V_sub[:, t], W_sub[:, t])
    
    error = np.max(np.abs(reconstructed - S))
    
    if error > 1e-6:
        # Fallback to schoolbook if Strassen decomposition failed
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][column] = 1
                    V[k * p + j][column] = 1
                    W[j * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the full decomposition using the sub-decomposition
    rank_total = 6 * 7 + 4  # 46
    U_full = np.zeros((n * m, rank_total))
    V_full = np.zeros((m * p, rank_total))
    W_full = np.zeros((p * n, rank_total))
    
    col_offset = 0
    for block in range(6):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # For this block, the local indices map to global as:
        # A local a: i*m+k where i=a//2, k=a%2 (but a is 0..3, so i=a//2, k=a%2)
        # Wait, A local index a: a=0->(i=0,k=0), a=1->(i=0,k=1), a=2->(i=1,k=0), a=3->(i=1,k=1)
        # So global A index = a (since n*m=4 and local is 4)
        
        # B local index b: b=0->(k=0,j=k0), b=1->(k=0,j=k1), b=2->(k=1,j=k0), b=3->(k=1,j=k1)
        # Global B index: k*p+j
        # b=0: k=0, j=k0 -> global = 0*13+k0 = k0
        # b=1: k=0, j=k1 -> global = 0*13+k1 = k1
        # b=2: k=1, j=k0 -> global = 1*13+k0 = 13+k0
        # b=3: k=1, j=k1 -> global = 1*13+k1 = 13+k1
        
        # C local index c: c=0->(j=k0,i=0), c=1->(j=k0,i=1), c=2->(j=k1,i=0), c=3->(j=k1,i=1)
        # Global C index: j*n+i
        # c=0: j=k0, i=0 -> global = k0*2+0 = 2*k0
        # c=1: j=k0, i=1 -> global = k0*2+1 = 2*k0+1
        # c=2: j=k1, i=0 -> global = k1*2+0 = 2*k1
        # c=3: j=k1, i=1 -> global = k1*2+1 = 2*k1+1
        
        for t in range(7):
            # U: local a -> global a (same index)
            for a in range(4):
                if U_sub[a, t] != 0:
                    U_full[a, col_offset + t] = U_sub[a, t]
            
            # V: local b -> global
            b_to_global = {0: k0, 1: k1, 2: 13 + k0, 3: 13 + k1}
            for b in range(4):
                if V_sub[b, t] != 0:
                    V_full[b_to_global[b], col_offset + t] = V_sub[b, t]
            
            # W: local c -> global
            c_to_global = {0: 2*k0, 1: 2*k0+1, 2: 2*k1, 3: 2*k1+1}
            for c in range(4):
                if W_sub[c, t] != 0:
                    W_full[c_to_global[c], col_offset + t] = W_sub[c, t]
        
        col_offset += 7
    
    # Last block: column 12 (single column)
    k0 = 12
    # Schoolbook for single column: 4 terms
    for i in range(n):
        for k in range(m):
            U_full[i * m + k, col_offset] = 1
            V_full[k * p + k0, col_offset] = 1
            W_full[k0 * n + i, col_offset] = 1
            col_offset += 1
    
    # Convert to lists for return
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

