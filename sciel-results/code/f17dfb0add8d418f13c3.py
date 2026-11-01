"""Program searching for a low-rank exact tensor decomposition of 2x2 * 2x3 matrix multiplication."""
import numpy as np
import itertools
import random
import time

DIMENSIONS = [2, 2, 3]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 6
PN = P * N   # 6

def _build_T():
    """Build the tensor T[k*N + i, j*P + k, i*M + j] = delta."""
    T = np.zeros((PN, MP, NM), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[k * N + i, j * P + k, i * M + j] = 1.0
    return T

T_tensor = _build_T()

def _check_validity(U, V, W, tol=1e-8):
    """Check if U, V, W exactly decompose the tensor."""
    rank = U.shape[1]
    # T_computed[a,b,c] = sum_r U[c,r] * V[b,r] * W[a,r]
    T2 = W @ V.T  # (PN, MP)
    T_computed = np.einsum('ab,cr->abc', T2, U)  # (PN, MP, NM)
    return np.allclose(T_computed, T_tensor, atol=tol, rtol=tol)

def _schoolbook():
    """Return the rank-12 schoolbook decomposition."""
    rank = N * M * P
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U_s (4,7), V_s (4,7), W_s (4,7) where:
    U_s: input A (2x2), V_s: input B (2x2), W_s: output C (2x2)
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Input indices: 0=a11, 1=a12, 2=a21, 3=a22
    # Output indices: 0=c11, 1=c12, 2=c21, 3=c22
    
    # M1 = (a11+a22)(b11+b22)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1  # c11 += M1, c22 += M1
    
    # M2 = (a21+a22)b11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1  # c21 += M2, c22 -= M2
    
    # M3 = a11(b12-b22)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1  # c12 += M3, c22 += M3
    
    # M4 = a22(b21-b11)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1  # c11 += M4, c21 += M4
    
    # M5 = (a11+a12)b22
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1  # c11 -= M5, c12 += M5
    
    # M6 = (a21-a11)(b11+b12)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1  # c22 += M6
    
    # M7 = (a12-a22)(b21+b22)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1  # c11 += M7
    
    return U_s, V_s, W_s

def _try_strassen_hybrid():
    """
    Combine Strassen (rank 7) for 2x2 block with schoolbook for remaining column.
    A (2x2) * B (2x3) = C (2x3)
    Split B = [B_left (2x2), B_right (2x1)]
    C = [C_left (2x2), C_right (2x1)]
    
    C_left = A * B_left: use Strassen (rank 7)
    C_right = A * B_right: use schoolbook (rank 4)
    
    Total rank: 7 + 4 = 11
    """
    rank = 11
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    
    # Strassen part for columns 0,1 of B (i.e., B_left)
    U_s, V_s, W_s = _strassen_2x2()
    
    # Map Strassen indices to full problem
    # Input A: same indices 0-3
    # Input B: columns 0,1 -> indices 0,1,2,3 in full B (j*P+k, j=0,1; k=0,1)
    # Output C: columns 0,1 -> indices 0,1,2,3 in full C (k*N+i, k=0,1; i=0,1)
    
    for r in range(7):
        # U: input A indices
        U[:, r] = U_s[:, r]
        
        # V: input B, only columns 0,1 (j=0,1; k=0,1)
        # B index = j*P + k = j*3 + k
        # For j=0: k=0 -> 0, k=1 -> 1
        # For j=1: k=0 -> 3, k=1 -> 4
        # V_s indices: 0=b11(j=0,k=0), 1=b12(j=0,k=1), 2=b21(j=1,k=0), 3=b22(j=1,k=1)
        v_map = [0, 1, 3, 4]  # V_s index -> full V index
        for idx_s in range(4):
            if V_s[idx_s, r] != 0:
                V[v_map[idx_s], r] = V_s[idx_s, r]
        
        # W: output C, only columns 0,1 (k=0,1; i=0,1)
        # C index = k*N + i = k*2 + i
        # For k=0: i=0 -> 0, i=1 -> 1
        # For k=1: i=0 -> 2, i=1 -> 3
        # W_s indices: 0=c11(k=0,i=0), 1=c12(k=0,i=1), 2=c21(k=1,i=0), 3=c22(k=1,i=1)
        w_map = [0, 1, 2, 3]  # W_s index -> full W index
        for idx_s in range(4):
            if W_s[idx_s, r] != 0:
                W[w_map[idx_s], r] = W_s[idx_s, r]
    
    # Schoolbook part for column 2 of B (i.e., B_right)
    # A (2x2) * B_col2 (2x1) = C_col2 (2x1)
    # 4 products: a_ij * b_2j
    col_offset = 7
    for i in range(N):  # i = 0,1
        for j in range(M):  # j = 0,1
            # U: input A index i*M + j
            U[i * M + j, col_offset] = 1.0
            # V: input B index j*P + 2 (column 2)
            V[j * P + 2, col_offset] = 1.0
            # W: output C index 2*N + i (column 2, row i)
            W[2 * N + i, col_offset] = 1.0
            col_offset += 1
    
    return U, V, W

def _try_rank10_structured(seed=0):
    """
    Try to find a rank-10 decomposition by modifying the rank-11 Strassen+schoolbook
    solution to merge one pair of terms.
    """
    rng = np.random.RandomState(seed)
    U11, V11, W11 = _try_strassen_hybrid()
    
    # Try to find a linear combination that reduces rank by 1
    # If we can find vectors u, v, w such that adding them to some terms
    # allows us to eliminate one term...
    
    # Alternative: try random perturbations and see if rank drops
    for attempt in range(100):
        U = U11.copy()
        V = V11.copy()
        W = W11.copy()
        
        # Try to merge two schoolbook terms
        # Terms 7,8,9,10 are schoolbook for column 2
        # See if we can express one as a combination of others
        
        # Random approach: try to find a rank-10 solution by optimization
        # This is complex; for now, just return the rank-11 solution
        pass
    
    return None

def _try_random_rank_k(k, max_iter=2000, seed=0, deadline=None):
    """
    Try to find a rank-k decomposition using random initialization and iterative refinement.
    """
    if deadline is not None and time.time() > deadline - 5:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Initialize with schoolbook and project to rank k
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Random projections
    A = rng.randn(12, k)
    B = rng.randn(12, k)
    C = rng.randn(12, k)
    
    U = U_sb @ A
    V = V_sb @ B
    W = W_sb @ C
    
    # Check if this works (it won't in general)
    # Use iterative method: alternate solving for one factor
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Solve for U: T[a,b,c] = sum_r U[c,r] * V[b,r] * W[a,r]
        # For fixed V, W: T[a,b,:] = sum_r W[a,r] * V[b,r] * U[:,r]
        # This is a linear system in U
        
        # Reshape T to (PN*MP, NM) and solve
        T_flat = T_tensor.reshape(PN * MP, NM)
        
        # For each a, b: T[a,b,:] = sum_r W[a,r] * V[b,r] * U[:,r]
        # Let M_ab = W[a,:] * V[b,:] (element-wise), shape (k,)
        # Then T[a,b,:] = M_ab @ U (where U is (NM, k), so M_ab @ U gives (NM,) if we do M_ab @ U.T... no)
        # T[a,b,c] = sum_r M_ab[r] * U[c,r]
        # So T[a,b,:] = (M_ab @ U.T).T = U @ M_ab
        
        # We want: for all a,b: T[a,b,:] = U @ (W[a,:] * V[b,:])
        # Let's set up the linear system:
        # For each (a,b) pair and each c: T[a,b,c] = sum_r U[c,r] * W[a,r] * V[b,r]
        # This is: sum_r U[c,r] * (W[a,r] * V[b,r]) = T[a,b,c]
        # For fixed c: sum_r U[c,r] * M_ab[r] = T[a,b,c]
        # Where M_ab = W[a,:] * V[b,:]
        
        # We have PN*MP*NM equations and NM*k unknowns
        # Let's build the system:
        # Unknowns: U[c,r] for c in 0..NM-1, r in 0..k-1
        # Equations: for each (a,b,c): sum_r U[c,r] * W[a,r] * V[b,r] = T[a,b,c]
        
        # This is a large system; use least squares
        # Build matrix: for each equation (a,b,c), the row has nonzeros at positions (c, r)
        # with value W[a,r] * V[b,r]
        
        # For efficiency, note that for fixed c, the equations are:
        # For each (a,b): sum_r U[c,r] * W[a,r] * V[b,r] = T[a,b,c]
        # Let X_c = U[c,:] (shape k,)
        # Then: for each (a,b): X_c @ (W[a,:] * V[b,:]) = T[a,b,c]
        # This is: M_c @ X_c.T = T[:, :, c].reshape(-1)
        # Where M_c[a,b] = W[a,:] * V[b,:] ... but this is not a standard matrix product
        
        # Actually: sum_r X_c[r] * W[a,r] * V[b,r] = (diag(W[a,:]) @ V[b,:].T @ X_c.T)[0]... no
        # = trace(diag(X_c) @ W[a,:] @ V[b,:].T)... this is getting complex
        
        # Simpler: just use the tensor equation directly
        # T_computed[a,b,c] = sum_r U[c,r] * V[b,r] * W[a,r]
        # Residual: R = T_computed - T_tensor
        # Gradient w.r.t. U[c,r]: V[b,r] * W[a,r] summed appropriately
        
        # Use a simple gradient step
        T2 = W @ V.T  # (PN, MP)
        T_comp = np.einsum('ab,cr->abc', T2, U)
        R = T_comp - T_tensor
        err = np.linalg.norm(R)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            return U, V, W
        
        # Gradient descent step
        lr = 0.01
        # dL/dU[c,r] = sum_{a,b} R[a,b,c] * V[b,r] * W[a,r]
        # = sum_{a,b} R[a,b,c] * V[b,r] * W[a,r]
        # Let's compute: for each r, dU[:,r] = sum_{a,b} R[:,:,r] ... no
        # dU[c,r] = sum_{a,b} R[a,b,c] * V[b,r] * W[a,r]
        # = sum_b (sum_a R[a,b,c] * W[a,r]) * V[b,r]
        # Let S[b,c,r] = sum_a R[a,b,c] * W[a,r]
        # Then dU[c,r] = sum_b S[b,c,r] * V[b,r]
        
        # S = np.einsum('abc,ar->bcr', R, W)  # (MP, NM, k)
        # dU = np.einsum('bcr,br->cr', S, V)  # (NM, k)
        
        S = np.einsum('abc,ar->bcr', R, W)
        dU = np.einsum('bcr,br->cr', S, V)
        U -= lr * dU
        
        # Similarly for V and W
        S2 = np.einsum('abc,cr->abr', R, U)
        dV = np.einsum('abr,ar->br', S2, W)
        V -= lr * dV
        
        S3 = np.einsum('abc,br->acr', R, V)
        dW = np.einsum('acr,cr->ar', S3, U)
        W -= lr * dW
    
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    deadline = time.time() + 120
    
    # First, try the known rank-11 Strassen+schoolbook hybrid
    U11, V11, W11 = _try_strassen_hybrid()
    if _check_validity(U11, V11, W11):
        best_rank = 11
        best_U, best_V, best_W = U11, V11, W11
    else:
        best_rank = 12
        U12, V12, W12 = _schoolbook()
        best_U, best_V, best_W = U12, V12, W12
    
    # Try to find rank-10
    if time.time() < deadline - 30:
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            result = _try_random_rank_k(10, max_iter=500, seed=seed, deadline=deadline)
            if result is not None:
                U10, V10, W10 = result
                if _check_validity(U10, V10, W10):
                    best_rank = 10
                    best_U, best_V, best_W = U10, V10, W10
                    break
    
    # Return the best result
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
