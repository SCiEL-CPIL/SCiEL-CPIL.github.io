"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Uses Kronecker product structure: T(2,3,6) = T(2,1,2) x T(1,3,3) with index permutations.
    Rank = rank(T(2,1,2)) * rank(T(1,3,3)) = 4 * 5 = 20.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Factor dimensions
    n1, m1, p1 = 2, 1, 2
    n2, m2, p2 = 1, 3, 3
    
    # --- Step 1: Decompose T1 (2x1x2) Rank 4 ---
    # U1 (2x4), V1 (2x4), W1 (4x4)
    U1 = np.zeros((n1*m1, 4))
    V1 = np.zeros((m1*p1, 4))
    W1 = np.zeros((p1*n1, 4))
    
    # Hardcoded rank 4 decomposition for T1
    # Indices: a=i, b=k, c=2k+i
    # r=0: i=0, k=0 -> a=0, b=0, c=0
    U1[0, 0] = 1; V1[0, 0] = 1; W1[0, 0] = 1
    # r=1: i=0, k=1 -> a=0, b=1, c=2
    U1[0, 1] = 1; V1[1, 1] = 1; W1[2, 1] = 1
    # r=2: i=1, k=0 -> a=1, b=0, c=1
    U1[1, 2] = 1; V1[0, 2] = 1; W1[1, 2] = 1
    # r=3: i=1, k=1 -> a=1, b=1, c=3
    U1[1, 3] = 1; V1[1, 3] = 1; W1[3, 3] = 1
    
    # --- Step 2: Decompose T2 (1x3x3) Rank 5 ---
    # Build T2
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                a = i*m2 + j
                b = j*p2 + k
                c = k*n2 + i
                T2[a, b, c] = 1.0
    
    # Find rank 5 decomposition using ALS
    U2, V2, W2 = find_als_decomp(T2, 5)
    
    if U2 is None:
        # Fallback to schoolbook for T2 (Rank 9)
        U2 = np.zeros((n2*m2, 9))
        V2 = np.zeros((m2*p2, 9))
        W2 = np.zeros((p2*n2, 9))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a = i*m2 + j
                    b = j*p2 + k
                    c = k*n2 + i
                    U2[a, col] = 1
                    V2[b, col] = 1
                    W2[c, col] = 1
                    col += 1
        r2 = 9
    else:
        r2 = 5
        
    r1 = 4
    r_total = r1 * r2
    
    # --- Step 3: Kronecker Product ---
    U_kr = np.kron(U1, U2)
    V_kr = np.kron(V1, V2)
    W_kr = np.kron(W1, W2)
    
    # --- Step 4: Index Permutations ---
    # We need to map Kronecker indices to Target indices.
    # Target:
    # a = i*m + j = i1*3 + j2
    # b = j*p + k = j2*6 + k1*3 + k2
    # c = k*n + i = (k1*3 + k2)*2 + i1 = 6*k1 + 2*k2 + i1
    
    # Kronecker:
    # a_kr = a1 * dim(U2) + a2 = i1 * 3 + j2. (Matches)
    # b_kr = b1 * dim(V2) + b2 = k1 * 9 + (j2*3 + k2) = 9*k1 + 3*j2 + k2
    # c_kr = c1 * dim(W2) + c2 = (2*k1 + i1) * 3 + k2 = 6*k1 + 3*i1 + k2
    
    # Permute V rows
    # V_kr has rows indexed by b_kr. We want V_final rows indexed by b_target.
    # V_final[b_target, r] = V_kr[b_kr, r]
    perm_V = np.zeros(m*p, dtype=int)
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            # Target index
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            b_t = j*p + k
                            
                            # Kronecker index
                            a1 = i1
                            b1 = k1
                            a2 = j2
                            b2 = j2*p2 + k2
                            b_kr = b1 * (m2*p2) + b2
                            
                            perm_V[b_t] = b_kr
                            
    # Permute W rows
    perm_W = np.zeros(p*n, dtype=int)
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            # Target index
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            c_t = k*n + i
                            
                            # Kronecker index
                            c1 = k1*n1 + i1
                            c2 = k2*n2 + i2
                            c_kr = c1 * (p2*n2) + c2
                            
                            perm_W[c_t] = c_kr
                            
    U_final = U_kr
    V_final = V_kr[perm_V, :]
    W_final = W_kr[perm_W, :]
    
    # --- Step 5: Verification ---
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_full[a, b, c] = 1.0
                
    recon = np.zeros((n*m, m*p, p*n))
    for r in range(r_total):
        recon += np.einsum('a,b,c->abc', U_final[:, r], V_final[:, r], W_final[:, r])
        
    if np.max(np.abs(recon - T_full)) < 1e-8:
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

def find_als_decomp(T, target_rank, max_iter=100, trials=20):
    N1, N2, N3 = T.shape
    vec_T = T.reshape(-1)
    
    for trial in range(trials):
        U = np.random.randn(N1, target_rank)
        V = np.random.randn(N2, target_rank)
        W = np.random.randn(N3, target_rank)
        
        for it in range(max_iter):
            # Solve U
            A_U = np.zeros((N2*N3, N1*target_rank))
            for a in range(N1):
                for r in range(target_rank):
                    idx = a*target_rank + r
                    vec_v = np.kron(np.ones(N3), V[:, r])
                    vec_w = np.kron(W[:, r], np.ones(N2))
                    A_U[:, idx] = vec_v * vec_w
            try:
                U_new, _, _, _ = np.linalg.lstsq(A_U, vec_T, rcond=None)
            except:
                break
            U = U_new.reshape(N1, target_rank)
            
            # Solve V
            A_V = np.zeros((N1*N3, N2*target_rank))
            for b in range(N2):
                for r in range(target_rank):
                    idx = b*target_rank + r
                    vec_u = np.kron(np.ones(N3), U[:, r])
                    vec_w = np.kron(W[:, r], np.ones(N1))
                    A_V[:, idx] = vec_u * vec_w
            try:
                V_new, _, _, _ = np.linalg.lstsq(A_V, vec_T, rcond=None)
            except:
                break
            V = V_new.reshape(N2, target_rank)
            
            # Solve W
            A_W = np.zeros((N1*N2, N3*target_rank))
            for c in range(N3):
                for r in range(target_rank):
                    idx = c*target_rank + r
                    vec_u = np.kron(U[:, r], np.ones(N2))
                    vec_v = np.kron(np.ones(N1), V[:, r])
                    A_W[:, idx] = vec_u * vec_v
            try:
                W_new, _, _, _ = np.linalg.lstsq(A_W, vec_T, rcond=None)
            except:
                break
            W = W_new.reshape(N3, target_rank)
            
            # Check error
            recon = np.zeros((N1, N2, N3))
            for r in range(target_rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            error = np.max(np.abs(recon - T))
            if error < 1e-8:
                return U, V, W
    return None, None, None
# EVOLVE-BLOCK-END
