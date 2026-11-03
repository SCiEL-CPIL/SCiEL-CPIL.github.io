"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s wall-clock

    result = _build_winograd_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p):
            # Try to reduce rank via optimization
            U_opt, V_opt, W_opt, rank_opt = _try_reduce_rank(U, V, W, n, m, p, deadline)
            if rank_opt < rank and _verify_fast(U_opt, V_opt, W_opt, n, m, p):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U is 4x7 (A indices: row-major [i*2+j, c])
    V is 4x7 (B indices: row-major [j*2+k, c])
    W is 4x7 (C indices: row-major [k*2+i, c])
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    # A = [[a, b], [c, d]] indexed as [0,1,2,3] = [a,b,c,d]
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V: B factors
    # B = [[e, f], [g, h]] indexed as [0,1,2,3] = [e,f,g,h]
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W: C factors (W[k*2+i, c])
    # C = [[c00, c01], [c10, c11]] indexed as [0,1,2,3] = [c00,c01,c10,c11]
    W[0, 0] = 1   # m0 in C[0,0] (k=0,i=0)
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    W[2, 2] = 1   # m2 in C[0,1] (k=0,i=1)
    W[2, 4] = 1   # m4 in C[0,1]
    W[1, 1] = 1   # m1 in C[1,0] (k=1,i=0)
    W[1, 3] = 1   # m3 in C[1,0]
    W[3, 0] = 1   # m0 in C[1,1] (k=1,i=1)
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _build_winograd_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    
    Block partition:
    - Rows i: {0,1} (top, width 2) and {2} (bottom, width 1)
    - Cols j: {0,1}, {2,3}, {4} (widths 2, 2, 1)
    - Rows k: {0,1}, {2,3}, {4,5} (widths 2, 2, 2)
    
    Rank calculation:
    Top (i in {0,1}):
      - j_width=2, k_width=2: 2x2x2 Strassen, rank 7. Count: 2*3=6 blocks. 6*7=42
      - j_width=1, k_width=2: 2x1x2 schoolbook, rank 4. Count: 1*3=3 blocks. 3*4=12
      Top subtotal: 54
    Bottom (i=2):
      - j_width=2, k_width=2: 1x2x2 schoolbook, rank 4. Count: 2*3=6 blocks. 6*4=24
      - j_width=1, k_width=2: 1x1x2 schoolbook, rank 2. Count: 1*3=3 blocks. 3*2=6
      Bottom subtotal: 30
    Grand total: 84
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Top 2 rows (i=0,1)
    for j_block in range(3):
        j0 = j_block * 2
        j1 = min(j0 + 2, m)
        j_width = j1 - j0
        
        for k_block in range(3):
            k0 = k_block * 2
            k1 = min(k0 + 2, p)
            k_width = k1 - k0
            
            if j_width == 2 and k_width == 2:
                for c in range(7):
                    U[j0, col] = sU[0, c]
                    U[j0+1, col] = sU[1, c]
                    U[5+j0, col] = sU[2, c]
                    U[5+j0+1, col] = sU[3, c]
                    
                    V[j0*6 + k0, col] = sV[0, c]
                    V[j0*6 + k0+1, col] = sV[1, c]
                    V[(j0+1)*6 + k0, col] = sV[2, c]
                    V[(j0+1)*6 + k0+1, col] = sV[3, c]
                    
                    W[k0*3 + 0, col] = sW[0, c]
                    W[k0*3 + 1, col] = sW[1, c]
                    W[(k0+1)*3 + 0, col] = sW[2, c]
                    W[(k0+1)*3 + 1, col] = sW[3, c]
                    col += 1
            elif j_width == 1 and k_width == 2:
                for i in range(2):
                    for kk in range(2):
                        U[i*5 + j0, col] = 1
                        V[j0*6 + k0 + kk, col] = 1
                        W[(k0 + kk)*3 + i, col] = 1
                        col += 1
            else:
                raise ValueError(f"Unexpected top block: {j_width}x{k_width}")
    
    # Bottom 1 row (i=2)
    for j_block in range(3):
        j0 = j_block * 2
        j1 = min(j0 + 2, m)
        j_width = j1 - j0
        
        for k_block in range(3):
            k0 = k_block * 2
            k1 = min(k0 + 2, p)
            k_width = k1 - k0
            
            if j_width == 2 and k_width == 2:
                for jj in range(2):
                    for kk in range(2):
                        U[10 + j0 + jj, col] = 1
                        V[(j0 + jj)*6 + k0 + kk, col] = 1
                        W[(k0 + kk)*3 + 2, col] = 1
                        col += 1
            elif j_width == 1 and k_width == 2:
                for kk in range(2):
                    U[10 + j0, col] = 1
                    V[j0*6 + k0 + kk, col] = 1
                    W[(k0 + kk)*3 + 2, col] = 1
                    col += 1
            else:
                raise ValueError(f"Unexpected bottom block: {j_width}x{k_width}")
    
    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank of the decomposition using optimization.
    Uses a penalty-based approach to encourage sparsity in the factor matrices.
    """
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Compute the target tensor
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    target_T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Try different target ranks below current
    for target_rank in range(rank - 1, max(rank - 20, 29), -1):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random perturbation of a subset of columns
        np.random.seed(42)
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Select a subset of columns to keep
            cols_to_keep = np.random.choice(rank, target_rank, replace=False)
            cols_to_keep.sort()
            
            U_init = U[:, cols_to_keep].copy()
            V_init = V[:, cols_to_keep].copy()
            W_init = W[:, cols_to_keep].copy()
            
            # Add small random perturbation
            U_init += np.random.randn(*U_init.shape) * 0.01
            V_init += np.random.randn(*V_init.shape) * 0.01
            W_init += np.random.randn(*W_init.shape) * 0.01
            
            # Optimize using alternating least squares
            U_opt, V_opt, W_opt = _als_optimize(U_init, V_init, W_init, target_T, n, m, p, deadline)
            
            # Verify
            if _verify_fast(U_opt, V_opt, W_opt, n, m, p):
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = target_rank
                break
        
        if best_rank < rank:
            break
    
    return best_U, best_V, best_W, best_rank


def _als_optimize(U, V, W, target_T, n, m, p, deadline, max_iters=50):
    """
    Alternating least squares optimization for tensor decomposition.
    Minimize ||U⊗V⊗W - target_T||^2
    """
    r = U.shape[1]
    
    for it in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # Build RHS and matrix for this (i,j) pair
                rhs = np.zeros(r)
                A_mat = np.zeros((r, r))
                for k in range(p):
                    vk = V[j*p+k, :]
                    wk = W[k*n+i, :]
                    rhs += target_T[i,j,k] * vk * wk
                    A_mat += np.outer(vk * wk, vk * wk)
                
                # Solve (A_mat + lambda*I) * u = rhs
                lambda_reg = 1e-6
                try:
                    U[i*m+j, :] = np.linalg.solve(A_mat + lambda_reg * np.eye(r), rhs)
                except np.linalg.LinAlgError:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat + lambda_reg * np.eye(r), rhs, rcond=None)[0]
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                rhs = np.zeros(r)
                A_mat = np.zeros((r, r))
                for i in range(n):
                    ui = U[i*m+j, :]
                    wi = W[k*n+i, :]
                    rhs += target_T[i,j,k] * ui * wi
                    A_mat += np.outer(ui * wi, ui * wi)
                
                lambda_reg = 1e-6
                try:
                    V[j*p+k, :] = np.linalg.solve(A_mat + lambda_reg * np.eye(r), rhs)
                except np.linalg.LinAlgError:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat + lambda_reg * np.eye(r), rhs, rcond=None)[0]
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                rhs = np.zeros(r)
                A_mat = np.zeros((r, r))
                for j in range(m):
                    uj = U[i*m+j, :]
                    vj = V[j*p+k, :]
                    rhs += target_T[i,j,k] * uj * vj
                    A_mat += np.outer(uj * vj, uj * vj)
                
                lambda_reg = 1e-6
                try:
                    W[k*n+i, :] = np.linalg.solve(A_mat + lambda_reg * np.eye(r), rhs)
                except np.linalg.LinAlgError:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat + lambda_reg * np.eye(r), rhs, rcond=None)[0]
    
    return U, V, W

# EVOLVE-BLOCK-END

