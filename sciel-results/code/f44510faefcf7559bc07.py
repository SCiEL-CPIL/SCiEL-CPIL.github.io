"""
Strategy: Sparse Recovery for Low-Rank Tensor Decomposition
Instead of static block algorithms, we use a compressed sensing approach (L1 minimization 
with sign restrictions) to find a decomposition with minimal rank and coefficients in {0, 0.5, 1}.
We iterate ranks from a known lower bound up to the current best (113).
"""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # Expected: 1 if j==j' and k==k', else 0
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            expected = 1.0 if (j == j2 and k == k2) else 0.0
                            actual = T[a, b, c]
                            if abs(actual - expected) > 1e-9:
                                errors += 1
                                if errors > 10:
                                    return False
    return errors == 0

def _schoolbook(n, m, p):
    """Generate schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _block_decomposition_1_4(n, m, p):
    """
    Re-implementation of the 1+4 block decomposition (rank 113) to serve as baseline.
    This ensures we have a valid fallback.
    """
    # We use a simpler, verified construction for fallback if sparse search fails.
    # Actually, let's just use the schoolbook for now if sparse fails, 
    # but we want to beat 113. 
    # For a robust fallback, I will implement a known good 113-rank decomposition 
    # or just return schoolbook if we can't find better.
    # Given time constraints, returning schoolbook (125) is safe but not an improvement.
    # However, the prompt implies we should try to beat the current best.
    # The current best is 113. If we fail to find < 113, we should return 113.
    # Let's include a hardcoded 113-rank decomposition from the current program's logic 
    # to ensure we don't regress.
    
    # Re-using the logic from the current program for the 113 rank case
    total_rank = 113
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2
    def _add_strassen_2x2_block(u_r0, u_c0, v_r0, v_c0, w_r0, w_c0, c):
        sU = [
            [0,0,0,1],
            [0,0,1,1],
            [1,0,0,0],
            [0,0,0,1]
        ]
        # Wait, Strassen factors need to be consistent. 
        # Let's use the standard Strassen 2x2 factors:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U (A) factors for M1..M7
        # A is 2x2, indices 00, 01, 10, 11
        # M1: A00=1, A11=1
        # M2: A10=1, A11=1
        # M3: A00=1
        # M4: A11=1
        # M5: A00=1, A01=1
        # M6: A10=1, A00=-1
        # M7: A01=1, A11=-1
        
        u_factors = [
            [1, 0, 1, 0], # M1: A00, A11? No, A00=1, A11=1. 
            # Let's map local indices: 0=00, 1=01, 2=10, 3=11
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0],# M6
            [0, 1, 0, -1] # M7
        ]
        
        # V (B) factors
        # M1: B00=1, B11=1
        # M2: B00=1
        # M3: B10=1, B11=-1
        # M4: B01=1, B00=-1
        # M5: B11=1
        # M6: B00=1, B01=1
        # M7: B10=1, B11=1
        v_factors = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 0, 1, -1],# M3
            [-1, 1, 0, 0],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # W (C) factors
        # C = A*B
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M5 + M7
        
        w_factors = [
            [1, 0, 0, 0], # M1 -> C00
            [0, 0, 0, 0], # M2 -> C10
            [0, 1, 0, 0], # M3 -> C01
            [-1, 0, 0, 0],# M4 -> C00
            [0, 1, 0, 0], # M5 -> C01
            [1, 0, 0, 0], # M6 -> C00
            [-1, 0, 0, 1] # M7 -> C00, C11
        ]
        # Correction for W factors based on standard Strassen:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M5 + M7
        
        w_factors_correct = [
            [1, 0, 0, 0], # M1: C00=1, C11=1
            [0, 0, 0, 0], # M2: C10=1
            [0, 1, 0, 0], # M3: C01=1
            [-1, 0, 0, 0],# M4: C00=-1, C10=1
            [0, 1, 0, 0], # M5: C01=1, C11=1
            [1, 0, 0, 0], # M6: C00=1
            [-1, 0, 0, 1] # M7: C00=-1, C11=1
        ]
        
        # Let's just use the known working 113 construction from the previous program
        # to avoid bugs in re-derivation.
        pass 

    # Since re-deriving the 113 rank is error-prone in this short format, 
    # and the sparse search is the new strategy, I will rely on the sparse search 
    # to find a rank <= 113. If it finds 113, great. If it finds < 113, better.
    # If it fails, I will return a schoolbook (125) which is valid but worse.
    # However, the task says "score is the rank". Lower is better.
    # The current best is 113.
    # To ensure I don't regress, I should probably just return the 113 rank solution 
    # if the sparse search doesn't find something better.
    
    # Let's implement a simple 113 rank solution using the block method 
    # from the inspiration, but simplified.
    
    # Actually, the previous program's `_block_decomposition_1_4` works.
    # I will copy-paste the core logic of that function to ensure a 113 baseline.
    
    def _add_strassen_2x2_block(U, V, W, u_r0, u_c0, v_r0, v_c0, w_r0, w_c0, col):
        sU, sV, sW = _strassen_2x2()
        for r in range(7):
            for local_idx in range(4):
                li = local_idx // 2
                lj = local_idx % 2
                u_global = (u_r0 + li) * 5 + (u_c0 + lj)
                v_global = (v_r0 + li) * 5 + (v_c0 + lj)
                w_global = (w_r0 + li) * 5 + (w_c0 + lj)
                U[u_global][col + r] += sU[local_idx][r]
                V[v_global][col + r] += sV[local_idx][r]
                W[w_global][col + r] += sW[local_idx][r]
        return col + 7

    def _add_schoolbook_block(U, V, W, u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, col):
        na = len(u_rows)
        nb = len(u_cols)
        nc = len(v_cols)
        for i in range(na):
            for j in range(nb):
                for k in range(nc):
                    u_idx = u_rows[i] * 5 + u_cols[j]
                    v_idx = v_rows[j] * 5 + v_cols[k]
                    w_idx = w_rows[k] * 5 + w_cols[i]
                    U[u_idx][col] += 1
                    V[v_idx][col] += 1
                    W[w_idx][col] += 1
                    col += 1
        return col

    def _add_4x4_strassen(U, V, W, u_r0, u_c0, v_r0, v_c0, w_r0, w_c0, col):
        col = _add_strassen_2x2_block(U, V, W, u_r0, u_c0, v_r0, v_c0, w_r0, w_c0, col)
        col = _add_strassen_2x2_block(U, V, W, u_r0, u_c0+2, v_r0+2, v_c0, w_r0, w_c0, col)
        col = _add_strassen_2x2_block(U, V, W, u_r0, u_c0, v_r0, v_c0+2, w_r0, w_c0+2, col)
        col = _add_strassen_2x2_block(U, V, W, u_r0, u_c0+2, v_r0+2, v_c0+2, w_r0, w_c0+2, col)
        col = _add_strassen_2x2_block(U, V, W, u_r0+2, u_c0, v_r0, v_c0, w_r0+2, w_c0, col)
        col = _add_strassen_2x2_block(U, V, W, u_r0+2, u_c0+2, v_r0+2, v_c0, w_r0+2, w_c0, col)
        col = _add_strassen_2x2_block(U, V, W, u_r0+2, u_c0, v_r0, v_c0+2, w_r0+2, w_c0+2, col)
        col = _add_strassen_2x2_block(U, V, W, u_r0+2, u_c0+2, v_r0+2, v_c0+2, w_r0+2, w_c0+2, col)
        return col

    # 1+4 Split
    # 1. A11(1x1)*B11(1x1)->C11(1x1) Rank 1
    U[0][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
    
    # 2. A11(1x1)*B12(1x4)->C12(1x4) Rank 4
    col = _add_schoolbook_block(U, V, W, [0], [0], [0], [1,2,3,4], [1,2,3,4], [0], col)
    
    # 3. A12(1x4)*B22(4x4)->C12(1x4) Rank 16
    col = _add_schoolbook_block(U, V, W, [0], [1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4], [0], col)
    
    # 4. A21(4x1)*B11(1x1)->C21(4x1) Rank 4
    col = _add_schoolbook_block(U, V, W, [1,2,3,4], [0], [0], [0], [0], [1,2,3,4], col)
    
    # 5. A22(4x4)*B21(4x1)->C21(4x1) Rank 16
    col = _add_schoolbook_block(U, V, W, [1,2,3,4], [1,2,3,4], [1,2,3,4], [0], [0], [1,2,3,4], col)
    
    # 6. A21(4x1)*B12(1x4)->C22(4x4) Rank 16
    col = _add_schoolbook_block(U, V, W, [1,2,3,4], [0], [0], [1,2,3,4], [1,2,3,4], [1,2,3,4], col)
    
    # 7. A22(4x4)*B22(4x4)->C22(4x4) Rank 56 (Strassen 4x4)
    col = _add_4x4_strassen(U, V, W, 1, 1, 1, 1, 1, 1, col)
    
    assert col == 113, f"Rank mismatch: {col}"
    return U, V, W

def _strassen_2x2():
    """
    Return U, V, W factor matrices for 2x2x2 Strassen decomposition.
    """
    factors = [
        ([1,0,0,1],[1,0,0,1],[1,0,0,0]),  # M1
        ([0,0,1,1],[1,0,0,0],[0,0,0,0]),  # M2
        ([1,0,0,0],[0,0,1,-1],[0,1,0,0]),  # M3
        ([0,0,0,1],[-1,1,0,0],[-1,0,0,0]),  # M4
        ([1,1,0,0],[0,0,0,1],[0,1,0,0]),  # M5
        ([-1,0,1,0],[1,1,0,0],[1,0,0,0]),  # M6
        ([0,1,0,-1],[0,0,1,1],[-1,0,0,1])   # M7
    ]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    for r, (u, v, w) in enumerate(factors):
        for idx in range(4):
            U[idx][r] = u[idx]
            V[idx][r] = v[idx]
            W[idx][r] = w[idx]
            
    return U, V, W

def _try_sparse_recovery(n, m, p, target_rank, time_limit=10.0):
    """
    Attempt to find a decomposition of rank `target_rank` using sparse recovery.
    We solve for U, V, W such that sum_r U[:,r] V[:,r]^T W[:,r]^T = T.
    This is a tensor train/MPS style problem but solved via iterative L1 projection.
    """
    start_time = time.time()
    N = n * m
    M = m * p
    P = p * n
    
    # Initialize with random small values
    U = np.random.randn(N, target_rank) * 0.1
    V = np.random.randn(M, target_rank) * 0.1
    W = np.random.randn(P, target_rank) * 0.1
    
    # Target tensor T (flattened)
    # T[a, b, c] = 1 if (a%m == b//p and b%p == c//n) else 0
    T_flat = np.zeros(N * M * P)
    for a in range(N):
        i = a // m
        j = a % m
        for b in range(M):
            j2 = b // p
            k = b % p
            if j != j2:
                continue
            for c in range(P):
                k2 = c // n
                i2 = c % n
                if k != k2:
                    continue
                T_flat[a * M * P + b * P + c] = 1.0
    
    T_matrix = T_flat.reshape(N * M, P)
    
    for it in range(500):
        if time.time() - start_time > time_limit:
            return None
            
        # Fix U, V -> Solve for W
        # T = U V^T W ... this is not linear in W directly in this form.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Let M_uv[a, r] = U[a, r]
        # Let M_vw[b, r] = V[b, r]
        # T[a,b,c] = sum_r M_uv[a,r] M_vw[b,r] W[c,r]
        
        # This is a bilinear problem. We can use ALS (Alternating Least Squares) 
        # with L1 penalty for sparsity.
        
        # 1. Fix V, W -> Solve for U
        # T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
        # For each a, we have a vector of equations.
        # Actually, let's vectorize.
        # T is N x M x P.
        # Reshape T to (N, M*P).
        # T_mat[a, b*P+c] = sum_r U[a,r] * (V[b,r] W[c,r])
        # Let X[a, r] = U[a, r]
        # Let Y[a, r, b*P+c] ... no.
        
        # Let's use the property:
        # T = sum_r U_r \otimes V_r \otimes W_r
        # This is a tensor of rank R.
        
        # ALS Step 1: Fix V, W. Solve for U.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For fixed a, we want to minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
        # This is a linear least squares problem for U[a, :].
        
        # Let's construct the design matrix for U.
        # For each a, the variables are U[a, 0..R-1].
        # The equations are for all b, c.
        # T[a, b, c] = sum_r U[a, r] * (V[b, r] * W[c, r])
        # Let Z[b, c, r] = V[b, r] * W[c, r]
        # Then T[a, b, c] = sum_r U[a, r] Z[b, c, r]
        
        Z = np.einsum('br,cr->bcr', V, W)
        Z_flat = Z.reshape(M * P, target_rank)
        
        # Solve U[a, :] such that U[a, :] @ Z_flat^T approx T[a, :, :].reshape(M*P)
        T_flat_a = T_flat.reshape(N, M * P)
        
        U_new = np.zeros_like(U)
        for a in range(N):
            # Minimize || T_flat_a[a] - U_new[a] @ Z_flat.T ||^2
            # U_new[a] = argmin ...
            # This is a standard linear regression: y = X theta
            # y = T_flat_a[a] (M*P,)
            # X = Z_flat (M*P, R)
            # theta = U_new[a] (R,)
            # theta = (X^T X)^-1 X^T y
            try:
                theta, residuals, rank, sv = np.linalg.lstsq(Z_flat, T_flat_a[a], rcond=None)
                U_new[a] = theta
            except:
                U_new[a] = U[a]
        
        U = U_new
        
        # 2. Fix U, W -> Solve for V
        # T[a, b, c] = sum_r V[b, r] (U[a, r] W[c, r])
        # Let Z2[a, c, r] = U[a, r] * W[c, r]
        Z2 = np.einsum('ar,cr->acr', U, W)
        Z2_flat = Z2.reshape(N * P, target_rank)
        
        V_new = np.zeros_like(V)
        for b in range(M):
            # We need to map (a, c) to a single index.
            # T_flat is indexed by (a, b, c).
            # For fixed b, we have a set of equations for all a, c.
            # y = T_flat[:, b, :].reshape(N*P)
            # X = Z2_flat (N*P, R)
            y_b = T_flat.reshape(N, M, P)[:, b, :].reshape(N * P)
            try:
                theta, residuals, rank, sv = np.linalg.lstsq(Z2_flat, y_b, rcond=None)
                V_new[b] = theta
            except:
                V_new[b] = V[b]
        
        V = V_new
        
        # 3. Fix U, V -> Solve for W
        # T[a, b, c] = sum_r W[c, r] (U[a, r] V[b, r])
        # Let Z3[a, b, r] = U[a, r] * V[b, r]
        Z3 = np.einsum('ar,br->abr', U, V)
        Z3_flat = Z3.reshape(N * M, target_rank)
        
        W_new = np.zeros_like(W)
        for c in range(P):
            y_c = T_flat.reshape(N, M, P)[:, :, c].reshape(N * M)
            try:
                theta, residuals, rank, sv = np.linalg.lstsq(Z3_flat, y_c, rcond=None)
                W_new[c] = theta
            except:
                W_new[c] = W[c]
        
        W = W_new
        
        # Sparsification / Quantization Step
        # We want entries in {0, 0.5, 1}.
        # Apply soft thresholding and then rounding.
        # Threshold small values to 0.
        U[np.abs(U) < 0.1] = 0
        V[np.abs(V) < 0.1] = 0
        W[np.abs(W) < 0.1] = 0
        
        # Round to nearest 0.5
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Check if solution is exact
        if it % 10 == 0:
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
                
    return None

def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    Tries sparse recovery to find a rank < 113.
    Falls back to 113-rank block decomposition.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Baseline: 113 rank
    try:
        U_base, V_base, W_base = _block_decomposition_1_4(n, m, p)
        if _verify_decomposition(U_base, V_base, W_base, n, m, p):
            best_rank = 113
            best_result = {"U": U_base, "V": V_base, "W": W_base}
        else:
            # Fallback to schoolbook if 113 fails verification (shouldn't happen)
            U_s, V_s, W_s = _schoolbook(n, m, p)
            best_rank = 125
            best_result = {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        U_s, V_s, W_s = _schoolbook(n, m, p)
        best_rank = 125
        best_result = {"U": U_s, "V": V_s, "W": W_s}
        
    # Try to improve using sparse recovery
    # Start from known lower bounds or slightly below best
    # Lower bound for 5x5 is likely around 100-110.
    # Let's try ranks 112 down to 100.
    
    for rank in range(best_rank - 1, 99, -1):
        if time.time() > deadline - 20:
            break
            
        # Allocate time per rank
        time_per_rank = min(10.0, (deadline - time.time()) / 2)
        if time_per_rank < 1.0:
            break
            
        res = _try_sparse_recovery(n, m, p, rank, time_limit=time_per_rank)
        if res is not None:
            U, V, W = res
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                # If we found a lower rank, we can try to go even lower, 
                # but we should stop if we hit a known good low rank or time runs out.
                # Continue loop to try lower ranks.
                continue
        # If failed, try next lower rank
        
    return best_result

# EVOLVE-BLOCK-END
