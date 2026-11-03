import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Test with random matrices (Vectorized verification)
    rng = np.random.default_rng(12345)
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape to 3D tensors for easier manipulation if needed, but let's stick to matrix ops
    # U is (n*m, rank) -> (n, m, rank)
    U_3d = U_np.reshape(n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    V_3d = V_np.reshape(m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    for _ in range(3):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        
        # Compute C_dec
        # C_dec[i, k] = sum_c W[k, i, c] * ( sum_j U[i, j, c] * A[i, j] * V[j, k, c] )
        # Note: The index in W is k*n+i, so W_3d[k, i, c] corresponds to W[k*n+i, c]
        
        # Step 1: Compute M[i, j, c] = U[i, j, c] * A[i, j]
        M = U_3d * A[:, :, np.newaxis]
        
        # Step 2: Compute N[j, k, c] = V[j, k, c] * B[j, k]
        N = V_3d * B[:, :, np.newaxis]
        
        # Step 3: For each c, compute S_c[i, k] = sum_j M[i, j, c] * N[j, k, c]
        # This is a batched matrix multiplication where the batch is 'c'
        # M is (n, m, rank), N is (m, p, rank)
        # We want to sum over j (axis 1 of M, axis 0 of N)
        # Result S_c should be (n, p, rank)
        # S_c[i, k, c] = sum_j M[i, j, c] * N[j, k, c]
        
        # np.einsum('ijc,jkc->ikc', M, N)
        S = np.einsum('ijc,jkc->ikc', M, N)
        
        # Step 4: C_dec[i, k] = sum_c S[i, k, c] * W[k, i, c]
        # W_3d is (p, n, rank). We need W[k, i, c].
        # S is (n, p, rank).
        # We need to align indices.
        # C_dec[i, k] = sum_c S[i, k, c] * W_3d[k, i, c]
        
        # Let's transpose S to (p, n, rank) to match W's first two dims?
        # S[i, k, c] -> S_T[k, i, c]
        S_T = np.transpose(S, (1, 0, 2)) # shape (p, n, rank)
        
        # Now element-wise multiply with W_3d (p, n, rank) and sum over c (axis 2)
        C_dec = np.sum(S_T * W_3d, axis=2)
        
        if np.max(np.abs(C - C_dec)) > tol:
            return False
    return True

def _to_list(mat):
    # mat is a list of lists or np array
    # Ensure we handle numpy arrays correctly
    if isinstance(mat, np.ndarray):
        mat = mat.tolist()
    return [[int(round(v)) if abs(v - round(v)) < 1e-9 else (round(v * 2) / 2 if abs(v * 2 - round(v * 2)) < 1e-9 else v) for v in row] for row in mat]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    rng = np.random.default_rng(42)
    
    # Precompute the tensor T flat
    T_flat = np.ones(n * m * p)
    
    # Map (i,j,k) to index
    def idx_ijk(i, j, k):
        return i * mp + j * p + k
        
    # Map (k,i,r) to unknown index
    def uidx_kir(k, i, r, R):
        return (k * n + i) * R + r

    # Try ranks from a reasonable lower bound up
    for r in range(32, min(97, best_rank)):
        if time.time() > deadline - 5:
            break
        
        # Try several random initializations
        max_iters = 50
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_rand = rng.standard_normal((nm, r))
            V_rand = rng.standard_normal((mp, r))
            
            # Construct A matrix
            A = np.zeros((n * m * p, pn * r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = idx_ijk(i, j, k)
                        for c in range(r):
                            uid = uidx_kir(k, i, c, r)
                            A[idx, uid] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
            
            # Solve A W_flat = T_flat
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
                
            # Reshape W_flat
            W_rand = W_flat.reshape((pn, r))
            
            # Check if W is close to integers or half-integers
            W_rounded = np.round(W_rand * 2) / 2
            if np.max(np.abs(W_rand - W_rounded)) < 1e-6:
                # Also check U and V
                U_rounded = np.round(U_rand * 2) / 2
                V_rounded = np.round(V_rand * 2) / 2
                if np.max(np.abs(U_rand - U_rounded)) < 1e-6 and np.max(np.abs(V_rand - V_rounded)) < 1e-6:
                    # Verify
                    U_list = _to_list(U_rounded)
                    V_list = _to_list(V_rounded)
                    W_list = _to_list(W_rounded)
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        break
        
        if best_rank == r:
            break
            
    # Convert final result
    U_final = best["U"]
    V_final = best["V"]
    W_final = best["W"]
    
    # Ensure integer/half-integer entries
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    U_final = clean_matrix(U_final)
    V_final = clean_matrix(V_final)
    W_final = clean_matrix(W_final)
    
    return {"U": U_final, "V": V_final, "W": W_final}
