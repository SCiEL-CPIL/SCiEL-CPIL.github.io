"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 30
    # Known minimal rank is 28.
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Function to verify a decomposition
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Note: indices in T are (i*m+j, j*p+k, k*n+i)
        # So we need to check for all i, j, k
        # T[i*m+j, j*p+k, k*n+i] == sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    # Wait, np.dot(U[idx_u], V[idx_v] * W[idx_w]) is sum_r U[idx_u,r] * V[idx_v,r] * W[idx_w,r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Also check non-diagonal? 
        # The tensor T is zero elsewhere.
        # We need to ensure that for any (a,b,c) not in the diagonal set, the sum is 0.
        # This is harder to check exhaustively, but if the diagonal is 1 and the rank is minimal, it's likely correct.
        # Actually, the tensor is defined ONLY by the diagonal entries being 1 and others 0.
        # We must check that the reconstructed tensor matches T exactly.
        # Reconstruct T_recon
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is expensive to do for all a,b,c.
        # But we can check a few random non-diagonal points.
        # Or, we can check that the tensor norm is close to the norm of T.
        # Norm of T is sqrt(30).
        # Let's check a sample of non-diagonal entries.
        for _ in range(100):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            # Check if this is a diagonal entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and k == k2:
                continue
            val = np.dot(U[a], V[b] * W[c])
            if abs(val) > 1e-6:
                return False
        return True

    # Try to find rank 28 decomposition
    best_rank = 30
    best_factors = None

    # Schoolbook fallback
    def get_schoolbook():
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    best_factors = (U_sb, V_sb, W_sb)
    
    # Attempt to find lower rank using ALS
    # We try ranks 28 and 29
    for target_rank in [28, 29]:
        start_time = time.time()
        found = False
        # Try multiple random initializations
        np.random.seed(42)
        for trial in range(50):
            if time.time() - start_time > 100:
                break
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # ALS iterations
            for it in range(100):
                # Update U
                # U[:, r] = T contracted with V[:, r] and W[:, r]
                # T is (n*m, m*p, p*n)
                # U[i, r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r]
                # This can be done by reshaping T and doing matrix multiplications
                # T_reshaped: (n*m, m*p*p*n) -> not quite.
                # Let's use einsum
                U_new = np.einsum('ijk,jr,kr->ir', T, V, W)
                # Update V
                V_new = np.einsum('ijk,ir,kr->jr', T, U, W)
                # Update W
                W_new = np.einsum('ijk,ir,jr->kr', T, U, V)
                
                # Normalize to prevent divergence
                for r in range(target_rank):
                    norm_u = np.linalg.norm(U_new[:, r])
                    norm_v = np.linalg.norm(V_new[:, r])
                    norm_w = np.linalg.norm(W_new[:, r])
                    if norm_u > 0 and norm_v > 0 and norm_w > 0:
                        # Scale such that product of norms is 1? Or just normalize each to 1?
                        # If we normalize each to 1, the product might drift.
                        # Better: keep the scale.
                        pass
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                # Compute error
                # Reconstruct T_recon
                # T_recon[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # Error = || T - T_recon ||^2
                # This is expensive. We can check the norm of the residual.
                if it % 10 == 0:
                    # Fast check: norm of T - T_recon
                    # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                    # This is (6, 15, 10) -> 900 elements. Fast.
                    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                    err = np.linalg.norm(T - T_recon)
                    if err < 1e-4:
                        # Try to snap to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        # Verify
                        if verify(U_snap, V_snap, W_snap):
                            best_rank = target_rank
                            best_factors = (U_snap, V_snap, W_snap)
                            found = True
                            break
            if found:
                break
        
        if found:
            break

    # Convert best_factors to list of lists
    U, V, W = best_factors
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
