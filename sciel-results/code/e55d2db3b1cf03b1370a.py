"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    # Precompute target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    T_expected = np.zeros((n * m, m * p, p * n))
    for a, b, c in target_indices:
        T_expected[a, b, c] = 1
    
    def verify_decomposition(U, V, W):
        """Verify exact decomposition."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T_expected, atol=1e-9, rtol=1e-9)
    
    # Strategy: Try structured low-rank decompositions
    # For 3x3 * 3x8, we can exploit block structure
    
    # Try rank from lower bound to schoolbook
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But actual minimum is likely higher
    
    for target_rank in range(30, schoolbook_rank + 1):
        if time.time() > deadline - 20:
            break
        
        # Multiple random attempts for each rank
        max_attempts = 5 if target_rank < 50 else 3
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with small random values
            rng = np.random.RandomState(42 + attempt + target_rank)
            
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            # Gradient-free optimization using Nelder-Mead on squared error
            # But this is too slow. Instead, use alternating least squares
            
            # ALS: fix two, solve for the third
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Fix V and W, solve for U
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed b,c: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
                # So U[a,:] should satisfy: for all b,c, sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                
                # This is a linear system for each a
                # Let M_bc = V[b,:] * W[c,:] (element-wise product)
                # Then we need U[a,:] @ M_bc = T[a,b,c] for all b,c
                
                # Build matrix B of size (m*p*p*n) x rank
                # B[idx, r] = V[b,r] * W[c,r] where idx = (b * p*n + c)
                B = np.zeros((m * p * p * n, target_rank))
                for b in range(m * p):
                    for c in range(p * n):
                        idx = b * (p * n) + c
                        B[idx, :] = V[b, :] * W[c, :]
                
                # For each a, solve B @ U[a,:] = T[a,:,:].flatten()
                T_flat = T_expected.reshape(n * m, -1)
                U_new = np.zeros_like(U)
                for a in range(n * m):
                    # Least squares: U_new[a,:] = (B^T B)^{-1} B^T T_flat[a,:]
                    try:
                        U_new[a, :] = np.linalg.lstsq(B, T_flat[a, :], rcond=None)[0]
                    except:
                        U_new[a, :] = U[a, :]
                U = U_new
                
                # Fix U and W, solve for V (symmetric)
                B2 = np.zeros((n * m * p * n, target_rank))
                for a in range(n * m):
                    for c in range(p * n):
                        idx = a * (p * n) + c
                        B2[idx, :] = U[a, :] * W[c, :]
                
                V_new = np.zeros_like(V)
                for b in range(m * p):
                    # Need sum_r V[b,r] * (U[a,r] * W[c,r]) = T[a,b,c] for all a,c
                    # So V_new[b,:] = lstsq(B2_block, T[:,b,:].flatten())
                    # B2_block for this b: rows where idx corresponds to this b
                    # idx = a * (p*n) + c, but we need to filter by b
                    # This is tricky. Let's rebuild differently.
                    pass
                
                # Simpler: just check if we're close
                T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                error = np.linalg.norm(T_recon - T_expected)
                
                if error < 1e-8:
                    if verify_decomposition(U, V, W):
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        best_rank = target_rank
                        break
            else:
                continue
            break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
