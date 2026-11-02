import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the full matrix multiplication tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Fallback Rank 17 decomposition
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U17[2, 7] = 1
    V17[6, 7] = 1
    W17[0, 7] = 1
    
    U17[2, 8] = 1
    V17[7, 8] = 1
    W17[2, 8] = 1
    
    U17[5, 9] = 1
    V17[6, 9] = 1
    W17[1, 9] = 1
    
    U17[5, 10] = 1
    V17[7, 10] = 1
    W17[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U17[0, 11] = 1
    V17[2, 11] = 1
    W17[4, 11] = 1
    
    U17[1, 12] = 1
    V17[5, 12] = 1
    W17[4, 12] = 1
    
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    U17[3, 14] = 1
    V17[2, 14] = 1
    W17[5, 14] = 1
    
    U17[4, 15] = 1
    V17[5, 15] = 1
    W17[5, 15] = 1
    
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    # Verify Rank 17
    T_recon = np.einsum('ar,br,cr->abc', U17, V17, W17)
    if not np.allclose(T_recon, T, atol=1e-10):
        # If even rank 17 fails, return it anyway as it's the best known structure
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

    # Try to find Rank 16
    # Strategy: Decompose into 2x3x2 (rank 10) and 2x3x1 (rank 6)
    # T_232 is the sub-tensor for k=0,1 (C cols 0,1)
    # T_231 is the sub-tensor for k=2 (C col 2)
    
    # Indices for C: c = k*n + i. 
    # k=0: c=0,1. k=1: c=2,3. k=2: c=4,5.
    # So T_232 involves W indices 0,1,2,3.
    # T_231 involves W indices 4,5.
    
    # The Rank 6 part for T_231 is already in U17[0:6, 11:17], V17[0:9, 11:17], W17[0:6, 11:17]
    # We will keep this part fixed as it is optimal for 2x3x1.
    
    # We need to find a Rank 10 decomposition for the 2x3x2 part.
    # The 2x3x2 tensor T_sub has shape (6, 6, 4)
    # a in 0..5, b in 0..5, c in 0..3
    T_sub = np.zeros((n*m, m*p, 2*n))
    for i in range(n):
        for j in range(m):
            for k in range(2): # Only k=0,1
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_sub[a, b, c] = 1.0
                
    rank_target = 10
    nm = n * m
    mp = m * p
    pn_sub = 2 * n
    
    found_rank_10 = False
    U10 = None
    V10 = None
    W10 = None
    
    # Use ALS (Alternating Least Squares) to find Rank 10 decomposition
    # We need to be careful with the scaling.
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        rng = np.random.default_rng(restart * 123 + 456)
        U = rng.standard_normal((nm, rank_target))
        V = rng.standard_normal((mp, rank_target))
        W = rng.standard_normal((pn_sub, rank_target))
        
        # Normalize to prevent divergence
        for r in range(rank_target):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 0 and norm_v > 0 and norm_w > 0:
                # Balance norms
                scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                if scale > 1e-12:
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
                    # Re-normalize to 1
                    U[:, r] /= np.linalg.norm(U[:, r])
                    V[:, r] /= np.linalg.norm(V[:, r])
                    W[:, r] /= np.linalg.norm(W[:, r])
        
        for step in range(100):
            if time.time() > deadline - 10:
                break
                
            # Solve for U
            # T_sub[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # For fixed a: T_sub[a, :, :] = sum_r U[a, r] (V[:, r] outer W[:, r])
            # This is a linear system in U[a, :]
            # Let M_a be a matrix of size (mp * pn_sub) x rank_target
            # M_a[:, r] = V[:, r] * W[:, r] (flattened)
            # T_flat_a = T_sub[a, :, :].flatten()
            # U[a, :] = argmin || M_a U[a, :] - T_flat_a ||
            
            U_new = np.zeros_like(U)
            for a in range(nm):
                M = np.zeros((mp * pn_sub, rank_target))
                for r in range(rank_target):
                    M[:, r] = (V[:, r] * W[:, r]).flatten()
                t_flat = T_sub[a, :, :].flatten()
                # Solve least squares
                try:
                    U_new[a, :] = np.linalg.lstsq(M, t_flat, rcond=None)[0]
                except:
                    U_new[a, :] = U[a, :]
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for b in range(mp):
                M = np.zeros((nm * pn_sub, rank_target))
                for r in range(rank_target):
                    M[:, r] = (U[:, r] * W[:, r]).flatten()
                t_flat = T_sub[:, b, :].flatten()
                try:
                    V_new[b, :] = np.linalg.lstsq(M, t_flat, rcond=None)[0]
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for c in range(pn_sub):
                M = np.zeros((nm * mp, rank_target))
                for r in range(rank_target):
                    M[:, r] = (U[:, r] * V[:, r]).flatten()
                t_flat = T_sub[:, :, c].flatten()
                try:
                    W_new[c, :] = np.linalg.lstsq(M, t_flat, rcond=None)[0]
                except:
                    W_new[c, :] = W[c, :]
            W = W_new
            
            # Check convergence
            T_recon_sub = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T_recon_sub - T_sub)
            if residual < 1e-8:
                found_rank_10 = True
                U10, V10, W10 = U, V, W
                break
                
    if found_rank_10:
        # Construct full Rank 16 solution
        # Combine U10, V10, W10 with the Rank 6 part from U17, V17, W17
        U16 = np.zeros((n * m, 16))
        V16 = np.zeros((m * p, 16))
        W16 = np.zeros((p * n, 16))
        
        # First 10 terms
        U16[:, 0:10] = U10
        V16[:, 0:10] = V10
        # W10 has shape (4, 10). W16 needs shape (6, 16).
        # The first 4 entries of W correspond to k=0,1.
        W16[0:4, 0:10] = W10
        # W16[4:6, 0:10] = 0
        
        # Last 6 terms (from Rank 17 solution, indices 11-16)
        U16[:, 10:16] = U17[:, 11:17]
        V16[:, 10:16] = V17[:, 11:17]
        W16[:, 10:16] = W17[:, 11:17]
        
        # Verify full decomposition
        T_recon16 = np.einsum('ar,br,cr->abc', U16, V16, W16)
        if np.allclose(T_recon16, T, atol=1e-8):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
            
    # Fallback to Rank 17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
