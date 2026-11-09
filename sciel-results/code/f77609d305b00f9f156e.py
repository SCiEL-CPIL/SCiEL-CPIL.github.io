"""Rank-minimal tensor decomposition search for 3x4x6 matrix multiplication."""
import time
import random
import numpy as np
import scipy.optimize as optimize

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS
A_DIM = N * M  # 12
B_DIM = M * P  # 24
C_DIM = P * N  # 18
TENSOR_SIZE = A_DIM * B_DIM * C_DIM  # 576
SCHOOLBOOK_RANK = N * M * P  # 72

def _build_tensor():
    """Build the matrix multiplication tensor T[a,b,c] = 1 iff a=M+i, b=M+j, c=i*N+k."""
    T = np.zeros((A_DIM, B_DIM, C_DIM), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T

def _check_decomposition(U, V, W, T):
    """Check if U,V,W exactly decompose the tensor T."""
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, T, atol=1e-8, rtol=1e-10)

def _snap_to_half_int(x):
    """Snap values to nearest half-integer."""
    return np.round(x * 2) / 2

def _search_3x3_rank(target_rank, deadline, seed_offset=0):
    """
    Search for a rank decomposition of the 3x3x3 matrix multiplication tensor.
    Returns (U, V, W) of shape (9, target_rank) if found, else None.
    """
    # Build 3x3x3 tensor
    N3, M3, P3 = 3, 3, 3
    T3 = np.zeros((N3*M3, M3*P3, P3*N3), dtype=np.float64)
    for i in range(N3):
        for j in range(M3):
            for k in range(P3):
                T3[i*M3+j, j*P3+k, k*N3+i] = 1.0
    
    A3, B3, C3 = N3*M3, M3*P3, P3*N3
    T3_flat = T3.flatten()
    
    best_result = None
    best_err = float('inf')
    
    for attempt in range(50): # Try 50 random starts
        if time.time() > deadline:
            break
            
        rng = random.Random(42 + seed_offset + attempt)
        
        # Initialize with random values
        U = np.random.randn(A3, target_rank)
        V = np.random.randn(B3, target_rank)
        W = np.random.randn(C3, target_rank)
        
        # Flatten parameters for optimization
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_flat = params[:A3*target_rank]
            V_flat = params[A3*target_rank:A3*target_rank + B3*target_rank]
            W_flat = params[A3*target_rank + B3*target_rank:]
            
            U_m = U_flat.reshape(A3, target_rank)
            V_m = V_flat.reshape(B3, target_rank)
            W_m = W_flat.reshape(C3, target_rank)
            
            # Compute decomposition
            decomp = np.einsum('ar,br,cr->abc', U_m, V_m, W_m)
            diff = decomp.flatten() - T3_flat
            return np.sum(diff**2)
        
        # Use L-BFGS-B
        try:
            res = optimize.minimize(objective, params, method='L-BFGS-B', 
                                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                U_opt = res.x[:A3*target_rank].reshape(A3, target_rank)
                V_opt = res.x[A3*target_rank:A3*target_rank + B3*target_rank].reshape(B3, target_rank)
                W_opt = res.x[A3*target_rank + B3*target_rank:].reshape(C3, target_rank)
                
                # Snap to half-integers
                U_snapped = _snap_to_half_int(U_opt)
                V_snapped = _snap_to_half_int(V_opt)
                W_snapped = _snap_to_half_int(W_opt)
                
                # Verify
                if _check_decomposition(U_snapped, V_snapped, W_snapped, T3):
                    return U_snapped, V_snapped, W_snapped
                
                # If snapping fails, check if the raw solution is close enough to snap
                # and if the snapped version is valid
                # Sometimes L-BFGS finds a solution that is integer/half-integer
                # but not perfectly snapped due to floating point
                # We can try to refine the snapped version with a few gradient steps
                # restricted to the nearest half-integers? No, that's discrete.
                
                # Check error of snapped version
                decomp_snapped = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
                err = np.sum((decomp_snapped - T3)**2)
                if err < 1e-6:
                    return U_snapped, V_snapped, W_snapped
                    
        except Exception:
            continue
            
    return None

def _construct_block_decomposition(deadline):
    """
    Construct the 3x4x6 decomposition using block decomposition.
    Block 1: j in 0..2, k in 0..2 (3x3x3)
    Block 2: j = 3, k in 0..5 (3x1x6)
    Block 3: j in 0..2, k in 3..5 (3x3x3)
    """
    T_full = _build_tensor()
    
    # Initialize full matrices with zeros
    U_full = np.zeros((A_DIM, 0))
    V_full = np.zeros((B_DIM, 0))
    W_full = np.zeros((C_DIM, 0))
    
    # --- Block 2: j=3, k in 0..5 (Rank 18) ---
    # a = 4*i + 3
    # b = 6*3 + k = 18 + k
    # c = 3*k + i
    # i in 0..2, k in 0..5
    # This is 18 terms.
    U_b2 = np.zeros((A_DIM, 18))
    V_b2 = np.zeros((B_DIM, 18))
    W_b2 = np.zeros((C_DIM, 18))
    col = 0
    for i in range(3):
        for k in range(6):
            a = 4*i + 3
            b = 18 + k
            c = 3*k + i
            U_b2[a, col] = 1
            V_b2[b, col] = 1
            W_b2[c, col] = 1
            col += 1
            
    U_full = np.hstack([U_full, U_b2])
    V_full = np.hstack([V_full, V_b2])
    W_full = np.hstack([W_full, W_b2])
    
    # --- Block 1: j in 0..2, k in 0..2 (3x3x3) ---
    # a = 4*i + j
    # b = 6*j + k
    # c = 3*k + i
    # i,j,k in 0..2
    # Local indices for 3x3x3:
    # a_loc = 3*i + j
    # b_loc = 3*j + k
    # c_loc = 3*k + i
    
    U_b1, V_b1, W_b1 = _search_3x3_rank(23, deadline, seed_offset=100)
    
    if U_b1 is None:
        # Fallback to schoolbook for this block (Rank 27)
        U_b1 = np.zeros((9, 27))
        V_b1 = np.zeros((9, 27))
        W_b1 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a_loc = 3*i + j
                    b_loc = 3*j + k
                    c_loc = 3*k + i
                    U_b1[a_loc, col] = 1
                    V_b1[b_loc, col] = 1
                    W_b1[c_loc, col] = 1
                    col += 1
        # Map to full indices
        U_b1_full = np.zeros((A_DIM, 27))
        V_b1_full = np.zeros((B_DIM, 27))
        W_b1_full = np.zeros((C_DIM, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = 4*i + j
                    b = 6*j + k
                    c = 3*k + i
                    U_b1_full[a, :] = U_b1[:, :] # This is wrong, U_b1 is already mapped?
                    # No, U_b1 is 9x27. We need to place it into the correct rows of U_full.
                    # Actually, the search returns 9x27. We need to embed it.
                    pass
        # Let's do the embedding properly
        U_b1_embedded = np.zeros((A_DIM, U_b1.shape[1]))
        V_b1_embedded = np.zeros((B_DIM, V_b1.shape[1]))
        W_b1_embedded = np.zeros((C_DIM, W_b1.shape[1]))
        
        for i in range(3):
            for j in range(3):
                a = 4*i + j
                a_loc = 3*i + j
                U_b1_embedded[a, :] = U_b1[a_loc, :]
                
                b = 6*j + k # k is in the loop? No, k is in the tensor definition.
                # The local b index is 3*j + k.
                # We need to map local b to global b.
                # Local b = 3*j + k.
                # Global b = 6*j + k.
                # For a fixed j, local k maps to global k.
                # So for each local b_loc, we can find j, k such that b_loc = 3*j + k.
                # j = b_loc // 3, k = b_loc % 3.
                # Global b = 6*j + k.
                
                for b_loc in range(9):
                    j_loc = b_loc // 3
                    k_loc = b_loc % 3
                    b_glob = 6*j_loc + k_loc
                    V_b1_embedded[b_glob, :] = V_b1[b_loc, :]
                
                for c_loc in range(9):
                    k_loc = c_loc // 3
                    i_loc = c_loc % 3
                    c_glob = 3*k_loc + i_loc
                    W_b1_embedded[c_glob, :] = W_b1[c_loc, :]
        
        U_full = np.hstack([U_full, U_b1_embedded])
        V_full = np.hstack([V_full, V_b1_embedded])
        W_full = np.hstack([W_full, W_b1_embedded])
    else:
        # Embed the found decomposition
        rank_b1 = U_b1.shape[1]
        U_b1_embedded = np.zeros((A_DIM, rank_b1))
        V_b1_embedded = np.zeros((B_DIM, rank_b1))
        W_b1_embedded = np.zeros((C_DIM, rank_b1))
        
        for i in range(3):
            for j in range(3):
                a = 4*i + j
                a_loc = 3*i + j
                U_b1_embedded[a, :] = U_b1[a_loc, :]
                
        for b_loc in range(9):
            j_loc = b_loc // 3
            k_loc = b_loc % 3
            b_glob = 6*j_loc + k_loc
            V_b1_embedded[b_glob, :] = V_b1[b_loc, :]
            
        for c_loc in range(9):
            k_loc = c_loc // 3
            i_loc = c_loc % 3
            c_glob = 3*k_loc + i_loc
            W_b1_embedded[c_glob, :] = W_b1[c_loc, :]
            
        U_full = np.hstack([U_full, U_b1_embedded])
        V_full = np.hstack([V_full, V_b1_embedded])
        W_full = np.hstack([W_full, W_b1_embedded])

    # --- Block 3: j in 0..2, k in 3..5 (3x3x3) ---
    # a = 4*i + j
    # b = 6*j + k
    # c = 3*k + i
    # i,j in 0..2, k in 3..5
    # Local indices for 3x3x3:
    # a_loc = 3*i + j
    # b_loc = 3*j + (k-3)
    # c_loc = 3*(k-3) + i
    
    U_b3, V_b3, W_b3 = _search_3x3_rank(23, deadline, seed_offset=200)
    
    if U_b3 is None:
        # Fallback to schoolbook for this block (Rank 27)
        U_b3 = np.zeros((9, 27))
        V_b3 = np.zeros((9, 27))
        W_b3 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a_loc = 3*i + j
                    b_loc = 3*j + k
                    c_loc = 3*k + i
                    U_b3[a_loc, col] = 1
                    V_b3[b_loc, col] = 1
                    W_b3[c_loc, col] = 1
                    col += 1
        
        U_b3_embedded = np.zeros((A_DIM, 27))
        V_b3_embedded = np.zeros((B_DIM, 27))
        W_b3_embedded = np.zeros((C_DIM, 27))
        
        for i in range(3):
            for j in range(3):
                a = 4*i + j
                a_loc = 3*i + j
                U_b3_embedded[a, :] = U_b3[a_loc, :]
                
        for b_loc in range(9):
            j_loc = b_loc // 3
            k_loc = b_loc % 3
            k_glob = 3 + k_loc
            b_glob = 6*j_loc + k_glob
            V_b3_embedded[b_glob, :] = V_b3[b_loc, :]
            
        for c_loc in range(9):
            k_loc = c_loc // 3
            i_loc = c_loc % 3
            k_glob = 3 + k_loc
            c_glob = 3*k_glob + i_loc
            W_b3_embedded[c_glob, :] = W_b3[c_loc, :]
            
        U_full = np.hstack([U_full, U_b3_embedded])
        V_full = np.hstack([V_full, V_b3_embedded])
        W_full = np.hstack([W_full, W_b3_embedded])
    else:
        rank_b3 = U_b3.shape[1]
        U_b3_embedded = np.zeros((A_DIM, rank_b3))
        V_b3_embedded = np.zeros((B_DIM, rank_b3))
        W_b3_embedded = np.zeros((C_DIM, rank_b3))
        
        for i in range(3):
            for j in range(3):
                a = 4*i + j
                a_loc = 3*i + j
                U_b3_embedded[a, :] = U_b3[a_loc, :]
                
        for b_loc in range(9):
            j_loc = b_loc // 3
            k_loc = b_loc % 3
            k_glob = 3 + k_loc
            b_glob = 6*j_loc + k_glob
            V_b3_embedded[b_glob, :] = V_b3[b_loc, :]
            
        for c_loc in range(9):
            k_loc = c_loc // 3
            i_loc = c_loc % 3
            k_glob = 3 + k_loc
            c_glob = 3*k_glob + i_loc
            W_b3_embedded[c_glob, :] = W_b3[c_loc, :]
            
        U_full = np.hstack([U_full, U_b3_embedded])
        V_full = np.hstack([V_full, V_b3_embedded])
        W_full = np.hstack([W_full, W_b3_embedded])

    # Verify the full decomposition
    if _check_decomposition(U_full, V_full, W_full, T_full):
        return U_full, V_full, W_full
    else:
        return None

def _schoolbook():
    """Return the schoolbook decomposition (rank 72)."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((A_DIM, rank))
    V = np.zeros((B_DIM, rank))
    W = np.zeros((C_DIM, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try block decomposition
    result = _construct_block_decomposition(deadline)
    if result is not None:
        U, V, W = result
        # Verify again
        T_full = _build_tensor()
        if _check_decomposition(U, V, W, T_full):
            U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
            V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
            W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    U, V, W = _schoolbook()
    U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}
