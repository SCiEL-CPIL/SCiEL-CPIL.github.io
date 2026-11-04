"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(result, T, atol=tol)

def _make_rank17():
    """Return the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = [
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _try_block_rank16(deadline):
    """
    Exploit block structure: 2x3 * 3x3 = [row0; row1] * B
    where row0 * B gives first 3 outputs, row1 * B gives last 3.
    
    For each row i (i=0,1), we need a 1x3 x 3x3 decomposition.
    The full tensor has entries T[a,b,c]=1 where a=i*m+j, b=j*p+k, c=k*n+i.
    
    Key insight: For row i, the relevant tensor slice is:
    T[i*m+j, j*p+k, k*n+i] = 1 for j=0,1,2 and k=0,1,2
    
    This is a 3x3x3 tensor (the 3x3 matrix multiplication tensor).
    
    We can try to find a rank-8 decomposition for each row's 3x3x3 tensor,
    then combine them. If the two rows share some rank-1 terms, total rank < 16.
    
    Strategy: 
    1. Find rank-8 decompositions for each 3x3x3 row tensor
    2. Try to align/share terms between the two rows
    3. If sharing is possible, get rank < 16
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    # The 3x3x3 matrix multiplication tensor for a single row
    # For row i: T_row[j, k, l] = 1 where the output index is determined by i
    # Actually, let's think more carefully.
    
    # For the full 2x3 * 3x3 multiplication:
    # A is 2x3: A[i, j] for i in {0,1}, j in {0,1,2}
    # B is 3x3: B[j, k] for j in {0,1,2}, k in {0,1,2}
    # C = A*B is 2x3: C[i, k] = sum_j A[i,j] * B[j,k]
    
    # Tensor T[a, b, c] where:
    # a = i*m + j = i*3 + j (index into A)
    # b = j*p + k = j*3 + k (index into B)  
    # c = k*n + i = k*2 + i (index into C)
    
    # For row i=0: a = j, b = j*3+k, c = k*2
    #   T[j, j*3+k, k*2] = 1
    # For row i=1: a = 3+j, b = j*3+k, c = k*2+1
    #   T[3+j, j*3+k, k*2+1] = 1
    
    # Each row's tensor is a 3x9x3 tensor (or effectively 3x3x3 if we reindex)
    
    # Let's try a different approach: direct search with structure awareness
    # Initialize with the rank-17 decomposition, then try to merge two columns
    
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        U, V, W = _make_rank17()
        
        # Try to merge two random columns into one
        r1 = random.randint(0, 16)
        r2 = random.randint(0, 16)
        if r1 == r2:
            continue
        
        # Try to find coefficients alpha, beta such that
        # alpha * U[:,r1] + beta * U[:,r2] works for all three matrices
        # This is a linear system problem
        
        # For each entry position, we need:
        # alpha * U[a,r1] * V[b,r1] * W[c,r1] + beta * U[a,r2] * V[b,r2] * W[c,r2]
        # to match the original sum minus these two terms
        
        # This is complex. Let's try a simpler heuristic:
        # Check if U[:,r1] is proportional to U[:,r2], etc.
        
        u1, u2 = U[:, r1], U[:, r2]
        v1, v2 = V[:, r1], V[:, r2]
        w1, w2 = W[:, r1], W[:, r2]
        
        # Check if the rank-1 tensors are "similar"
        # Compute correlation-like measure
        norm_u1 = np.linalg.norm(u1)
        norm_u2 = np.linalg.norm(u2)
        if norm_u1 < 1e-10 or norm_u2 < 1e-10:
            continue
        
        cos_u = np.dot(u1, u2) / (norm_u1 * norm_u2)
        
        # If they're very similar, try to merge
        if abs(cos_u) > 0.9:
            # Try to find a linear combination that preserves the tensor
            # This is a 2x2 system for each (a,b,c) position
            # Too complex for now, skip
            
            pass
    
    # Alternative: try to find a rank-16 by zeroing out one entry and adjusting
    # Use the fact that some entries might be redundant
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        U, V, W = _make_rank17()
        rank = 16
        
        # Remove one column
        drop = random.randint(0, 16)
        U = np.delete(U, drop, axis=1)
        V = np.delete(V, drop, axis=1)
        W = np.delete(W, drop, axis=1)
        
        # Now we have a rank-16 decomposition with error
        # Try to fix it by adjusting other entries
        
        # Compute the error tensor
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = result - T
        
        # The error is the negative of the removed rank-1 term
        # Try to distribute this error across other terms
        
        # For each term r, try to add a small correction
        # This is a nonlinear problem, so use local search
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.sum(err ** 2)
        
        for it in range(200):
            if time.time() > deadline - 3:
                break
            
            # Pick a random entry to modify
            which = random.randint(0, 2)
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                old = U[a, r]
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                old = V[b, r]
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                old = W[c, r]
            
            # Try different values
            candidates = [-1, 0, 1, 0.5, -0.5]
            if old in candidates:
                candidates = [v for v in candidates if v != old]
            else:
                candidates.append(old)
            
            best_val = old
            best_local_err = best_err
            
            for new_val in candidates:
                if which == 0:
                    U[a, r] = new_val
                elif which == 1:
                    V[b, r] = new_val
                else:
                    W[c, r] = new_val
                
                result_new = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err_new = np.sum((result_new - T) ** 2)
                
                if err_new < best_local_err:
                    best_local_err = err_new
                    best_val = new_val
                
                # Restore
                if which == 0:
                    U[a, r] = old
                elif which == 1:
                    V[b, r] = old
                else:
                    W[c, r] = old
            
            # Apply best
            if which == 0:
                U[a, r] = best_val
            elif which == 1:
                V[b, r] = best_val
            else:
                W[c, r] = best_val
            
            best_err = best_local_err
            
            if best_err < 1e-8:
                break
        
        if best_err < 1e-8:
            return U, V, W
    
    return None

def _try_structural_rank16(deadline):
    """
    Placeholder for structural search.
    Returns None immediately to prioritize other paths and avoid timeout.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    # First, try to find rank 16 using block-structure-aware search
    T = _build_tensor(n, m, p)
    
    # Try block-structure search for rank 16
    result_16 = _try_block_rank16(deadline)
    if result_16 is not None:
        U, V, W = result_16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _make_rank17()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
