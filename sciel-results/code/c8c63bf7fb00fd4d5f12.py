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
        ([0, 4], [1, 1], [1, 1], [1, 1], [0, 3], [1, 1]),
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

def _try_structured_rank16(deadline):
    """
    Try to find a rank-16 decomposition by using a structured ansatz that embeds
    the Strassen decomposition for the 2x2x2 sub-block and treats remaining
    interactions as low-rank corrections. Uses linear dependencies to eliminate
    one rank term.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    # Precompute non-zero tensor entries
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                nz_indices.append((a, b, c))
    
    # Build set for fast lookup
    nz_set = set(nz_indices)
    
    def compute_error(U, V, W):
        """Compute squared error at all positions efficiently."""
        err = 0.0
        # Check non-zero positions
        for a, b, c in nz_indices:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            err += (val - 1.0) ** 2
        # Sample zero positions for additional constraint
        for _ in range(30):
            a = random.randint(0, n_a-1)
            b = random.randint(0, n_b-1)
            c = random.randint(0, n_c-1)
            if (a, b, c) not in nz_set:
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                err += val ** 2
        return err
    
    # Strategy 1: Start from rank-17, drop one column, and do informed local search
    for drop in range(17):
        if time.time() > deadline - 15:
            break
        
        U, V, W = _make_rank17()
        U = np.delete(U, drop, axis=1).astype(float)
        V = np.delete(V, drop, axis=1).astype(float)
        W = np.delete(W, drop, axis=1).astype(float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = compute_error(best_U, best_V, best_W)
        
        # Multi-phase local search with increasing step sizes
        phases = [
            (200, [0.1, 0.25, 0.5, 1.0]),
            (150, [0.25, 0.5, 1.0, 2.0]),
            (100, [0.5, 1.0, 2.0, 4.0]),
        ]
        
        for it_count, steps in phases:
            for it in range(it_count):
                if time.time() > deadline - 10:
                    break
                
                # Pick which matrix and which entry to modify
                which = random.randint(0, 2)
                if which == 0:
                    a = random.randint(0, n_a-1)
                    r = random.randint(0, rank-1)
                    old = U[a, r]
                elif which == 1:
                    b = random.randint(0, n_b-1)
                    r = random.randint(0, rank-1)
                    old = V[b, r]
                else:
                    c = random.randint(0, n_c-1)
                    r = random.randint(0, rank-1)
                    old = W[c, r]
                
                # Try candidate values
                candidates = set()
                for s in steps:
                    candidates.add(old + s)
                    candidates.add(old - s)
                # Also try snapping to half-integers
                candidates.add(round(old * 2) / 2)
                candidates.add(round(old * 2) / 2 + 0.5)
                candidates.add(round(old * 2) / 2 - 0.5)
                candidates.add(0.0)
                
                best_val = old
                best_local_err = best_err
                
                for new_val in candidates:
                    if which == 0:
                        U[a, r] = new_val
                    elif which == 1:
                        V[b, r] = new_val
                    else:
                        W[c, r] = new_val
                    
                    err_new = compute_error(U, V, W)
                    
                    if err_new < best_local_err - 1e-12:
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
                
                if best_err < 1e-10:
                    break
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            # Final polish: snap to nearest half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            final_err = compute_error(U, V, W)
            if final_err < 1e-9:
                return U, V, W
    
    # Strategy 2: Try a completely different initialization
    # Use the structure: rows 0,1,3,4 for U (Strassen part), rows 2,5 for direct part
    # Try to find a rank-16 solution by parameterizing the correction terms
    
    if time.time() < deadline - 20:
        # Try random restarts with structured initialization
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            U = np.zeros((n_a, rank))
            V = np.zeros((n_b, rank))
            W = np.zeros((n_c, rank))
            
            # Initialize with Strassen-like structure for first 7 columns
            # and random for remaining 9 columns
            for r in range(7):
                # Strassen pattern (simplified)
                U[0, r] = random.choice([0, 1, -1])
                U[1, r] = random.choice([0, 1, -1])
                U[3, r] = random.choice([0, 1, -1])
                U[4, r] = random.choice([0, 1, -1])
                V[0, r] = random.choice([0, 1, -1])
                V[1, r] = random.choice([0, 1, -1])
                V[3, r] = random.choice([0, 1, -1])
                V[4, r] = random.choice([0, 1, -1])
                W[0, r] = random.choice([0, 1, -1])
                W[1, r] = random.choice([0, 1, -1])
                W[2, r] = random.choice([0, 1, -1])
                W[3, r] = random.choice([0, 1, -1])
            
            # Random initialization for remaining columns
            for r in range(7, rank):
                for a in range(n_a):
                    U[a, r] = random.choice([0, 0.5, 1, -0.5, -1])
                for b in range(n_b):
                    V[b, r] = random.choice([0, 0.5, 1, -0.5, -1])
                for c in range(n_c):
                    W[c, r] = random.choice([0, 0.5, 1, -0.5, -1])
            
            best_err = compute_error(U, V, W)
            
            # Local search
            for it in range(300):
                if time.time() > deadline - 5:
                    break
                
                which = random.randint(0, 2)
                if which == 0:
                    a = random.randint(0, n_a-1)
                    r = random.randint(0, rank-1)
                    old = U[a, r]
                elif which == 1:
                    b = random.randint(0, n_b-1)
                    r = random.randint(0, rank-1)
                    old = V[b, r]
                else:
                    c = random.randint(0, n_c-1)
                    r = random.randint(0, rank-1)
                    old = W[c, r]
                
                candidates = [old + d for d in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]]
                candidates.extend([0, 0.5, 1, -0.5, -1])
                
                best_val = old
                best_local_err = best_err
                
                for new_val in candidates:
                    if which == 0:
                        U[a, r] = new_val
                    elif which == 1:
                        V[b, r] = new_val
                    else:
                        W[c, r] = new_val
                    
                    err_new = compute_error(U, V, W)
                    
                    if err_new < best_local_err - 1e-12:
                        best_local_err = err_new
                        best_val = new_val
                    
                    if which == 0:
                        U[a, r] = old
                    elif which == 1:
                        V[b, r] = old
                    else:
                        W[c, r] = old
                
                if which == 0:
                    U[a, r] = best_val
                elif which == 1:
                    V[b, r] = best_val
                else:
                    W[c, r] = best_val
                
                best_err = best_local_err
                
                if best_err < 1e-10:
                    break
            
            if best_err < 1e-10:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                final_err = compute_error(U, V, W)
                if final_err < 1e-9:
                    return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    T = _build_tensor(n, m, p)
    
    # Try structured rank-16 search
    result_16 = _try_structured_rank16(deadline)
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
