"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_rational(arr, tol=1e-9):
    """Snap values to nearest integer or half-integer if close enough."""
    result = np.array(arr, dtype=float)
    # Try integers first
    int_snapped = np.round(result)
    if np.max(np.abs(result - int_snapped)) < tol:
        return int_snapped.astype(int).tolist()
    # Try half-integers
    half_snapped = np.round(result * 2) / 2
    if np.max(np.abs(result - half_snapped)) < tol:
        lst = []
        for row in half_snapped:
            r = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(int(round(v * 2)) / 2.0)
            lst.append(r)
        return lst
    # Return as-is (floats)
    return result.tolist()

def _als_step(U, V, W, n, m, p, rank):
    """Perform one ALS step."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Update W
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    # M[k, i, c] = sum_j U[i, j, c] * V[j, k, c]
    M = np.einsum('ijc,jkc->kic', U3, V3)
    for k in range(p):
        for i in range(n):
            vec = M[k, i, :]
            norm_sq = np.dot(vec, vec)
            if norm_sq > 1e-12:
                W[k*n+i, :] = vec / norm_sq
            else:
                W[k*n+i, :] = 0.0

    # Update U
    W3 = W.reshape(p, n, rank)
    # N[i, j, c] = sum_k V[j, k, c] * W[k, i, c]
    N = np.einsum('jkc,kic->ijc', V3, W3)
    for i in range(n):
        for j in range(m):
            vec = N[i, j, :]
            norm_sq = np.dot(vec, vec)
            if norm_sq > 1e-12:
                U[i*m+j, :] = vec / norm_sq
            else:
                U[i*m+j, :] = 0.0

    # Update V
    U3 = U.reshape(n, m, rank)
    # P[j, k, c] = sum_i U[i, j, c] * W[k, i, c]
    P = np.einsum('ijc,kic->jkc', U3, W3)
    for j in range(m):
        for k in range(p):
            vec = P[j, k, :]
            norm_sq = np.dot(vec, vec)
            if norm_sq > 1e-12:
                V[j*p+k, :] = vec / norm_sq
            else:
                V[j*p+k, :] = 0.0

def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline (rank 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Try to find a better rank using ALS
    # Known upper bound from composition is 16.
    # We will try ranks 14, 15, 16.
    
    for rank in [14, 15, 16]:
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = 50 if rank < 16 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(attempt * 101 + rank * 7)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Run ALS
            converged = False
            for it in range(1000):
                if time.time() > deadline - 10:
                    break
                _als_step(U, V, W, n, m, p, rank)
                
                # Check convergence every 50 steps
                if it % 50 == 49:
                    U3 = U.reshape(n, m, rank)
                    V3 = V.reshape(m, p, rank)
                    W3 = W.reshape(p, n, rank)
                    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    err = np.max(np.abs(recon - 1.0))
                    if err < 1e-10:
                        converged = True
                        break
            
            # Final check
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.max(np.abs(recon - 1.0))
            
            if err < 1e-6:
                # Snap to rationals
                U_list = _snap_to_rational(U)
                V_list = _snap_to_rational(V)
                W_list = _snap_to_rational(W)
                
                # Re-verify with snapped values
                if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank, tol=1e-4):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # If we found a rank 14, we can stop. If 15, we might try 14? 
            # The loop goes 14, 15, 16. So if 14 works, we break out of the rank loop?
            # No, the loop is `for rank in [14, 15, 16]`.
            # If we find 14, we set best_rank=14 and found=True.
            # We should probably break out of the rank loop if we find the lowest possible.
            # But we don't know the lowest possible.
            # However, 14 is a strong candidate.
            # Let's just let the loop finish or break if we find 14.
            if rank == 14:
                break

    return best
