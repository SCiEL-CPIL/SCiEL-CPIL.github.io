"""Rewritten program for 4x4 @ 4x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2.
    Each term: (U_dict, V_dict, W_dict) with (row,col)->coeff
    U indexed by (i,j), V by (j,k), W by (k,i) for C[k,i] = sum_j A[i,j]B[j,k]
    """
    terms = []
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
    terms.append((
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
    terms.append((
        {(1,1): 1},
        {(0,1): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _build_4x4x4_strassen():
    """Build 4x4x4 Strassen decomposition (rank 49)."""
    terms_2x2 = _strassen_2x2_terms()
    rank = 49
    U = [[0.0]*rank for _ in range(16)]
    V = [[0.0]*rank for _ in range(16)]
    W = [[0.0]*rank for _ in range(16)]
    
    for t1, (U1, V1, W1) in enumerate(terms_2x2):
        for t2, (U2, V2, W2) in enumerate(terms_2x2):
            idx = t1*7 + t2
            for (bi, bj), cu in U1.items():
                for (li, lj), cu2 in U2.items():
                    i = bi*2 + li
                    j = bj*2 + lj
                    U[i*4+j][idx] += cu*cu2
            for (bj, bk), cv in V1.items():
                for (lj, lk), cv2 in V2.items():
                    j = bj*2 + lj
                    k = bk*2 + lk
                    V[j*4+k][idx] += cv*cv2
            for (bk, bi), cw in W1.items():
                for (lk, li), cw2 in W2.items():
                    k = bk*2 + lk
                    i = bi*2 + li
                    W[k*4+i][idx] += cw*cw2
    return U, V, W

def _build_4x4x5_strassen():
    """Build 4x4x5: 4x4x4 Strassen (49) + 4x4x1 schoolbook (4) = rank 53."""
    n, m, p = 4, 4, 5
    U4, V4, W4 = _build_4x4x4_strassen()
    rank_base = 49
    rank = rank_base + 4
    
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    for i in range(16):
        for r in range(rank_base):
            U[i][r] = U4[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank_base):
                V[j*5+k][r] = V4[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_base):
                W[k*4+i][r] = W4[k*4+i][r]
    
    # 4x4x1 part: C[i,4] = sum_j A[i,j]*B[j,4]
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i*m+j][r] = 1.0
        V[j*p+4][r] = 1.0
        for i in range(n):
            W[4*n+i][r] = 1.0  # k=4, index = 4*4+i = 16+i
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U_np[i*m+j, :]
                v = V_np[j*p+k, :]
                w = W_np[k*n+i, :]
                total = np.dot(u*v, w)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _is_half_integer(M, tol=1e-8):
    for row in M:
        for v in row:
            if abs(v*2 - round(v*2)) > tol:
                return False
    return True

def _snap_half_integer(M):
    return [[float(round(v*2.0))/2.0 for v in row] for row in M]

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    # For each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Reshape to check all at once
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u = U_np[i*m+j, :]
            for k in range(p):
                v = V_np[j*p+k, :]
                w = W_np[k*n+i, :]
                total = np.dot(u*v, w)
                err = abs(total - 1.0)
                if err > max_err:
                    max_err = err
    return max_err <= tol

def _als_optimize(U_init, V_init, W_init, n, m, p, rank, max_iter=200, tol=1e-10, time_limit=30, seed=42):
    """ALS (Alternating Least Squares) optimization for tensor decomposition.
    
    Given initial U,V,W of shape (n*m, rank), (m*p, rank), (p*n, rank),
    iteratively refine to minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    """
    start = time.time()
    np.random.seed(seed)
    
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    
    # Precompute index mappings
    # For each r, we want to solve for U[:,r] given V[:,r] and W[:,r]
    # The constraint: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Build the linear system for U:
    # For fixed r, U[:,r] appears in constraints where V[j*p+k,r] and W[k*n+i,r] are nonzero
    # This is complex, so we use a simpler approach: treat each (i,j,k) as a sample
    
    # Better approach: vectorize the loss and use gradient descent with momentum
    # Loss = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    # Precompute all (i,j,k) triples
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
    indices = np.array(indices, dtype=np.int32)  # (nmp, 3)
    
    nmp = n*m*p
    U_flat = U.reshape(-1, rank)  # (n*m, rank)
    V_flat = V.reshape(-1, rank)  # (m*p, rank)
    W_flat = W.reshape(-1, rank)  # (p*n, rank)
    
    lr = 0.01
    momentum = 0.9
    vU = np.zeros_like(U_flat)
    vV = np.zeros_like(V_flat)
    vW = np.zeros_like(W_flat)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U_flat.copy(), V_flat.copy(), W_flat.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Compute residuals
        # For each (i,j,k): resid = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
        # Vectorize: for each r, contrib_r = U[indices[:,0],r] * V[indices[:,1],r] * W[indices[:,2],r]
        total = np.zeros(nmp)
        for r in range(rank):
            total += U_flat[indices[:,0], r] * V_flat[indices[:,1], r] * W_flat[indices[:,2], r]
        resid = total - 1.0
        loss = np.sum(resid*resid)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_flat.copy(), V_flat.copy(), W_flat.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        gU = np.zeros_like(U_flat)
        gV = np.zeros_like(V_flat)
        gW = np.zeros_like(W_flat)
        
        for r in range(rank):
            # d(loss)/d(U[a,r]) = 2 * sum over (i,j,k) where index_a = i*m+j
            # = 2 * sum_{j,k} resid[i,j,k] * V[j*p+k,r] * W[k*n+i,r]
            # For each a = i*m+j, the contributing (i,j,k) are all k for fixed i,j
            # gU[a,r] = 2 * sum_k resid[i,j,k] * V[j*p+k,r] * W[k*n+i,r]
            
            # Vectorize over (i,j,k)
            u_vals = U_flat[indices[:,0], r]
            v_vals = V_flat[indices[:,1], r]
            w_vals = W_flat[indices[:,2], r]
            
            # grad for U: gU[indices[:,0], r] += 2*resid*v_vals*w_vals
            # Use np.add.at for accumulation
            np.add.at(gU, (indices[:,0], r), 2.0*resid*v_vals*w_vals)
            np.add.at(gV, (indices[:,1], r), 2.0*resid*u_vals*w_vals)
            np.add.at(gW, (indices[:,2], r), 2.0*resid*u_vals*v_vals)
        
        # Update with momentum
        vU = momentum*vU + lr*gU
        vV = momentum*vV + lr*gV
        vW = momentum*vW + lr*gW
        U_flat -= vU
        V_flat -= vV
        W_flat -= vW
    
    if best_loss < loss:
        U_flat, V_flat, W_flat = best_U, best_V, best_W
        loss = best_loss
    
    return U_flat, V_flat, W_flat, loss

def _try_eliminate_terms(U, V, W, n, m, p, max_eliminate=5, time_limit=30):
    """Try to eliminate terms by re-solving remaining terms.
    
    Strategy: for each term r, set it to zero, then re-optimize all terms
    via ALS to see if we can still satisfy the constraints.
    """
    start = time.time()
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    best_rank = rank
    
    # Try eliminating terms one at a time
    for elim_r in range(rank):
        if time.time() - start > time_limit:
            break
        
        # Create modified tensors with term elim_r zeroed
        U_mod = U_np.copy()
        V_mod = V_np.copy()
        W_mod = W_np.copy()
        U_mod[:, elim_r] = 0
        V_mod[:, elim_r] = 0
        W_mod[:, elim_r] = 0
        
        # Try ALS optimization
        U_opt, V_opt, W_opt, loss = _als_optimize(
            U_mod.tolist(), V_mod.tolist(), W_mod.tolist(),
            n, m, p, rank, max_iter=100, tol=1e-10, time_limit=5, seed=elim_r
        )
        
        if loss < 1e-8:
            # Check if the eliminated term is actually zero (or nearly)
            if np.allclose(U_opt[:, elim_r], 0, atol=1e-6) and \
               np.allclose(V_opt[:, elim_r], 0, atol=1e-6) and \
               np.allclose(W_opt[:, elim_r], 0, atol=1e-6):
                # Remove this term
                keep = [r for r in range(rank) if r != elim_r]
                U_new = U_opt[:, keep]
                V_new = V_opt[:, keep]
                W_new = W_opt[:, keep]
                new_rank = len(keep)
                if new_rank < best_rank:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = new_rank
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _try_random_search(target_rank, n, m, p, time_limit=30, seeds=None):
    """Try to find a valid decomposition of given rank via random initialization + ALS."""
    start = time.time()
    if seeds is None:
        seeds = [target_rank, target_rank+100, target_rank+200, target_rank+300, target_rank+400]
    
    for seed in seeds:
        if time.time() - start > time_limit:
            break
        
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        U_opt, V_opt, W_opt, loss = _als_optimize(
            U.tolist(), V.tolist(), W.tolist(),
            n, m, p, target_rank, max_iter=300, tol=1e-10, time_limit=8, seed=seed
        )
        
        if loss < 1e-8:
            # Try to snap to half-integers
            U_snap = np.round(U_opt*2.0)/2.0
            V_snap = np.round(V_opt*2.0)/2.0
            W_snap = np.round(W_opt*2.0)/2.0
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
            
            # Try without snapping
            U_list = U_opt.tolist()
            V_list = V_opt.tolist()
            W_list = W_opt.tolist()
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based (rank 53)
    try:
        U, V, W = _build_4x4x5_strassen()
        rank = len(U[0])
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
    except Exception:
        pass
    
    # Approach 2: Try to eliminate redundant terms from Strassen
    if best_result is not None and time.time() < deadline - 40:
        try:
            result = _try_eliminate_terms(
                best_result["U"], best_result["V"], best_result["W"],
                n, m, p, max_eliminate=5, time_limit=30
            )
            if result is not None:
                U_new, V_new, W_new = result
                new_rank = len(U_new[0])
                if new_rank < best_rank and _verify(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = new_rank
        except Exception:
            pass
    
    # Approach 3: Random search for lower rank
    if best_result is not None and time.time() < deadline - 20:
        for target_rank in range(best_rank - 1, max(best_rank - 5, 40), -1):
            if time.time() > deadline - 15:
                break
            try:
                result = _try_random_search(target_rank, n, m, p, time_limit=15, seeds=[target_rank, target_rank+50, target_rank+100])
                if result is not None:
                    U_new, V_new, W_new = result
                    if _verify(U_new, V_new, W_new, n, m, p):
                        best_result = {"U": U_new, "V": V_new, "W": W_new}
                        best_rank = target_rank
                        break
            except Exception:
                pass
    
    if best_result is not None:
        U = [[float(x) for x in row] for row in best_result["U"]]
        V = [[float(x) for x in row] for row in best_result["V"]]
        W = [[float(x) for x in row] for row in best_result["W"]]
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _build_schoolbook()

# EVOLVE-BLOCK-END
