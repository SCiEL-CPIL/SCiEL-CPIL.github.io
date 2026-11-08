import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank17_v2():
    """Alternative partitioning: Strassen(2x2x2) + schoolbook(2x2x1) + schoolbook(2x1x2) + schoolbook(2x1x1)"""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Strassen for A[:,:2]*B[:2,:2] -> C[:,:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # A[:,:2]*B[:2,2] -> C[:,2] (2x2 * 2x1)
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    
    # A[:,2]*B[2,:2] -> C[:,:2] (2x1 * 1x2)
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[5, 13] = 1; V[6, 13] = 1; W[1, 13] = 1
    U[5, 14] = 1; V[7, 14] = 1; W[3, 14] = 1
    
    # A[:,2]*B[2,2] -> C[:,2] (2x1 * 1x1)
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _build_rank17_v3():
    """Another partitioning variant"""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Same Strassen core
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # A[:,2]*B[2,:] -> C[:,0:3] (2x1 * 1x3) - 3 terms
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[2, 9] = 1; V[8, 9] = 1; W[4, 9] = 1
    
    U[5, 10] = 1; V[6, 10] = 1; W[1, 10] = 1
    U[5, 11] = 1; V[7, 11] = 1; W[3, 11] = 1
    U[5, 12] = 1; V[8, 12] = 1; W[5, 12] = 1
    
    # A[:,:2]*B[:2,2] -> C[:,2] (2x2 * 2x1) - 4 terms
    U[0, 13] = 1; V[2, 13] = 1; W[4, 13] = 1
    U[1, 14] = 1; V[5, 14] = 1; W[4, 14] = 1
    U[3, 15] = 1; V[2, 15] = 1; W[5, 15] = 1
    U[4, 16] = 1; V[5, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _compute_tensor(U, V, W, n, m, p):
    """Compute the full multiplication tensor from factorization"""
    rank = U.shape[1]
    tensor = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ia = i*m + j
                    ib = j*p + k
                    out_idx = k*n + i
                    tensor[ia, ib, out_idx] += U[ia, r] * V[ib, r] * W[out_idx, r]
    return tensor

def _target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i*m + j
                ib = j*p + k
                out_idx = k*n + i
                T[ia, ib, out_idx] = 1.0
    return T

def _try_merge_terms(U, V, W, n, m, p):
    """Try to find if any pair of terms can be merged (linear dependence check)"""
    rank = U.shape[1]
    target = _target_tensor(n, m, p)
    
    for r1 in range(rank):
        for r2 in range(r1+1, rank):
            # Check if removing r2 and adjusting r1 could work
            # This is a heuristic - check if the contribution of r2 is a scalar multiple pattern
            # that could be absorbed into r1
            U1 = U[:, r1].copy()
            V1 = V[:, r1].copy()
            W1 = W[:, r1].copy()
            U2 = U[:, r2].copy()
            V2 = V[:, r2].copy()
            W2 = W[:, r2].copy()
            
            # Check if U2 is proportional to U1, V2 to V1, W2 to W1
            # (which would mean they're redundant)
            if np.allclose(U2, 0) or np.allclose(V2, 0) or np.allclose(W2, 0):
                continue
            
            # Try to find alpha such that U2 = alpha*U1, etc.
            # Use least squares
            if np.linalg.norm(U1) > 1e-12:
                alpha_U = np.dot(U2, U1) / np.dot(U1, U1)
            else:
                continue
            if np.linalg.norm(V1) > 1e-12:
                alpha_V = np.dot(V2, V1) / np.dot(V1, V1)
            else:
                continue
            if np.linalg.norm(W1) > 1e-12:
                alpha_W = np.dot(W2, W1) / np.dot(W1, W1)
            else:
                continue
            
            # If all alphas are approximately equal, the terms might be mergeable
            if abs(alpha_U - alpha_V) < 1e-6 and abs(alpha_V - alpha_W) < 1e-6:
                # Try removing r2 and scaling r1
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                U_new[:, r1] += alpha_U * U_new[:, r2]
                V_new[:, r1] += alpha_V * V_new[:, r2]
                W_new[:, r1] += alpha_W * W_new[:, r2]
                U_new[:, r2] = 0
                V_new[:, r2] = 0
                W_new[:, r2] = 0
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    mask = np.ones(rank, dtype=bool)
                    mask[r2] = False
                    return U_new[:, mask], V_new[:, mask], W_new[:, mask]
    
    return None

def _perturbation_search(U, V, W, n, m, p, max_iters=20000):
    """More sophisticated perturbation search for rank 16"""
    rank = U.shape[1]
    target = _target_tensor(n, m, p)
    
    best_err = float('inf')
    best_result = None
    
    for _ in range(max_iters):
        # Start from current solution, zero out one random term
        idx = random.randint(0, rank-1)
        U2 = U.copy()
        V2 = V.copy()
        W2 = W.copy()
        U2[:, idx] = 0
        V2[:, idx] = 0
        W2[:, idx] = 0
        
        # Compute residual
        tensor2 = _compute_tensor(U2, V2, W2, n, m, p)
        residual = target - tensor2
        
        # Try to fix by adjusting remaining terms
        # For each remaining term, see if we can adjust it to absorb the residual
        for r in range(rank):
            if r == idx:
                continue
            
            # Try small perturbations
            for scale in [0.5, 1.0, 1.5, 2.0, -0.5, -1.0]:
                U3 = U2.copy()
                V3 = V2.copy()
                W3 = W2.copy()
                
                # Perturb term r
                U3[:, r] *= scale
                V3[:, r] *= scale
                W3[:, r] *= scale
                
                tensor3 = _compute_tensor(U3, V3, W3, n, m, p)
                err = np.max(np.abs(tensor3 - target))
                
                if err < best_err:
                    best_err = err
                    best_result = (U3, V3, W3, r, scale)
                
                if err < 1e-9:
                    mask = np.ones(rank, dtype=bool)
                    mask[idx] = False
                    return U3[:, mask], V3[:, mask], W3[:, mask]
        
        # Also try adding a new random term to absorb residual
        if random.random() < 0.3:
            U3 = U2.copy()
            V3 = V2.copy()
            W3 = W2.copy()
            
            # Add a new term at position idx with random values
            for i in range(n*m):
                if random.random() < 0.3:
                    U3[i, idx] = random.choice([-2, -1, 1, 2])
            for i in range(m*p):
                if random.random() < 0.3:
                    V3[i, idx] = random.choice([-2, -1, 1, 2])
            for i in range(p*n):
                if random.random() < 0.3:
                    W3[i, idx] = random.choice([-2, -1, 1, 2])
            
            tensor3 = _compute_tensor(U3, V3, W3, n, m, p)
            err = np.max(np.abs(tensor3 - target))
            
            if err < 1e-9:
                mask = np.ones(rank, dtype=bool)
                # Check if we can remove any other term
                for r in range(rank):
                    if r == idx:
                        continue
                    U4 = U3.copy()
                    V4 = V3.copy()
                    W4 = W3.copy()
                    U4[:, r] = 0
                    V4[:, r] = 0
                    W4[:, r] = 0
                    tensor4 = _compute_tensor(U4, V4, W4, n, m, p)
                    if np.max(np.abs(tensor4 - target)) < 1e-9:
                        mask[r] = False
                        mask[idx] = False
                        return U3[:, mask], V3[:, mask], W3[:, mask]
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    
    n, m, p = DIMENSIONS
    
    # Get baseline rank 17
    U, V, W = _build_rank17()
    
    # Verify baseline
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 17
    best_U, best_V, best_W = U, V, W
    
    # Try removing individual terms from rank 17
    for idx in range(17):
        if time.time() > deadline:
            break
        mask = np.ones(17, dtype=bool)
        mask[idx] = False
        U16 = U[:, mask]
        V16 = V[:, mask]
        W16 = W[:, mask]
        if _verify(U16, V16, W16, n, m, p):
            best_rank = 16
            best_U, best_V, best_W = U16, V16, W16
            break
    
    # Try alternative constructions
    if best_rank > 16 and time.time() < deadline - 10:
        for builder in [_build_rank17_v2, _build_rank17_v3]:
            if time.time() > deadline:
                break
            U_alt, V_alt, W_alt = builder()
            if _verify(U_alt, V_alt, W_alt, n, m, p):
                # Try removing terms from this alternative
                for idx in range(17):
                    if time.time() > deadline:
                        break
                    mask = np.ones(17, dtype=bool)
                    mask[idx] = False
                    U16 = U_alt[:, mask]
                    V16 = V_alt[:, mask]
                    W16 = W_alt[:, mask]
                    if _verify(U16, V16, W16, n, m, p):
                        best_rank = 16
                        best_U, best_V, best_W = U16, V16, W16
                        break
                if best_rank == 16:
                    break
    
    # Try term merging
    if best_rank > 16 and time.time() < deadline - 10:
        merged = _try_merge_terms(U, V, W, n, m, p)
        if merged is not None:
            U_m, V_m, W_m = merged
            if _verify(U_m, V_m, W_m, n, m, p):
                best_rank = 16
                best_U, best_V, best_W = U_m, V_m, W_m
    
    # Perturbation search for rank 16
    if best_rank > 16 and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        max_iters = int(remaining_time * 5000)  # Scale iterations to available time
        result = _perturbation_search(U, V, W, n, m, p, max_iters=max(1000, max_iters))
        if result is not None:
            best_rank = 16
            best_U, best_V, best_W = result
    
    # Try rank 15 by removing two terms
    if best_rank > 15 and time.time() < deadline - 5:
        for idx1 in range(17):
            if time.time() > deadline:
                break
            for idx2 in range(idx1 + 1, 17):
                mask = np.ones(17, dtype=bool)
                mask[idx1] = False
                mask[idx2] = False
                U15 = U[:, mask]
                V15 = V[:, mask]
                W15 = W[:, mask]
                if _verify(U15, V15, W15, n, m, p):
                    best_rank = 15
                    best_U, best_V, best_W = U15, V15, W15
                    break
            if best_rank == 15:
                break
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        U, V, W = _build_rank17()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
