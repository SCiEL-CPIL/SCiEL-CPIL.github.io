import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
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
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _verify_fast(U, V, W, tol=1e-9):
    """Fast vectorized verification using the tensor contraction structure."""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can vectorize this: for each r, the contribution to T[i,j,k] is
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the tensor by summing outer products
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U[i,j,r] = U[i*m+j, r], etc.
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # This is a 3-way contraction
        for i in range(n):
            for j in range(m):
                T[i,j,:] += u[i,j] * v[j,:] * w[:,i]
    
    return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

def _verify_vectorized(U, V, W, tol=1e-9):
    """Fully vectorized verification - much faster."""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape and use einsum
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # For each r, the contribution is:
    #   sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # We can compute this as:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's build it step by step:
    # For each i, j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
    
    # Precompute for each (j,k): V[j*p+k, :] has shape (rank,)
    # For each (k,i): W[k*n+i, :] has shape (rank,)
    # For each (i,j): U[i*m+j, :] has shape (rank,)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            u_vec = U[i*m + j, :]  # shape (rank,)
            for k in range(p):
                v_vec = V[j*p + k, :]  # shape (rank,)
                w_vec = W[k*n + i, :]  # shape (rank,)
                T[i, j, k] = np.dot(u_vec, v_vec * w_vec)
    
    return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

def _compute_tensor(U, V, W):
    """Compute the decomposed tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            u_vec = U[i*m + j, :]
            for k in range(p):
                v_vec = V[j*p + k, :]
                w_vec = W[k*n + i, :]
                T[i, j, k] = np.dot(u_vec, v_vec * w_vec)
    return T

def _try_merge_terms(U, V, W, deadline):
    """
    Try to merge pairs of terms that have overlapping structure.
    If two terms r1 and r2 can be replaced by a single term, we reduce rank by 1.
    
    Strategy: For each pair (r1, r2), check if their combined contribution
    T[r1] + T[r2] can be expressed as a single rank-1 tensor.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Compute individual term tensors
    term_tensors = []
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        t = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                t[i,j,:] = u[i,j] * v[j,:] * w[:,i]
        term_tensors.append(t)
    
    # For each pair, try to see if their sum is rank-1
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            if time.time() > deadline:
                return None
            
            combined = term_tensors[r1] + term_tensors[r2]
            
            # Check if combined is rank-1: all 2x2 minors should be zero
            # A tensor T is rank-1 iff T[i,j,k]*T[i',j',k'] == T[i,j',k']*T[i',j,k'] for all i,i',j,j',k,k'
            # This is expensive, so use a simpler check:
            # Flatten and check if the tensor can be written as outer product
            
            # For a rank-1 tensor, T[i,j,k] = a[i]*b[j]*c[k]
            # We can try to extract a, b, c by:
            # a[i] = T[i, j0, k0] / (b[j0]*c[k0]) for some fixed j0, k0
            
            # Try all possible (j0, k0) anchors
            found = False
            for j0 in range(m):
                for k0 in range(p):
                    # Check if T[:, j0, k0] is nonzero
                    col = combined[:, j0, k0]
                    if np.all(np.abs(col) < 1e-12):
                        continue
                    
                    # Assume T[i,j,k] = a[i] * b[j] * c[k]
                    # Then a[i] = T[i,j0,k0] / (b[j0]*c[k0])
                    # Let's set b[j0] = 1, c[k0] = 1 for normalization
                    # Then a[i] = T[i,j0,k0]
                    
                    a = col.copy()
                    
                    # Check if this works
                    valid = True
                    for i in range(n):
                        if np.abs(a[i]) < 1e-12:
                            # a[i] is zero, so T[i,:,:] should be all zero
                            if np.any(np.abs(combined[i,:,:]) > 1e-10):
                                valid = False
                                break
                            continue
                        
                        # b[j] = T[i,j,k0] / (a[i] * c[k0]) = T[i,j,k0] / a[i] (since c[k0]=1)
                        # c[k] = T[i,j0,k] / (a[i] * b[j0]) = T[i,j0,k] / a[i] (since b[j0]=1)
                        
                        b = combined[i, :, k0] / a[i]
                        c = combined[i, j0, :] / a[i]
                        
                        # Verify
                        reconstructed = np.einsum('i,j,k->ijk', a, b, c)
                        if not np.allclose(reconstructed, combined, atol=1e-10):
                            valid = False
                            break
                    
                    if valid:
                        # Found a rank-1 decomposition of the combined tensor
                        # Now we need to express this as U_new, V_new, W_new
                        # such that U_new[i*m+j] * V_new[j*p+k] * W_new[k*n+i] = a[i]*b[j]*c[k]
                        
                        # We can set:
                        # U_new[i*m+j] = a[i] * 1 (for all j)
                        # V_new[j*p+k] = b[j] * 1 (for all k)  
                        # W_new[k*n+i] = c[k] * 1 (for all i)
                        
                        # But this doesn't work because the indices are mixed.
                        # We need: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = a[i]*b[j]*c[k]
                        
                        # One way: U[i*m+j, r] = a[i], V[j*p+k, r] = b[j], W[k*n+i, r] = c[k]
                        # Then product = a[i]*b[j]*c[k] ✓
                        
                        U_new = np.zeros((nm, 1))
                        V_new = np.zeros((mp, 1))
                        W_new = np.zeros((pn, 1))
                        
                        for i in range(n):
                            for j in range(m):
                                U_new[i*m+j, 0] = a[i]
                        for j in range(m):
                            for k in range(p):
                                V_new[j*p+k, 0] = b[j]
                        for k in range(p):
                            for i in range(n):
                                W_new[k*n+i, 0] = c[k]
                        
                        # Build the new rank-16 matrices
                        keep = [r for r in range(rank) if r != r1 and r != r2]
                        U16 = np.hstack([U[:, keep], U_new])
                        V16 = np.hstack([V[:, keep], V_new])
                        W16 = np.hstack([W[:, keep], W_new])
                        
                        if _verify_fast(U16, V16, W16, tol=1e-9):
                            return U16, V16, W16
                        found = True
                        break
                if found:
                    break
            if found:
                continue
    
    return None

def _try_linear_combination(U, V, W, deadline):
    """
    Try to find a linear combination of two terms that can replace them with one.
    More general than _try_merge_terms.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # For each pair (r1, r2), try alpha*term_r1 + beta*term_r2 = single_term
    # The single term must be rank-1 in the tensor sense
    
    # Compute term tensors
    term_tensors = []
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        t = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                t[i,j,:] = u[i,j] * v[j,:] * w[:,i]
        term_tensors.append(t)
    
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            if time.time() > deadline:
                return None
            
            # Try several alpha/beta ratios
            for alpha_beta in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25]:
                combined = term_tensors[r1] + alpha_beta * term_tensors[r2]
                
                # Check if combined is rank-1
                for j0 in range(m):
                    for k0 in range(p):
                        col = combined[:, j0, k0]
                        if np.all(np.abs(col) < 1e-12):
                            continue
                        
                        a = col.copy()
                        valid = True
                        for i in range(n):
                            if np.abs(a[i]) < 1e-12:
                                if np.any(np.abs(combined[i,:,:]) > 1e-10):
                                    valid = False
                                    break
                                continue
                            
                            b = combined[i, :, k0] / a[i]
                            c = combined[i, j0, :] / a[i]
                            
                            reconstructed = np.einsum('i,j,k->ijk', a, b, c)
                            if not np.allclose(reconstructed, combined, atol=1e-10):
                                valid = False
                                break
                        
                        if valid:
                            # Build new term
                            U_new = np.zeros((nm, 1))
                            V_new = np.zeros((mp, 1))
                            W_new = np.zeros((pn, 1))
                            
                            for i in range(n):
                                for j in range(m):
                                    U_new[i*m+j, 0] = a[i]
                            for j in range(m):
                                for k in range(p):
                                    V_new[j*p+k, 0] = b[j]
                            for k in range(p):
                                for i in range(n):
                                    W_new[k*n+i, 0] = c[k]
                            
                            # The new term replaces r1 and r2
                            # But we need to adjust: the new term's contribution should be
                            # term_r1 + alpha_beta * term_r2
                            # So we keep all other terms, remove r1 and r2, add the new term
                            
                            keep = [r for r in range(rank) if r != r1 and r != r2]
                            U16 = np.hstack([U[:, keep], U_new])
                            V16 = np.hstack([V[:, keep], V_new])
                            W16 = np.hstack([W[:, keep], W_new])
                            
                            if _verify_fast(U16, V16, W16, tol=1e-9):
                                return U16, V16, W16
    
    return None

def _try_optimized_rank16(U, V, W, deadline):
    """
    Use the structure of the rank-17 solution to try to find rank 16.
    
    The key insight: some terms in the rank-17 solution might be linearly
    dependent in a way that allows merging.
    
    Specifically, look at terms that share the same W pattern or U pattern.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Group terms by their W pattern (up to scaling)
    # Two terms can potentially be merged if they have the same W pattern
    # and their U and V patterns are compatible
    
    # For each pair with similar W patterns, try to merge
    w_patterns = {}
    for r in range(rank):
        w_vec = W[:, r]
        # Normalize
        norm = np.linalg.norm(w_vec)
        if norm > 1e-12:
            w_norm = w_vec / norm
            # Round to identify similar patterns
            w_key = tuple(np.round(w_norm, 6))
            if w_key not in w_patterns:
                w_patterns[w_key] = []
            w_patterns[w_key].append(r)
    
    # Try merging terms within each group
    for w_key, indices in w_patterns.items():
        if len(indices) < 2:
            continue
        
        for i1 in range(len(indices)):
            for i2 in range(i1 + 1, len(indices)):
                r1 = indices[i1]
                r2 = indices[i2]
                if time.time() > deadline:
                    return None
                
                # These two terms have the same W pattern
                # Try to merge them
                # T_combined[i,j,k] = U1[i,j]*V1[j,k]*W[k,i] + U2[i,j]*V2[j,k]*W[k,i]
                # = W[k,i] * (U1[i,j]*V1[j,k] + U2[i,j]*V2[j,k])
                
                # For this to be rank-1, we need:
                # U1[i,j]*V1[j,k] + U2[i,j]*V2[j,k] = a[i]*b[j]*c[k] for some a,b,c
                # But since W is fixed, the product must factor as (something in i,j) * (something in j,k)
                
                # Actually, since W is the same, we need:
                # (U1[i,j]*V1[j,k] + U2[i,j]*V2[j,k]) * W[k,i] = A[i,j,k]
                # where A is rank-1.
                
                # This is hard to check directly. Let's just try the merge and verify.
                
                # Simple approach: average the U and V vectors
                U_avg = (U[:, r1] + U[:, r2]) / 2
                V_avg = (V[:, r1] + V[:, r2]) / 2
                W_avg = W[:, r1]  # Same as W[:, r2] up to scaling
                
                # Scale to match the original contribution
                # Original: U1*V1*W + U2*V2*W
                # New: U_avg * V_avg * W_avg
                # We need: U_avg * V_avg * W_avg ≈ U1*V1*W + U2*V2*W
                
                # This won't work in general. Let's try a different approach.
                
                # Try to find U_new, V_new such that:
                # U_new[i,j] * V_new[j,k] * W[k,i] = U1[i,j]*V1[j,k]*W[k,i] + U2[i,j]*V2[j,k]*W[k,i]
                # => U_new[i,j] * V_new[j,k] = U1[i,j]*V1[j,k] + U2[i,j]*V2[j,k]
                
                # The RHS is a matrix M[i,j,k]... wait, it's not a matrix, it's a tensor.
                # Actually for fixed i,k: sum_j ... no, it's U1[i,j]*V1[j,k] + U2[i,j]*V2[j,k]
                # This is a matrix in (i,k) for each... no, it's a 3-tensor.
                
                # Let's just try several simple combinations
                for scale_u in [1.0, 0.5, 2.0]:
                    for scale_v in [1.0, 0.5, 2.0]:
                        U_new = scale_u * (U[:, r1] + U[:, r2])
                        V_new = scale_v * (V[:, r1] + V[:, r2])
                        
                        keep = [r for r in range(rank) if r != r1 and r != r2]
                        U16 = np.hstack([U[:, keep], U_new.reshape(-1, 1)])
                        V16 = np.hstack([V[:, keep], V_new.reshape(-1, 1)])
                        W16 = np.hstack([W[:, keep], W[:, r1].reshape(-1, 1)])
                        
                        if _verify_fast(U16, V16, W16, tol=1e-9):
                            return U16, V16, W16
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-17 solution
    U17, V17, W17 = _build_rank17()
    
    # Verify it works
    if not _verify_fast(U17, V17, W17, tol=1e-9):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Try to find rank-16 by merging terms
    if time.time() < deadline:
        result = _try_merge_terms(U17, V17, W17, deadline)
        if result is not None:
            U16, V16, W16 = result
            if _verify_fast(U16, V16, W16, tol=1e-9):
                return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Try linear combination approach
    if time.time() < deadline - 10:
        result = _try_linear_combination(U17, V17, W17, deadline)
        if result is not None:
            U16, V16, W16 = result
            if _verify_fast(U16, V16, W16, tol=1e-9):
                return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Try optimized rank-16 search
    if time.time() < deadline - 5:
        result = _try_optimized_rank16(U17, V17, W17, deadline)
        if result is not None:
            U16, V16, W16 = result
            if _verify_fast(U16, V16, W16, tol=1e-9):
                return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return rank-17 as fallback
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

# EVOLVE-BLOCK-END
