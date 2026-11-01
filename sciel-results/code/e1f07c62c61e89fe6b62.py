"""Matrix multiplication 2x3x7: rank-minimal exact tensor decomposition."""
import time
import numpy as np
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def to_list(mat):
        return mat.tolist()

    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the tensor."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Compute the tensor decomposition using einsum for speed
        T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        if not np.allclose(T_comp, T, atol=1e-9):
            return False
            
        # Check if U, V, W are close to half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9):
            return False
            
        return True

    # Schoolbook decomposition as baseline
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Winograd-style approach: exploit cyclic convolution structure in m=3 dimension
    # The key insight: for m=3, we can use a 3-point DFT-like transform
    # to reduce multiplications. The transform matrix for cyclic group Z_3 is:
    #   [1 1 1]
    #   [1 ω ω²]
    #   [1 ω² ω]
    # where ω = e^(2πi/3) = -1/2 + i√3/2
    # 
    # However, we need real/half-integer entries. Instead, let's try a different
    # structured decomposition that exploits the small m=3 dimension.
    #
    # Alternative: Use a structured search that tries to merge schoolbook terms
    # by exploiting linearity. If two terms share the same W factor, we can merge
    # their U and V factors.
    
    def try_merge_search(target_rank, max_iters=1000):
        """
        Try to find a decomposition of target_rank by starting with schoolbook
        and attempting to merge terms.
        """
        if time.time() > deadline:
            return None
        
        # Start with schoolbook terms
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = np.zeros(n * m)
                    v_vec = np.zeros(m * p)
                    w_vec = np.zeros(p * n)
                    u_vec[i * m + j] = 1.0
                    v_vec[j * p + k] = 1.0
                    w_vec[k * n + i] = 1.0
                    terms.append((u_vec, v_vec, w_vec))
        
        best_error = np.inf
        best_terms = None
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Randomly select terms to try merging
            # Strategy: pick two terms that share a W index, and see if we can
            # combine their U and V vectors
            
            current_terms = terms[:]
            
            # Try to merge pairs of terms
            merged = False
            for a in range(len(current_terms)):
                if len(current_terms) <= target_rank:
                    break
                for b in range(a + 1, len(current_terms)):
                    if len(current_terms) <= target_rank:
                        break
                    
                    ua, va, wa = current_terms[a]
                    ub, vb, wb = current_terms[b]
                    
                    # Check if W vectors are proportional (i.e., same support)
                    # For merging, we need wa and wb to have the same support pattern
                    # or one to be a scalar multiple of the other
                    
                    # Simple check: if they have the same non-zero positions
                    nonzero_a = np.nonzero(wa)[0]
                    nonzero_b = np.nonzero(wb)[0]
                    
                    if len(nonzero_a) != len(nonzero_b):
                        continue
                    if not np.array_equal(nonzero_a, nonzero_b):
                        continue
                    
                    # Try to find scalars alpha, beta such that:
                    # alpha * ua + beta * ub has the same support as needed
                    # For now, just try simple combinations
                    
                    # Try: new_u = ua + ub, new_v = va + vb, new_w = wa (if wa == wb)
                    if np.allclose(wa, wb):
                        new_u = ua + ub
                        new_v = va + vb
                        new_w = wa.copy()
                        
                        # Check if this creates a valid term (no unexpected zeros)
                        # The merged term should still be a valid rank-1 term
                        # For now, just try it
                        
                        # Remove terms a and b, add merged term
                        current_terms = [t for idx, t in enumerate(current_terms) if idx != a and idx != b]
                        current_terms.append((new_u, new_v, new_w))
                        merged = True
                        break
                
                if merged:
                    break
            
            if not merged:
                # No more merges possible, try a different random perturbation
                # Randomly perturb one of the terms
                if len(current_terms) > 0:
                    idx = np.random.randint(len(current_terms))
                    u, v, w = current_terms[idx]
                    # Small perturbation
                    if np.random.random() < 0.5:
                        u = u + np.random.choice([-0.5, 0.5])
                    else:
                        v = v + np.random.choice([-0.5, 0.5])
                    current_terms[idx] = (u, v, w)
            
            # Build U, V, W from current terms
            rank = len(current_terms)
            if rank != target_rank:
                continue
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for r_idx, (u, v, w) in enumerate(current_terms):
                U[:, r_idx] = u
                V[:, r_idx] = v
                W[:, r_idx] = w
            
            # Compute error
            T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            error = np.sum((T_comp - T) ** 2)
            
            if error < best_error:
                best_error = error
                best_terms = [t.copy() for t in current_terms]
                
                if error < 1e-9:
                    # Round to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int
            
            # Update terms for next iteration
            terms = current_terms[:]
        
        return None
    
    # Try structured search for rank 41
    result = try_merge_search(41, max_iters=500)
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Try rank 40
    result = try_merge_search(40, max_iters=300)
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # If all fails, return schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
