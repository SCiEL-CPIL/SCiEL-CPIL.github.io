"""Matrix multiplication 3,3,5 - low rank decomposition via structured search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank decomposition for 3x3 x 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Build the target multiplication tensor
    # Shape: (n*m, m*p, p*n) = (9, 15, 15)
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify_decomposition(U, V, W, target_tensor, tol=1e-9):
        """Verify that U, V, W decompose the target tensor."""
        rank = U.shape[1]
        reconstructed = np.zeros_like(target_tensor)
        for r in range(rank):
            reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(reconstructed, target_tensor, atol=tol)

    def schoolbook_decomposition():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1

        return U, V, W, rank

    def try_structured_decomposition(rank):
        """
        Try to find a structured decomposition of given rank.
        Uses the fact that we can group (i,j,k) triples into rank-1 terms.
        """
        if time.time() > deadline - 5:
            return None

        # For a rank-1 term to be valid, we need vectors u, v, w such that:
        # u[a] * v[b] * w[c] = 1 for (a,b,c) in the support of this term
        # u[a] * v[b] * w[c] = 0 for all other (a,b,c)
        
        # Strategy: Try to partition the 45 (i,j,k) triples into groups
        # where each group can be covered by a single rank-1 term.
        
        # A rank-1 term covering a set S of (i,j,k) triples requires:
        # - u[i*m+j] is non-zero only for (i,j) that appear in S
        # - v[j*p+k] is non-zero only for (j,k) that appear in S  
        # - w[k*n+i] is non-zero only for (k,i) that appear in S
        # - For each (i,j,k) in S: u[i*m+j]*v[j*p+k]*w[k*n+i] = 1
        # - For any (a,b,c) not in S but where all three indices are "active": product must be 0
        
        # Let's try a simpler approach: use known algebraic identities.
        # For 3x3x5, we can think of it as 3x3x3 + 3x3x2.
        # The 3x3x3 part can potentially use rank 23 (known bound) or less.
        # The 3x3x2 part can use rank 6 (schoolbook) or less.
        
        # Actually, let's try a direct search for small ranks using random initialization
        # with ALS, but with better initialization strategies.
        
        def try_als_for_rank(r, seed=42):
            """Try ALS to find rank-r decomposition."""
            if time.time() > deadline - 5:
                return None
            
            rng = np.random.RandomState(seed)
            
            # Initialize with sparse random vectors
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            for col in range(r):
                # Random sparse initialization
                u_support = rng.choice(n * m, size=min(3, n * m), replace=False)
                v_support = rng.choice(m * p, size=min(5, m * p), replace=False)
                w_support = rng.choice(p * n, size=min(3, p * n), replace=False)
                
                for idx in u_support:
                    U[idx, col] = rng.choice([-1, 1])
                for idx in v_support:
                    V[idx, col] = rng.choice([-1, 1])
                for idx in w_support:
                    W[idx, col] = rng.choice([-1, 1])
            
            # ALS iterations
            for step in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                M_U = np.zeros((r, m * p * p * n))
                for col in range(r):
                    M_U[col, :] = np.outer(V[:, col], W[:, col]).flatten()
                T_flat = target.reshape(n * m, m * p * p * n)
                try:
                    U.T = np.linalg.lstsq(M_U, T_flat.T, rcond=None)[0]
                except:
                    return None
                
                # Update V
                M_V = np.zeros((r, n * m * p * n))
                for col in range(r):
                    M_V[col, :] = np.outer(U[:, col], W[:, col]).flatten()
                T_flat_v = np.zeros((m * p, n * m * p * n))
                for b in range(m * p):
                    T_flat_v[b, :] = target[:, b, :].flatten()
                try:
                    V.T = np.linalg.lstsq(M_V, T_flat_v.T, rcond=None)[0]
                except:
                    return None
                
                # Update W
                M_W = np.zeros((r, n * m * m * p))
                for col in range(r):
                    M_W[col, :] = np.outer(U[:, col], V[:, col]).flatten()
                T_flat_w = np.zeros((p * n, n * m * m * p))
                for c in range(p * n):
                    T_flat_w[c, :] = target[:, :, c].flatten()
                try:
                    W.T = np.linalg.lstsq(M_W, T_flat_w.T, rcond=None)[0]
                except:
                    return None
            
            # Check if we got a good solution
            recon = np.zeros_like(target)
            for col in range(r):
                recon += np.einsum('a,b,c->abc', U[:, col], V[:, col], W[:, col])
            err = np.linalg.norm(recon - target)
            
            if err < 1e-6:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify_decomposition(U_r, V_r, W_r, target, tol=1e-5):
                    return U_r, V_r, W_r
                
                # Try rounding to integers
                U_r = np.round(U)
                V_r = np.round(V)
                W_r = np.round(W)
                if verify_decomposition(U_r, V_r, W_r, target, tol=1e-5):
                    return U_r, V_r, W_r
                
                return U, V, W
            
            return None
        
        # Try multiple seeds
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = try_als_for_rank(rank, seed=seed)
            if result is not None:
                return result
        
        return None

    # First, get schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook_decomposition()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # Try to find better decompositions
    # Start from a reasonable lower bound and go up
    # For 3x3x5, let's try from 27 to 44
    for rank in range(27, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_structured_decomposition(rank)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break  # Found a valid decomposition at this rank

    U, V, W = best_result

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
