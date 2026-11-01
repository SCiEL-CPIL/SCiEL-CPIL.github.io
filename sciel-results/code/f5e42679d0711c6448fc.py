"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor indices
    # Target T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We only need the non-zero entries for the residual calculation
    target_indices_a = []
    target_indices_b = []
    target_indices_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices_a.append(a)
                target_indices_b.append(b)
                target_indices_c.append(c)
                
    target_indices_a = np.array(target_indices_a)
    target_indices_b = np.array(target_indices_b)
    target_indices_c = np.array(target_indices_c)
    num_nonzero = len(target_indices_a)
    
    # Full tensor size
    total_size = nm * mp * pn
    
    def unpack_params(theta, r):
        """Unpack flattened parameter vector into U, V, W."""
        U_flat = theta[:nm * r]
        V_flat = theta[nm * r : nm * r + mp * r]
        W_flat = theta[nm * r + mp * r :]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        return U, V, W

    def residuals(theta, r):
        """Compute residuals for least_squares."""
        U, V, W = unpack_params(theta, r)
        
        # Compute reconstructed tensor values at the non-zero positions of the target
        # R[a, b, c] = sum_k U[a, k] * V[b, k] * W[c, k]
        # We only need R at (target_indices_a, target_indices_b, target_indices_c)
        
        # Efficient calculation:
        # For each component k, we have U[:, k], V[:, k], W[:, k]
        # Contribution to R at index (a, b, c) is U[a, k] * V[b, k] * W[c, k]
        
        # We can compute this using matrix multiplications or einsum on the subsets
        # Since the number of non-zeros is small (42), we can just loop over them or use advanced indexing
        # However, doing it for every residual evaluation in least_squares might be slow if not vectorized.
        
        # Let's vectorize the reconstruction for the specific indices.
        # U_vals: shape (num_nonzero, r) where U_vals[i, k] = U[target_indices_a[i], k]
        # V_vals: shape (num_nonzero, r) where V_vals[i, k] = V[target_indices_b[i], k]
        # W_vals: shape (num_nonzero, r) where W_vals[i, k] = W[target_indices_c[i], k]
        
        U_sub = U[target_indices_a, :]  # (42, r)
        V_sub = V[target_indices_b, :]  # (42, r)
        W_sub = W[target_indices_c, :]  # (42, r)
        
        # R_vals[i] = sum_k U_sub[i, k] * V_sub[i, k] * W_sub[i, k]
        R_vals = np.einsum('ik,ik,ik->i', U_sub, V_sub, W_sub)
        
        # The target is 1.0 at these positions.
        # The residual is R_vals - 1.0
        # What about the other positions?
        # The tensor must be 0 at all other positions.
        # If we only fit the non-zero positions, we might get a solution that is non-zero elsewhere.
        # We must include the zero positions in the residual.
        
        # This requires generating the indices of all zero positions.
        # Total size 1764. Non-zeros 42. Zeros 1722.
        # This makes the residual vector length 1764.
        
        # Let's create the full index arrays once.
        # We can precompute the full target tensor as a sparse representation or just the indices.
        
        return None # Placeholder, logic below is better

    # Precompute full index arrays for residual
    # Create a list of all (a, b, c) indices
    all_indices = []
    target_vals = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                all_indices.append((a, b, c))
                # Check if this is a target index
                # We can use a set of tuples for O(1) lookup
                pass
    
    # Better: Create a boolean mask or just use the fact that target is sparse.
    # Let's create the full target array for simplicity, as 1764 floats is small.
    target_tensor = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
                
    target_flat = target_tensor.flatten()
    
    def residuals_full(theta, r):
        """Compute full residuals."""
        U, V, W = unpack_params(theta, r)
        
        # Compute full reconstructed tensor
        # R = sum_k outer(outer(U[:, k], V[:, k]), W[:, k])
        # This is equivalent to:
        # R_flat[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        
        # Using einsum on the full tensors
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # R: (nm, mp, pn)
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        
        return (R - target_tensor).flatten()

    def verify(U, V, W):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        # Recompute R
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        
        if np.max(np.abs(R - target_tensor)) > 1e-6:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            # Check if 2*val is close to an integer
            if not np.allclose(2.0 * mat, np.round(2.0 * mat), atol=1e-6):
                return False
        return True

    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_result = None
    best_rank = 42

    # Try ranks from 30 to 41
    # We start from lower ranks to find the minimum
    for rank in range(30, 42):
        if time.time() > deadline - 15:
            break
            
        param_size = rank * (nm + mp + pn)
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(hash((rank, attempt)) % 2**32)
            
            # Initialize with small random values
            theta0 = np.random.randn(param_size) * 0.5
            
            try:
                # Use least_squares
                # 'trf' is a good default for medium-sized problems
                res = least_squares(
                    residuals_full, 
                    theta0, 
                    args=(rank,),
                    method='trf',
                    max_nfev=1000
                )
                
                # Check if solution is good
                if res.cost < 1e-10:
                    U, V, W = unpack_params(res.x, rank)
                    
                    # Round to half-integers
                    U_rounded = round_to_half_integer(U)
                    V_rounded = round_to_half_integer(V)
                    W_rounded = round_to_half_integer(W)
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        best_result = (U_rounded, V_rounded, W_rounded)
                        best_rank = rank
                        break # Break attempt loop, found valid for this rank
                        # We want the minimum rank, so if we find one at rank r, 
                        # we should check if we can go lower? 
                        # The loop goes 30->41, so if we find 35, we break inner loop.
                        # Then outer loop continues to 31? No, range is 30, 31... 41.
                        # So if we find 30, we break outer loop too.
                        
            except Exception as e:
                pass
        
        if best_result is not None:
            # Found a valid decomposition at this rank. Since we iterate upwards from 30,
            # this is the lowest rank found so far. We can stop.
            break

    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42

    # Convert to lists as required
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
