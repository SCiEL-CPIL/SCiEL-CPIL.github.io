"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin for overhead
    
    n, m, p = DIMENSIONS
    
    # Trivial rank-48 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
    
    # The tensor T for matrix multiplication M_{n,m,p} is defined by:
    # T[i, j, k] = 1 if the operation involves A[i,j] and B[j,k] to produce C[i,k]
    # Wait, the standard definition is:
    # (A * B)[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, j, k) where i in [0, n), j in [0, m), k in [0, p).
    # T[i, j, k] = 1 for all valid i, j, k.
    # Actually, the tensor rank is the minimum number of rank-1 tensors that sum to T.
    # A rank-1 tensor is u_i v_j w_k.
    # We want sum_r u_r[i] v_r[j] w_r[k] = T[i, j, k] = 1 (for all i,j,k).
    # Wait, this is not quite right.
    # The multiplication tensor is T_{i, j, k} = delta_{i, j, k}? No.
    # Let's look at the structure.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # This is a trilinear form.
    # The tensor T has entries T_{i, j, k} = 1.
    # We want to decompose T as sum_r u_r[i] v_r[j] w_r[k].
    # So sum_r u_r[i] v_r[j] w_r[k] = 1 for all i, j, k.
    # This implies that the matrices U, V, W satisfy:
    # sum_r U[:, r] V[:, r]^T ... no, that's not it.
    # The condition is:
    # For all i, j, k: sum_{r=1}^R u_r[i] v_r[j] w_r[k] = 1.
    
    # Let's verify this interpretation.
    # If U[i, r] = u_r[i], V[j, r] = v_r[j], W[k, r] = w_r[k].
    # Then the sum is sum_r U[i, r] V[j, r] W[k, r].
    # We want this to be 1 for all i, j, k.
    # This means the tensor formed by the outer product sum is the all-ones tensor.
    # Is the matrix multiplication tensor the all-ones tensor?
    # No.
    # The matrix multiplication tensor T_{n,m,p} is defined such that:
    # (A B)_{i, k} = sum_j T_{i, j, k} A_{i, j} B_{j, k}.
    # So T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} appears in the sum for C_{i, k}.
    # It does appear for all valid i, j, k.
    # So T_{i, j, k} = 1 for all i, j, k.
    # Yes, the tensor is the all-ones tensor of size n x m x p.
    
    # So the problem reduces to:
    # Find minimum R such that there exist vectors u_r (length n), v_r (length m), w_r (length p)
    # such that sum_{r=1}^R u_r[i] v_r[j] w_r[k] = 1 for all i, j, k.
    
    # This is equivalent to finding a low-rank decomposition of the all-ones tensor.
    # The all-ones tensor J_{n,m,p} has rank 1?
    # J_{i, j, k} = 1.
    # Can we write 1 = sum_r u_r[i] v_r[j] w_r[k]?
    # If we take R=1, u[i]=1, v[j]=1, w[k]=1.
    # Then sum is 1*1*1 = 1.
    # So the rank is 1?
    # That can't be right. The rank of matrix multiplication is known to be much higher.
    
    # I am confusing the tensor definition.
    # The matrix multiplication tensor T is NOT the all-ones tensor in the standard basis of the input/output spaces.
    # The inputs are A (n*m dims) and B (m*p dims). The output is C (n*p dims).
    # The tensor T is an element of (R^{nm}*) \otimes (R^{mp}*) \otimes (R^{np}).
    # The entry T_{a, b, c} is 1 if the coefficient of A_a B_b in C_c is 1.
    # Let a = i*m + j, b = j'*p + k, c = k'*n + i'.
    # The term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So T_{a, b, c} = 1 if a corresponds to (i, j), b corresponds to (j, k), and c corresponds to (i, k).
    # Note the index j must match.
    # So T_{i, j, k, i', j', k'} is non-zero only if i=i', j=j', k=k'.
    # Wait, the indices are:
    # Input A index: (i, j)
    # Input B index: (j, k)
    # Output C index: (i, k)
    # The tensor T has indices (A_idx, B_idx, C_idx).
    # T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(i,k).
    # Otherwise 0.
    
    # So T is not the all-ones tensor. It is a sparse tensor.
    # Specifically, T is the "structure tensor" of the multiplication.
    
    # Let's re-read the task interface.
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # The decomposition is:
    # sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    # where T[a, b, c] is 1 if a=(i,j), b=(j,k), c=(i,k) and 0 otherwise.
    
    # So we are looking for a decomposition of this specific sparse tensor.
    
    # Let's check the trivial decomposition again.
    # For each (i, j, k), we have a rank-1 term.
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    # This matches the condition.
    
    # Now, can we do better?
    # We want to minimize R.
    # This is a hard NP-hard problem in general, but for small n, m, p we can search.
    
    # Strategy: Greedy Random Search.
    # Start with R = lower_bound.
    # Lower bound is max(n*m, m*p, p*n) = 24.
    # Let's try R from 24 upwards.
    # For each R, try to find U, V, W.
    # We can use a greedy approach:
    # 1. Initialize U, V, W randomly.
    # 2. Iteratively update one factor to minimize the error.
    # 3. Alternatively, use a "greedy selection" of rank-1 components.
    
    # Let's try a different approach: Local Search / Simulated Annealing on the coefficients.
    # Or better: Use the fact that the tensor is sparse.
    
    # Let's try to construct a decomposition with R < 48.
    # The Strassen algorithm suggests R=42 might be possible.
    # The current program's Strassen attempt failed to construct a valid rank-42 decomposition because the block products were not decomposed correctly or the rank count was wrong.
    # Let's try to implement a correct Strassen-like decomposition if possible, or use a search to find it.
    
    # Given the time limit, a full search for R=24..47 might be too slow.
    # However, we can try to verify if a known decomposition exists.
    # For 2x4x6, is there a known rank?
    # The rank of 2x2x2 is 7 (Strassen).
    # The rank of 2x4x4 is 18? (Bini et al.)
    # The rank of 2x4x6 is likely 42 or lower.
    
    # Let's try to implement a simple local optimization to refine a random initialization for a given R.
    # Objective: Minimize || sum_r U[:,r] V[:,r]^T ... || - T ||
    # Actually, we want sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    # This is a tensor completion / decomposition problem.
    
    # Let's use a greedy rank-1 update approach.
    # 1. Set residual R = T.
    # 2. For step = 1 to MaxSteps:
    #    Find u, v, w that best approximate R.
    #    This is the best rank-1 approximation of R.
    #    We can find this by maximizing the inner product <R, u v w>.
    #    This is a non-convex problem.
    #    We can use a simple heuristic:
    #    - Randomly initialize u, v, w.
    #    - Update u = R \cdot (v, w) (contract R with v and w).
    #    - Update v = R \cdot (u, w).
    #    - Update w = R \cdot (u, v).
    #    - Normalize.
    #    - Subtract u v w from R.
    #    - Add u, v, w to the decomposition.
    
    # This is the "Greedy Rank-1" algorithm. It is not guaranteed to find the global minimum, but it often works well for sparse tensors.
    
    # Let's implement this.
    
    # First, construct the tensor T as a 3D array.
    # T[i, j, k] is not quite right because the indices are flattened.
    # Let's create a 3D array T_3d of shape (n, m, p) where T_3d[i, j, k] = 1.
    # Wait, the tensor T has indices (A_idx, B_idx, C_idx).
    # A_idx = i*m + j
    # B_idx = j*p + k
    # C_idx = k*n + i
    # The condition is that the j in A_idx and B_idx must be the same.
    # So T is not a simple 3D array of ones.
    # T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(i,k).
    
    # Let's create the tensor T as a 3D array of size (n*m, m*p, n*p).
    # This is 8 * 24 * 12 = 2304 elements. Small enough.
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1
    
    # Greedy Rank-1 Decomposition
    def greedy_decompose(T, max_rank, time_limit=60):
        """
        Greedy algorithm to find a rank-R decomposition of tensor T.
        Returns U, V, W if successful, else None.
        """
        n_a, n_b, n_c = T.shape
        R = max_rank
        U = np.zeros((n_a, R))
        V = np.zeros((n_b, R))
        W = np.zeros((n_c, R))
        
        residual = T.copy()
        
        for r in range(R):
            # Find best rank-1 approximation of residual
            # Use a few random restarts to find a good u, v, w
            best_err = np.inf
            best_u = None
            best_v = None
            best_w = None
            
            num_restarts = 10
            for _ in range(num_restarts):
                u = np.random.randn(n_a)
                v = np.random.randn(n_b)
                w = np.random.randn(n_c)
                
                # Iterative updates
                for _ in range(20):
                    # Update u: u_new[i] = sum_{b,c} residual[i, b, c] v[b] w[c]
                    # This is a contraction.
                    # u = np.einsum('ibc,b,c->i', residual, v, w)
                    # v = np.einsum('ibc,i,c->b', residual, u, w)
                    # w = np.einsum('ibc,i,b->c', residual, u, v)
                    
                    # Normalize to prevent divergence
                    u_norm = np.linalg.norm(u)
                    v_norm = np.linalg.norm(v)
                    w_norm = np.linalg.norm(w)
                    if u_norm > 0: u /= u_norm
                    if v_norm > 0: v /= v_norm
                    if w_norm > 0: w /= w_norm
                    
                    # Calculate error
                    # err = np.linalg.norm(residual - np.einsum('i,b,c->ibc', u, v, w))
                    # This is expensive. We can just check the inner product.
                    # The goal is to maximize <residual, u v w>.
                    # If we subtract the projection, the residual decreases.
                    
                    # Let's just do the update and normalization.
                    # We need to be careful with the scaling.
                    # The optimal scaling for u, given v, w is:
                    # u = (residual * v * w) / (v * w * u)
                    # Let's implement a simpler version:
                    # 1. Compute u_proj = einsum('ibc,b,c->i', residual, v, w)
                    # 2. Compute v_proj = einsum('ibc,i,c->b', residual, u, w)
                    # 3. Compute w_proj = einsum('ibc,i,b->c', residual, u, v)
                    # 4. Set u = u_proj, v = v_proj, w = w_proj
                    # 5. Normalize u, v, w such that u.v.w = 1? Or just normalize each to unit norm?
                    # If we normalize to unit norm, the scaling factor is lost.
                    # We need to keep the scaling.
                    
                    # Let's use the standard multiplicative update for tensor rank-1:
                    # u <- u * (residual * v * w) / (u * (residual * u * v * w))
                    # This is complex.
                    
                    # Simpler: Just use the projected vectors and renormalize to unit norm,
                    # and store the scaling factor separately?
                    # No, we just want to find the direction. The magnitude is determined by the residual.
                    # Actually, for the greedy step, we want the BEST rank-1 approximation.
                    # The best rank-1 approximation of a tensor T is given by the top eigenvector of the matricizations.
                    # We can approximate this with power iteration.
                    
                    # Let's do power iteration on the matricization.
                    # Matricize T to T_a (n_a x n_b n_c).
                    # T_a[i, b*n_c+c] = T[i, b, c].
                    # The top eigenvector of T_a T_a^T is u.
                    # Then v, w can be found from the corresponding singular vectors.
                    
                    # This is more robust.
                    # But it's expensive to do for every restart.
                    
                    # Let's stick to the simple gradient-like update.
                    # u = einsum('ibc,b,c->i', residual, v, w)
                    # v = einsum('ibc,i,c->b', residual, u, w)
                    # w = einsum('ibc,i,b->c', residual, u, v)
                    
                    # And then normalize u, v, w to unit norm.
                    # The actual rank-1 tensor is alpha * u v w.
                    # We want to maximize <residual, u v w>.
                    # Since u, v, w are unit norm, the max value is the spectral norm.
                    # We just need the directions.
                    
                    u_new = np.einsum('ibc,b,c->i', residual, v, w)
                    v_new = np.einsum('ibc,i,c->b', residual, u, w)
                    w_new = np.einsum('ibc,i,b->c', residual, u, v)
                    
                    u_norm = np.linalg.norm(u_new)
                    v_norm = np.linalg.norm(v_new)
                    w_norm = np.linalg.norm(w_new)
                    
                    if u_norm < 1e-9 or v_norm < 1e-9 or w_norm < 1e-9:
                        break
                    
                    u = u_new / u_norm
                    v = v_new / v_norm
                    w = w_new / w_norm
                    
                # After convergence, calculate the error
                # The rank-1 tensor is u v w.
                # We want to subtract it from residual.
                # But we need the correct scaling.
                # The projection of residual onto u v w is <residual, u v w> u v w.
                # Let's compute this.
                proj_val = np.einsum('ibc,i,b,c', residual, u, v, w)
                
                # Check if this is a good improvement
                # The reduction in norm squared is proj_val^2.
                # We want to maximize proj_val^2.
                err_reduction = proj_val * proj_val
                
                if err_reduction > best_err:
                    best_err = err_reduction
                    best_u = u.copy()
                    best_v = v.copy()
                    best_w = w.copy()
                    best_proj_val = proj_val
            
            # Add the best component
            if best_u is not None:
                U[:, r] = best_u
                V[:, r] = best_v
                W[:, r] = best_w
                # Subtract the component from residual
                # The component is best_proj_val * best_u best_v best_w
                # Wait, U, V, W are the factors.
                # The tensor is sum_r U[:,r] V[:,r] W[:,r].
                # So we should set U[:,r] = best_proj_val * best_u?
                # No, we can absorb the scaling into one of the vectors.
                # Let's set U[:,r] = best_u, V[:,r] = best_v, W[:,r] = best_w * best_proj_val.
                # Or just keep them unit norm and adjust later?
                # The evaluator checks the exact decomposition.
                # So we need sum_r U[:,r] V[:,r] W[:,r] = T.
                # If we set U[:,r] = best_u, V[:,r] = best_v, W[:,r] = best_w * best_proj_val,
                # then the term is best_u best_v (best_w * best_proj_val) = best_proj_val best_u best_v best_w.
                # This is the projection.
                # So we subtract this from residual.
                
                U[:, r] = best_u
                V[:, r] = best_v
                W[:, r] = best_w * best_proj_val
                
                residual -= np.einsum('i,b,c->ibc', U[:, r], V[:, r], W[:, r])
                
                # Check if residual is zero
                if np.linalg.norm(residual) < 1e-9:
                    # We found an exact decomposition with rank r+1
                    return U[:, :r+1], V[:, :r+1], W[:, :r+1]
            
            # Check time
            if time.time() > time_limit:
                return None
        
        return None
    
    # Try ranks from 24 to 47
    # We start with a high rank and go down? Or low and go up?
    # Greedy is better if we start with the tensor and subtract components.
    # So we should try to find a decomposition with R components.
    # If we fail, we try R+1.
    # But if we find a decomposition with R, it might not be minimal.
    # We want the MINIMAL rank.
    # So we should try R=24, 25, ... until we find one.
    # But if R=24 is not possible, the greedy algorithm might fail or produce a non-exact result.
    # We check for exactness (norm < 1e-9).
    
    best_result = None
    min_rank_found = 48 # Fallback
    
    for R in range(24, 48):
        if time.time() > deadline - 10:
            break
            
        # Try to find a decomposition of rank R
        # We might need multiple attempts with different random seeds
        success = False
        for attempt in range(3):
            np.random.seed(attempt * 1000 + R)
            U, V, W = greedy_decompose(T, R, time_limit=10)
            if U is not None:
                # Verify the decomposition
                # sum_r U[:,r] V[:,r] W[:,r] == T
                reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(reconstructed, T, atol=1e-9):
                    success = True
                    best_result = (U, V, W)
                    min_rank_found = R
                    break
        
        if success:
            # Found a valid decomposition with rank R
            # Since we iterate from low to high, this is likely minimal (or close to it)
            # We can break here if we want to be fast, but we might find an even lower one?
            # No, we started from 24. So 24 is the lowest.
            # If we found one at 24, it's the best possible (given the lower bound).
            # If we found one at 42, it's better than 48.
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Ensure entries are integer/half-integer?
        # The task says "integer/half-integer entries".
        # The greedy algorithm produces real numbers.
        # We need to round them?
        # If we round, the decomposition might break.
        # The task says "exactly decompose".
        # So we must ensure the entries are exactly representable.
        # If the solution is rational, we can find it.
        # But greedy might produce irrational numbers.
        # However, for integer tensors, the optimal decomposition often has rational/integer entries.
        # Let's check if the found U, V, W are close to integers/half-integers.
        # If not, we might have to fall back to the trivial one.
        
        # Let's check if the entries are close to multiples of 0.5
        def is_close_to_half_int(arr):
            # Multiply by 2 and check if close to integer
            arr2 = arr * 2
            rounded = np.round(arr2)
            return np.allclose(arr2, rounded, atol=1e-6)
        
        if is_close_to_half_int(U) and is_close_to_half_int(V) and is_close_to_half_int(W):
            # Round to nearest half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Re-verify
            reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(reconstructed, T, atol=1e-6):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to trivial rank 48
    return {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }

# EVOLVE-BLOCK-END
