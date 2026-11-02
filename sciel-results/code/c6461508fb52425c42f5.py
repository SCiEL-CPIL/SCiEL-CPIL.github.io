import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, rank, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For all i, j, jp, k, ip, kp:
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = 1 if (i==ip and k==kp and j==jp) else 0
    for i in range(n):
        for j in range(m):
            U_row = U_np[i*m+j, :]
            for jp in range(m):
                for k in range(p):
                    # Compute V[jp*p+k, :] * W[kp*n+ip, :] for all ip, kp
                    V_row = V_np[jp*p+k, :]
                    # We need: for all ip, kp: sum_r U_row[r] * V_row[r] * W[kp*n+ip, r] = target
                    # target = 1 if i==ip and k==kp and j==jp else 0
                    for ip in range(n):
                        for kp in range(p):
                            W_row = W_np[kp*n+ip, :]
                            val = np.dot(U_row * V_row, W_row)
                            target = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                            if abs(val - target) > tol:
                                return False
    return True

def _round_to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    if np.all(np.abs(doubled - rounded) < tol):
        return (rounded / 2.0).tolist()
    return x.tolist()

def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, seed=0, time_limit=None, start_time=None):
    """ALS decomposition of the matrix multiplication tensor."""
    if start_time is None:
        start_time = time.time()
    
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            return None, it
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # Build system: for each (jp, k, ip, kp), target = 1 if (i==ip and k==kp and j==jp) else 0
                # Equation: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = target
                # Let A[r] = sum over (jp,k,ip,kp) of target * V[jp*p+k, r] * W[kp*n+ip, r]
                # Let b = sum over (jp,k,ip,kp) of target^2
                # Then U[i*m+j, :] = A / b (for rank-1 per row, but we need full rank)
                # Actually, this is a linear system in U[i*m+j, :]:
                # For each (jp, k, ip, kp): dot(U[i*m+j,:], V[jp*p+k,:] * W[kp*n+ip,:]) = target
                # This is: M @ U[i*m+j, :] = b where M is (num_eqns x rank)
                
                # More efficient: build the matrix M and vector b
                # M[row, r] = V[jp*p+k, r] * W[kp*n+ip, r]
                # b[row] = target
                # Then solve M @ u = b in least squares sense
                
                # Number of equations: m * p * n * p
                # For fixed i, j, the target is 1 only when jp==j, ip==i, k==kp
                # So b is sparse: only m*p... no, only p entries are 1 (when jp==j, ip==i, k==kp for k=0..p-1)
                
                # Actually for fixed i, j:
                # target(jp, k, ip, kp) = 1 if jp==j and ip==i and k==kp, else 0
                # So the non-zero targets are at: (jp=j, ip=i, k=kp) for kp=0..p-1
                # That's p non-zero entries.
                
                # M has rows indexed by (jp, k, ip, kp), total m*p*n*p rows
                # But we only need to consider all rows for the LS problem.
                
                # Efficient approach:
                # M @ u = b
                # M[jp, k, ip, kp, r] = V[jp*p+k, r] * W[kp*n+ip, r]
                # b[jp, k, ip, kp] = 1 if jp==j and ip==i and k==kp else 0
                
                # We can write: M = reshape of outer products
                # Let's build it efficiently.
                
                # For each r, M[:, r] = V_flat * W_flat where V_flat is (m*p, n*p) reshaped and W_flat similarly
                # Actually: M[jp*p+k, ip, kp, r]... this is getting complex.
                
                # Simpler: just build the full matrix for small sizes
                num_eqns = m * p * n * p
                M = np.zeros((num_eqns, rank))
                b = np.zeros(num_eqns)
                
                idx = 0
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                V_row = V[jp*p+k, :]
                                W_row = W[kp*n+ip, :]
                                M[idx, :] = V_row * W_row
                                if jp == j and ip == i and k == kp:
                                    b[idx] = 1.0
                                idx += 1
                
                # Solve least squares
                try:
                    U[i*m+j, :], residuals, rank_ls, sv = np.linalg.lstsq(M, b, rcond=None)
                except:
                    pass
        
        # Update V: for each (jp, k), solve least squares
        for jp in range(m):
            for k in range(p):
                num_eqns = m * p * n * p
                M = np.zeros((num_eqns, rank))
                b = np.zeros(num_eqns)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for ip in range(n):
                            for kp in range(p):
                                # Equation: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = target
                                # target = 1 if i==ip and k==kp and j==jp else 0
                                # For fixed (jp, k), unknown is V[jp*p+k, :]
                                # M[row, r] = U[i*m+j, r] * W[kp*n+ip, r]
                                # b[row] = target
                                U_row = U[i*m+j, :]
                                W_row = W[kp*n+ip, :]
                                M[idx, :] = U_row * W_row
                                if i == ip and k == kp and j == jp:
                                    b[idx] = 1.0
                                idx += 1
                
                try:
                    V[jp*p+k, :], residuals, rank_ls, sv = np.linalg.lstsq(M, b, rcond=None)
                except:
                    pass
        
        # Update W: for each (kp, ip), solve least squares
        for kp in range(p):
            for ip in range(n):
                num_eqns = m * p * n * p
                M = np.zeros((num_eqns, rank))
                b = np.zeros(num_eqns)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                # Equation: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = target
                                # target = 1 if i==ip and k==kp and j==jp else 0
                                # For fixed (kp, ip), unknown is W[kp*n+ip, :]
                                # M[row, r] = U[i*m+j, r] * V[jp*p+k, r]
                                # b[row] = target
                                U_row = U[i*m+j, :]
                                V_row = V[jp*p+k, :]
                                M[idx, :] = U_row * V_row
                                if i == ip and k == kp and j == jp:
                                    b[idx] = 1.0
                                idx += 1
                
                try:
                    W[kp*n+ip, :], residuals, rank_ls, sv = np.linalg.lstsq(M, b, rcond=None)
                except:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            # Quick check on a few entries
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                        err += (val - 1.0)**2
            if err < tol:
                break
    
    return (U, V, W), it

def _construct_trivial(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Start with trivial solution
    U_triv, V_triv, W_triv, rank_trivial = _construct_trivial(n, m, p)
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = rank_trivial
    
    # Try to find lower rank decompositions
    # Try ranks from just below trivial down
    for rank in range(rank_trivial - 1, max(1, rank_trivial - 30), -1):
        if time.time() - start_time > deadline:
            break
        
        for seed in range(3):
            if time.time() - start_time > deadline:
                break
            
            result, iters = _als_decompose(n, m, p, rank, max_iter=100, seed=seed, 
                                           time_limit=deadline - (time.time() - start_time),
                                           start_time=start_time)
            if result is None:
                continue
            
            U, V, W = result
            if _verify(U, V, W, rank, n, m, p):
                # Round to half-integers if possible
                U_rounded = _round_to_half_int(U)
                V_rounded = _round_to_half_int(V)
                W_rounded = _round_to_half_int(W)
                
                # Verify rounded version
                if _verify(U_rounded, V_rounded, W_rounded, rank, n, m, p, tol=1e-6):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank
                    break
                else:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break
        
        if best_rank < rank_trivial:
            break
    
    return best_result
    
    # Pre-compute target tensor T[i,j,k] = 1 (scalar product structure)
    # The tensor to decompose is T such that C[i,k] = sum_j A[i,j] B[j,k]
    # This is equivalent to decomposing the tensor T_{i,j,k} where T_{i,j,k} = delta_{i,i'}delta_{j,j'}delta_{k,k'} in the appropriate basis
    # Actually, the matrix multiplication tensor has components T_{i,j,k} = 1 for all i,j,k in the standard basis
    # No, the multiplication tensor T is defined such that T(e_i, e_j, e_k) = delta_{i,k} (if we identify rows and columns)
    # Let's use the standard definition: 
    # The tensor T in V* x W* x U* (dual spaces) or V x W x U
    # T(a, b, c) = <a, c> * <b, ...> ? No.
    # T(A, B, C) = tr(C^T A B) ? No, that's a scalar.
    # The tensor is T in V* ⊗ W* ⊗ U* such that T(a, b, c) = sum_{i,j} a_i b_j c_{ij} ? No.
    # Standard: T_{ijk} = 1 if i==k and j==j? 
    # Let's stick to the factorization: C[i,k] = sum_j A[i,j] B[j,k]
    # We want U, V, W such that:
    # U[i, r] V[j, r] W[k, r] summed over r gives the coefficient for A[i,j]B[j,k] in C[i,k]
    # The coefficient is 1 if i==i', j==j', k==k' in the expansion?
    # Actually, the tensor T_{i, j, k} is defined by:
    # T_{i, j, k} = 1 if we are computing C[i,k] from A[i,j] and B[j,k]?
    # The structure is: For fixed i, j, k, the term A[i,j] B[j,k] contributes to C[i,k].
    # So the tensor T has indices i (row of A), j (col of A / row of B), k (col of B).
    # T_{i, j, k} = 1 for all i, j, k? No.
    # T_{i, j, k} is the coefficient of A[i,j] B[j,k] in C[i,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j] B[j,k] in C[i,k] is 1 if the indices match.
    # Wait, the tensor is T in R^{n x m x p} ?
    # T_{i, j, k} = 1 ?
    # If T_{i, j, k} = 1, then sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k?
    # That would mean U V^T W^T is the all-ones tensor.
    # But matrix multiplication is not just all-ones.
    # The tensor for matrix multiplication is T_{i, j, k} = delta_{i, i'} delta_{j, j'} delta_{k, k'}?
    # Let's look at the indices again.
    # C_{i,k} = \sum_j A_{i,j} B_{j,k}
    # The tensor T_{i, j, k} satisfies: \sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = \sum_{i,k} C_{i,k} (something)?
    # No, the standard definition is:
    # T_{i, j, k} = 1 if i==i', j==j', k==k' is not right.
    # T_{i, j, k} is the tensor in U* ⊗ V* ⊗ W*?
    # Let's use the factorization: C_{i,k} = \sum_r U_{i, r} V_{k, r} (sum_j A_{i,j} W_{j, r})? No.
    # The factors are U (n x R), V (p x R), W (m x R)?
    # The task says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # This is different from the standard U, V, W sizes.
    # Standard: U (n x R), V (m x R), W (p x R) ?
    # If U is (n*m, R), it flattens A.
    # If V is (m*p, R), it flattens B.
    # If W is (p*n, R), it flattens C.
    # Then the decomposition is:
    # C_flat = \sum_r U_{idx_A, r} V_{idx_B, r} W_{idx_C, r} ?
    # No, that's not right.
    # The condition is:
    # \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'} ?
    # No.
    # The condition for matrix multiplication tensor decomposition with these shapes is:
    # For all i, j, k:
    # \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 ?
    # If this is the case, the tensor is the all-ones tensor.
    # But matrix multiplication tensor is NOT the all-ones tensor.
    # Let's re-read the task carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 6 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This implies the tensor T_{i, j, k} is defined such that:
    # \sum_r U_{i,j} V_{j,k} W_{k,i} ? No.
    # The standard multiplication tensor T is in R^{n} \otimes R^{m} \otimes R^{p} ?
    # No, T is in (R^n \otimes R^p)^* \otimes R^m ?
    # Let's assume the standard definition:
    # T_{i, j, k} = 1 if i==i', j==j', k==k' is wrong.
    # T_{i, j, k} = \delta_{i, k} ? No.
    # Let's look at the "official trace convention".
    # Usually, the multiplication tensor is T(A, B) = A B.
    # As a bilinear map, it's linear in A and B.
    # As a trilinear form, T(A, B, C) = tr(C^T A B).
    # Then T(e_i, e_j, e_k) = tr(E_k^T E_i E_j) = tr(E_{k,i,j})?
    # E_i E_j is zero unless i==j? No, E_i is row i, E_j is col j?
    # Let's use the index notation:
    # (A B)_{i,k} = \sum_j A_{i,j} B_{j,k}
    # The tensor T_{i, j, k} is such that \sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = \sum_{i,k} C_{i,k} ? No.
    # The tensor is T_{i, j, k} = 1 for all i,j,k?
    # If T_{i,j,k} = 1, then \sum_{i,j,k} A_{i,j} B_{j,k} = (sum A) (sum B).
    # That is not matrix multiplication.
    
    # Correct Tensor Definition:
    # The multiplication tensor T is in U^* \otimes V^* \otimes W^* where U=R^n, V=R^m, W=R^p?
    # No, the spaces are:
    # A is in R^{n x m}
    # B is in R^{m x p}
    # C is in R^{n x p}
    # The tensor T is in (R^{n x m})^* \otimes (R^{m x p})^* \otimes (R^{n x p})^* ?
    # No, it's a map (R^{n x m}) \otimes (R^{m x p}) -> R^{n x p}
    # This is an element of (R^{n x m})^* \otimes (R^{m x p})^* \otimes R^{n x p}
    # The components are T_{(i,j), (j',k'), (i',k')} ?
    # The condition is: C_{i',k'} = \sum_{i,j,j',k'} T_{(i,j), (j',k'), (i',k')} A_{i,j} B_{j',k'}
    # We need C_{i,k} = \sum_j A_{i,j} B_{j,k}
    # So T_{(i,j), (j,k), (i',k')} = \delta_{i,i'} \delta_{k,k'} \delta_{j,j} ?
    # No, the indices must match.
    # T_{(i,j), (j',k'), (i',k')} = 1 if i==i' and k==k' and j==j' ?
    # If so, then \sum_{i,j,j',k'} T_{...} A_{i,j} B_{j',k'} = \sum_{i,j} A_{i,j} B_{j,k} ?
    # If j==j', then yes.
    # So T_{(i,j), (j',k'), (i',k')} = \delta_{i,i'} \delta_{k,k'} \delta_{j,j'}
    # This means T is the identity tensor on the "matching" indices.
    
    # Now, the factors are:
    # U: (n*m, R) -> U_{(i,j), r}
    # V: (m*p, R) -> V_{(j',k'), r}
    # W: (p*n, R) -> W_{(k',i'), r} ?
    # The task says W[k*n+i, r]. So W_{(k,i), r}.
    # The decomposition is:
    # T_{(i,j), (j',k'), (i',k')} = \sum_r U_{(i,j), r} V_{(j',k'), r} W_{(k',i'), r} ?
    # Note the W index is (k',i') in the target, but W is stored as (k,i).
    # So we need: \sum_r U_{i*m+j, r} V_{j'*p+k', r} W_{k'*n+i', r} = \delta_{i,i'} \delta_{k,k'} \delta_{j,j'}
    # Wait, the target is \delta_{i,i'} \delta_{k,k'} \delta_{j,j'}.
    # So we are decomposing the tensor T where T_{i,j,j',k',i',k'} = \delta_{i,i'} \delta_{k,k'} \delta_{j,j'}
    # This is a tensor in (R^{n*m})^* \otimes (R^{m*p})^* \otimes (R^{p*n})^* ?
    # No, it's a tensor in R^{n*m} \otimes R^{m*p} \otimes R^{p*n} ?
    # The rank is the minimum R such that T = \sum_r U_r \otimes V_r \otimes W_r
    # where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
    # U_{i*m+j, r} is the r-th component of U_r.
    
    # So the target tensor T has components:
    # T_{(i,j), (j',k'), (k',i')} = 1 if i==i' and k==k' and j==j' else 0.
    # Note that the third index of T is (k', i') to match W[k*n+i].
    # So T_{(i,j), (j',k'), (k',i')} = \delta_{i,i'} \delta_{k,k'} \delta_{j,j'}
    
    # Let's verify with the trivial solution.
    # U_{i*m+j, r} = 1 if r corresponds to (i,j,k) ?
    # The trivial solution has rank n*m*p = 90.
    # For each (i,j,k), we have a term.
    # U_{i*m+j, r_{ijk}} = 1
    # V_{j*p+k, r_{ijk}} = 1
    # W_{k*n+i, r_{ijk}} = 1
    # Sum over r:
    # \sum_r U_{i*m+j, r} V_{j'*p+k', r} W_{k'*n+i', r}
    # = 1 if there exists r such that U_{i*m+j, r}=1, V_{j'*p+k', r}=1, W_{k'*n+i', r}=1
    # In the trivial solution, U_{i*m+j, r}=1 only if r corresponds to (i,j,k) for some k.
    # So U_{i*m+j, r_{ijk}}=1 for any k.
    # V_{j'*p+k', r}=1 only if r corresponds to (i,j',k') for some i.
    # W_{k'*n+i', r}=1 only if r corresponds to (i',j,k') for some j.
    # For the sum to be non-zero, we need r such that:
    # r = (i, j, k) from U
    # r = (i, j', k') from V
    # r = (i', j, k') from W
    # This implies i=i', j=j', k=k'.
    # So the sum is 1 if i=i', j=j', k=k', else 0.
    # This matches the target.
    
    # So the target tensor is T_{(i,j), (j',k'), (k',i')} = \delta_{i,i'} \delta_{k,k'} \delta_{j,j'}
    # This is the "identity" tensor for the bilinear map.
    
    # To find a lower rank, we need to find U, V, W of rank R < 90 such that:
    # \sum_r U_{i*m+j, r} V_{j'*p+k', r} W_{k'*n+i', r} = \delta_{i,i'} \delta_{k,k'} \delta_{j,j'}
    
    # This is a system of equations.
    # Number of equations: n*p*n*m = 3*6*3*5 = 270 ?
    # No, the indices are i,j,j',k',i',k'.
    # But the condition is \delta_{i,i'} \delta_{k,k'} \delta_{j,j'}.
    # This means the tensor is zero unless i=i', k=k', j=j'.
    # So we only need to satisfy the conditions where i=i', k=k', j=j'.
    # For i=i', k=k', j=j', the value must be 1.
    # For all other cases, the value must be 0.
    
    # Let's count the constraints.
    # Non-zero constraints: i=i', k=k', j=j'.
    # i in [0,2], k in [0,5], j in [0,4].
    # So 3*6*5 = 90 constraints where the sum must be 1.
    # Zero constraints: all other (i,j,j',k',i',k') combinations.
    # Total combinations: (n*m) * (m*p) * (p*n) = 15 * 30 * 18 = 8100.
    # 90 of them are 1, 8010 are 0.
    
    # We can solve this using ALS or other methods.
    # Given the time constraints and the difficulty, we will implement a simple ALS solver.
    
    def verify(U, V, W, rank):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Check non-zero constraints
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        
        # Check zero constraints (sample a few if too many)
        # Total zero constraints: 8100 - 90 = 8010
        # We can check all if fast enough
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            if i != k and j != jp and k != kp:
                                # This is not sufficient. We need to check all where NOT (i==i' and k==k' and j==j')
                                # Since i' is i, k' is k in the non-zero case?
                                # No, the indices are i, j, j', k', i', k'.
                                # The condition is i=i', k=k', j=j'.
                                # So for a fixed i, j, j', k', i', k':
                                # If i==i' and k==k' and j==j', it must be 1.
                                # Otherwise, it must be 0.
                                # In the loop above, we fixed i, j, k.
                                # The variables are j', k', i'.
                                # So for fixed i, j, k:
                                # We need to check all j', k', i' such that NOT (i==i' and k==k' and j==j')
                                # But i' is a variable, not fixed to i.
                                # So we need a 6-loop or nested loops.
                                pass
        
        # Simpler verification:
        # Construct the tensor T from U, V, W and compare with target.
        # T_{i,j,j',k',i',k'} = sum_r U[i*m+j, r] V[j'*p+k', r] W[k'*n+i', r]
        # Target is 1 if i==i', k==k', j==j' else 0.
        
        # We can check this by iterating over all 8100 combinations.
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                target = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                                val = 0.0
                                for r in range(rank):
                                    val += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                                if abs(val - target) > 1e-6:
                                    return False
        return True

    # ALS Solver
    def als_solve(n, m, p, rank, max_iter=50, tol=1e-8, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for it in range(max_iter):
            # Update U
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, :] = argmin_u sum_{jp,k,ip,kp} (target - u_jp_k_ip_kp) * V[jp*p+k, :] * W[kp*n+ip, :]
                    # This is a least squares problem.
                    # Let A = V[jp*p+k, :] * W[kp*n+ip, :]
                    # We want to minimize || A u - b ||^2
                    # Where b is the vector of targets for fixed i,j.
                    
                    # Targets for fixed i,j:
                    # b_{jp,k,ip,kp} = 1 if i==ip, k==kp, j==jp else 0
                    # Note: k is fixed in the loop? No, k is a variable in the target.
                    # The target depends on i, j, jp, k, ip, kp.
                    # For fixed i, j, the variables are jp, k, ip, kp.
                    # b_{jp,k,ip,kp} = 1 if jp==j and k==k (wait, k is a variable)
                    # No, the target is \delta_{i,ip} \delta_{k,kp} \delta_{j,jp}
                    # For fixed i, j:
                    # b_{jp,k,ip,kp} = 1 if jp==j and k==kp and i==ip
                    # Otherwise 0.
                    
                    # So for fixed i, j:
                    # b is non-zero only when jp==j, ip==i, and k==kp.
                    # The variables are jp (m), k (p), ip (n), kp (p).
                    # Number of equations: m*p*n*p = 5*6*3*6 = 540.
                    # Unknowns: rank R.
                    
                    # We can form the matrix A and vector b.
                    # A has shape (540, R)
                    # b has shape (540,)
                    
                    # But this is expensive to form for each i,j.
                    # We can use a more efficient method.
                    
                    # Alternatively, we can update U, V, W using a simpler method.
                    # Given the time, we will skip the full ALS implementation and just return the trivial solution.
                    # If we have time, we can try a few random restarts with rank 80, 70, etc.
                    pass
        
        return None

    # Due to complexity and time, we return the trivial solution.
    return best_result
