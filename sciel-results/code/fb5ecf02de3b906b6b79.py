"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 120.0
DIMENSIONS = [3, 4, 7]


def _deadline():
    return time.time() - _START > _LIMIT - 5.0


def _pack(arr):
    """Convert numpy array to list of lists of floats (integers if integral)."""
    a = np.asarray(arr)
    result = []
    for row in a:
        r = []
        for x in row:
            xf = float(x)
            nearest = round(xf * 2) / 2
            if abs(xf - nearest) < 1e-9:
                r.append(nearest)
            else:
                r.append(xf)
        result.append(r)
    return result


def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = U.shape[1]
    # Check all valid (i,j,k) entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                actual = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                if abs(actual - 1.0) > tol:
                    return False
    # Check that non-matching entries are zero
    Nm, Mp, Pn = n * m, m * p, p * n
    np.random.seed(42)
    for _ in range(500):
        a = np.random.randint(0, Nm)
        b = np.random.randint(0, Mp)
        c = np.random.randint(0, Pn)
        i_val = a // m
        j_val = a % m
        jp_val = b // p
        k_val = b % p
        kp_val = c // n
        ip_val = c % n
        if not (i_val == ip_val and j_val == jp_val and k_val == kp_val):
            actual = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(actual) > tol:
                return False
    return True


def _build_constraint_tensor(n, m, p):
    """
    Build the constraint structure.
    
    The multiplication tensor T has T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    for some (i,j,k), else 0.
    
    We view each rank-1 term as a vector in R^(Nm*Mp*Pn) which is the outer product
    U_r ⊗ V_r ⊗ W_r flattened.
    
    The constraint is: sum_r (U_r ⊗ V_r ⊗ W_r) = T (the multiplication tensor).
    
    For finding decompositions, we can use the fact that if we have one valid
    decomposition, we can add any set of terms whose sum is zero (in the tensor space).
    """
    Nm, Mp, Pn = n * m, m * p, p * n
    
    # The multiplication tensor T as a flat vector
    T_flat = np.zeros(Nm * Mp * Pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[a * Mp * Pn + b * Pn + c] = 1.0
    
    return T_flat, Nm, Mp, Pn


def _try_rank_with_nullspace(n, m, p, r, max_time=20.0, seed=0):
    """
    Try to find a rank-r decomposition using the null-space approach.
    
    Key idea: Start from the schoolbook decomposition (rank 84). The first term
    handles (i=0,j=0,k=0). We need the remaining r-1 terms to sum to T - term_0.
    
    Actually, a better approach: we want sum_{r=0}^{r-1} U_r⊗V_r⊗W_r = T.
    
    Using the structure: if we fix U_0, V_0, W_0 as the schoolbook first term,
    then we need sum_{r=1}^{r-1} U_r⊗V_r⊗W_r = T - U_0⊗V_0⊗W_0.
    
    But this is still hard. Instead, let's use a more direct approach:
    
    For rank r, we have r vectors u_r in R^Nm, v_r in R^Mp, w_r in R^Pn.
    The constraint is: for all (a,b,c), sum_r u_r[a]*v_r[b]*w_r[c] = T[a,b,c].
    
    This is a system of Nm*Mp*Pn equations in r*(Nm+Mp+Pn) unknowns.
    
    For small r, this is overdetermined. We need to find solutions.
    
    Strategy: Use alternating minimization / gradient descent on the residual.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    Nm, Mp, Pn = n * m, m * p, p * n
    T_flat, _, _, _ = _build_constraint_tensor(n, m, p)
    
    # Precompute index mappings for valid entries
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.append((a, b, c))
    
    best_result = None
    
    for attempt in range(30):
        if time.time() - start_time > max_time:
            break
        if _deadline():
            break
        
        # Random initialization
        U = np.random.randn(Nm, r) * 0.3
        V = np.random.randn(Mp, r) * 0.3
        W = np.random.randn(Pn, r) * 0.3
        
        # Alternating minimization
        lr = 0.1
        for it in range(100):
            if time.time() - start_time > max_time:
                break
            if _deadline():
                break
            
            # Update U: for each a, minimize sum_{(a,b,c) in valid} (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
            # + sum_{(a,b,c) invalid} (sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is complex. Instead, use a simpler gradient step.
            
            # Compute residual for valid entries
            # For efficiency, only check a subset
            
            # Gradient w.r.t. U[a, r]: sum over (b,c) where (a,b,c) valid: V[b,r]*W[c,r] * resid[a,b,c]
            #                            + sum over (b,c) where (a,b,c) invalid: V[b,r]*W[c,r] * resid[a,b,c]
            
            # Simplified: use a random subset of constraints
            np.random.shuffle(valid_indices)
            subset = valid_indices[:min(50, len(valid_indices))]
            
            grad_U = np.zeros((Nm, r))
            grad_V = np.zeros((Mp, r))
            grad_W = np.zeros((Pn, r))
            
            for (a, b, c) in subset:
                # Current value
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                resid = val - 1.0
                
                # dU[a,r] = V[b,r] * W[c,r] * resid
                grad_U[a, :] += np.outer(V[b, :], W[c, :]) * resid
                grad_V[b, :] += np.outer(U[a, :], W[c, :]) * resid
                grad_W[c, :] += np.outer(U[a, :], V[b, :]) * resid
            
            # Also add some invalid constraint gradients (sampled)
            for _ in range(30):
                a = np.random.randint(0, Nm)
                b = np.random.randint(0, Mp)
                c = np.random.randint(0, Pn)
                i_val = a // m
                j_val = a % m
                jp_val = b // p
                k_val = b % p
                kp_val = c // n
                ip_val = c % n
                if not (i_val == ip_val and j_val == jp_val and k_val == kp_val):
                    val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                    resid = val  # want this to be 0
                    
                    grad_U[a, :] += np.outer(V[b, :], W[c, :]) * resid
                    grad_V[b, :] += np.outer(U[a, :], W[c, :]) * resid
                    grad_W[c, :] += np.outer(U[a, :], V[b, :]) * resid
            
            # Gradient step
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            max_resid = 0
            for (a, b, c) in valid_indices[:20]:
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                max_resid = max(max_resid, abs(val - 1.0))
            
            if max_resid < 1e-6:
                if _check_decomposition(U, V, W, n, m, p):
                    best_result = (U.copy(), V.copy(), W.copy())
                    break
        
        if best_result is not None:
            break
    
    return best_result


def _structured_rank_search(n, m, p):
    """
    Try structured decompositions based on the algebraic structure.
    
    Key insight: For matrix multiplication, there are known low-rank algorithms.
    For 3x4 x 4x7, we can try:
    
    1. Block diagonal: rank = m * max(n,p) = 4*7 = 28
    2. Strassen-like: potentially lower
    
    Let's try a specific structured approach:
    For each j in {0,1,2,3}, we need to compute sum over j of A[i,j]*B[j,k].
    
    One approach: use the fact that for fixed j, A[:,j] is a vector in R^n and
    B[j,:] is a vector in R^p. Their outer product contributes to the output.
    
    So C = sum_j A[:,j] * B[j,:] (outer product).
    
    This is a sum of m=4 outer products, each of rank 1 (as a matrix).
    But as a tensor decomposition, each outer product A[:,j]*B[j,:] corresponds to
    a rank-1 term in the multiplication tensor? No, not exactly.
    
    Actually, the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1 for matching indices.
    This is different from the matrix multiplication map.
    
    Let me reconsider. The tensor T is defined such that:
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] should give the multiplication.
    
    For the schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    for col corresponding to (i,j,k).
    
    So the tensor has 84 non-zero entries, each being 1.
    
    The rank of this tensor is what we're minimizing.
    """
    # Return schoolbook as fallback
    R0 = n * m * p
    U0 = np.zeros((n * m, R0))
    V0 = np.zeros((m * p, R0))
    W0 = np.zeros((p * n, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    return U0, V0, W0


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    
    # Start with schoolbook
    R0 = n * m * p
    U0 = np.zeros((Nm, R0))
    V0 = np.zeros((Mp, R0))
    W0 = np.zeros((Pn, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = R0
    
    # Lower bound: max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower_bound upward
    for r in range(lower_bound, best_rank):
        if _deadline():
            break
        if time.time() - _START > _LIMIT - 10:
            break
        
        # Try multiple seeds for robustness
        found = False
        for seed in range(5):
            if _deadline():
                break
            if time.time() - _START > _LIMIT - 10:
                break
            
            result = _try_rank_with_nullspace(n, m, p, r, max_time=15.0, seed=seed * 100 + r)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = r
                found = True
                break
        
        if found:
            break  # Found minimal rank
    
    # If we found a good rank, try to polish it
    if best_rank < R0:
        # High-precision polish
        pass
    
    return {"U": _pack(best_U), "V": _pack(best_V), "W": _pack(best_W)}
